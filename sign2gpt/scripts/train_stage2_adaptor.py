#!/usr/bin/env python3
"""
Stage 2 training script using Zero-Gated Cross-Attention Adaptors.
"""

import argparse
import sys
import yaml
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_rs = str(_REPO_ROOT)
if _rs not in sys.path:
    sys.path.insert(0, _rs)

import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

from src.data.lmdb_dataset import ISignDataset
from src.data.transforms import VideoTransform
from src.models.stage2_model_adaptor import Stage2ModelAdaptor
from src.training.stage2_trainer import Stage2Trainer


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phoenix2014t.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Load --resume checkpoint, run one validation pass, and exit",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        choices=["dev", "test"],
        default="dev",
        help="Dataset split to evaluate in --eval_only mode (default: dev)",
    )
    parser.add_argument(
        "--moe_variant",
        type=str,
        choices=["temporal", "fusion"],
        default=None,
        help="Optionally override MoE mode for configs that carry both temporal and fusion schedules.",
    )
    args = parser.parse_args()
    if args.eval_only and not args.resume:
        parser.error("--eval_only requires an explicit --resume checkpoint")
    eval_split = args.eval_split if args.eval_only else "dev"

    cfg = load_config(args.config)
    if args.moe_variant is not None:
        model_cfg = cfg.setdefault("model", {})
        training_cfg = cfg.setdefault("training", {})
        if args.moe_variant == "temporal":
            model_cfg["temporal_moe"] = True
            model_cfg["fusion_moe"] = False
        elif args.moe_variant == "fusion":
            model_cfg["temporal_moe"] = False
            model_cfg["fusion_moe"] = True

        variant_ckpt_dirs = training_cfg.get("moe_variant_ckpt_dirs", {})
        if args.moe_variant in variant_ckpt_dirs:
            training_cfg["ckpt_dir"] = variant_ckpt_dirs[args.moe_variant]

    set_seed(cfg.get("seed", 42))

    # MoE routing can leave some expert params unused in a given step;
    # DDP needs find_unused_parameters=True to handle this.
    model_cfg = cfg.get("model", {})
    use_moe = model_cfg.get("temporal_moe", False) or model_cfg.get("fusion_moe", False)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=use_moe)

    accelerator = Accelerator(
        mixed_precision=cfg["training"].get("mixed_precision", "bf16"),
        gradient_accumulation_steps=cfg["training"].get("grad_accum_steps", 1),
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        print("=" * 80)
        print("Stage 2 Training: Zero-Gated Cross-Attention Adaptors")
        print("=" * 80)
        print(f"Config: {args.config}")
        print(f"GPUs: {accelerator.num_processes}")
        print(f"Mixed precision: {cfg['training'].get('mixed_precision', 'bf16')}")
        if args.eval_only:
            print(f"Eval split: {eval_split}")
        print("=" * 80)

    # Data transforms
    train_tf = VideoTransform(
        training=True,
        img_size=cfg["data"].get("img_size", 224),
        random_shift=cfg["data"].get("random_shift", 4),
        stride=cfg["data"].get("stride", 2),
        max_seq_len=cfg["data"].get("max_frames", 256),
        horizontal_flip=cfg["data"].get("horizontal_flip", False),
        aug_strength=cfg["data"].get("aug_strength", 0.2),
    )
    val_tf = VideoTransform(
        training=False,
        img_size=cfg["data"].get("img_size", 224),
        random_shift=cfg["data"].get("random_shift", 4),
        stride=cfg["data"].get("stride", 2),
        max_seq_len=cfg["data"].get("max_frames", 256),
    )

    # Datasets
    min_trans_len = cfg["data"].get("min_translation_length", 0)
    train_ds = ISignDataset(
        metadata_csv=cfg["data"]["metadata_csv"],
        lmdb_root=cfg["data"]["lmdb_root"],
        split="train",
        transform=train_tf,
        max_frames=cfg["data"].get("max_frames", 256),
        stride=cfg["data"].get("stride", 2),
        random_shift=cfg["data"].get("random_shift", 4),
        pseudo_gloss_dict=None,
        min_translation_length=min_trans_len,
    )
    val_ds = ISignDataset(
        metadata_csv=cfg["data"]["metadata_csv"],
        lmdb_root=cfg["data"]["lmdb_root"],
        split=eval_split,
        transform=val_tf,
        max_frames=cfg["data"].get("max_frames", 256),
        stride=cfg["data"].get("stride", 2),
        random_shift=cfg["data"].get("random_shift", 4),
        pseudo_gloss_dict=None,
        min_translation_length=min_trans_len,
    )

    # Limit samples
    train_limit = cfg["data"].get("train_samples", None)
    val_limit = cfg["data"].get("val_samples", None)
    if train_limit is not None and train_limit > 0 and len(train_ds) > train_limit:
        train_ds = Subset(train_ds, list(range(train_limit)))
        if accelerator.is_main_process:
            print(f"Train limited to {train_limit} samples")
    if val_limit is not None and val_limit > 0 and len(val_ds) > val_limit:
        val_ds = Subset(val_ds, list(range(val_limit)))
        if accelerator.is_main_process:
            print(f"Val limited to {val_limit} samples")


    if accelerator.is_main_process:
        split_label = "Test" if eval_split == "test" else "Val"
        print(f"Train samples: {len(train_ds)}, {split_label} samples: {len(val_ds)} (split={eval_split})")

    # Collate function
    def stage2_collate(batch):
        frames = [sample["frames"] for sample in batch]
        texts = [sample["sentence"] for sample in batch]
        file_names = [sample["file_name"] for sample in batch]
        return {"frames": frames, "text": texts, "file_name": file_names}

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"].get("batch_size", 4),
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 8),
        collate_fn=stage2_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"].get("batch_size", 4),
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 8),
        collate_fn=stage2_collate,
        pin_memory=True,
    )

    # Build model
    model_cfg = cfg["model"]
    vision_config = model_cfg.get("vision_encoder", {})

    if accelerator.is_main_process:
        print("\nBuilding Stage2ModelAdaptor...")
    
    model = Stage2ModelAdaptor(
        vision_config=vision_config,
        temporal_layers=model_cfg.get("temporal_layers", [2, 2]),
        temporal_mlp_ratio=model_cfg.get("temporal_mlp_ratio", 4.0),
        temporal_dropout=model_cfg.get("temporal_dropout", 0.1),
        temporal_drop_path=model_cfg.get("temporal_drop_path", 0.1),
        use_layer_scale=model_cfg.get("use_layer_scale", True),
        use_downsampler=model_cfg.get("use_downsampler", True),
        use_pos_embed=model_cfg.get("use_pos_embed", True),
        projection_dropout=model_cfg.get("projection_dropout", 0.1),
        lm_path=model_cfg.get("lm_path", "/mnt/lustre-grete/projects/intern_agc_emmy/hf_models/facebook__xglm-1.7B"),
        lora_rank=model_cfg.get("lora_rank", 4),
        lora_alpha=model_cfg.get("lora_alpha", 4.0),
        lora_dropout=model_cfg.get("lora_dropout", 0.1),
        adaptor_layers=model_cfg.get("adaptor_layers", None),
        lora_layers=model_cfg.get("lora_layers", None),
        gate_type=model_cfg.get("gate_type", "clamp"),
        label_smoothing=cfg["training"].get("label_smoothing", 0.0),
        w_lora_ff=model_cfg.get("w_lora_ff", False),
        gate_init=model_cfg.get("gate_init", 0.0),
        pretext=model_cfg.get("pretext", ""),
        freeze_vision=model_cfg.get("freeze_vision", False),
        freeze_temporal=model_cfg.get("freeze_temporal", False),
        # MoE Plan A: Temporal MoE
        temporal_moe=model_cfg.get("temporal_moe", False),
        temporal_moe_block_indices=model_cfg.get("temporal_moe_block_indices", None),
        temporal_num_experts=model_cfg.get("temporal_num_experts", 4),
        temporal_top_k=model_cfg.get("temporal_top_k", 1),
        temporal_use_shared_expert=model_cfg.get("temporal_use_shared_expert", True),
        temporal_moe_aux_loss_weight=model_cfg.get("temporal_moe_aux_loss_weight", 0.01),
        temporal_moe_router_noise=model_cfg.get("temporal_moe_router_noise", 0.1),
        temporal_router_temperature=model_cfg.get("temporal_router_temperature", 1.0),
        temporal_moe_schedule=model_cfg.get("temporal_moe_schedule", None),
        temporal_moe_expert_scope=model_cfg.get(
            "temporal_moe_expert_scope", "ffn"
        ),
        temporal_routing_granularity=model_cfg.get(
            "temporal_routing_granularity", "token"
        ),
        temporal_segment_size=model_cfg.get("temporal_segment_size", 8),
        temporal_shared_mix_mode=model_cfg.get("temporal_shared_mix_mode", "average"),
        temporal_shared_expert_gate_init=model_cfg.get(
            "temporal_shared_expert_gate_init", -2.0
        ),
        temporal_expert_dropout=model_cfg.get("temporal_expert_dropout", 0.0),
        temporal_moe_load_balance_type=model_cfg.get(
            "temporal_moe_load_balance_type", "switch"
        ),
        # MoE Plan B: Fusion MoE
        fusion_moe=model_cfg.get("fusion_moe", False),
        fusion_num_experts=model_cfg.get("fusion_num_experts", 4),
        fusion_top_k=model_cfg.get("fusion_top_k", 1),
        fusion_use_shared_expert=model_cfg.get("fusion_use_shared_expert", True),
        fusion_moe_aux_loss_weight=model_cfg.get("fusion_moe_aux_loss_weight", 0.01),
        fusion_moe_router_noise=model_cfg.get("fusion_moe_router_noise", 0.1),
        fusion_router_temperature=model_cfg.get("fusion_router_temperature", 1.0),
        fusion_moe_schedule=model_cfg.get("fusion_moe_schedule", None),
    )

    # Load Stage 1 checkpoint (all ranks must load before accelerator.prepare)
    stage1_ckpt = model_cfg.get("stage1_checkpoint", None)
    if stage1_ckpt and Path(stage1_ckpt).exists():
        if accelerator.is_main_process:
            print(f"\nLoading Stage 1 checkpoint: {stage1_ckpt}")
        model.load_stage1_checkpoint(stage1_ckpt)
    else:
        if accelerator.is_main_process:
            print(f"⚠️ Stage 1 checkpoint not found: {stage1_ckpt}")

    stage2_warmstart_ckpt = model_cfg.get("stage2_warmstart_checkpoint", None)
    if stage2_warmstart_ckpt and Path(stage2_warmstart_ckpt).exists():
        if accelerator.is_main_process:
            print(f"\nLoading Stage 2 warm-start checkpoint: {stage2_warmstart_ckpt}")
        model.load_stage2_warmstart_checkpoint(stage2_warmstart_ckpt)
    elif stage2_warmstart_ckpt:
        if accelerator.is_main_process:
            print(f"⚠️ Stage 2 warm-start checkpoint not found: {stage2_warmstart_ckpt}")

    # Count parameters
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Separate weight-decay and no-decay groups (matching official Sign2GPT)
    decay_params = []
    no_decay_params = []
    no_decay_modules = (torch.nn.Embedding, torch.nn.LayerNorm, torch.nn.BatchNorm1d,
                        torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if pn.endswith("bias") or isinstance(m, no_decay_modules):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
    wd = cfg["training"].get("weight_decay", 0.001)
    optimizer = AdamW([
        {"params": decay_params, "weight_decay": wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg["training"].get("lr", 3e-4))

    # Prepare with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Trainer
    trainer = Stage2Trainer(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        tokenizer=model.module.tokenizer if hasattr(model, 'module') else model.tokenizer,
        log_interval=cfg["training"].get("log_interval", 500),
        ckpt_dir=cfg["training"].get("ckpt_dir", "checkpoints"),
        grad_clip_norm=cfg["training"].get("grad_clip_norm", 1.0),
        grad_clip_value=cfg["training"].get("grad_clip_value", None),
        early_stopping_patience=cfg["training"].get("early_stopping_patience", 15),
        warmup_epochs=cfg["training"].get("warmup_epochs", 5),
        max_epochs=cfg["training"].get("epochs", 100),
        max_gen_tokens=cfg["training"].get("max_gen_tokens", 64),
        min_gen_tokens=cfg["training"].get("min_gen_tokens", 10),  # Force minimum output
        num_beams=cfg["training"].get("num_beams", 4),
        label_smoothing=cfg["training"].get("label_smoothing", 0.1),
        warmup_start_factor=cfg["training"].get("warmup_start_factor", 0.1),
        warmup_end_factor=cfg["training"].get("warmup_end_factor", 1.0),
        cosine_end_factor=cfg["training"].get("cosine_end_factor", 0.2),
        gen_temperature=cfg["training"].get("gen_temperature", 1.0),
        gen_length_penalty=cfg["training"].get("gen_length_penalty", 1.0),
        # Prefer training.* (new configs), but keep backward compatibility with root-level keys.
        gate_grad_multiplier=cfg["training"].get(
            "gate_grad_multiplier",
            cfg.get("gate_grad_multiplier", model_cfg.get("gate_grad_multiplier", 1.0)),
        ),
        apply_metric_splitter=cfg["training"].get("apply_metric_splitter", cfg.get("apply_metric_splitter", False)),
        append_string=cfg["training"].get("append_string", cfg.get("append_string", "")),
        moe_aux_loss_weight=cfg["training"].get("moe_aux_loss_weight", 1.0),
    )

    trainer.build_scheduler(train_loader)

    # Resume
    start_epoch = 1
    if args.resume:
        if accelerator.is_main_process:
            print(f"Resuming from: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume) + 1
    elif cfg["training"].get("resume", False):
        latest = Path(cfg["training"].get("ckpt_dir", "checkpoints")) / "stage2_latest.pt"
        if latest.exists():
            if accelerator.is_main_process:
                print(f"Auto-resuming from: {latest}")
            start_epoch = trainer.load_checkpoint(str(latest)) + 1

    if args.eval_only:
        checkpoint_epoch = start_epoch - 1
        if accelerator.is_main_process:
            print(f"\nEvaluating checkpoint from epoch {checkpoint_epoch} on split={eval_split}")
        trainer._apply_moe_schedule(max(checkpoint_epoch, 1))
        trainer.validate(val_loader, checkpoint_epoch, generate=True)
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "print_moe_diagnostics"):
                unwrapped_model.print_moe_diagnostics(
                    prefix=f"  Epoch {checkpoint_epoch} [{eval_split}]: "
                )
            print(f"\nEvaluation complete! (split={eval_split})")
        return

    # Training loop
    num_epochs = cfg["training"].get("epochs", 100)
    eval_every = cfg["training"].get("eval_every", 1)
    generate_during_val = cfg["training"].get("generate_during_val", False)
    save_ckpt = cfg["training"].get("save_ckpt", True)

    if accelerator.is_main_process:
        print(f"\nStarting training from epoch {start_epoch}")
        print(f"Total epochs: {num_epochs}")
        print(f"Eval every: {eval_every} epochs")
        print(f"Architecture: Zero-Gated Cross-Attention Adaptors")

    for epoch in range(start_epoch, num_epochs + 1):
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")

        epoch_start = time.time()
        train_metrics = trainer.train_one_epoch(train_loader, epoch)
        epoch_time = time.time() - epoch_start

        if accelerator.is_main_process:
            print(f"Epoch {epoch} completed in {epoch_time/60:.1f} min")

        # Validation
        if epoch % eval_every == 0:
            val_metrics = trainer.validate(val_loader, epoch, generate=generate_during_val)
            
            # Print adaptor gate diagnostics (monitoring if LM is "listening" to video)
            # Healthy: gates should reach 0.2-0.4 after training
            # Warning: if gates stay < 0.02 after 10 epochs, LM not using visual info
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                if hasattr(unwrapped_model, 'print_gate_diagnostics'):
                    unwrapped_model.print_gate_diagnostics(prefix=f"  Epoch {epoch}: ")
                if hasattr(unwrapped_model, "print_moe_diagnostics"):
                    unwrapped_model.print_moe_diagnostics(prefix=f"  Epoch {epoch}: ")

            if save_ckpt:
                trainer.save_checkpoint(epoch, is_best=val_metrics.get("is_best", False))
            
            if val_metrics.get("should_stop", False):
                if accelerator.is_main_process:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                break

    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()

