"""Stage 2 trainer using Accelerate (multi-GPU)."""

import torch
import torch.nn as nn
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import PreTrainedTokenizer

from accelerate import Accelerator
from accelerate.utils import gather_object

from src.metrics.stage2_metrics import Stage2Metrics


class Stage2Trainer:
    def __init__(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        optimizer,
        tokenizer: PreTrainedTokenizer,
        log_interval: int = 10,
        ckpt_dir: str = "checkpoints",
        grad_clip_norm: Optional[float] = 1.0,
        grad_clip_value: Optional[float] = None,
        early_stopping_patience: int = 5,  # Stop if no improvement for N epochs
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        steps_per_epoch: Optional[int] = None,
        max_gen_tokens: int = 128,
        min_gen_tokens: int = 10,  # NEW: Force at least N tokens before allowing EOS
        num_beams: int = 4,
        label_smoothing: float = 0.0,  # Cross-entropy label smoothing
        warmup_start_factor: float = 0.1,  # Start warmup at this fraction of LR
        warmup_end_factor: float = 1.0,  # End warmup at this fraction of base LR (old Sign2GPT: 0.5)
        cosine_end_factor: float = 0.1,  # End cosine at this fraction of peak LR
        gen_temperature: float = 1.0,
        gen_length_penalty: float = 1.0,
        gate_grad_multiplier: float = 1.0,
        apply_metric_splitter: bool = False,
        append_string: str = "",
        # MoE auxiliary loss weight (applied on top of per-component weights)
        moe_aux_loss_weight: float = 1.0,
    ):
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.log_interval = log_interval
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_value = grad_clip_value
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.max_gen_tokens = max_gen_tokens
        self.min_gen_tokens = min_gen_tokens
        self.num_beams = num_beams
        self.label_smoothing = label_smoothing
        self.warmup_start_factor = warmup_start_factor
        self.warmup_end_factor = warmup_end_factor
        self.cosine_end_factor = cosine_end_factor
        self.gen_temperature = gen_temperature
        self.gen_length_penalty = gen_length_penalty
        self.gate_grad_multiplier = gate_grad_multiplier
        self.apply_metric_splitter = apply_metric_splitter
        self.append_string = append_string
        self.moe_aux_loss_weight = moe_aux_loss_weight

        self._gate_grad_hooks = []
        if self.gate_grad_multiplier is not None and float(self.gate_grad_multiplier) != 1.0:
            mult = float(self.gate_grad_multiplier)
            for name, p in self.model.named_parameters():
                if p.requires_grad and "adaptor_gate" in name:
                    self._gate_grad_hooks.append(p.register_hook(lambda g, m=mult: g * m))
        
        self.metrics = Stage2Metrics()
        self.best_val_loss = float("inf")
        self.best_val_bleu4 = float("-inf")
        self.scheduler = None
        self.early_stopping_patience = early_stopping_patience
        self.epochs_without_improvement = 0
        
        # Create loss function with label smoothing
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        ) if label_smoothing > 0 else None
    
    def build_scheduler(self, train_loader: DataLoader):
        if self.steps_per_epoch is None:
            steps_per_epoch = len(train_loader)
        else:
            steps_per_epoch = self.steps_per_epoch
        
        warmup_steps = self.warmup_epochs * steps_per_epoch
        total_steps = self.max_epochs * steps_per_epoch
        cosine_steps = max(1, total_steps - warmup_steps)
        
        # - Warmup from (warmup_start_factor * base_lr) to (warmup_end_factor * base_lr)
        # - Then cosine decay from peak_lr to (cosine_end_factor * peak_lr)
        #
        # IMPORTANT: CosineAnnealingLR captures optimizer.base_lrs at construction time.
        # If we leave optimizer lr at base_lr, we get an LR "jump" at the warmup→cosine switch.
        # So we set optimizer lr to peak_lr first, and express warmup in terms of peak_lr.
        base_lr = float(self.optimizer.param_groups[0]["lr"])
        peak_lr = base_lr * float(self.warmup_end_factor)
        eta_min = peak_lr * float(self.cosine_end_factor)

        # Set optimizer lr to peak before creating schedulers so cosine base_lrs == peak_lr
        for group in self.optimizer.param_groups:
            group["lr"] = peak_lr

        # Warmup relative to peak_lr: start at (warmup_start_factor/warmup_end_factor) * peak_lr
        # which equals warmup_start_factor * base_lr.
        denom = float(self.warmup_end_factor) if float(self.warmup_end_factor) != 0.0 else 1.0
        start_factor_rel_peak = float(self.warmup_start_factor) / denom
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=start_factor_rel_peak,
            end_factor=1.0,
            total_iters=max(1, warmup_steps),
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_steps,
            eta_min=eta_min,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_bleu4": self.best_val_bleu4,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        
        # Latest
        latest_path = self.ckpt_dir / "stage2_latest.pt"
        self.accelerator.save(state, latest_path)
        
        # Best
        if is_best:
            best_path = self.ckpt_dir / "stage2_best.pt"
            self.accelerator.save(state, best_path)
            if self.accelerator.is_main_process:
                print(f"  ✓ New best checkpoint saved (bleu4={self.best_val_bleu4:.2f}, val_loss={self.best_val_loss:.4f})")
    
    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location="cpu")
        
        # Handle DDP module prefix mismatch
        state_dict = ckpt["model"]
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Check if state_dict has module. prefix but model doesn't expect it (or vice versa)
        model_keys = set(unwrapped_model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        # If checkpoint has module. prefix but model doesn't, strip it
        if any(k.startswith("module.") for k in ckpt_keys) and not any(k.startswith("module.") for k in model_keys):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        # If model has module. prefix but checkpoint doesn't, add it
        elif any(k.startswith("module.") for k in model_keys) and not any(k.startswith("module.") for k in ckpt_keys):
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}

        current_state = unwrapped_model.state_dict()
        backfilled_keys = []
        for key, value in current_state.items():
            if key not in state_dict and key.endswith("shared_expert_gate_logits"):
                state_dict[key] = value
                backfilled_keys.append(key)

        load_result = unwrapped_model.load_state_dict(state_dict, strict=False)
        missing_keys = [k for k in load_result.missing_keys if k not in backfilled_keys]
        unexpected_keys = list(load_result.unexpected_keys)
        if self.accelerator.is_main_process:
            if backfilled_keys:
                print(
                    f"  ⚠ Backfilled {len(backfilled_keys)} shared-expert gate parameter(s) "
                    f"while loading checkpoint: {backfilled_keys[:4]}"
                )
            if missing_keys:
                print(f"  ⚠ Missing model keys on resume (sample): {missing_keys[:5]}")
            if unexpected_keys:
                print(f"  ⚠ Unexpected model keys on resume (sample): {unexpected_keys[:5]}")

        optimizer_loaded = False
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            optimizer_loaded = True
        except ValueError as exc:
            if self.accelerator.is_main_process:
                print(f"  ⚠ Skipping optimizer state load due to parameter mismatch: {exc}")
        if optimizer_loaded and self.scheduler is not None and "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.best_val_bleu4 = ckpt.get("best_val_bleu4", float("-inf"))
        return ckpt.get("epoch", 0)
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """Tokenize text and prepare labels."""
        device = self.accelerator.device
        
        frames = [f.to(device, non_blocking=True) for f in batch["frames"]]
        texts = batch["text"]
        
        # This ensures the model learns to output EOS to stop generation.
        # XGLM tokenizer adds EOS at start (token 2) automatically, but we need EOS at end too.
        eos_token = self.tokenizer.eos_token or ""

        unwrapped = self.accelerator.unwrap_model(self.model)
        pretext = getattr(unwrapped, "pretext", "") or ""
        pretext_length = int(getattr(unwrapped, "pretext_length", 0) or 0)

        texts_with_eos = [pretext + t + eos_token for t in texts]
        
        # Tokenize with add_special_tokens=True (default) to get [EOS, ...tokens..., EOS]
        encoded = self.tokenizer(
            texts_with_eos,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
            add_special_tokens=True,  # Explicit: adds EOS at start (XGLM quirk)
        )
        # This ensures training/inference consistency (generation doesn't start with final EOS)
        input_ids_full = encoded.input_ids.to(device)
        attention_mask_full = encoded.attention_mask.to(device)

        input_ids = input_ids_full[:, :-1]  # Remove final EOS (like old Sign2GPT text_ids)
        attention_mask = attention_mask_full[:, :-1]  # Adjust mask accordingly
        
        # Use tokenizer vocab_size (may differ from model vocab_size)
        vocab_size = getattr(self.tokenizer, 'vocab_size', len(self.tokenizer))
        # Always clamp to ensure all token IDs are valid (defensive programming)
        max_valid_id = vocab_size - 1
        if input_ids.max().item() > max_valid_id:
            if not hasattr(self, '_invalid_input_ids_warned'):
                num_invalid = (input_ids > max_valid_id).sum().item()
                max_input_id = input_ids.max().item()
                print(f"[WARNING] Found {num_invalid} input_ids > vocab_size (max={max_input_id}, vocab_size={vocab_size}). Clamping to valid range.")
                self._invalid_input_ids_warned = True
            # Replace invalid tokens with pad_token_id (or 0)
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            invalid_mask = input_ids > max_valid_id
            input_ids = torch.where(invalid_mask, 
                                   torch.tensor(pad_token_id, device=input_ids.device, dtype=input_ids.dtype), 
                                   input_ids)
        # Always clamp to ensure all values are in valid range [0, vocab_size-1]
        input_ids = torch.clamp(input_ids, min=0, max=max_valid_id)

        # If pretext is empty, XGLM still starts with EOS so pretext_length should be 1.
        if pretext_length <= 0:
            pretext_length = 1
        labels = input_ids_full[:, pretext_length:]
        labels_mask = attention_mask_full[:, pretext_length:]
        labels[labels_mask == 0] = -100
        
        return {
            "frames": frames,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _apply_moe_schedule(self, epoch: int) -> None:
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if not hasattr(unwrapped_model, "update_moe_routing"):
            return

        messages = unwrapped_model.update_moe_routing(epoch)
        if self.accelerator.is_main_process:
            for message in messages:
                print(message)
    
    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, Any]:
        self._apply_moe_schedule(epoch)
        self.model.train()
        total_loss = 0.0
        num_steps = 0
        epoch_start_time = time.time()
        
        nan_count = 0
        for step, batch in enumerate(dataloader):
            with self.accelerator.accumulate(self.model):
                prepared = self._prepare_batch(batch)


                # Print shapes (handles lists and GPU tensors)
                frames_info = f"frames: list of {len(prepared['frames'])} tensors"
                if prepared["frames"]:
                    frames_info += f", shapes={[f.shape for f in prepared['frames']]}"
                #print(frames_info)
                #print(f"input_ids: {prepared['input_ids'].shape} (device={prepared['input_ids'].device})")
                #print(f"attention_mask: {prepared['attention_mask'].shape} (device={prepared['attention_mask'].device})")
                #print(f"labels: {prepared['labels'].shape} (device={prepared['labels'].device})")

                with self.accelerator.autocast():
                    outputs = self.model(
                        frames=prepared["frames"],
                        input_ids=prepared["input_ids"],
                        attention_mask=prepared["attention_mask"],
                        labels=prepared["labels"],
                    )
                    loss = outputs["loss"]

                    aux_loss = outputs.get("aux_loss", None)
                    if aux_loss is not None and self.moe_aux_loss_weight > 0:
                        loss = loss + self.moe_aux_loss_weight * aux_loss

                # Skip NaN losses to prevent gradient corruption
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    if self.accelerator.is_main_process:
                        try:
                            max_frames = max(f.shape[0] for f in prepared["frames"]) if prepared["frames"] else 0
                        except Exception:
                            max_frames = "unknown"
                        print(f"  ⚠️ NaN/Inf loss at step {step+1} (count={nan_count}), max_frames={max_frames}, skipping...", flush=True)
                    self.optimizer.zero_grad()
                    continue
                
                self.accelerator.backward(loss)
                
                # Calculate gradient norm and apply clipping (reuse norm for logging)
                grad_norm = None
                if self.grad_clip_norm is not None or self.grad_clip_value is not None:
                    self.accelerator.unscale_gradients()
                    # clip_grad_norm_ returns the norm before clipping - reuse it for logging
                    if self.grad_clip_norm is not None:
                        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    if self.grad_clip_value is not None:
                        nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip_value)
                elif (step + 1) % self.log_interval == 0 and self.accelerator.is_main_process:
                    # Only calculate gradient norm for logging when not clipping
                    try:
                        self.accelerator.unscale_gradients()
                        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                    except (RuntimeError, AttributeError):
                        # If unscale not available (no mixed precision), calculate norm directly
                        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_steps += 1
            
            if (step + 1) % self.log_interval == 0 and self.accelerator.is_main_process:
                lr = self.optimizer.param_groups[0]["lr"]
                avg = total_loss / num_steps
                elapsed = time.time() - epoch_start_time
                eta = elapsed / (step + 1) * (len(dataloader) - step - 1)
                grad_str = f" | Grad: {grad_norm:.6f}" if grad_norm is not None else ""
                aux_str = ""
                if aux_loss is not None and isinstance(aux_loss, torch.Tensor):
                    aux_str = f" | AuxMoE: {aux_loss.item():.4f}"
                print(f"  Epoch {epoch} Step {step+1}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg: {avg:.4f} | LR: {lr:.2e}{grad_str}{aux_str} | ETA: {eta/60:.1f}min")
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                if hasattr(unwrapped_model, "print_moe_diagnostics"):
                    unwrapped_model.print_moe_diagnostics(prefix="    ")
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / max(1, num_steps)
        
        if self.accelerator.is_main_process:
            print(f"Epoch {epoch} TRAIN | Loss: {avg_loss:.4f} | Time: {epoch_time/60:.1f}min")
        
        return {"loss": avg_loss, "epoch_time": epoch_time}
    
    def validate(self, dataloader: DataLoader, epoch: int, generate: bool = True) -> Dict[str, Any]:
        """Validate."""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        num_steps = 0
        
        # Teacher-forced predictions (old: "obleu"), keep file_name to de-dup accelerator padding
        obleu_triplets: List[tuple] = []  # (file_name, pred_str, ref_str)

        # Autoregressive predictions (old: "ableu"), keep file_name to de-dup accelerator padding
        gen_triplets: List[tuple] = []  # (file_name, pred_str, ref_str)
        gen_count = 0
        
        # Get unwrapped model once for generation
        unwrapped_model = self.accelerator.unwrap_model(self.model) if generate else None
        
        with torch.no_grad():
            for batch in dataloader:
                prepared = self._prepare_batch(batch)
                
                with self.accelerator.autocast():
                    outputs = self.model(
                        frames=prepared["frames"],
                        input_ids=prepared["input_ids"],
                        attention_mask=prepared["attention_mask"],
                        labels=prepared["labels"],
                    )
                    loss = outputs["loss"]
                
                total_loss += loss.item()
                num_steps += 1

                # --- OBLEU: teacher-forced argmax on logits ---
                try:
                    logits = outputs.get("logits", None)
                    labels = prepared.get("labels", None)
                    if logits is not None and labels is not None:
                        mask = labels != -100  # gt_text_mask
                        pred_ids = logits.argmax(dim=-1).detach()
                        eos_id = self.tokenizer.eos_token_id
                        for pred_row, mask_row, ref_text, fid in zip(pred_ids, mask, batch["text"], batch.get("file_name", [None] * pred_ids.size(0))):
                            tok = pred_row[mask_row].detach().cpu()
                            eos_pos = (tok == eos_id).nonzero(as_tuple=False)
                            if eos_pos.numel() > 0:
                                tok = tok[: int(eos_pos[0].item())]
                            pred_str = self.tokenizer.decode(tok, skip_special_tokens=True)
                            if self.apply_metric_splitter:
                                pred_str = " ".join(list(pred_str))
                                ref_str = " ".join(list(ref_text))
                            else:
                                ref_str = ref_text
                            obleu_triplets.append((fid, pred_str + self.append_string, ref_str + self.append_string))
                except Exception as e:
                    if self.accelerator.is_main_process:
                        print(f"  [OBLEU] Error computing teacher-forced predictions: {e}")
                
                # Generate on ALL ranks (parallelized to avoid NCCL timeout)
                if generate:
                    frames = [f.to(self.accelerator.device) for f in batch["frames"]]
                    try:
                        generated = unwrapped_model.generate(
                            frames=frames,
                            max_length=self.max_gen_tokens,
                            min_new_tokens=self.min_gen_tokens,  # Force minimum output
                            num_beams=self.num_beams,
                            temperature=self.gen_temperature,
                            length_penalty=self.gen_length_penalty,
                        )
                        for pred_str, ref_str, fid in zip(generated, batch["text"], batch.get("file_name", [None] * len(generated))):
                            if self.apply_metric_splitter:
                                pred_str = " ".join(list(pred_str))
                                ref_str = " ".join(list(ref_str))
                            pred_str = pred_str + self.append_string
                            ref_str = ref_str + self.append_string
                            gen_triplets.append((fid, pred_str, ref_str))
                        gen_count += len(generated)
                        
                        # Progress logging every 2000 samples (main process only)
                        if self.accelerator.is_main_process and gen_count % 2000 < len(batch["text"]):
                            print(f"  [Val Gen] {gen_count} samples generated on this rank...")
                        
                        del frames
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            if self.accelerator.is_main_process:
                                print(f"  ⚠️ OOM during generation at sample {gen_count}, skipping batch")
                        else:
                            raise e
        
        # Synchronize after validation loop
        self.accelerator.wait_for_everyone()
        
        avg_loss = total_loss / max(1, num_steps)
        
        # Gather predictions from all ranks for BLEU computation
        results = {"val_loss": avg_loss}
        # Always compute OBLEU if we have teacher-forced preds
        if obleu_triplets:
            gathered = gather_object(obleu_triplets)
            if self.accelerator.is_main_process:
                if len(gathered) > 0 and isinstance(gathered[0], (list, tuple)) and len(gathered) > 0 and isinstance(gathered[0][0], (list, tuple)):
                    flat = [x for sublist in gathered for x in sublist]
                else:
                    flat = gathered
                # De-dup by file_name (Accelerate may pad shards → repeats)
                dedup = {}
                for fid, p, r in flat:
                    if fid is None:
                        fid = f"_idx_{len(dedup)}"
                    if fid not in dedup:
                        dedup[fid] = (p, r)
                preds = [v[0] for v in dedup.values()]
                refs = [v[1] for v in dedup.values()]
                self.metrics.update(preds, refs)
                results.update(self.metrics.compute())

        # ABLEU (generation) under a_*
        if generate and gen_triplets:
            gathered = gather_object(gen_triplets)
            if self.accelerator.is_main_process:
                if len(gathered) > 0 and isinstance(gathered[0], (list, tuple)) and len(gathered) > 0 and isinstance(gathered[0][0], (list, tuple)):
                    flat = [x for sublist in gathered for x in sublist]
                else:
                    flat = gathered
                dedup = {}
                for fid, p, r in flat:
                    if fid is None:
                        fid = f"_idx_{len(dedup)}"
                    if fid not in dedup:
                        dedup[fid] = (p, r)
                preds = [v[0] for v in dedup.values()]
                refs = [v[1] for v in dedup.values()]
                print(f"  [Val Gen] Unique samples: {len(dedup)} (raw gathered={len(flat)})")
                tmp = Stage2Metrics()
                tmp.update(preds, refs)
                m2 = tmp.compute()
                results.update({f"a_{k}": v for k, v in m2.items() if k.startswith("bleu")})

                # Track 10 fixed samples across epochs (choose once)
                if not hasattr(self, "_tracked_val_ids"):
                    self._tracked_val_ids = sorted(list(dedup.keys()))[:10]
                print("  [Val Samples] Fixed 10 examples (GT vs Pred):")
                for fid in self._tracked_val_ids:
                    p, r = dedup.get(fid, ("", ""))
                    # Keep printing compact
                    print(f"    - {fid}:")
                    print(f"      GT  : {r[:180]}")
                    print(f"      Pred: {p[:180]}")

        # (generation metrics handled above)
        
        is_best = False
        bleu4 = results.get("bleu4", None)
        if bleu4 is not None:
            if bleu4 > self.best_val_bleu4:
                self.best_val_bleu4 = float(bleu4)
                is_best = True
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
        else:
            # Fallback to loss if BLEU not available
            is_best = avg_loss < self.best_val_loss and not torch.isnan(torch.tensor(avg_loss))
            if is_best:
                self.best_val_loss = avg_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

        # Always keep best loss for reference
        if avg_loss < self.best_val_loss and not torch.isnan(torch.tensor(avg_loss)):
            self.best_val_loss = avg_loss
        results["is_best"] = is_best
        
        # Check early stopping
        should_stop = self.epochs_without_improvement >= self.early_stopping_patience
        results["should_stop"] = should_stop
        
        if self.accelerator.is_main_process:
            msg = f"Epoch {epoch} VAL | Loss: {avg_loss:.4f}"
            if "bleu4" in results:
                msg += f" | BLEU-4: {results['bleu4']:.2f}"
            if "rouge_l" in results:
                msg += f" | ROUGE-L: {results['rouge_l']:.2f}"
            print(msg)
            
            if "bleu1" in results:
                bleu_details = f"  BLEU-1: {results['bleu1']:.2f} | BLEU-2: {results['bleu2']:.2f} | BLEU-3: {results['bleu3']:.2f} | BLEU-4: {results['bleu4']:.2f}"
                print(bleu_details)
            
            if should_stop:
                print(f"  ⚠️ Early stopping: No improvement for {self.early_stopping_patience} epochs")
        
        return results

