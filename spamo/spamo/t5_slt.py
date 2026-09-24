import json
import os
import torch
import torch.nn as nn
import random
import math
from typing import Dict, List, Optional, Tuple, Any

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import BertConfig, BertModel
from peft import LoraConfig, get_peft_model, TaskType

from spamo.tconv import SparseMoETemporalConv, TemporalConv
from utils.helpers import create_mask, derangement
from spamo.mm_projector import build_vision_projector
from utils.evaluate import evaluate_results
from spamo.clip_loss import clip_loss
from spamo.asb import AbstractSLT
from transformers import get_cosine_schedule_with_warmup


os.environ["TOKENIZERS_PARALLELISM"] = "false"


torch.set_float32_matmul_precision('high')


class FlanT5SLT(AbstractSLT):
    """
    FlanT5-based Sign Language Translation model with multimodal capabilities.
    """
    def __init__(
        self, 
        tuning_type: str = 'lora', 
        model_name: Optional[str] = None, 
        frame_sample_rate: int = 1, 
        prompt: str = '',
        input_size: int = 1024,
        fusion_mode: str = 'joint',
        inter_hidden: int = 768,
        max_frame_len: int = 1024,
        max_txt_len: int = 64,
        cross_modal_align: bool = False,
        warm_up_steps: Optional[int] = None,
        combined_loss: bool = False,
        alpha: float = 0.1,
        use_resampler: bool = False,
        sampling_length: int = 64,
        cache_dir: str = "/data3/models",
        use_in_context: bool = False,
        num_in_context: int = 0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_temporal_moe: bool = False,
        temporal_num_experts: int = 4,
        temporal_top_k: int = 1,
        temporal_use_shared_expert: bool = True,
        temporal_moe_aux_loss_weight: float = 0.01,
        temporal_moe_router_noise: float = 0.1,
        temporal_router_temperature: float = 1.0,
        temporal_routing_granularity: str = "token",
        temporal_segment_size: int = 8,
        temporal_moe_per_modality: bool = False,
        temporal_spatial_num_experts: Optional[int] = None,
        temporal_motion_num_experts: Optional[int] = None,
        temporal_spatial_top_k: Optional[int] = None,
        temporal_motion_top_k: Optional[int] = None,
        temporal_shared_expert_gate_init: float = -2.0,
        temporal_router_z_loss_weight: float = 0.0,
        temporal_router_entropy_loss_weight: float = 0.0,
        temporal_moe_load_balance_type: str = "switch",
        temporal_shared_mix_mode: str = "average",
        temporal_expert_dropout: float = 0.0,
        temporal_router_noise_anneal_frac: float = 0.0,
        temporal_router_lr: Optional[float] = None,
        temporal_upcycle_noise: float = 0.0,
        eval_do_sample: bool = True,
        bleu_tokenizer: str = "13a",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Configuration parameters
        self.input_size = input_size
        self.prompt = prompt
        self.model_name = model_name
        self.frame_sample_rate = frame_sample_rate
        self.fusion_mode = fusion_mode
        self.inter_hidden = inter_hidden
        self.max_frame_len = max_frame_len
        self.max_txt_len = max_txt_len
        self.tuning_type = tuning_type
        self.cross_modal_align = cross_modal_align
        self.warm_up_steps = warm_up_steps
        self.combined_loss = combined_loss
        self.alpha = alpha
        self.use_resampler = use_resampler
        self.sampling_length = sampling_length
        self.cache_dir = cache_dir
        self.use_in_context = use_in_context
        self.num_in_context = num_in_context
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_temporal_moe = use_temporal_moe
        self.temporal_num_experts = temporal_num_experts
        self.temporal_top_k = temporal_top_k
        self.temporal_use_shared_expert = temporal_use_shared_expert
        self.temporal_moe_aux_loss_weight = temporal_moe_aux_loss_weight
        self.temporal_moe_router_noise = temporal_moe_router_noise
        self.temporal_router_temperature = temporal_router_temperature
        granularity = str(temporal_routing_granularity).lower()
        if granularity not in {"token", "segment", "sequence"}:
            raise ValueError(
                "temporal_routing_granularity must be 'token', 'segment', or "
                f"'sequence'; got {temporal_routing_granularity!r}"
            )
        self.temporal_routing_granularity = granularity
        self.temporal_segment_size = max(1, int(temporal_segment_size))
        self.temporal_moe_per_modality = bool(temporal_moe_per_modality)
        self.temporal_spatial_num_experts = (
            temporal_num_experts
            if temporal_spatial_num_experts is None
            else int(temporal_spatial_num_experts)
        )
        self.temporal_motion_num_experts = (
            max(1, temporal_num_experts // 3)
            if temporal_motion_num_experts is None
            else int(temporal_motion_num_experts)
        )
        self.temporal_spatial_top_k = (
            temporal_top_k
            if temporal_spatial_top_k is None
            else int(temporal_spatial_top_k)
        )
        self.temporal_motion_top_k = (
            temporal_top_k
            if temporal_motion_top_k is None
            else int(temporal_motion_top_k)
        )
        self.temporal_shared_expert_gate_init = float(temporal_shared_expert_gate_init)
        self.temporal_router_z_loss_weight = max(float(temporal_router_z_loss_weight), 0.0)
        self.temporal_router_entropy_loss_weight = max(
            float(temporal_router_entropy_loss_weight), 0.0
        )
        self.temporal_moe_load_balance_type = str(
            temporal_moe_load_balance_type
        ).lower()
        self.temporal_shared_mix_mode = str(temporal_shared_mix_mode).lower()
        self.temporal_expert_dropout = min(max(float(temporal_expert_dropout), 0.0), 1.0)
        self.temporal_router_noise_anneal_frac = max(
            float(temporal_router_noise_anneal_frac), 0.0
        )
        self.temporal_router_lr = (
            None if temporal_router_lr is None else float(temporal_router_lr)
        )
        self.temporal_upcycle_noise = max(float(temporal_upcycle_noise), 0.0)
        self.eval_do_sample = bool(eval_do_sample)
        self.bleu_tokenizer = bleu_tokenizer
        
        self.prepare_models(model_name)

        # Apply the selected tuning strategy
        if tuning_type == 'freeze':
            self._freeze_model()
        elif tuning_type == 'lora':
            self._apply_lora()

        self.set_container()
        
    # def load_pretrained_weights(self, checkpoint_path: str) -> None:
    #     """Load weights from a pretrained checkpoint."""
    #     checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
    #     # Get model's state dict
    #     model_state_dict = self.state_dict()
    #     checkpoint_state_dict = checkpoint['state_dict']
        
    #     # Filter out mismatched keys
    #     filtered_state_dict = {}
    #     for k, v in checkpoint_state_dict.items():
    #         if k in model_state_dict and v.size() == model_state_dict[k].size():
    #             filtered_state_dict[k] = v
        
    #     # Load the filtered state dict
    #     self.load_state_dict(filtered_state_dict)
    #     print(f'Checkpoint loaded from {checkpoint_path}. Loaded {len(filtered_state_dict)}/{len(checkpoint_state_dict)} parameters.')
    
    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        src = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        dst = self.state_dict()
        loaded = {}
        skipped_shape = 0
        for key, value in src.items():
            if key in dst and dst[key].shape == value.shape:
                loaded[key] = value
            elif key in dst:
                skipped_shape += 1
        incompatible = self.load_state_dict(loaded, strict=False)
        n_upcycle = 0
        if self.use_temporal_moe:
            n_upcycle = self._upcycle_temporal_conv(src)
        print(
            f"Checkpoint loaded from {checkpoint_path}. "
            f"matched={len(loaded)}, skipped_shape={skipped_shape}, "
            f"upcycled_tensors={n_upcycle}, missing={len(incompatible.missing_keys)}."
        )

    def _src_temporal_conv_state(self, src: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prefix = "temporal_encoder.temporal_conv."
        return {
            f"temporal_conv.{key[len(prefix):]}": value
            for key, value in src.items()
            if key.startswith(prefix)
        }

    def _copy_temporal_conv(
        self,
        module: nn.Module,
        tc_state: Dict[str, torch.Tensor],
        noise_std: float,
        generator: torch.Generator,
    ) -> int:
        copied = 0
        for name, param in module.named_parameters():
            if name not in tc_state or tc_state[name].shape != param.shape:
                continue
            data = tc_state[name].to(dtype=param.dtype, device=param.device).clone()
            if noise_std > 0:
                noise = torch.randn(data.shape, generator=generator, dtype=torch.float32)
                data = data + noise_std * noise.to(dtype=data.dtype, device=data.device)
            param.data.copy_(data)
            copied += 1
        for name, buf in module.named_buffers():
            if name not in tc_state or tc_state[name].shape != buf.shape:
                continue
            buf.data.copy_(tc_state[name].to(dtype=buf.dtype, device=buf.device))
            copied += 1
        return copied

    def _upcycle_temporal_conv(self, src: Dict[str, torch.Tensor]) -> int:
        tc_state = self._src_temporal_conv_state(src)
        if not tc_state:
            return 0
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        copied = 0
        for encoder in self._iter_temporal_moe():
            if encoder.use_shared_expert:
                copied += self._copy_temporal_conv(
                    encoder.shared_expert, tc_state, 0.0, generator
                )
            for expert in encoder.experts:
                copied += self._copy_temporal_conv(
                    expert, tc_state, self.temporal_upcycle_noise, generator
                )
        print(
            f"Upcycled baseline TemporalConv into {copied} MoE expert tensors "
            f"(expert noise={self.temporal_upcycle_noise})."
        )
        return copied

    def _iter_temporal_moe(self):
        if not self.use_temporal_moe:
            return
        if self.temporal_moe_per_modality:
            yield self.spatial_temporal_encoder
            yield self.motion_temporal_encoder
        else:
            yield self.temporal_encoder

    def _make_temporal_moe(self, num_experts: int, top_k: int) -> SparseMoETemporalConv:
        return SparseMoETemporalConv(
            self.inter_hidden,
            self.inter_hidden,
            num_experts=num_experts,
            top_k=top_k,
            router_noise=self.temporal_moe_router_noise,
            router_temperature=self.temporal_router_temperature,
            use_shared_expert=self.temporal_use_shared_expert,
            shared_expert_gate_init=self.temporal_shared_expert_gate_init,
            routing_granularity=self.temporal_routing_granularity,
            segment_size=self.temporal_segment_size,
            router_z_loss_weight=self.temporal_router_z_loss_weight,
            expert_dropout=self.temporal_expert_dropout,
            load_balance_type=self.temporal_moe_load_balance_type,
            shared_mix_mode=self.temporal_shared_mix_mode,
        )

    def _apply_lora(self) -> None:
        """Apply LoRA adapter to the T5 model."""
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q", "v"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.t5_model = get_peft_model(self.t5_model, lora_config)
        print("LoRA adapter applied to T5 model.")

    def _freeze_model(self) -> None:
        """Freeze the T5 model parameters."""
        self.t5_model.eval()
        for params in self.t5_model.parameters():
            params.requires_grad = False
        print("T5 model frozen.")

    def set_container(self) -> None:
        self.generated = []
        self.references = []

    def prepare_models(self, t5_model: str) -> None:
        """Prepare the textual and visual models."""
        
        # Load the textual model (Auto resolves mT5/mT0 as well as T5)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(
            t5_model, 
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16, 
        )
        
        # Load the tokenizer
        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            t5_model, 
            cache_dir=self.cache_dir,
            max_length=self.max_txt_len,
        )

        # Load the vision projectors
        self.spatio_proj = build_vision_projector('linear', self.input_size, self.inter_hidden)
        self.spatiotemp_proj = build_vision_projector('linear', 1024, self.inter_hidden)
        self.fusion_proj = build_vision_projector('mlp2x_gelu', self.inter_hidden, self.t5_model.config.hidden_size)
        
        # Load the temporal encoder
        if self.use_temporal_moe:
            if self.temporal_moe_per_modality:
                self.spatial_temporal_encoder = self._make_temporal_moe(
                    self.temporal_spatial_num_experts,
                    self.temporal_spatial_top_k,
                )
                self.motion_temporal_encoder = self._make_temporal_moe(
                    self.temporal_motion_num_experts,
                    self.temporal_motion_top_k,
                )
                routing_msg = (
                    f"routing={self.temporal_routing_granularity}"
                    + (
                        f", segment_size={self.temporal_segment_size}"
                        if self.temporal_routing_granularity == "segment"
                        else ""
                    )
                )
                print(
                    "Per-modality sparse temporal MoE applied: "
                    f"spatial experts={self.temporal_spatial_num_experts}, "
                    f"top_k={self.temporal_spatial_top_k}; "
                    f"motion experts={self.temporal_motion_num_experts}, "
                    f"top_k={self.temporal_motion_top_k}; {routing_msg}"
                )
            else:
                self.temporal_encoder = self._make_temporal_moe(
                    self.temporal_num_experts,
                    self.temporal_top_k,
                )
                routing_msg = (
                    f"routing={self.temporal_routing_granularity}"
                    + (
                        f", segment_size={self.temporal_segment_size}"
                        if self.temporal_routing_granularity == "segment"
                        else ""
                    )
                )
                print(
                    "Sparse temporal MoE applied: "
                    f"experts={self.temporal_num_experts}, top_k={self.temporal_top_k}, "
                    f"{routing_msg}"
                )
        else:
            self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)

        # if self.cross_modal_align:
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def prepare_inputs(
        self, 
        visual_outputs: torch.Tensor, 
        visual_mask: torch.Tensor, 
        samples: Dict, 
        split: str, 
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        """Prepare combined inputs for the T5 model."""
        bs = visual_outputs.shape[0]
        
        # Prepare the prompt with language information
        prompts = [f'{self.prompt}'] * bs
        prompts = [p.format(l) for p, l in zip(prompts, samples['lang'])]
        
        if self.use_in_context:
            prompts = [f"{p} {c}" for p, c in zip(prompts, samples['ex_lang_trans'])]
        
        # Tokenize prompts
        input_tokens = self.t5_tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Get lengths for visual and prompt sequences
        visual_lengths = visual_mask.sum(1)
        prompt_lengths = input_tokens.attention_mask.sum(1)
        new_lengths = visual_lengths + prompt_lengths
        
        # Convert tokens to embeddings
        input_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        
        # Concatenate visual and text embeddings
        joint_outputs = []
        for i in range(bs):
            vis_out = visual_outputs[i, :visual_lengths[i], :]
            prompt_embeds = input_embeds[i, :prompt_lengths[i], :]
            concat_sample = torch.cat((vis_out, prompt_embeds), dim=0)
            joint_outputs.append(concat_sample)
        
        # Pad the combined embeddings
        joint_outputs = pad_sequence(joint_outputs, batch_first=True)
        joint_mask = create_mask(seq_lengths=new_lengths.tolist(), device=self.device)
        
        # Tokenize target texts
        output_tokens = self.t5_tokenizer(
            samples['text'],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)
        
        # Prepare target labels (replace pad tokens with -100)
        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )
        
        return joint_outputs, joint_mask, output_tokens, targets

    def prepare_visual_inputs(self, samples: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare visual inputs based on the fusion mode."""
        # Determine which visual features to use based on fusion mode
        if self.fusion_mode in ['joint']:
            spatial = spatiotemporal = True
        else:
            spatial = self.fusion_mode == 'spatial'
            spatiotemporal = self.fusion_mode == 'spatiotemporal'

        # Process spatial features if needed
        if spatial:
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)
            spatial_outputs = self.spatio_proj(pixel_values)
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)
        
        # Process spatiotemporal features if needed
        if spatiotemporal:
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)
        
        # Combine features for joint mode
        if self.fusion_mode == 'joint':
            bs = spatial_outputs.shape[0]
            spatial_length = spatial_mask.sum(1)
            spatiotemporal_length = spatiotemporal_mask.sum(1)
            if self.use_temporal_moe and self.temporal_moe_per_modality:
                spatial_conv = self.spatial_temporal_encoder(
                    spatial_outputs.permute(0, 2, 1),
                    spatial_length.to(dtype=torch.long),
                )
                motion_conv = self.motion_temporal_encoder(
                    spatiotemporal_outputs.permute(0, 2, 1),
                    spatiotemporal_length.to(dtype=torch.long),
                )
                spatial_feat = spatial_conv["visual_feat"].permute(1, 0, 2)
                motion_feat = motion_conv["visual_feat"].permute(1, 0, 2)
                spatial_out_len = spatial_conv["feat_len"].to(torch.int).tolist()
                motion_out_len = motion_conv["feat_len"].to(torch.int).tolist()
                joint_outputs = []
                for i in range(bs):
                    joint_outputs.append(
                        torch.cat(
                            (
                                spatial_feat[i, : int(spatial_out_len[i])],
                                motion_feat[i, : int(motion_out_len[i])],
                            ),
                            dim=0,
                        )
                    )
                visual_outputs = pad_sequence(joint_outputs, batch_first=True)
                visual_masks = create_mask(
                    seq_lengths=[
                        int(spatial_out_len[i]) + int(motion_out_len[i])
                        for i in range(bs)
                    ],
                    device=self.device,
                )
            else:
                new_length = spatial_length + spatiotemporal_length

                # Concatenate spatial and spatiotemporal features for each sample
                joint_outputs = []
                for i in range(bs):
                    valid_spatial_output = spatial_outputs[i, :spatial_length[i], :]
                    valid_spatiotemporal_output = spatiotemporal_outputs[i, :spatiotemporal_length[i], :]
                    concat_sample = torch.cat((valid_spatial_output, valid_spatiotemporal_output), dim=0)
                    joint_outputs.append(concat_sample)
                joint_outputs = pad_sequence(joint_outputs, batch_first=True)

                # Apply temporal encoder
                visual_conv_outputs = self.temporal_encoder(
                    joint_outputs.permute(0,2,1), torch.tensor(new_length.tolist(), device=self.device)
                )

                visual_outputs = visual_conv_outputs['visual_feat'].permute(1,0,2)
                visual_masks = create_mask(
                    seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(),
                    device=self.device
                ) 
        else:
            # Use single feature type
            if spatial:
                spatial_conv_outputs = self.temporal_encoder(
                    spatial_outputs.permute(0,2,1), torch.tensor(samples['num_frames'], device=self.device)
                )
                visual_outputs = spatial_conv_outputs['visual_feat'].permute(1,0,2)
                visual_masks = create_mask(
                    seq_lengths=spatial_conv_outputs['feat_len'].to(torch.int).tolist(), 
                    device=self.device
                )
            elif spatiotemporal:
                visual_outputs = spatiotemporal_outputs
                visual_masks = spatiotemporal_mask
            else:
                raise NotImplementedError("Invalid fusion mode")
        
        return visual_outputs, visual_masks

    def get_inputs(self, batch: List) -> Dict:
        """Process batch inputs into a structured dictionary."""
        pixel_values, glor_values, masks, ids = [], [], [], []
        texts, glosses = [], []
        num_frames, glor_lengths, langs = [], [], []
        ex_lang_translations = []
        
        max_frame_len = self.max_frame_len

        for sample in batch:
            if sample['pixel_value'].shape[0] != 0:
                # Calculate number of frames after sampling
                nframe = math.ceil(sample['num_frames'] / self.frame_sample_rate)
                pval = sample['pixel_value'][::self.frame_sample_rate]

                # Collect metadata
                ids.append(sample['id'])
                texts.append(sample['text'].lower())
                glosses.append(sample['gloss'])
                langs.append(sample['lang'])

                if self.use_in_context:
                    _ex_lang_trans = []
                    for key in ('en_text', 'fr_text', 'es_text'):
                        if sample.get(key):
                            _ex_lang_trans.append(f"{sample[key]}={sample['text']}")
                    _ex_lang_trans = _ex_lang_trans[:self.num_in_context]
                    ex_lang_translations.append(' '.join(_ex_lang_trans))
                
                # Handle too long sequences with random cropping
                if nframe > max_frame_len:
                    nframe = max_frame_len
                    start_index = random.randint(0, pval.size(0) - max_frame_len)
                    pval = pval[start_index:start_index + max_frame_len]
                
                # Store processed visual features
                num_frames.append(nframe)
                pixel_values.append(pval)
                
                # Process glor values if available
                if sample['glor_value'] is not None:
                    if isinstance(sample['glor_value'], list):
                        glor_values.append(torch.cat(sample['glor_value'], dim=0))
                        glor_lengths.append(sum(len(g) for g in sample['glor_value']))
                    else:
                        glor_values.append(sample['glor_value'])
                        glor_lengths.append(len(sample['glor_value']))
        
        if self.use_in_context and len(ex_lang_translations) > 1:
            ex_lang_translations = derangement(ex_lang_translations)
        
        # Return structured dictionary
        return {
            'pixel_values': pixel_values,
            'glor_values': glor_values,
            'bool_mask_pos': masks,
            'ids': ids,
            'text': texts,
            'ex_lang_trans': ex_lang_translations,
            'gloss': glosses,
            'lang': langs,
            'num_frames': num_frames,
            'glor_lengths': glor_lengths,
        }

    def visual_textual_align(self, visual_outputs: torch.Tensor, visual_masks: torch.Tensor, samples: Dict) -> torch.Tensor:
        """Calculate visual-textual alignment loss."""
        # Tokenize target texts
        output_tokens = self.t5_tokenizer(
            samples['text'],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)
        
        # Get text embeddings
        text_embeds = self.t5_model.encoder.embed_tokens(output_tokens.input_ids)
        
        # Mean pooling for visual and text embeddings
        image_embeds = visual_outputs.mean(1)  # global pooling
        text_embeds = text_embeds.mean(1)  # global pooling
        
        # Normalize features
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Calculate cosine similarities with temperature scaling
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        # Calculate contrastive loss
        loss = clip_loss(logits_per_text)
        
        return loss

    def shared_step(self, inputs: Dict, split: str, batch_idx: int) -> Tuple[torch.Tensor, Dict]:
        """Shared logic for training, validation and testing steps."""
        # Prepare visual inputs and project to match text embedding dimensions
        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)
        visual_outputs = self.fusion_proj(visual_outputs)
        
        # Initialize logging dictionary
        log_dict = {}
        
        # STEP 1: Determine training mode and prepare inputs accordingly
        if self.cross_modal_align:
            # For pure contrastive learning or warm-up phase
            if self.warm_up_steps is None and not self.combined_loss:
                # Pure contrastive learning mode
                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )
                
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                log_dict[f"{split}/contra_loss"] = cont_loss
                loss = cont_loss
                
            elif self.warm_up_steps is not None and self.global_step <= self.warm_up_steps:
                # Warm-up phase with contrastive learning
                with torch.no_grad():
                    input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                        visual_outputs, visual_masks, inputs, split, batch_idx
                    )
                
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                log_dict[f"{split}/contra_loss"] = cont_loss
                loss = cont_loss
                
            else:
                # Combined loss mode (regular training + contrastive)
                input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                    visual_outputs, visual_masks, inputs, split, batch_idx
                )
                
                # Forward pass through T5 model
                outputs = self.t5_model(
                    inputs_embeds=input_embeds,
                    attention_mask=input_masks,
                    decoder_attention_mask=output_tokens.attention_mask,
                    labels=targets,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                t5_loss = outputs.loss
                log_dict[f"{split}/loss"] = t5_loss
                
                # Add contrastive component if using combined loss
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                loss = t5_loss + self.alpha * cont_loss
                
                log_dict[f"{split}/contra_loss"] = cont_loss
                log_dict[f"{split}/combined_loss"] = loss
        else:
            # Standard training without contrastive learning
            input_embeds, input_masks, output_tokens, targets = self.prepare_inputs(
                visual_outputs, visual_masks, inputs, split, batch_idx
            )
            
            # Forward pass through T5 model
            outputs = self.t5_model(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                decoder_attention_mask=output_tokens.attention_mask,
                labels=targets,
                output_hidden_states=True,
                return_dict=True
            )
            
            loss = outputs.loss
            log_dict[f"{split}/loss"] = loss

        moe_aux_loss = None
        moe_z_loss = None
        moe_entropy_loss = None
        if self.use_temporal_moe:
            if split == "train" and self.temporal_router_noise_anneal_frac > 0:
                for encoder in self._iter_temporal_moe():
                    encoder.anneal_router_noise(self.global_step)
            aux_terms = []
            z_terms = []
            entropy_terms = []
            for encoder in self._iter_temporal_moe():
                aux = encoder.get_aux_loss()
                if aux is not None:
                    aux_terms.append(aux)
                z_loss = getattr(encoder, "last_z_loss", None)
                if z_loss is not None:
                    z_terms.append(z_loss)
                entropy_loss = getattr(encoder, "last_entropy_loss", None)
                if entropy_loss is not None:
                    entropy_terms.append(entropy_loss)
            if aux_terms:
                moe_aux_loss = sum(aux_terms)
                log_dict[f"{split}/temporal_moe_aux_loss"] = moe_aux_loss
                if split == "train" and self.temporal_moe_aux_loss_weight > 0:
                    loss = loss + self.temporal_moe_aux_loss_weight * moe_aux_loss
                    log_dict[f"{split}/loss_with_temporal_moe"] = loss
            if z_terms:
                moe_z_loss = sum(z_terms)
                log_dict[f"{split}/temporal_moe_z_loss"] = moe_z_loss
                if split == "train" and self.temporal_router_z_loss_weight > 0:
                    loss = loss + self.temporal_router_z_loss_weight * moe_z_loss
                    log_dict[f"{split}/loss_with_temporal_moe"] = loss
            if entropy_terms:
                moe_entropy_loss = sum(entropy_terms)
                log_dict[f"{split}/temporal_moe_entropy_loss"] = moe_entropy_loss
                if (
                    split == "train"
                    and self.temporal_router_entropy_loss_weight > 0
                ):
                    loss = (
                        loss
                        + self.temporal_router_entropy_loss_weight
                        * moe_entropy_loss
                    )
                    log_dict[f"{split}/loss_with_temporal_moe"] = loss

        # STEP 2: Handle evaluation phase (validation/testing)
        if split != "train":
            # Prepare inputs for text generation
            input_embeds, input_masks, _, _ = self.prepare_inputs(
                visual_outputs, visual_masks, inputs, split, batch_idx
            )
            
            # Generate translations
            generate_kwargs = {
                "inputs_embeds": input_embeds,
                "attention_mask": input_masks,
                "num_beams": self.beam_size,
                "max_length": self.max_txt_len,
                "do_sample": self.eval_do_sample,
            }
            if self.eval_do_sample:
                generate_kwargs["top_p"] = 0.9
            generated = self.t5_model.generate(**generate_kwargs)
            
            # Decode generated outputs and references
            generated_strings = self.t5_tokenizer.batch_decode(generated, skip_special_tokens=True)
            generated_strings = [gen.lower() for gen in generated_strings]
            
            reference_strings = self.t5_tokenizer.batch_decode(output_tokens.input_ids, skip_special_tokens=True)
            reference_strings = [ref.lower() for ref in reference_strings]

            self.generated.extend(generated_strings)
            self.references.extend(reference_strings)
            
            # Calculate evaluation metrics
            # eval_res = evaluate_results(
            #     predictions=generated_strings,
            #     references=reference_strings,
            #     split=split,
            #     tokenizer='zh' if inputs['lang'][0] == 'Chinese' else '13a',
            #     device=self.device
            # )
            
            # Add evaluation results to logging
            # log_dict.update(eval_res)

        return loss, log_dict

    def on_validation_epoch_end(self) -> None:
        # Print some examples of generated translations and references with colors
        print("\n===== Validation Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m")  # Blue color for references
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")    # Green color for generated
            print("-" * 50)
            
        # Calculate evaluation metrics
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='val',
            tokenizer=self.bleu_tokenizer,
            device=self.device
        )
        
        # Add evaluation results to logging
        # log_dict.update(eval_res)

        self.log_dict(eval_res, sync_dist=True)

        self.set_container()

    def on_test_epoch_end(self) -> None:
        # Print some examples of generated translations and references with colors
        print("\n===== Validation Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m")  # Blue color for references
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")    # Green color for generated
            print("-" * 50)
            
        # Calculate evaluation metrics
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='test',
            tokenizer=self.bleu_tokenizer,
            device=self.device
        )

        self.log_dict(eval_res, sync_dist=True)
        logdir = getattr(self.trainer, "log_dir", None) or os.environ.get("TMPDIR") or "."
        try:
            os.makedirs(logdir, exist_ok=True)
            out_path = os.path.join(logdir, "test_generations.json")
            with open(out_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "predictions": self.generated,
                        "references": self.references,
                    },
                    handle,
                    ensure_ascii=False,
                )
        except OSError as exc:
            print(f"Could not save test generations: {exc}")
        self.set_container()

    def on_train_start(self) -> None:
        if not self.use_temporal_moe or self.temporal_router_noise_anneal_frac <= 0:
            return
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if not total_steps:
            return
        anneal_steps = int(float(total_steps) * self.temporal_router_noise_anneal_frac)
        for encoder in self._iter_temporal_moe():
            encoder.router_noise_anneal_steps = anneal_steps
        print(f"Temporal MoE router noise anneal steps={anneal_steps}.")

    def configure_optimizers(self):
        if self.use_temporal_moe and self.temporal_router_lr is not None:
            router_params = []
            other_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if ".router." in name:
                    router_params.append(param)
                else:
                    other_params.append(param)
            param_groups = [
                {"params": other_params, "lr": self.lr},
                {"params": router_params, "lr": self.temporal_router_lr},
            ]
            print(
                f"Optimizer param groups: {len(other_params)} tensors at lr={self.lr}, "
                f"{len(router_params)} router tensors at lr={self.temporal_router_lr}."
            )
            optimizer = torch.optim.AdamW(
                param_groups,
                eps=1e-8,
                weight_decay=0.01,
                betas=(0.9, 0.98),
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                eps=1e-8,
                weight_decay=0.01,
                betas=(0.9, 0.98)
            )
        
        # Calculate total steps based on PyTorch Lightning trainer settings
        if hasattr(self.trainer, 'estimated_stepping_batches'):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            # Fallback calculation if the attribute doesn't exist
            max_epochs = self.trainer.max_epochs
            train_dataloader = self.trainer.train_dataloader
            if hasattr(train_dataloader, 'dataloader'):
                train_dataloader = train_dataloader.dataloader
            
            batches_per_epoch = len(train_dataloader)
            total_steps = batches_per_epoch * max_epochs
            
            # Account for gradient accumulation if used
            if hasattr(self.trainer, 'accumulate_grad_batches'):
                total_steps = total_steps // self.trainer.accumulate_grad_batches
        
        warmup_steps = int(total_steps * 0.1)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }