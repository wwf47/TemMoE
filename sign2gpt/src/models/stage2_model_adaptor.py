"""Stage 2 Model with Zero-Gated Cross-Attention Adaptors."""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from transformers import AutoTokenizer

from src.models.vision_encoder import VisionEncoder
from src.models.vision_encoder_legacy import create_legacy_vision_encoder
from src.models.stage1_model import TemporalEncoder, WholeTemporalEncoderMoE
from src.models.huggingface.modeling_xglm import XGLMForCausalLM
from src.models.moe import MoEAdaptor


class SinePositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding (same as old Sign2GPT)."""
    
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class VisualProjection(nn.Module):
    """Project visual features to LM dimension."""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        pos_type: str = "sine",
        pre_pos: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pre_pos = pre_pos
        
        pos_dim = in_dim if pre_pos else out_dim
        if pos_type == "sine":
            self.pos_encoder = SinePositionalEmbedding(pos_dim)
        else:
            self.pos_encoder = nn.Identity()
        
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            #nn.LayerNorm(out_dim),
            #nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Args:"""
        if self.pre_pos:
            x = self.pos_encoder(x)
        x = self.proj(x)
        if not self.pre_pos:
            x = self.pos_encoder(x)
        return x, mask


class MoEVisualProjection(nn.Module):
    """Visual projection using MoE adaptor (Plan B: Fusion MoE)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_experts: int = 4,
        top_k: int = 1,
        dropout: float = 0.1,
        router_noise: float = 0.1,
        router_temperature: float = 1.0,
        use_shared_expert: bool = True,
    ):
        super().__init__()
        self.pos_encoder = SinePositionalEmbedding(in_dim)
        self.moe_adaptor = MoEAdaptor(
            in_dim=in_dim,
            out_dim=out_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            router_noise=router_noise,
            router_temperature=router_temperature,
            use_shared_expert=use_shared_expert,
        )
        self._aux_loss = 0.0

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.pos_encoder(x)
        output, mask, self._aux_loss = self.moe_adaptor(x, mask)
        return output, mask

    def set_routing_config(
        self,
        top_k: Optional[int] = None,
        router_noise: Optional[float] = None,
        router_temperature: Optional[float] = None,
    ) -> None:
        self.moe_adaptor.set_routing_config(
            top_k=top_k,
            router_noise=router_noise,
            router_temperature=router_temperature,
        )

    def get_moe_diagnostics(self) -> Dict:
        return self.moe_adaptor.get_diagnostics()


class Stage2ModelAdaptor(nn.Module):
    """Stage 2 model using zero-gated cross-attention adaptors."""
    
    def __init__(
        self,
        # Vision encoder config
        vision_config: Dict,
        # Temporal encoder config
        temporal_layers: List[int] = [2, 2],
        temporal_mlp_ratio: float = 4.0,
        temporal_dropout: float = 0.1,
        temporal_drop_path: float = 0.1,
        use_layer_scale: bool = True,
        use_downsampler: bool = True,
        use_pos_embed: bool = True,
        # Projection config
        projection_dropout: float = 0.1,
        # Language model config
        lm_path: str = "/mnt/lustre-grete/projects/intern_agc_emmy/hf_models/facebook__xglm-1.7B",
        lora_rank: int = 4,
        lora_alpha: float = 4.0,
        lora_dropout: float = 0.1,
        adaptor_layers: Optional[List[int]] = None,
        lora_layers: Optional[List[int]] = None,
        gate_type: str = "clamp",
        label_smoothing: float = 0.0,
        w_lora_ff: bool = False,  # Match old Sign2GPT (config controls this)
        gate_init: float = 0.0,   # Match old Sign2GPT: gates start at 0
        pretext: str = "",  # NEW: Pretext to prepend to all sequences (like old Sign2GPT)
        # Freezing options
        freeze_vision: bool = False,
        freeze_temporal: bool = False,
        # --- MoE Plan A: Temporal MoE ---
        temporal_moe: bool = False,
        temporal_moe_block_indices: Optional[List[int]] = None,
        temporal_num_experts: int = 4,
        temporal_top_k: int = 1,
        temporal_use_shared_expert: bool = True,
        temporal_moe_aux_loss_weight: float = 0.01,
        temporal_moe_router_noise: float = 0.1,
        temporal_router_temperature: float = 1.0,
        temporal_moe_schedule: Optional[List[Dict]] = None,
        temporal_moe_expert_scope: str = "ffn",
        temporal_routing_granularity: str = "token",
        temporal_segment_size: int = 8,
        temporal_shared_mix_mode: str = "average",
        temporal_shared_expert_gate_init: float = -2.0,
        temporal_expert_dropout: float = 0.0,
        temporal_moe_load_balance_type: str = "switch",
        # --- MoE Plan B: Fusion MoE ---
        fusion_moe: bool = False,
        fusion_num_experts: int = 4,
        fusion_top_k: int = 1,
        fusion_use_shared_expert: bool = True,
        fusion_moe_aux_loss_weight: float = 0.01,
        fusion_moe_router_noise: float = 0.1,
        fusion_router_temperature: float = 1.0,
        fusion_moe_schedule: Optional[List[Dict]] = None,
    ):
        super().__init__()
        
        # Vision encoder
        # Supports:
        # - Modern path: `VisionEncoder` (HF DinoV2/V3 + optional PEFT)
        self.use_legacy_vision = bool(vision_config.get("use_legacy_encoder", False))
        if self.use_legacy_vision:
            self.vision_encoder = create_legacy_vision_encoder(
                model_name=vision_config.get("model_name", "dinov2-small"),
                peft_config=vision_config.get("peft_config", {}),
                out_dim=int(vision_config.get("out_dim", 512)),
                device=str(vision_config.get("device", "cuda")),
                checkpoint_path=vision_config.get("checkpoint_path", None),
            )
            vision_dim = int(vision_config.get("out_dim", 512))
        else:
            vision_config_clean = {k: v for k, v in vision_config.items() if k != "device"}
            self.vision_encoder = VisionEncoder(**vision_config_clean)
            vision_dim = self.vision_encoder.out_dim
        
        # MoE flags
        self.temporal_moe = temporal_moe
        self.fusion_moe = fusion_moe
        self.temporal_moe_aux_loss_weight = temporal_moe_aux_loss_weight
        self.fusion_moe_aux_loss_weight = fusion_moe_aux_loss_weight
        self._temporal_moe_base = {
            "top_k": temporal_top_k,
            "router_noise": temporal_moe_router_noise,
            "router_temperature": temporal_router_temperature,
            "aux_loss_weight": temporal_moe_aux_loss_weight,
        }
        self._fusion_moe_base = {
            "top_k": fusion_top_k,
            "router_noise": fusion_moe_router_noise,
            "router_temperature": fusion_router_temperature,
            "aux_loss_weight": fusion_moe_aux_loss_weight,
        }
        self.temporal_moe_schedule = self._normalize_moe_schedule(temporal_moe_schedule)
        self.fusion_moe_schedule = self._normalize_moe_schedule(fusion_moe_schedule)
        self._last_temporal_schedule_state: Optional[Tuple[int, float, float, float]] = None
        self._last_fusion_schedule_state: Optional[Tuple[int, float, float, float]] = None

        # Temporal encoder: optionally route complete MetaFormer encoders.
        if temporal_moe and temporal_moe_expert_scope == "encoder":
            self.temporal_encoder = WholeTemporalEncoderMoE(
                dim=vision_dim,
                layers=temporal_layers,
                mlp_ratio=temporal_mlp_ratio,
                dropout=temporal_dropout,
                drop_path_rate=temporal_drop_path,
                use_layer_scale=use_layer_scale,
                use_downsampler=use_downsampler,
                use_pos_embed=use_pos_embed,
                num_experts=temporal_num_experts,
                top_k=temporal_top_k,
                router_noise=temporal_moe_router_noise,
                router_temperature=temporal_router_temperature,
                use_shared_expert=temporal_use_shared_expert,
            )
        else:
            moe_block_indices = temporal_moe_block_indices if temporal_moe else None
            self.temporal_encoder = TemporalEncoder(
                dim=vision_dim,
                layers=temporal_layers,
                mlp_ratio=temporal_mlp_ratio,
                dropout=temporal_dropout,
                drop_path_rate=temporal_drop_path,
                use_layer_scale=use_layer_scale,
                use_downsampler=use_downsampler,
                use_pos_embed=use_pos_embed,
                moe_block_indices=moe_block_indices,
                moe_num_experts=temporal_num_experts,
                moe_top_k=temporal_top_k,
                moe_router_noise=temporal_moe_router_noise,
                moe_router_temperature=temporal_router_temperature,
                moe_use_shared_expert=temporal_use_shared_expert,
                moe_shared_expert_gate_init=temporal_shared_expert_gate_init,
                moe_shared_mix_mode=temporal_shared_mix_mode,
                moe_expert_dropout=temporal_expert_dropout,
                moe_load_balance_type=temporal_moe_load_balance_type,
                moe_expert_scope=temporal_moe_expert_scope,
                moe_routing_granularity=temporal_routing_granularity,
                moe_segment_size=temporal_segment_size,
            )
        temporal_out_dim = vision_dim  # TemporalEncoder preserves dimension
        
        #   self.lang_model = modeling_xglm.XGLMForCausalLM.from_pretrained(...)
        #   freeze all params
        #   self.lang_model.init_adaptor(...)
        self.lm = XGLMForCausalLM.from_pretrained(lm_path, local_files_only=True)
        for _, p in self.lm.named_parameters():
            p.requires_grad = False

        num_layers = int(getattr(self.lm.config, "num_layers", 24))
        if adaptor_layers is None:
            adaptor_layers = list(range(num_layers))
        if lora_layers is None:
            lora_layers = list(range(num_layers))

        self.lm.init_adaptor(
            adapt_layers=adaptor_layers,
            lora_layers=lora_layers,
            w_lora_ff=w_lora_ff,
            lora_rank=lora_rank,
            lora_drop=lora_dropout,
            gate_type=gate_type,
            lora_a=lora_alpha,
            adapt_tokens=False,
        )

        lm_dim = self.lm.embed_dim
        
        # Projection: temporal features -> LM dimension
        if fusion_moe:
            self.projection = MoEVisualProjection(
                in_dim=temporal_out_dim,
                out_dim=lm_dim,
                num_experts=fusion_num_experts,
                top_k=fusion_top_k,
                dropout=projection_dropout,
                router_noise=fusion_moe_router_noise,
                router_temperature=fusion_router_temperature,
                use_shared_expert=fusion_use_shared_expert,
            )
        else:
            self.projection = VisualProjection(
                in_dim=temporal_out_dim,
                out_dim=lm_dim,
                pos_type="sine",
                pre_pos=True,
                dropout=projection_dropout,
            )
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(lm_path, local_files_only=True)

        self.pretext = pretext
        if self.pretext:
            self.pretext_tokens = self.tokenizer(self.pretext)["input_ids"]
            self.pretext_length = len(self.pretext_tokens)
            if self.pretext[-1] == " ":
                self.pretext_tokens = self.pretext_tokens[:-1]
                self.pretext_length = self.pretext_length - 1
        else:
            self.pretext = ""
            self.pretext_tokens = []
            self.pretext_length = 0

        # Loss config (label smoothing handled here since we compute loss inside the model)
        self.label_smoothing = float(label_smoothing)

        # Freezing
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        if freeze_temporal:
            for p in self.temporal_encoder.parameters():
                p.requires_grad = False

    @staticmethod
    def _normalize_moe_schedule(schedule: Optional[List[Dict]]) -> List[Dict]:
        if not schedule:
            return []

        normalized = []
        for phase in schedule:
            if phase is None:
                continue
            phase_copy = dict(phase)
            phase_copy["start_epoch"] = int(phase_copy.get("start_epoch", 1))
            if phase_copy.get("end_epoch") is not None:
                phase_copy["end_epoch"] = int(phase_copy["end_epoch"])
            normalized.append(phase_copy)

        return sorted(normalized, key=lambda item: item["start_epoch"])

    @staticmethod
    def _select_schedule_phase(schedule: List[Dict], epoch: int) -> Optional[Dict]:
        for phase in schedule:
            start_epoch = int(phase.get("start_epoch", 1))
            end_epoch = phase.get("end_epoch")
            if end_epoch is None and epoch >= start_epoch:
                return phase
            if end_epoch is not None and start_epoch <= epoch <= int(end_epoch):
                return phase
        return None

    @staticmethod
    def _resolve_phase_value(phase: Dict, key: str, epoch: int):
        if key in phase and phase[key] is not None:
            return phase[key]

        start_key = f"{key}_start"
        end_key = f"{key}_end"
        has_start = phase.get(start_key) is not None
        has_end = phase.get(end_key) is not None
        if not (has_start or has_end):
            return None

        start_val = float(phase[start_key] if has_start else phase[end_key])
        end_val = float(phase[end_key] if has_end else phase[start_key])
        start_epoch = int(phase.get("start_epoch", epoch))
        end_epoch = phase.get("end_epoch")
        if end_epoch is None or int(end_epoch) <= start_epoch:
            return end_val

        progress = (epoch - start_epoch) / max(1, int(end_epoch) - start_epoch)
        progress = min(max(progress, 0.0), 1.0)
        return start_val + (end_val - start_val) * progress

    @staticmethod
    def _value_or_default(value, default):
        return default if value is None else value

    @staticmethod
    def _format_routing_message(
        prefix: str,
        top_k: int,
        router_temperature: float,
        router_noise: float,
        aux_weight: float,
    ) -> str:
        return (
            f"  {prefix} routing schedule: top_k={top_k} | "
            f"temp={router_temperature:.3f} | noise={router_noise:.3f} | aux={aux_weight:.4f}"
        )

    def update_moe_routing(self, epoch: int) -> List[str]:
        messages = []

        if self.temporal_moe and self.temporal_moe_schedule:
            phase = self._select_schedule_phase(self.temporal_moe_schedule, epoch)
            if phase is not None:
                top_k = int(phase.get("top_k", self._temporal_moe_base["top_k"]))
                router_noise = float(
                    self._value_or_default(
                        self._resolve_phase_value(phase, "router_noise", epoch),
                        self._temporal_moe_base["router_noise"],
                    )
                )
                router_temperature = float(
                    self._value_or_default(
                        self._resolve_phase_value(phase, "router_temperature", epoch),
                        self._temporal_moe_base["router_temperature"],
                    )
                )
                aux_weight = float(
                    self._value_or_default(
                        self._resolve_phase_value(phase, "aux_loss_weight", epoch),
                        self._temporal_moe_base["aux_loss_weight"],
                    )
                )
                self.temporal_encoder.set_moe_routing_config(
                    top_k=top_k,
                    router_noise=router_noise,
                    router_temperature=router_temperature,
                )
                self.temporal_moe_aux_loss_weight = aux_weight

                state = (top_k, round(router_temperature, 6), round(router_noise, 6), round(aux_weight, 6))
                if state != self._last_temporal_schedule_state:
                    messages.append(
                        self._format_routing_message(
                            "Temporal MoE",
                            top_k=top_k,
                            router_temperature=router_temperature,
                            router_noise=router_noise,
                            aux_weight=aux_weight,
                        )
                    )
                    self._last_temporal_schedule_state = state

        if self.fusion_moe and self.fusion_moe_schedule and hasattr(self.projection, "set_routing_config"):
            phase = self._select_schedule_phase(self.fusion_moe_schedule, epoch)
            if phase is not None:
                top_k = int(phase.get("top_k", self._fusion_moe_base["top_k"]))
                router_noise = float(
                    self._value_or_default(
                        self._resolve_phase_value(phase, "router_noise", epoch),
                        self._fusion_moe_base["router_noise"],
                    )
                )
                router_temperature = float(
                    self._value_or_default(
                        self._resolve_phase_value(phase, "router_temperature", epoch),
                        self._fusion_moe_base["router_temperature"],
                    )
                )
                aux_weight = float(
                    self._value_or_default(
                        self._resolve_phase_value(phase, "aux_loss_weight", epoch),
                        self._fusion_moe_base["aux_loss_weight"],
                    )
                )
                self.projection.set_routing_config(
                    top_k=top_k,
                    router_noise=router_noise,
                    router_temperature=router_temperature,
                )
                self.fusion_moe_aux_loss_weight = aux_weight

                state = (top_k, round(router_temperature, 6), round(router_noise, 6), round(aux_weight, 6))
                if state != self._last_fusion_schedule_state:
                    messages.append(
                        self._format_routing_message(
                            "Fusion MoE",
                            top_k=top_k,
                            router_temperature=router_temperature,
                            router_noise=router_noise,
                            aux_weight=aux_weight,
                        )
                    )
                    self._last_fusion_schedule_state = state

        return messages

    def encode_video(self, frames: List[torch.Tensor]) -> tuple:
        """Encode video frames to visual features."""
        features = []
        masks = []

        # Match the original Sign2GPT legacy DINO path: encode the whole batch
        # together so BatchNorm1d sees the same frame distribution as Stage 1.
        if self.use_legacy_vision:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                frame_feats, frame_masks = self.vision_encoder(
                    frames,
                    return_cls_token=True,
                    max_len=None,
                )
                temporal_feats, temporal_masks = self.temporal_encoder(
                    frame_feats,
                    mask=frame_masks,
                    return_mask=True,
                )

            for feats_i, mask_i in zip(temporal_feats, temporal_masks):
                valid_len = int(mask_i.sum().item())
                features.append(feats_i[:valid_len])
                masks.append(mask_i[:valid_len])
            return features, masks

        for video in frames:
            # Modern path: (T, C, H, W) → (T, D_vision) → (T', D_temporal)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                frame_feats, _ = self.vision_encoder(video)  # ignore cls_tokens
                temporal_feats = self.temporal_encoder(frame_feats.unsqueeze(0)).squeeze(0)

            features.append(temporal_feats)
            masks.append(torch.ones(temporal_feats.size(0), device=temporal_feats.device, dtype=torch.bool))
        
        return features, masks
    
    def forward(
        self,
        frames: List[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """Forward pass for training."""
        device = input_ids.device
        B = input_ids.size(0)
        
        # Encode videos
        video_feats, video_masks = self.encode_video(frames)  # all masks are True
        
        # Pad and stack
        max_len = max(f.size(0) for f in video_feats)
        padded_feats = []
        padded_masks = []
        
        for f, m in zip(video_feats, video_masks):
            T = f.size(0)
            if T < max_len:
                pad = torch.zeros(max_len - T, f.size(-1), device=f.device, dtype=f.dtype)
                f = torch.cat([f, pad], dim=0)
                m = torch.cat([m, torch.zeros(max_len - T, device=device, dtype=torch.bool)])
            padded_feats.append(f)
            padded_masks.append(m)
        
        video_feats = torch.stack(padded_feats)  # (B, T, D)
        video_masks = torch.stack(padded_masks)  # (B, T)
        
        # Project to LM dimension
        visual_embeds, visual_mask = self.projection(video_feats, video_masks)

        # Use model's vocab_size (may differ from tokenizer vocab_size if embeddings were resized)
        vocab_size = self.lm.config.vocab_size
        max_valid_id = vocab_size - 1
        # Always clamp to ensure all token IDs are valid (defensive programming)
        if input_ids.max().item() > max_valid_id:
            if not hasattr(self, '_invalid_input_ids_warned'):
                num_invalid = (input_ids > max_valid_id).sum().item()
                max_input_id = input_ids.max().item()
                print(f"[WARNING] Found {num_invalid} input_ids > model vocab_size (max={max_input_id}, vocab_size={vocab_size}). Clamping to valid range.")
                self._invalid_input_ids_warned = True
            # Replace invalid tokens with pad_token_id (or 0)
            pad_token_id = getattr(self.lm.config, 'pad_token_id', None)
            if pad_token_id is None:
                pad_token_id = 0
            invalid_mask = input_ids > max_valid_id
            input_ids = torch.where(invalid_mask, 
                                   torch.tensor(pad_token_id, device=input_ids.device, dtype=input_ids.dtype), 
                                   input_ids)
        # Always clamp to ensure all values are in valid range [0, vocab_size-1]
        input_ids = torch.clamp(input_ids, min=0, max=max_valid_id)

        # Forward through LM with cross-attention to visual features.
        # We do NOT pass labels to the LM; Stage2 uses externally-aligned labels.
        # Disable KV cache during training to save memory.
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_adaptors=visual_embeds,
            adaptor_mask=visual_mask.float() if visual_mask is not None else None,
            labels=None,
            use_cache=False,
        )
        
        # Empty pretext -> tokenizer("") = [2], pretext_length=1.
        pre_len = self.pretext_length if self.pretext_length > 0 else 1
        logits = outputs.logits[:, pre_len - 1:]  # Slice to align with labels

        # Labels are [sentence, EOS] = [1234, 5678, 2], aligned with logits
        # In autoregressive: logits[i] predicts labels[i] (next token at position i)
        # Ensure logits and labels have matching sequence lengths
        loss = None
        if labels is not None:
            # Use vocab_size from logits (handles resized embeddings)
            vocab_size = logits.size(-1)
            
            # Align labels with logits length (logits come from input_ids which is shorter)
            # logits shape: (batch, seq_len, vocab) where seq_len = input_ids.size(1)
            # labels shape: (batch, labels_len) where labels_len might be different
            seq_len = logits.size(1)
            labels_len = labels.size(1)
            
            if not hasattr(self, '_loss_shape_warned'):
                if labels_len != seq_len:
                    print(f"[WARNING] Label/length mismatch: logits seq_len={seq_len}, labels len={labels_len}")
                self._loss_shape_warned = True
            
            if labels_len > seq_len:
                # Truncate labels to match logits length
                labels = labels[:, :seq_len]
            elif labels_len < seq_len:
                # Pad labels (shouldn't happen, but safety check)
                pad_length = seq_len - labels_len
                pad_tokens = torch.full((labels.size(0), pad_length), -100, 
                                      dtype=labels.dtype, device=labels.device)
                labels = torch.cat([labels, pad_tokens], dim=1)
            
            # Replace any invalid token IDs (>= vocab_size or < -100) with -100 (ignored in loss)
            # This prevents CUDA index out of bounds errors
            invalid_mask = (labels >= 0) & (labels >= vocab_size)
            if invalid_mask.any():
                if not hasattr(self, '_invalid_labels_warned'):
                    num_invalid = invalid_mask.sum().item()
                    max_label = labels.max().item()
                    print(f"[WARNING] Found {num_invalid} invalid label tokens (max={max_label}, vocab_size={vocab_size}). Replacing with -100.")
                    self._invalid_labels_warned = True
                # Replace invalid token IDs with -100 (ignore in loss)
                labels = torch.where(invalid_mask, 
                                    torch.tensor(-100, device=labels.device, dtype=labels.dtype), 
                                    labels)
            
            # Final safety: clamp all labels to valid range [-100, vocab_size-1]
            # This ensures no index out of bounds errors
            labels = torch.clamp(labels, min=-100, max=vocab_size - 1)
            
            # Final shape check
            if logits.size(1) != labels.size(1):
                raise ValueError(f"Shape mismatch: logits seq_len={logits.size(1)}, labels len={labels.size(1)}")
            
            if self.label_smoothing > 0:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=self.label_smoothing)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, vocab_size), labels.view(-1)
            )
        
        # Collect MoE auxiliary losses
        aux_loss = torch.tensor(0.0, device=device)
        if self.temporal_moe:
            temporal_aux = getattr(self.temporal_encoder, "_moe_aux_loss", 0.0)
            aux_loss = aux_loss + self.temporal_moe_aux_loss_weight * temporal_aux
        if self.fusion_moe:
            fusion_aux = getattr(self.projection, "_aux_loss", 0.0)
            aux_loss = aux_loss + self.fusion_moe_aux_loss_weight * fusion_aux

        return {
            "loss": loss,
            "logits": logits,
            "aux_loss": aux_loss,
        }
    
    @torch.no_grad()
    def generate(
        self,
        frames: List[torch.Tensor],
        masks: Optional[List[torch.Tensor]] = None,
        # We accept either `max_length` or `max_new_tokens` for convenience.
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = 64,
        min_new_tokens: int = 5,  # NEW: Force at least N tokens before stopping
        num_beams: int = 4,
        temperature: float = 1.0,
        length_penalty: float = 1.0,
        **generate_kwargs,
    ) -> List[str]:
        """Generate text from video."""
        device = frames[0].device
        B = len(frames)

        # Encode videos
        video_feats, video_masks = self.encode_video(frames)


        # Pad and stack
        max_len = max(f.size(0) for f in video_feats)
        padded_feats = []
        padded_masks = []

        for f, m in zip(video_feats, video_masks):
            T = f.size(0)
            if T < max_len:
                pad = torch.zeros(max_len - T, f.size(-1), device=f.device, dtype=f.dtype)
                f = torch.cat([f, pad], dim=0)
                m = torch.cat([m, torch.zeros(max_len - T, device=device, dtype=torch.bool)])
            padded_feats.append(f)
            padded_masks.append(m)

        video_feats = torch.stack(padded_feats)
        video_masks = torch.stack(padded_masks)

        # Project
        visual_embeds, visual_mask = self.projection(video_feats, video_masks)

        # Generate with cross-attention to visual features
        # This matches training where tokenizer("") returns [2]

        # Start tokens: pretext if provided, else empty prompt (XGLM starts with EOS token).
        if self.pretext_tokens:
            start_ids = torch.tensor(self.pretext_tokens, device=device, dtype=torch.long).unsqueeze(0).repeat(B, 1)
        else:
            start_ids = torch.tensor([self.tokenizer.eos_token_id], device=device, dtype=torch.long).unsqueeze(0).repeat(B, 1)

        prompt_len = start_ids.size(1)
        if max_length is None:
            # Convert "max_new_tokens" into old-style "max_length"
            max_length = int(prompt_len + (max_new_tokens or 0))

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_ids = self.lm.generate(
                input_ids=start_ids,
                inputs_adaptors=visual_embeds,
                adaptor_mask=visual_mask.float() if visual_mask is not None else None,
                max_length=max_length,
                min_new_tokens=min_new_tokens,  # Force minimum output length
                num_beams=num_beams,
                temperature=temperature,
                length_penalty=length_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                # bos_token_id is ignored - XGLM always starts with EOS (token 2)
            )
        
        # This removes the initial EOS (and any pretext tokens) before decoding.
        gen_ids = output_ids[:, prompt_len:]
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        return texts
    
    def print_gate_diagnostics(self, prefix: str = ""):
        """Print adaptor gate values for monitoring training progress."""
        # (Vendored `modeling_xglm.py` does not provide a convenience helper; omit by default.)
        return
    
    def get_gate_diagnostics(self) -> dict:
        """Get gate statistics for logging."""
        return {}

    def get_moe_diagnostics(self) -> Dict:
        diagnostics: Dict[str, object] = {}
        if self.temporal_moe and hasattr(self.temporal_encoder, "get_moe_diagnostics"):
            temporal_diag = self.temporal_encoder.get_moe_diagnostics()
            if temporal_diag:
                diagnostics["temporal"] = temporal_diag
        if self.fusion_moe and hasattr(self.projection, "get_moe_diagnostics"):
            fusion_diag = self.projection.get_moe_diagnostics()
            if fusion_diag:
                diagnostics["fusion"] = fusion_diag
        return diagnostics

    @staticmethod
    def _format_expert_vector(values: List[float]) -> str:
        return " ".join(f"e{i}:{v:.2f}" for i, v in enumerate(values))

    @classmethod
    def _format_single_moe_diag(cls, label: str, diagnostics: Dict) -> str:
        parts = [
            f"{label}",
            f"valid={int(diagnostics.get('num_valid_tokens', 0))}",
            f"H={float(diagnostics.get('router_entropy', 0.0)):.3f}",
            f"top_k={int(diagnostics.get('top_k', 0))}",
            f"temp={float(diagnostics.get('router_temperature', 0.0)):.3f}",
            f"noise={float(diagnostics.get('router_noise', 0.0)):.3f}",
        ]
        if "shared_gate" in diagnostics:
            parts.append(f"shared={float(diagnostics['shared_gate']):.3f}")
        if "shared_effective_ratio" in diagnostics:
            parts.append(f"shared_eff={float(diagnostics['shared_effective_ratio']):.3f}")
        expert_mass = diagnostics.get("expert_mass", [])
        expert_top1 = diagnostics.get("expert_top1", [])
        if expert_mass:
            parts.append(f"mass=[{cls._format_expert_vector(expert_mass)}]")
        if expert_top1:
            parts.append(f"top1=[{cls._format_expert_vector(expert_top1)}]")
        return " | ".join(parts)

    def format_moe_diagnostics(self, prefix: str = "") -> List[str]:
        diagnostics = self.get_moe_diagnostics()
        lines: List[str] = []

        temporal_diag = diagnostics.get("temporal", [])
        for diag in temporal_diag:
            label = f"{prefix}Temporal MoE {diag.get('block_name', 'unknown')}".rstrip()
            lines.append(self._format_single_moe_diag(label, diag))

        fusion_diag = diagnostics.get("fusion", None)
        if isinstance(fusion_diag, dict) and fusion_diag:
            label = f"{prefix}Fusion MoE".rstrip()
            lines.append(self._format_single_moe_diag(label, fusion_diag))

        return lines

    def print_moe_diagnostics(self, prefix: str = ""):
        for line in self.format_moe_diagnostics(prefix=prefix):
            print(line)
    
    def load_stage1_checkpoint(self, path: str, strict: bool = False):
        """Load pretrained Stage-1 weights into vision + temporal encoders."""
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

        def strip_wrappers(k: str) -> str:
            for p in ("model.", "module."):
                if k.startswith(p):
                    k = k[len(p):]
            return k

        # Accept multiple possible prefixes
        VISION_PREFIXES = ("vision_encoder.", "encoder.")
        TEMP_PREFIXES   = ("temporal_encoder.", "temporal.")

        vision_state = {}
        temporal_state = {}

        for k, v in state_dict.items():
            k2 = strip_wrappers(k)

            for vp in VISION_PREFIXES:
                if k2.startswith(vp):
                    vision_state[k2[len(vp):]] = v
                    break

            for tp in TEMP_PREFIXES:
                if k2.startswith(tp):
                    temporal_state[k2[len(tp):]] = v
                    break

        if not vision_state and not temporal_state:
            sample_keys = list(state_dict.keys())[:30]
            raise RuntimeError(
                "Stage-1 checkpoint load found 0 matching keys for vision/temporal.\n"
                f"Expected prefixes: {VISION_PREFIXES} and {TEMP_PREFIXES}\n"
                f"Sample checkpoint keys:\n  " + "\n  ".join(sample_keys)
            )

        if vision_state:
            ret_v = self.vision_encoder.load_state_dict(vision_state, strict=strict)
            print(f"✓ Loaded vision encoder ({len(vision_state)} keys). "
                f"missing={len(ret_v.missing_keys)} unexpected={len(ret_v.unexpected_keys)}")
            if ret_v.missing_keys[:5]:
                print("  vision missing (sample):", ret_v.missing_keys[:5])
            if ret_v.unexpected_keys[:5]:
                print("  vision unexpected (sample):", ret_v.unexpected_keys[:5])
        else:
            print("⚠ No vision keys matched. Check prefixes / module names.")

        if temporal_state:
            ret_t = self.temporal_encoder.load_state_dict(temporal_state, strict=strict)
            print(f"✓ Loaded temporal encoder ({len(temporal_state)} keys). "
                f"missing={len(ret_t.missing_keys)} unexpected={len(ret_t.unexpected_keys)}")
            if ret_t.missing_keys[:5]:
                print("  temporal missing (sample):", ret_t.missing_keys[:5])
            if ret_t.unexpected_keys[:5]:
                print("  temporal unexpected (sample):", ret_t.unexpected_keys[:5])

            # Warm-start MoE expert FFNs from the checkpoint's dense MLPs.
            # For MoE blocks, load_state_dict(strict=False) skips the old
            # mlp.* keys (unexpected) and leaves moe_ffn.* at random init.
            # Here we copy the dense FFN weights into every expert + shared.
            if self.temporal_moe:
                self._warmstart_temporal_moe(temporal_state)
        else:
            print("⚠ No temporal keys matched. Check prefixes / module names.")

    def load_stage2_warmstart_checkpoint(self, path: str, strict: bool = False):
        """Warm-start from a dense Stage-2 checkpoint."""
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

        def strip_wrappers(k: str) -> str:
            for p in ("model.", "module."):
                if k.startswith(p):
                    k = k[len(p):]
            return k

        state_dict = {strip_wrappers(k): v for k, v in state_dict.items()}

        model_state = self.state_dict()
        compatible_state = {}
        shape_mismatches = []
        for k, v in state_dict.items():
            if k not in model_state:
                continue
            if model_state[k].shape == v.shape:
                compatible_state[k] = v
            else:
                shape_mismatches.append((k, tuple(v.shape), tuple(model_state[k].shape)))

        ret = self.load_state_dict(compatible_state, strict=strict)
        print(
            f"✓ Loaded Stage-2 warm-start ({len(compatible_state)} exact-match keys). "
            f"missing={len(ret.missing_keys)} unexpected={len(ret.unexpected_keys)}"
        )
        if shape_mismatches[:5]:
            sample = [f"{k}: ckpt={src} model={dst}" for k, src, dst in shape_mismatches[:5]]
            print("  stage2 shape mismatch (sample):", sample)
        if ret.missing_keys[:5]:
            print("  stage2 missing (sample):", ret.missing_keys[:5])
        if ret.unexpected_keys[:5]:
            print("  stage2 unexpected (sample):", ret.unexpected_keys[:5])

        if self.fusion_moe:
            self._warmstart_fusion_moe_from_dense_projection(state_dict)

    @staticmethod
    def _init_fusion_expert_from_dense_projection(
        expert: nn.Module,
        dense_weight: torch.Tensor,
        dense_bias: Optional[torch.Tensor],
    ) -> bool:
        """Initialize a 2-layer GELU expert to equal a dense linear map."""
        if not hasattr(expert, "net") or len(expert.net) < 4:
            return False

        first = expert.net[0]
        second = expert.net[3]
        if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
            return False

        out_dim, in_dim = dense_weight.shape
        required_hidden = 2 * in_dim
        if first.in_features != in_dim or second.out_features != out_dim:
            return False
        if first.out_features < required_hidden or second.in_features < required_hidden:
            return False

        dense_weight = dense_weight.to(device=first.weight.device, dtype=first.weight.dtype)
        if dense_bias is not None:
            dense_bias = dense_bias.to(device=second.weight.device, dtype=second.weight.dtype)

        with torch.no_grad():
            first.weight.zero_()
            if first.bias is not None:
                first.bias.zero_()
            second.weight.zero_()
            if second.bias is not None:
                second.bias.zero_()

            eye = torch.eye(in_dim, device=first.weight.device, dtype=first.weight.dtype)
            first.weight[:in_dim].copy_(eye)
            first.weight[in_dim:required_hidden].copy_(-eye)

            second.weight[:, :in_dim].copy_(dense_weight)
            second.weight[:, in_dim:required_hidden].copy_(-dense_weight)
            if dense_bias is not None and second.bias is not None:
                second.bias.copy_(dense_bias)

        return True

    def _warmstart_fusion_moe_from_dense_projection(self, stage2_state: dict):
        """Seed fusion MoE experts from a dense Stage-2 projection checkpoint."""
        if not isinstance(self.projection, MoEVisualProjection):
            print("  ⚠ Fusion warm-start requested but projection is not MoEVisualProjection")
            return

        dense_weight = stage2_state.get("projection.proj.0.weight")
        dense_bias = stage2_state.get("projection.proj.0.bias")
        if dense_weight is None:
            print("  ⚠ No dense projection weights found at projection.proj.0.*; skipping fusion warm-start")
            return

        moe = self.projection.moe_adaptor
        targets = [(f"expert {idx}", expert) for idx, expert in enumerate(moe.experts)]
        if moe.use_shared_expert:
            targets.append(("shared expert", moe.shared_expert))

        warmed = 0
        for name, expert in targets:
            if self._init_fusion_expert_from_dense_projection(expert, dense_weight, dense_bias):
                print(f"  ✓ Warm-started fusion {name} from dense projection")
                warmed += 1
            else:
                print(f"  ⚠ Could not warm-start fusion {name}; incompatible expert shape")

        if warmed > 0:
            print(f"  ✓ Total fusion projection warm-start: {warmed} adaptor(s) initialized")

    def _warmstart_temporal_moe(self, temporal_state: dict):
        """Copy dense Stage-1 weights into temporal MoE experts."""
        te = self.temporal_encoder
        if getattr(te, "moe_expert_scope", None) == "encoder":
            targets = list(te.experts)
            if te.use_shared_expert:
                targets.append(te.shared_expert)
            for expert in targets:
                expert.load_state_dict(temporal_state, strict=False)
            print(
                f"  ✓ Warm-started {len(targets)} complete MetaFormer experts "
                f"from {len(temporal_state)} dense temporal parameters"
            )
            return

        layers_cfg = [len(te.stage1)]
        if te.stage2 is not None:
            layers_cfg.append(len(te.stage2))

        warm_count = 0
        global_idx = 0
        for stage_name, stage_blocks in [("stage1", te.stage1), ("stage2", te.stage2)]:
            if stage_blocks is None:
                continue
            for local_idx, blk in enumerate(stage_blocks):
                if not blk.use_moe:
                    global_idx += 1
                    continue

                if getattr(blk, "moe_expert_scope", "ffn") == "block":
                    prefix = f"{stage_name}.{local_idx}."
                    block_weights = {
                        k[len(prefix):]: v
                        for k, v in temporal_state.items()
                        if k.startswith(prefix)
                    }
                    if not block_weights:
                        print(
                            f"  ⚠ No dense block weights found for "
                            f"{stage_name}.{local_idx} (global {global_idx}), "
                            "skipping warm-start"
                        )
                        global_idx += 1
                        continue

                    targets = list(blk.experts)
                    if blk.use_shared_expert:
                        targets.append(blk.shared_expert)
                    for expert in targets:
                        expert.load_state_dict(block_weights, strict=False)
                    warm_count += len(block_weights) * len(targets)
                    print(
                        f"  ✓ Warm-started whole-block MoE "
                        f"{stage_name}.{local_idx} (global {global_idx}): "
                        f"{len(block_weights)} dense params -> "
                        f"{len(targets)} experts"
                    )
                    global_idx += 1
                    continue

                prefix = f"{stage_name}.{local_idx}.mlp."
                mlp_weights = {
                    k[len(prefix):]: v
                    for k, v in temporal_state.items()
                    if k.startswith(prefix)
                }

                if not mlp_weights:
                    print(f"  ⚠ No dense MLP weights found for {stage_name}.{local_idx} "
                          f"(global {global_idx}), skipping warm-start")
                    global_idx += 1
                    continue

                moe = blk.moe_ffn
                for expert in moe.experts:
                    for k, v in mlp_weights.items():
                        target_key = f"net.{k}"
                        param = dict(expert.named_parameters()).get(target_key)
                        if param is not None and param.shape == v.shape:
                            param.data.copy_(v)
                            warm_count += 1

                if moe.use_shared_expert:
                    for k, v in mlp_weights.items():
                        target_key = f"net.{k}"
                        param = dict(moe.shared_expert.named_parameters()).get(target_key)
                        if param is not None and param.shape == v.shape:
                            param.data.copy_(v)
                            warm_count += 1

                num_targets = moe.num_experts + (1 if moe.use_shared_expert else 0)
                print(f"  ✓ Warm-started MoE block {stage_name}.{local_idx} "
                      f"(global {global_idx}): {len(mlp_weights)} dense params "
                      f"-> {num_targets} experts")

                global_idx += 1

        if warm_count > 0:
            print(f"  ✓ Total MoE warm-start: {warm_count} parameter tensors copied")


if __name__ == "__main__":
    print("Testing Stage2ModelAdaptor...")
    
    model = Stage2ModelAdaptor(
        vision_config={
            "model_name": "dinov3-base",
            "out_dim": 1024,
            "use_peft": False,
        },
        temporal_layers=[1, 1],
        lora_rank=4,
        lora_alpha=4.0,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Test forward
    frames = [torch.randn(8, 3, 224, 224), torch.randn(10, 3, 224, 224)]
    input_ids = torch.randint(0, 1000, (2, 20))
    attention_mask = torch.ones(2, 20)
    labels = input_ids.clone()
    
    print("\nTesting forward pass...")
    outputs = model(frames, input_ids, attention_mask, labels)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    print("\n✓ Stage2ModelAdaptor test passed!")
