"""Legacy Vision Encoder using the old Sign2GPT's DinoV2 implementation."""
import torch
import torch.nn as nn
from typing import Optional, List, Union, Dict, Any, Tuple
from pathlib import Path

from src.models.dinov2.model.vision_transformer import vit_small, vit_base


# Default checkpoint path for pretrained DinoV2 weights
DEFAULT_DINOV2_CHECKPOINT = Path("/mnt/lustre-grete/projects/intern_agc_emmy/hf_models/dinov2/dinov2_vits14_pretrain.pth")


class LegacyVisionEncoder(nn.Module):
    """Vision encoder using the Sign2GPT's DinoV2 implementation."""
    
    def __init__(
        self,
        model_name: str = "dinov2-small",
        peft_config: Optional[dict] = None,
        out_dim: int = 512,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        
        if peft_config is None:
            peft_config = {}
        
        # LoRA parameters
        lora_rank = peft_config.get("rank", 4)
        lora_alpha = peft_config.get("alpha", 4.0)
        lora_dropout = peft_config.get("dropout", 0.1)
        layers_to_transform = peft_config.get("layers_to_transform", [9, 10, 11])
        
        # Create the DinoV2 model with LoRA adapters
        self.spatial_model = vit_small(
            img_size=518,  # Hardcoded in old Sign2GPT
            init_values=1.0,
            patch_size=14,  # Hardcoded in old Sign2GPT
            block_chunks=0,
            adaptor_layers=layers_to_transform,
            adapt_params={
                'w_lora': True,
                'w_lora_ff': True,
                'lora_rank': lora_rank,
                'lora_drop': lora_dropout,
                'lora_a': lora_alpha,
                'rng_init': False,  # Sign2GPT does not use rng_init for LoRA
            },
        )
        
        self.lin = nn.Linear(self.spatial_model.num_features, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)  # Old Sign2GPT uses BatchNorm1d
        
        # Load pretrained checkpoint
        checkpoint_path = checkpoint_path or str(DEFAULT_DINOV2_CHECKPOINT)
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            print(f"Loading DinoV2 checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'teacher' in checkpoint:
                    state_dict = checkpoint['teacher']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            result = self.spatial_model.load_state_dict(state_dict, strict=False)
            print(f"Checkpoint loaded: {len(state_dict)} keys, {len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected")
        else:
            print(f"WARNING: DinoV2 checkpoint not found at {checkpoint_path}")
            print("Model will use random weights for base model - this will affect performance!")
        
        # Freeze base model, unfreeze LoRA parameters
        lora_param_count = 0
        frozen_param_count = 0
        for name, param in self.spatial_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                lora_param_count += 1
            else:
                param.requires_grad = False
                frozen_param_count += 1
        
        print(f"DinoV2 LoRA trainable: {lora_param_count}, Frozen: {frozen_param_count}")
        
        # Projection layers are always trainable
        for param in self.lin.parameters():
            param.requires_grad = True
        for param in self.bn.parameters():
            param.requires_grad = True

    @staticmethod
    def _pad_to_length(tensor: torch.Tensor, length: int) -> torch.Tensor:
        """Pad first dimension to `length` with zeros."""
        if tensor.size(0) == length:
            return tensor
        if tensor.size(0) > length:
            return tensor[:length]
        pad = tensor.new_zeros(length - tensor.size(0), *tensor.size()[1:])
        return torch.cat([tensor, pad], dim=0)
        
    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        return_cls_token: bool = True,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass."""
        if isinstance(x, list):
            if len(x) == 0:
                raise ValueError("LegacyVisionEncoder received an empty list of frames.")
            lengths = torch.tensor([len(v) for v in x], device=x[0].device, dtype=torch.long)
            if max_len is None:
                max_len = int(lengths.max().item())

            flat_frames = torch.cat(x, dim=0)  # (sum(Ti), C, H, W)
            feats = self.spatial_model.forward_features(flat_frames)["x_norm_clstoken"]  # (sum(Ti), D)
            feats = self.bn(self.lin(feats))  # (sum(Ti), out_dim)

            # Pad back into (B, max_len, out_dim)
            out_list = []
            offset = 0
            for l in lengths.tolist():
                seg = feats[offset : offset + l]
                out_list.append(self._pad_to_length(seg, max_len))
                offset += l
            features = torch.stack(out_list, dim=0)  # (B, max_len, out_dim)

            mask = torch.zeros(features.shape[0], features.shape[1], device=features.device, dtype=torch.bool)
            for i, l in enumerate(lengths.tolist()):
                mask[i, :l] = True
            return features, mask

        # --- Compatibility path: already padded tensor (B, T, C, H, W) ---
        B, T, C, H, W = x.shape
        flat = x.view(B * T, C, H, W)
        feats = self.spatial_model.forward_features(flat)["x_norm_clstoken"]
        feats = self.bn(self.lin(feats))
        feats = feats.view(B, T, -1)
        return feats, None


def create_legacy_vision_encoder(
    model_name: str = "dinov2-small",
    peft_config: Optional[dict] = None,
    out_dim: int = 512,
    device: str = "cuda",
    checkpoint_path: Optional[str] = None,
) -> LegacyVisionEncoder:
    """
    Create legacy vision encoder using old Sign2GPT's DinoV2 implementation.
    """
    encoder = LegacyVisionEncoder(
        model_name=model_name,
        peft_config=peft_config,
        out_dim=out_dim,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    return encoder
