"""Stage 1 model: Vision encoder -> MetaFormer temporal encoder -> prototype head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from pathlib import Path
import pickle
import math

from src.models.vision_encoder import create_vision_encoder
from src.models.vision_encoder_legacy import create_legacy_vision_encoder
from src.models.moe import (
    MoEFFN,
    TopKRouter,
    _build_router_diagnostics,
    _resolve_valid_token_mask,
    load_balancing_loss,
)


SIGN2GPT_ROOT = Path("/mnt/lustre-grete/projects/intern_agc_emmy/Sign2GPT")
FASTTEXT_PROTOTYPES = SIGN2GPT_ROOT / "data/isign/fasttext_prototypes.pt"
PSEUDO_GLOSS_PKL = SIGN2GPT_ROOT / "data/isign/processed_words.minfreq3.isign_pkl"


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample for regularization."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) matching old Sign2GPT."""
    
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]
        
        if (self._seq_len_cached is None or 
            seq_len != self._seq_len_cached or 
            self._cos_cached.device != x.device or
            self._cos_cached.dtype != x.dtype):
            
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=x.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
            self._sin_cached = emb.sin()[None, None, :, :].to(x.dtype)
        
        return self._cos_cached, self._sin_cached
    
    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def _apply_rotary_pos_emb(self, x, cos, sin):
        cos = cos[:, :, :x.shape[-2], :]
        sin = sin[:, :, :x.shape[-2], :]
        return (x * cos) + (self._rotate_half(x) * sin)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """Apply rotary embeddings to Q and K."""
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)
        return (
            self._apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            self._apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class LocalMaskAttention(nn.Module):
    """Local Mask Attention matching old Sign2GPT."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        window_size: int = 7,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        
        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # Cache for local attention mask
        self._attention_mask = None
    
    def _get_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create local attention mask with window_size."""
        # Create sliding window pattern
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        half_window = self.window_size // 2
        for i in range(seq_len):
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            mask[i, start:end] = True
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings to Q and K
        q, k = self.rotary_emb(q, k)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply local attention mask
        if self._attention_mask is None or self._attention_mask.shape[0] != T:
            self._attention_mask = self._get_local_mask(T, x.device)
        
        # Combine local mask with padding mask
        local_mask = self._attention_mask.unsqueeze(0)  # (1, T, T)
        
        if att_mask is not None:
            # att_mask: (B, T) bool where True = valid
            # Expand to (B, 1, 1, T) for key masking
            key_mask = att_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            # Combine: both local AND valid key
            combined_mask = local_mask.unsqueeze(0) & key_mask  # (B, 1, T, T)
            # Also ensure self-attention (diagonal) is always allowed
            eye = torch.eye(T, device=x.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
            combined_mask = combined_mask | eye
        else:
            combined_mask = local_mask.unsqueeze(0)  # (1, 1, T, T)
        
        # Apply mask: set invalid positions to -inf
        attn = attn.masked_fill(~combined_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        
        return out


class MetaFormerBlock(nn.Module):
    """MetaFormer block matching old Sign2GPT architecture."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-5,
        use_layer_scale: bool = True,
        window_size: int = 7,
        # MoE options (Plan A)
        use_moe: bool = False,
        moe_num_experts: int = 4,
        moe_top_k: int = 1,
        moe_router_noise: float = 0.1,
        moe_router_temperature: float = 1.0,
        moe_use_shared_expert: bool = True,
        moe_shared_expert_gate_init: float = -2.0,
        moe_shared_mix_mode: str = "average",
        moe_expert_dropout: float = 0.0,
        moe_load_balance_type: str = "switch",
        moe_routing_granularity: str = "token",
        moe_segment_size: int = 8,
    ):
        super().__init__()
        self.use_layer_scale = use_layer_scale
        self.use_moe = use_moe
        
        self.token_mixer = LocalMaskAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            window_size=window_size,
        )
        
        # FFN: dense MLP or MoE
        if use_moe:
            self.moe_ffn = MoEFFN(
                dim=dim,
                mlp_ratio=mlp_ratio,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                dropout=dropout,
                router_noise=moe_router_noise,
                router_temperature=moe_router_temperature,
                use_shared_expert=moe_use_shared_expert,
                shared_expert_gate_init=moe_shared_expert_gate_init,
                shared_mix_mode=moe_shared_mix_mode,
                expert_dropout=moe_expert_dropout,
                load_balance_type=moe_load_balance_type,
                routing_granularity=moe_routing_granularity,
                segment_size=moe_segment_size,
            )
        else:
            hidden = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, dim),
                nn.Dropout(dropout),
            )

        self._aux_loss = 0.0
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init * torch.ones(dim))
            self.layer_scale_2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def set_moe_routing_config(
        self,
        top_k: Optional[int] = None,
        router_noise: Optional[float] = None,
        router_temperature: Optional[float] = None,
    ) -> None:
        if self.use_moe:
            self.moe_ffn.set_routing_config(
                top_k=top_k,
                router_noise=router_noise,
                router_temperature=router_temperature,
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.token_mixer(x, att_mask=mask)

        if self.use_moe:
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.layer_scale_1 * attn_out))
                mlp_out, self._aux_loss = self.moe_ffn(x, token_mask=mask)
                x = self.norm2(x + self.drop_path(self.layer_scale_2 * mlp_out))
            else:
                x = self.norm1(x + self.drop_path(attn_out))
                mlp_out, self._aux_loss = self.moe_ffn(x, token_mask=mask)
                x = self.norm2(x + self.drop_path(mlp_out))
        else:
            self._aux_loss = 0.0
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.layer_scale_1 * attn_out))
                x = self.norm2(x + self.drop_path(self.layer_scale_2 * self.mlp(x)))
            else:
                x = self.norm1(x + self.drop_path(attn_out))
                x = self.norm2(x + self.drop_path(self.mlp(x)))

        return x


class WholeMetaFormerMoE(nn.Module):
    """Token-routed MoE whose experts are complete MetaFormer blocks."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-5,
        use_layer_scale: bool = True,
        window_size: int = 7,
        num_experts: int = 4,
        top_k: int = 1,
        router_noise: float = 0.1,
        router_temperature: float = 1.0,
        use_shared_expert: bool = True,
        shared_expert_gate_init: float = -2.0,
    ):
        super().__init__()
        self.use_moe = True
        self.moe_expert_scope = "block"
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.use_shared_expert = bool(use_shared_expert)
        self._aux_loss = 0.0
        self._last_diagnostics: Dict = {}

        self.router = TopKRouter(
            dim=dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            noise_std=router_noise,
            temperature=router_temperature,
        )

        def make_expert() -> MetaFormerBlock:
            return MetaFormerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=drop_path,
                layer_scale_init=layer_scale_init,
                use_layer_scale=use_layer_scale,
                window_size=window_size,
                use_moe=False,
            )

        self.experts = nn.ModuleList([make_expert() for _ in range(self.num_experts)])
        if self.use_shared_expert:
            self.shared_expert = make_expert()
            self.shared_expert_gate_logits = nn.Parameter(
                torch.tensor(float(shared_expert_gate_init), dtype=torch.float32)
            )

    def set_moe_routing_config(
        self,
        top_k: Optional[int] = None,
        router_noise: Optional[float] = None,
        router_temperature: Optional[float] = None,
    ) -> None:
        self.router.set_routing_config(
            top_k=top_k,
            noise_std=router_noise,
            temperature=router_temperature,
        )
        self.top_k = self.router.top_k

    def get_moe_diagnostics(self) -> Dict:
        diagnostics: Dict = {}
        for key, value in self._last_diagnostics.items():
            if torch.is_tensor(value):
                if value.numel() == 1:
                    diagnostics[key] = float(value.detach().cpu().item())
                else:
                    diagnostics[key] = [
                        float(v) for v in value.detach().cpu().tolist()
                    ]
            else:
                diagnostics[key] = value
        return diagnostics

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        top_k_probs, expert_indices, router_logits = self.router(x)
        valid_mask = _resolve_valid_token_mask(mask, x.shape[:2], x.device)
        routed_output = torch.zeros_like(x)

        for expert_idx, expert in enumerate(self.experts):
            selected = ((expert_indices == expert_idx).any(dim=-1) & valid_mask)
            if not selected.any():
                continue
            expert_output = expert(x, mask=mask)
            expert_weight = torch.zeros_like(top_k_probs[..., 0])
            for route_idx in range(self.top_k):
                expert_weight = expert_weight + (
                    top_k_probs[..., route_idx]
                    * (expert_indices[..., route_idx] == expert_idx).to(
                        top_k_probs.dtype
                    )
                )
            routed_output = routed_output + expert_output * expert_weight.unsqueeze(-1)

        shared_output = None
        shared_gate = None
        if self.use_shared_expert:
            shared_output = self.shared_expert(x, mask=mask)
            shared_gate = torch.sigmoid(self.shared_expert_gate_logits).to(
                routed_output.dtype
            )
            output = (routed_output + shared_gate * shared_output) / (
                1.0 + shared_gate
            )
        else:
            output = routed_output

        self._aux_loss = load_balancing_loss(
            router_logits,
            top_k_probs,
            expert_indices,
            self.num_experts,
            token_mask=valid_mask,
        )
        self._last_diagnostics = _build_router_diagnostics(
            router_logits=router_logits.detach(),
            top_k_probs=top_k_probs.detach(),
            top_k_indices=expert_indices.detach(),
            valid_token_mask=valid_mask.detach(),
            top_k=self.top_k,
            router_temperature=self.router.temperature,
            router_noise=self.router.noise_std,
            shared_gate=None if shared_gate is None else shared_gate.detach(),
            routed_output=routed_output.detach(),
            shared_output=None if shared_output is None else shared_output.detach(),
        )

        if mask is not None:
            output = output * valid_mask.unsqueeze(-1).to(output.dtype)
        return output


class TemporalDownsampler(nn.Module):
    """Temporal downsampler (reduces sequence length by 2x)."""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # in_dim and out_dim must be equal (no dimension change)
        assert in_dim == out_dim, "Old Sign2GPT downsampler doesn't change dimensions"
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: (B, T, D) -> (B, D, T) for pooling
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x, kernel_size=3, stride=2, padding=1)
        x = x.transpose(1, 2)  # (B, T', D)
        
        if mask is not None:
            with torch.no_grad():
                mask_float = mask.float()
                mask_pooled = F.avg_pool1d(
                    mask_float.unsqueeze(1), 
                    kernel_size=3, stride=2, padding=1, 
                    count_include_pad=False
                ).squeeze(1)
                # Ensure at least first position is valid
                mask_pooled[:, 0] = 1.0
                mask = (mask_pooled > 0.0).bool()
        
        return x, mask


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for embedding input."""
    
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TemporalEncoder(nn.Module):
    """Two-stage MetaFormer temporal encoder matching old Sign2GPT."""

    def __init__(
        self,
        dim: int,
        layers: List[int] = [2, 2],  # [stage1_blocks, stage2_blocks]
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        use_layer_scale: bool = True,
        layer_scale_init: float = 1e-5,
        use_downsampler: bool = True,
        use_pos_embed: bool = True,
        window_size: int = 7,  # Local attention window (matching old Sign2GPT)
        # MoE options (Plan A)
        moe_block_indices: Optional[List[int]] = None,
        moe_num_experts: int = 4,
        moe_top_k: int = 1,
        moe_router_noise: float = 0.1,
        moe_router_temperature: float = 1.0,
        moe_use_shared_expert: bool = True,
        moe_shared_expert_gate_init: float = -2.0,
        moe_shared_mix_mode: str = "average",
        moe_expert_dropout: float = 0.0,
        moe_load_balance_type: str = "switch",
        moe_expert_scope: str = "ffn",
        moe_routing_granularity: str = "token",
        moe_segment_size: int = 8,
    ):
        super().__init__()
        self.use_downsampler = use_downsampler
        if moe_expert_scope not in {"ffn", "block"}:
            raise ValueError(
                f"moe_expert_scope must be 'ffn' or 'block', got {moe_expert_scope!r}"
            )
        self.moe_expert_scope = moe_expert_scope
        self.moe_routing_granularity = str(moe_routing_granularity).lower()
        self.moe_segment_size = max(1, int(moe_segment_size))
        num_heads = max(1, dim // 64)  # ~8 heads for dim=512
        total_blocks = sum(layers)
        self.num_layers = total_blocks

        if moe_block_indices is None:
            moe_block_indices = []
        self.moe_block_indices = set(moe_block_indices)
        
        if use_pos_embed:
            self.pos_embed = SinusoidalPositionalEncoding(dim)
        else:
            self.pos_embed = nn.Identity()
        
        def _make_block(global_idx: int, drop_path: float) -> MetaFormerBlock:
            is_moe = global_idx in self.moe_block_indices
            if is_moe and self.moe_expert_scope == "block":
                return WholeMetaFormerMoE(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_path,
                    layer_scale_init=layer_scale_init,
                    use_layer_scale=use_layer_scale,
                    window_size=window_size,
                    num_experts=moe_num_experts,
                    top_k=moe_top_k,
                    router_noise=moe_router_noise,
                    router_temperature=moe_router_temperature,
                    use_shared_expert=moe_use_shared_expert,
                )
            return MetaFormerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=drop_path,
                layer_scale_init=layer_scale_init,
                use_layer_scale=use_layer_scale,
                window_size=window_size,
                use_moe=is_moe,
                moe_num_experts=moe_num_experts,
                moe_top_k=moe_top_k,
                moe_router_noise=moe_router_noise,
                moe_router_temperature=moe_router_temperature,
                moe_use_shared_expert=moe_use_shared_expert,
                moe_shared_expert_gate_init=moe_shared_expert_gate_init,
                moe_shared_mix_mode=moe_shared_mix_mode,
                moe_expert_dropout=moe_expert_dropout,
                moe_load_balance_type=moe_load_balance_type,
                moe_routing_granularity=self.moe_routing_granularity,
                moe_segment_size=self.moe_segment_size,
            )

        # Stage 1
        stage1_blocks = []
        for i in range(layers[0]):
            dp = drop_path_rate * i / (total_blocks - 1) if total_blocks > 1 else 0.0
            stage1_blocks.append(_make_block(i, dp))
        self.stage1 = nn.ModuleList(stage1_blocks)
        
        if use_downsampler and len(layers) > 1:
            self.downsampler = TemporalDownsampler(dim, dim)
        else:
            self.downsampler = None
        
        # Stage 2
        if len(layers) > 1:
            stage2_blocks = []
            for i in range(layers[1]):
                block_idx = layers[0] + i
                dp = drop_path_rate * block_idx / (total_blocks - 1) if total_blocks > 1 else 0.0
                stage2_blocks.append(_make_block(block_idx, dp))
            self.stage2 = nn.ModuleList(stage2_blocks)
        else:
            self.stage2 = None
        
        self.norm = nn.LayerNorm(dim)

        # Accumulated MoE auxiliary loss (set during forward)
        self._moe_aux_loss = 0.0

        self._init_like_old_sign2gpt()

    def set_moe_routing_config(
        self,
        top_k: Optional[int] = None,
        router_noise: Optional[float] = None,
        router_temperature: Optional[float] = None,
    ) -> None:
        for stage_blocks in (self.stage1, self.stage2):
            if stage_blocks is None:
                continue
            for blk in stage_blocks:
                if getattr(blk, "use_moe", False):
                    blk.set_moe_routing_config(
                        top_k=top_k,
                        router_noise=router_noise,
                        router_temperature=router_temperature,
                    )

    def get_moe_router_maps(self) -> List[Dict]:
        maps: List[Dict] = []
        global_idx = 0
        for stage_name, stage_blocks in (("stage1", self.stage1), ("stage2", self.stage2)):
            if stage_blocks is None:
                continue
            for local_idx, blk in enumerate(stage_blocks):
                moe = getattr(blk, "moe_ffn", None)
                probs = getattr(moe, "last_token_gating_weights", None) if moe is not None else None
                if probs is None and moe is not None:
                    probs = getattr(moe, "last_token_router_probs", None)
                mask = getattr(moe, "last_valid_token_mask", None) if moe is not None else None
                if probs is not None:
                    maps.append(
                        {
                            "block_name": f"{stage_name}.{local_idx}",
                            "global_idx": global_idx,
                            "probs": probs.detach(),
                            "mask": None if mask is None else mask.detach(),
                        }
                    )
                global_idx += 1
        return maps

    def get_moe_diagnostics(self) -> List[Dict]:
        diagnostics: List[Dict] = []
        global_idx = 0
        for stage_name, stage_blocks in (("stage1", self.stage1), ("stage2", self.stage2)):
            if stage_blocks is None:
                continue
            for local_idx, blk in enumerate(stage_blocks):
                if getattr(blk, "use_moe", False) and hasattr(
                    blk, "get_moe_diagnostics"
                ):
                    diag = blk.get_moe_diagnostics()
                elif getattr(blk, "use_moe", False) and hasattr(blk, "moe_ffn"):
                    diag = blk.moe_ffn.get_diagnostics()
                else:
                    diag = {}
                if diag:
                    diagnostics.append(
                        {
                            "block_name": f"{stage_name}.{local_idx}",
                            "global_idx": global_idx,
                            **diag,
                        }
                    )
                global_idx += 1
        return diagnostics

    def _init_like_old_sign2gpt(self):
        """Match old Sign2GPT MetaFormer init from `models/metaformer/meta_model.py`."""
        def _init_weights(module: nn.Module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        self.apply(_init_weights)

        # Special scaled init for attention projection ("token_mixer.proj" in old code).
        scaled_std = 0.02 / math.sqrt(2 * max(1, self.num_layers))
        for name, p in self.named_parameters():
            if "token_mixer" in name and ("out_proj.weight" in name or "proj.weight" in name):
                p.data.normal_(mean=0.0, std=scaled_std)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_mask: bool = False,
    ):
        # Add positional encoding
        x = self.pos_embed(x)

        self._moe_aux_loss = 0.0
        
        # Stage 1
        for blk in self.stage1:
            x = blk(x, mask=mask)
            self._moe_aux_loss = self._moe_aux_loss + getattr(blk, "_aux_loss", 0.0)
        
        # Downsampler
        if self.downsampler is not None:
            x, mask = self.downsampler(x, mask)
        
        # Stage 2
        if self.stage2 is not None:
            for blk in self.stage2:
                x = blk(x, mask=mask)
                self._moe_aux_loss = self._moe_aux_loss + getattr(blk, "_aux_loss", 0.0)
        
        x = self.norm(x)
        if return_mask:
            return x, mask
        return x


class WholeTemporalEncoderMoE(nn.Module):
    """Token-routed MoE whose experts are complete temporal MetaFormers."""

    def __init__(
        self,
        dim: int,
        layers: List[int] = [2, 2],
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        use_layer_scale: bool = True,
        layer_scale_init: float = 1e-5,
        use_downsampler: bool = True,
        use_pos_embed: bool = True,
        window_size: int = 7,
        num_experts: int = 4,
        top_k: int = 1,
        router_noise: float = 0.1,
        router_temperature: float = 1.0,
        use_shared_expert: bool = True,
        shared_expert_gate_init: float = -2.0,
    ):
        super().__init__()
        self.use_moe = True
        self.moe_expert_scope = "encoder"
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.use_shared_expert = bool(use_shared_expert)
        self.use_downsampler = bool(use_downsampler and len(layers) > 1)
        self._moe_aux_loss = 0.0
        self._last_diagnostics: Dict = {}

        self.router_pos_embed = (
            SinusoidalPositionalEncoding(dim) if use_pos_embed else nn.Identity()
        )
        self.router = TopKRouter(
            dim=dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            noise_std=router_noise,
            temperature=router_temperature,
        )

        def make_expert() -> TemporalEncoder:
            return TemporalEncoder(
                dim=dim,
                layers=layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init=layer_scale_init,
                use_downsampler=use_downsampler,
                use_pos_embed=use_pos_embed,
                window_size=window_size,
                moe_block_indices=[],
                moe_expert_scope="ffn",
            )

        self.experts = nn.ModuleList([make_expert() for _ in range(self.num_experts)])
        if self.use_shared_expert:
            self.shared_expert = make_expert()
            self.shared_expert_gate_logits = nn.Parameter(
                torch.tensor(float(shared_expert_gate_init), dtype=torch.float32)
            )

    def _prepare_router_tokens(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]
    ):
        route_input = self.router_pos_embed(x)
        route_mask = mask
        if self.use_downsampler:
            route_input = F.avg_pool1d(
                route_input.transpose(1, 2),
                kernel_size=3,
                stride=2,
                padding=1,
            ).transpose(1, 2)
            if route_mask is not None:
                mask_pooled = F.avg_pool1d(
                    route_mask.float().unsqueeze(1),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    count_include_pad=False,
                ).squeeze(1)
                mask_pooled[:, 0] = 1.0
                route_mask = mask_pooled > 0.0
        return route_input, route_mask

    def set_moe_routing_config(
        self,
        top_k: Optional[int] = None,
        router_noise: Optional[float] = None,
        router_temperature: Optional[float] = None,
    ) -> None:
        self.router.set_routing_config(
            top_k=top_k,
            noise_std=router_noise,
            temperature=router_temperature,
        )
        self.top_k = self.router.top_k

    def get_moe_diagnostics(self) -> List[Dict]:
        diagnostics: Dict = {}
        for key, value in self._last_diagnostics.items():
            if torch.is_tensor(value):
                if value.numel() == 1:
                    diagnostics[key] = float(value.detach().cpu().item())
                else:
                    diagnostics[key] = [
                        float(v) for v in value.detach().cpu().tolist()
                    ]
            else:
                diagnostics[key] = value
        if not diagnostics:
            return []
        return [{"block_name": "full_metaformer", "global_idx": 0, **diagnostics}]

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_mask: bool = False,
    ):
        route_input, route_mask = self._prepare_router_tokens(x, mask)
        top_k_probs, expert_indices, router_logits = self.router(route_input)
        valid_mask = _resolve_valid_token_mask(
            route_mask, route_input.shape[:2], route_input.device
        )
        routed_output = torch.zeros_like(route_input)

        for expert_idx, expert in enumerate(self.experts):
            selected = ((expert_indices == expert_idx).any(dim=-1) & valid_mask)
            if not selected.any():
                continue
            expert_output = expert(x, mask=mask)
            expert_weight = torch.zeros_like(top_k_probs[..., 0])
            for route_idx in range(self.top_k):
                expert_weight = expert_weight + (
                    top_k_probs[..., route_idx]
                    * (expert_indices[..., route_idx] == expert_idx).to(
                        top_k_probs.dtype
                    )
                )
            routed_output = routed_output + expert_output * expert_weight.unsqueeze(-1)

        shared_output = None
        shared_gate = None
        if self.use_shared_expert:
            shared_output = self.shared_expert(x, mask=mask)
            shared_gate = torch.sigmoid(self.shared_expert_gate_logits).to(
                routed_output.dtype
            )
            output = (routed_output + shared_gate * shared_output) / (
                1.0 + shared_gate
            )
        else:
            output = routed_output

        self._moe_aux_loss = load_balancing_loss(
            router_logits,
            top_k_probs,
            expert_indices,
            self.num_experts,
            token_mask=valid_mask,
        )
        self._last_diagnostics = _build_router_diagnostics(
            router_logits=router_logits.detach(),
            top_k_probs=top_k_probs.detach(),
            top_k_indices=expert_indices.detach(),
            valid_token_mask=valid_mask.detach(),
            top_k=self.top_k,
            router_temperature=self.router.temperature,
            router_noise=self.router.noise_std,
            shared_gate=None if shared_gate is None else shared_gate.detach(),
            routed_output=routed_output.detach(),
            shared_output=None if shared_output is None else shared_output.detach(),
        )
        output = output * valid_mask.unsqueeze(-1).to(output.dtype)

        if return_mask:
            return output, route_mask
        return output


def load_fasttext_prototypes(
    prototypes_path: str = str(FASTTEXT_PROTOTYPES),
) -> torch.Tensor:
    """Load pre-computed FastText embeddings for pseudo-gloss vocabulary."""
    print(f"Loading FastText prototypes from: {prototypes_path}")
    embeddings = torch.load(prototypes_path, map_location="cpu")
    print(f"Loaded FastText embeddings: {embeddings.shape}")
    return embeddings


class PrototypeHead(nn.Module):
    """Prototype head with FastText embeddings and class/time temperatures (matching old Sign2GPT)."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        dropout: float = 0.2,
        class_temperature: float = 0.1,
        time_temperature: float = 0.1,
        dynamic_class_temp: bool = True,
        dynamic_time_temp: bool = True,
        trainable_prototypes: bool = False,
        prototypes_path: str = str(FASTTEXT_PROTOTYPES),
    ):
        super().__init__()
        self.num_classes = num_classes

        # Always use FastText embeddings (300-dim)
        proto_dim = 300  # FastText dimension
        self.fc_hidden = nn.Linear(in_dim, proto_dim)
        
        # Load pre-computed FastText embeddings
        embeddings = load_fasttext_prototypes(prototypes_path)
        self.prototypes = nn.Parameter(embeddings, requires_grad=trainable_prototypes)

        if dynamic_class_temp:
            self.class_temperature = nn.Parameter(torch.tensor(class_temperature))
        else:
            self.register_buffer("class_temperature", torch.tensor(class_temperature))

        if dynamic_time_temp:
            self.time_temperature = nn.Parameter(torch.tensor(time_temperature))
        else:
            self.register_buffer("time_temperature", torch.tensor(time_temperature))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape

        y = self.fc_hidden(self.dropout(x))
        y = F.normalize(y, dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)

        time_res = torch.einsum("btd,cd->btc", y, proto)

        cls_temp = torch.clamp(self.class_temperature, 0.01, 1.0)
        cls_softmax = (time_res / cls_temp).softmax(dim=-1)

        time_temp = torch.clamp(self.time_temperature, 0.01, 1.0)
        if mask is not None:
            time_mask = (~mask).unsqueeze(-1).expand_as(time_res)
            time_scores = time_res / time_temp
            time_scores = time_scores.masked_fill(time_mask, float("-inf"))
        else:
            time_scores = time_res / time_temp
        time_softmax = time_scores.softmax(dim=1)

        softmax_scores = cls_softmax * time_softmax
        class_scores = softmax_scores.sum(dim=1)
        logits = class_scores[:, :self.num_classes]

        return {
            "logits": logits,
            "time_res": time_res,
            "softmax_scores": softmax_scores,
        }


class Stage1Model(nn.Module):
    def __init__(
        self,
        num_classes: int = 14106,
        vision_config: Optional[dict] = None,
        dropout: float = 0.2,
        temporal_layers: List[int] = [2, 2],  # [stage1_blocks, stage2_blocks]
        temporal_mlp_ratio: float = 4.0,
        temporal_drop_path: float = 0.1,
        use_layer_scale: bool = True,
        use_downsampler: bool = True,
        use_pos_embed: bool = True,
        window_size: int = 7,  # Local attention window (matching old Sign2GPT)
        # Prototype head params
        class_temperature: float = 0.1,
        time_temperature: float = 0.1,
        dynamic_class_temp: bool = True,
        dynamic_time_temp: bool = True,
        trainable_prototypes: bool = False,
        prototypes_path: str = str(FASTTEXT_PROTOTYPES),
        # Vision encoder type
        use_legacy_encoder: bool = True,  # Use old Sign2GPT's combined QKV LoRA (faster learning)
    ):
        super().__init__()
        
        if vision_config is None:
            vision_config = {}

        # Choose vision encoder backend
        if use_legacy_encoder:
            print("Using Legacy Vision Encoder (Old Sign2GPT style - faster learning)")
            self.encoder = create_legacy_vision_encoder(**vision_config)
        else:
            # User-requested: disable PEFT encoder path for now
            raise RuntimeError(
                "PEFT Vision Encoder is disabled for Stage 1 right now. "
                "Set model.use_legacy_encoder: true in the config."
            )
        
        out_dim = vision_config.get("out_dim", 1024)

        self.temporal = TemporalEncoder(
            dim=out_dim,
            layers=temporal_layers,
            mlp_ratio=temporal_mlp_ratio,
            dropout=dropout,
            drop_path_rate=temporal_drop_path,
            use_layer_scale=use_layer_scale,
            use_downsampler=use_downsampler,
            use_pos_embed=use_pos_embed,
            window_size=window_size,
        )
        self.head = PrototypeHead(
            in_dim=out_dim,
            num_classes=num_classes,
            dropout=dropout,
            class_temperature=class_temperature,
            time_temperature=time_temperature,
            dynamic_class_temp=dynamic_class_temp,
            dynamic_time_temp=dynamic_time_temp,
            trainable_prototypes=trainable_prototypes,
            prototypes_path=prototypes_path,
        )

    def forward(self, frames: List[torch.Tensor]):
        """Old Sign2GPT-style forward:"""
        feats, mask = self.encoder(frames, return_cls_token=True, max_len=None)
        feats, mask = self.temporal(feats, mask=mask, return_mask=True)
        out = self.head(feats, mask=mask)
        return out


if __name__ == "__main__":
    model = Stage1Model(num_classes=100)
    dummy = [torch.randn(8, 3, 224, 224), torch.randn(6, 3, 224, 224)]
    out = model(dummy)
    print("logits:", out["logits"].shape)
