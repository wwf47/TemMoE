"""Mixture-of-Experts building blocks for Sign2GPT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple


class TopKRouter(nn.Module):
    """Token-level top-k router with optional noise for exploration."""

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 1,
        noise_std: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.temperature = max(float(temperature), 1e-4)
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def set_routing_config(
        self,
        top_k: Optional[int] = None,
        noise_std: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> None:
        if top_k is not None:
            self.top_k = max(1, min(int(top_k), self.num_experts))
        if noise_std is not None:
            self.noise_std = max(float(noise_std), 0.0)
        if temperature is not None:
            self.temperature = max(float(temperature), 1e-4)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:"""
        logits = self.gate(x)

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        logits = logits / self.temperature
        router_probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        return top_k_probs, top_k_indices, logits


def _resolve_valid_token_mask(
    token_mask: Optional[torch.Tensor],
    shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    if token_mask is None:
        return torch.ones(shape, device=device, dtype=torch.bool)
    return token_mask.to(device=device, dtype=torch.bool)


def _sequence_route_inputs(
    route_inputs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool valid tokens into one routing vector per sequence."""
    mask = valid_mask.unsqueeze(-1).to(route_inputs.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    summary = (route_inputs * mask).sum(dim=1) / denom
    return summary.unsqueeze(1), valid_mask.new_ones(route_inputs.shape[0], 1)


def _segment_route_inputs(
    route_inputs: torch.Tensor,
    valid_mask: torch.Tensor,
    segment_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool non-overlapping token windows for segment routing."""
    batch_size, num_tokens, channels = route_inputs.shape
    pad = (segment_size - (num_tokens % segment_size)) % segment_size
    if pad:
        route_inputs = F.pad(route_inputs, (0, 0, 0, pad))
        valid_mask = F.pad(valid_mask, (0, pad), value=False)
    num_segments = route_inputs.shape[1] // segment_size
    windowed = route_inputs.view(batch_size, num_segments, segment_size, channels)
    window_mask = valid_mask.view(batch_size, num_segments, segment_size)
    mask = window_mask.unsqueeze(-1).to(windowed.dtype)
    denom = mask.sum(dim=2).clamp_min(1.0)
    segment_inputs = (windowed * mask).sum(dim=2) / denom
    segment_mask = window_mask.any(dim=2)
    return segment_inputs, segment_mask


def _broadcast_route_matrix(
    values: torch.Tensor,
    num_tokens: int,
    routing_granularity: str,
    segment_size: int,
) -> torch.Tensor:
    """Expand sequence/segment router tensors back onto tokens."""
    if routing_granularity == "token":
        return values
    if routing_granularity == "sequence":
        return values.expand(-1, num_tokens, -1)
    batch_size, num_segments, last_dim = values.shape
    expanded_len = num_segments * segment_size
    expanded = (
        values.unsqueeze(2)
        .expand(-1, -1, segment_size, -1)
        .reshape(batch_size, expanded_len, last_dim)
    )
    return expanded[:, :num_tokens]


def _broadcast_route_decisions(
    top_k_probs: torch.Tensor,
    top_k_indices: torch.Tensor,
    num_tokens: int,
    routing_granularity: str,
    segment_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand sequence/segment router decisions back onto tokens."""
    return (
        _broadcast_route_matrix(
            top_k_probs, num_tokens, routing_granularity, segment_size
        ),
        _broadcast_route_matrix(
            top_k_indices, num_tokens, routing_granularity, segment_size
        ),
    )


def _compute_dispatch_tensor(
    router_probs: torch.Tensor,
    top_k_probs: torch.Tensor,
    top_k_indices: torch.Tensor,
) -> torch.Tensor:
    dispatch = torch.zeros_like(router_probs)
    dispatch.scatter_add_(dim=-1, index=top_k_indices, src=top_k_probs)
    return dispatch


def _build_router_diagnostics(
    router_logits: torch.Tensor,
    top_k_probs: torch.Tensor,
    top_k_indices: torch.Tensor,
    valid_token_mask: torch.Tensor,
    top_k: int,
    router_temperature: float,
    router_noise: float,
    shared_gate: Optional[torch.Tensor] = None,
    routed_output: Optional[torch.Tensor] = None,
    shared_output: Optional[torch.Tensor] = None,
    output_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    router_probs = F.softmax(router_logits, dim=-1)
    dispatch = _compute_dispatch_tensor(router_probs, top_k_probs, top_k_indices)

    valid_mask_f = valid_token_mask.unsqueeze(-1).to(dtype=router_probs.dtype)
    denom = valid_mask_f[..., 0].sum().clamp_min(1.0)

    dispatch = dispatch * valid_mask_f
    router_probs = router_probs * valid_mask_f

    entropy = -(router_probs.clamp_min(1e-9) * router_probs.clamp_min(1e-9).log()).sum(dim=-1)
    entropy = (entropy * valid_mask_f[..., 0]).sum() / denom

    top1_mask = F.one_hot(
        top_k_indices[..., 0], num_classes=router_probs.size(-1)
    ).to(router_probs.dtype)
    top1_mask = top1_mask * valid_mask_f

    diagnostics: Dict[str, Any] = {
        "num_valid_tokens": valid_mask_f[..., 0].sum().detach(),
        "router_entropy": entropy.detach(),
        "expert_mass": (dispatch.sum(dim=(0, 1)) / denom).detach(),
        "expert_top1": (top1_mask.sum(dim=(0, 1)) / denom).detach(),
        "top_k": int(top_k),
        "router_temperature": float(router_temperature),
        "router_noise": float(router_noise),
    }

    if shared_gate is not None:
        diagnostics["shared_gate"] = shared_gate.detach()
    output_valid = valid_token_mask if output_mask is None else output_mask
    if (
        routed_output is not None
        and shared_output is not None
        and output_valid.any()
        and routed_output.shape[:2] == output_valid.shape
    ):
        routed_norm = routed_output.norm(dim=-1)[output_valid].mean()
        shared_norm = shared_output.norm(dim=-1)[output_valid].mean()
        diagnostics["routed_norm"] = routed_norm.detach()
        diagnostics["shared_norm"] = shared_norm.detach()
        if shared_gate is not None:
            diagnostics["shared_effective_ratio"] = (
                shared_gate.to(shared_norm.dtype) * shared_norm
            ).detach() / routed_norm.detach().clamp_min(1e-6)

    return diagnostics


def load_balancing_loss(
    router_logits: torch.Tensor,
    top_k_probs: torch.Tensor,
    top_k_indices: torch.Tensor,
    num_experts: int,
    token_mask: Optional[torch.Tensor] = None,
    load_balance_type: str = "switch",
) -> torch.Tensor:
    """Top-k-aware auxiliary loss encouraging balanced expert utilization."""
    router_probs = F.softmax(router_logits, dim=-1)  # (B, T, E)
    valid_mask = _resolve_valid_token_mask(
        token_mask, router_logits.shape[:2], router_logits.device
    )

    num_valid = int(valid_mask.sum().item())
    if num_valid == 0:
        return router_logits.new_zeros(())

    dispatch = _compute_dispatch_tensor(router_probs, top_k_probs, top_k_indices)

    valid_mask_f = valid_mask.unsqueeze(-1).to(dtype=router_probs.dtype)
    dispatch = dispatch * valid_mask_f
    router_probs = router_probs * valid_mask_f

    denom = router_probs.new_tensor(float(num_valid))
    tokens_per_expert = dispatch.sum(dim=(0, 1)) / denom
    router_prob_per_expert = router_probs.sum(dim=(0, 1)) / denom

    balance_type = str(load_balance_type).lower()
    if balance_type == "cv2":
        expert_usage = tokens_per_expert + 1e-8
        usage_mean = expert_usage.mean()
        usage_std = expert_usage.std(unbiased=False)
        return (usage_std / usage_mean) ** 2
    if balance_type not in {"switch", "importance"}:
        raise ValueError(
            "load_balance_type must be 'switch' or 'cv2'; "
            f"got {load_balance_type!r}"
        )
    return num_experts * (tokens_per_expert * router_prob_per_expert).sum()


# ---------------------------------------------------------------------------
# Plan A helpers – Temporal MoE
# ---------------------------------------------------------------------------


class ExpertFFN(nn.Module):
    """Single expert FFN (mirrors MetaFormerBlock.mlp architecture)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEFFN(nn.Module):
    """Mixture-of-Experts FFN layer for temporal encoder blocks."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        num_experts: int = 4,
        top_k: int = 1,
        dropout: float = 0.1,
        router_noise: float = 0.1,
        router_temperature: float = 1.0,
        use_shared_expert: bool = True,
        shared_expert_gate_init: float = -2.0,
        shared_mix_mode: str = "average",
        expert_dropout: float = 0.0,
        load_balance_type: str = "switch",
        routing_granularity: str = "token",
        segment_size: int = 8,
    ):
        super().__init__()
        granularity = str(routing_granularity).lower()
        if granularity not in {"token", "segment", "sequence"}:
            raise ValueError(
                "routing_granularity must be 'token', 'segment', or 'sequence'; "
                f"got {routing_granularity!r}"
            )
        mix_mode = str(shared_mix_mode).lower()
        if mix_mode not in {"average", "residual"}:
            raise ValueError(
                "shared_mix_mode must be 'average' or 'residual'; "
                f"got {shared_mix_mode!r}"
            )
        hidden_dim = int(dim * mlp_ratio)
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_granularity = granularity
        self.segment_size = max(1, int(segment_size))
        self.shared_mix_mode = mix_mode
        self.expert_dropout = max(float(expert_dropout), 0.0)
        self.load_balance_type = str(load_balance_type).lower()
        self._last_diagnostics: Dict[str, Any] = {}
        self.last_token_router_probs: Optional[torch.Tensor] = None
        self.last_token_gating_weights: Optional[torch.Tensor] = None
        self.last_valid_token_mask: Optional[torch.Tensor] = None

        self.router = TopKRouter(
            dim,
            num_experts,
            top_k,
            noise_std=router_noise,
            temperature=router_temperature,
        )
        self.experts = nn.ModuleList(
            [ExpertFFN(dim, hidden_dim, dropout) for _ in range(num_experts)]
        )

        self.use_shared_expert = use_shared_expert
        if use_shared_expert:
            self.shared_expert = ExpertFFN(dim, hidden_dim, dropout)
            # Keep the shared path as a low-weight fallback instead of a full
            # dense bypass that can wash out routed specialization.
            self.shared_expert_gate_logits = nn.Parameter(
                torch.tensor(float(shared_expert_gate_init), dtype=torch.float32)
            )

    def set_routing_config(
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

    def get_diagnostics(self) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        for key, value in self._last_diagnostics.items():
            if torch.is_tensor(value):
                if value.numel() == 1:
                    diagnostics[key] = float(value.detach().cpu().item())
                else:
                    diagnostics[key] = [float(v) for v in value.detach().cpu().tolist()]
            else:
                diagnostics[key] = value
        return diagnostics

    def forward(
        self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:"""
        valid_token_mask = _resolve_valid_token_mask(token_mask, x.shape[:2], x.device)
        if self.routing_granularity == "sequence":
            route_input, route_mask = _sequence_route_inputs(x, valid_token_mask)
        elif self.routing_granularity == "segment":
            route_input, route_mask = _segment_route_inputs(
                x, valid_token_mask, self.segment_size
            )
        else:
            route_input, route_mask = x, valid_token_mask

        route_probs, route_indices, router_logits = self.router(route_input)
        self.last_token_router_probs = _broadcast_route_matrix(
            F.softmax(router_logits.detach(), dim=-1),
            num_tokens=x.shape[1],
            routing_granularity=self.routing_granularity,
            segment_size=self.segment_size,
        )
        self.last_valid_token_mask = valid_token_mask.detach()
        top_k_probs, expert_indices = _broadcast_route_decisions(
            route_probs,
            route_indices,
            num_tokens=x.shape[1],
            routing_granularity=self.routing_granularity,
            segment_size=self.segment_size,
        )
        gating_weights = x.new_zeros(*x.shape[:2], self.num_experts)
        gating_weights.scatter_add_(-1, expert_indices, top_k_probs)
        self.last_token_gating_weights = gating_weights.detach()

        routed_output = torch.zeros_like(x)

        for i in range(self.num_experts):
            for k in range(self.top_k):
                dispatch_mask = (expert_indices[:, :, k] == i) & valid_token_mask
                if dispatch_mask.any():
                    expert_input = x[dispatch_mask]
                    expert_output = self.experts[i](expert_input)
                    weights = top_k_probs[:, :, k][dispatch_mask].unsqueeze(-1)
                    routed_output[dispatch_mask] = (
                        routed_output[dispatch_mask] + weights * expert_output
                    )

        if self.training and self.expert_dropout > 0:
            keep = (
                torch.rand(
                    routed_output.shape[0],
                    routed_output.shape[1],
                    1,
                    device=routed_output.device,
                    dtype=routed_output.dtype,
                )
                >= self.expert_dropout
            )
            routed_output = routed_output * keep

        shared_output = None
        shared_gate = None
        if self.use_shared_expert:
            shared_output = self.shared_expert(x)
            shared_output = shared_output * valid_token_mask.unsqueeze(-1).to(shared_output.dtype)
            shared_gate = torch.sigmoid(self.shared_expert_gate_logits).to(routed_output.dtype)
            if self.shared_mix_mode == "residual":
                # Always-on dense path plus a gated sparse residual (SpaMo M5).
                output = shared_output + shared_gate * routed_output
            else:
                output = (routed_output + shared_gate * shared_output) / (1.0 + shared_gate)
        else:
            output = routed_output

        aux_loss = load_balancing_loss(
            router_logits,
            route_probs,
            route_indices,
            self.num_experts,
            token_mask=route_mask,
            load_balance_type=self.load_balance_type,
        )

        self._last_diagnostics = _build_router_diagnostics(
            router_logits=router_logits.detach(),
            top_k_probs=route_probs.detach(),
            top_k_indices=route_indices.detach(),
            valid_token_mask=route_mask.detach(),
            top_k=self.top_k,
            router_temperature=self.router.temperature,
            router_noise=self.router.noise_std,
            shared_gate=None if shared_gate is None else shared_gate.detach(),
            routed_output=routed_output.detach(),
            shared_output=None if shared_output is None else shared_output.detach(),
            output_mask=valid_token_mask.detach(),
        )
        self._last_diagnostics["routing_granularity"] = self.routing_granularity
        self._last_diagnostics["segment_size"] = int(self.segment_size)
        self._last_diagnostics["shared_mix_mode"] = self.shared_mix_mode
        self._last_diagnostics["load_balance_type"] = self.load_balance_type

        if token_mask is not None:
            output = output * valid_token_mask.unsqueeze(-1).to(output.dtype)

        return output, aux_loss


# ---------------------------------------------------------------------------
# Plan B helpers – Fusion / Adaptor MoE
# ---------------------------------------------------------------------------


class ExpertAdaptor(nn.Module):
    """Single expert adaptor: Dv -> hidden -> Dl  (2-layer MLP)."""

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEAdaptor(nn.Module):
    """MoE Adaptor / Fusion module."""

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
        hidden_ratio: float = 2.0,
        shared_expert_gate_init: float = -2.0,
    ):
        super().__init__()
        hidden_dim = int(max(in_dim, out_dim) * hidden_ratio)
        self.num_experts = num_experts
        self.top_k = top_k
        self.out_dim = out_dim
        self._last_diagnostics: Dict[str, Any] = {}

        self.norm = nn.LayerNorm(in_dim)
        self.router = TopKRouter(
            in_dim,
            num_experts,
            top_k,
            noise_std=router_noise,
            temperature=router_temperature,
        )
        self.experts = nn.ModuleList(
            [
                ExpertAdaptor(in_dim, hidden_dim, out_dim, dropout)
                for _ in range(num_experts)
            ]
        )

        self.use_shared_expert = use_shared_expert
        if use_shared_expert:
            self.shared_expert = ExpertAdaptor(in_dim, hidden_dim, out_dim, dropout)
            self.shared_expert_gate_logits = nn.Parameter(
                torch.tensor(float(shared_expert_gate_init), dtype=torch.float32)
            )

    def set_routing_config(
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

    def get_diagnostics(self) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        for key, value in self._last_diagnostics.items():
            if torch.is_tensor(value):
                if value.numel() == 1:
                    diagnostics[key] = float(value.detach().cpu().item())
                else:
                    diagnostics[key] = [float(v) for v in value.detach().cpu().tolist()]
            else:
                diagnostics[key] = value
        return diagnostics

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Args:"""
        B, T, _ = x.shape
        x_normed = self.norm(x)

        top_k_probs, expert_indices, router_logits = self.router(x_normed)
        valid_token_mask = _resolve_valid_token_mask(mask, x.shape[:2], x.device)

        routed_output = torch.zeros(B, T, self.out_dim, device=x.device, dtype=x.dtype)

        for i in range(self.num_experts):
            for k in range(self.top_k):
                dispatch_mask = (expert_indices[:, :, k] == i) & valid_token_mask
                if dispatch_mask.any():
                    expert_input = x_normed[dispatch_mask]
                    expert_output = self.experts[i](expert_input)
                    weights = top_k_probs[:, :, k][dispatch_mask].unsqueeze(-1)
                    routed_output[dispatch_mask] = (
                        routed_output[dispatch_mask] + weights * expert_output
                    )

        shared_output = None
        shared_gate = None
        if self.use_shared_expert:
            shared_output = self.shared_expert(x_normed)
            shared_output = shared_output * valid_token_mask.unsqueeze(-1).to(shared_output.dtype)
            shared_gate = torch.sigmoid(self.shared_expert_gate_logits).to(routed_output.dtype)
            output = (routed_output + shared_gate * shared_output) / (1.0 + shared_gate)
        else:
            output = routed_output

        aux_loss = load_balancing_loss(
            router_logits,
            top_k_probs,
            expert_indices,
            self.num_experts,
            token_mask=valid_token_mask,
        )

        self._last_diagnostics = _build_router_diagnostics(
            router_logits=router_logits.detach(),
            top_k_probs=top_k_probs.detach(),
            top_k_indices=expert_indices.detach(),
            valid_token_mask=valid_token_mask.detach(),
            top_k=self.top_k,
            router_temperature=self.router.temperature,
            router_noise=self.router.noise_std,
            shared_gate=None if shared_gate is None else shared_gate.detach(),
            routed_output=routed_output.detach(),
            shared_output=None if shared_output is None else shared_output.detach(),
        )

        if mask is not None:
            output = output * valid_token_mask.unsqueeze(-1).to(output.dtype)

        return output, mask, aux_loss
