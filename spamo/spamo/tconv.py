import pdb
import copy
import math
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, num_classes=-1):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K5', 'K5', "P2"]
        elif self.conv_type == 4:
            self.kernel_size = ['K5', 'K5']
        elif self.conv_type == 5:
            self.kernel_size = ['K5', "P2", 'K5']
        elif self.conv_type == 6:
            self.kernel_size = ["P2", 'K5', 'K5']
        elif self.conv_type == 7:
            self.kernel_size = ["P2", 'K5', "P2", 'K5']
        elif self.conv_type == 8:
            self.kernel_size = ["P2", "P2", 'K5', 'K5']

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                    #MultiScale_TemporalConv(input_sz, self.hidden_size)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, 2)
            else:
                feat_len -= int(ks[1]) - 1
                #pass
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1) if logits is not None else None,
            "feat_len": lgt.cpu(),
        }


class TopKTokenRouter(nn.Module):
    """Route each valid output token to a sparse set of temporal experts."""

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 1,
        noise_std: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.noise_std = max(float(noise_std), 0.0)
        self.temperature = max(float(temperature), 1e-4)
        self.gate = nn.Linear(dim, self.num_experts, bias=False)

    def set_noise_std(self, noise_std: float) -> None:
        self.noise_std = max(float(noise_std), 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        logits = logits / self.temperature
        router_probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return top_k_probs, top_k_indices, logits


def _token_route_inputs(
    frame_feat: torch.Tensor,
    input_lengths: torch.Tensor,
    output_lengths: torch.Tensor,
    max_output_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create one length-aware routing descriptor per output token."""
    batch_size, channels, max_input_len = frame_feat.shape
    input_lengths = input_lengths.to(frame_feat.device).long().clamp(
        min=1, max=max_input_len
    )
    output_lengths = output_lengths.to(frame_feat.device).long().clamp(
        min=1, max=max_output_len
    )
    route_inputs = frame_feat.new_zeros(batch_size, max_output_len, channels)

    for batch_idx in range(batch_size):
        input_len = int(input_lengths[batch_idx].item())
        output_len = int(output_lengths[batch_idx].item())
        pooled = F.adaptive_avg_pool1d(
            frame_feat[batch_idx : batch_idx + 1, :, :input_len],
            output_len,
        ).transpose(1, 2)
        route_inputs[batch_idx, :output_len] = pooled[0]

    valid_mask = (
        torch.arange(max_output_len, device=frame_feat.device).unsqueeze(0)
        < output_lengths.unsqueeze(1)
    )
    return route_inputs, valid_mask


def _sequence_route_inputs(
    route_inputs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool all valid tokens into one routing descriptor per video."""
    mask = valid_mask.unsqueeze(-1).to(route_inputs.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    summary = (route_inputs * mask).sum(dim=1) / denom
    return summary.unsqueeze(1), valid_mask.new_ones(route_inputs.shape[0], 1)


def _segment_route_inputs(
    route_inputs: torch.Tensor,
    valid_mask: torch.Tensor,
    segment_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool non-overlapping output-token windows for segment routing."""
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


def _broadcast_route_decisions(
    top_k_probs: torch.Tensor,
    top_k_indices: torch.Tensor,
    num_tokens: int,
    routing_granularity: str,
    segment_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand sequence/segment router decisions back onto output tokens."""
    if routing_granularity == "token":
        return top_k_probs, top_k_indices
    if routing_granularity == "sequence":
        return (
            top_k_probs.expand(-1, num_tokens, -1),
            top_k_indices.expand(-1, num_tokens, -1),
        )
    batch_size, num_segments, top_k = top_k_probs.shape
    expanded_len = num_segments * segment_size
    probs = (
        top_k_probs.unsqueeze(2)
        .expand(-1, -1, segment_size, -1)
        .reshape(batch_size, expanded_len, top_k)
    )
    indices = (
        top_k_indices.unsqueeze(2)
        .expand(-1, -1, segment_size, -1)
        .reshape(batch_size, expanded_len, top_k)
    )
    return probs[:, :num_tokens], indices[:, :num_tokens]


def _load_balancing_loss(
    router_logits: torch.Tensor,
    top_k_probs: torch.Tensor,
    top_k_indices: torch.Tensor,
    num_experts: int,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    router_probs = F.softmax(router_logits, dim=-1)
    dispatch = torch.zeros_like(router_probs)
    dispatch.scatter_add_(dim=-1, index=top_k_indices, src=top_k_probs)

    valid_mask = token_mask.unsqueeze(-1).to(router_probs.dtype)
    denom = valid_mask[..., 0].sum().clamp_min(1.0)
    tokens_per_expert = (dispatch * valid_mask).sum(dim=(0, 1)) / denom
    router_prob_per_expert = (router_probs * valid_mask).sum(dim=(0, 1)) / denom
    return num_experts * (tokens_per_expert * router_prob_per_expert).sum()


def _cv2_load_balancing_loss(
    router_logits: torch.Tensor,
    top_k_probs: torch.Tensor,
    top_k_indices: torch.Tensor,
    num_experts: int,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize squared imbalance in both router importance and sparse load."""
    router_probs = F.softmax(router_logits, dim=-1)
    dispatch = torch.zeros_like(router_probs)
    dispatch.scatter_add_(dim=-1, index=top_k_indices, src=top_k_probs)
    valid_mask = token_mask.unsqueeze(-1).to(router_probs.dtype)
    importance = (router_probs * valid_mask).sum(dim=(0, 1))
    load = (dispatch * valid_mask).sum(dim=(0, 1))
    importance = importance / importance.sum().clamp_min(1e-8)
    load = load / load.sum().clamp_min(1e-8)
    # Each term is zero at a uniform distribution.
    return (
        num_experts * importance.square().sum() - 1.0
        + num_experts * load.square().sum() - 1.0
    )


class SparseMoETemporalConv(nn.Module):
    """MoE over SpaMo TemporalConv experts with token/segment/sequence routing."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        conv_type: int = 2,
        num_experts: int = 4,
        top_k: int = 1,
        router_noise: float = 0.1,
        router_temperature: float = 1.0,
        use_shared_expert: bool = True,
        shared_expert_gate_init: float = -2.0,
        routing_granularity: str = "token",
        segment_size: int = 8,
        router_z_loss_weight: float = 0.0,
        expert_dropout: float = 0.0,
        router_noise_anneal_steps: int = 0,
        load_balance_type: str = "switch",
        shared_mix_mode: str = "average",
    ):
        super().__init__()
        granularity = str(routing_granularity).lower()
        if granularity not in {"token", "segment", "sequence"}:
            raise ValueError(
                "routing_granularity must be 'token', 'segment', or 'sequence'; "
                f"got {routing_granularity!r}"
            )
        self.routing_granularity = granularity
        self.segment_size = max(1, int(segment_size))
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.use_shared_expert = bool(use_shared_expert)
        self.router_noise_init = max(float(router_noise), 0.0)
        self.router_noise_anneal_steps = max(int(router_noise_anneal_steps), 0)
        self.router_z_loss_weight = max(float(router_z_loss_weight), 0.0)
        self.expert_dropout = min(max(float(expert_dropout), 0.0), 1.0)
        load_balance_type = str(load_balance_type).lower()
        if load_balance_type not in {"switch", "cv2"}:
            raise ValueError(
                "load_balance_type must be 'switch' or 'cv2'; "
                f"got {load_balance_type!r}"
            )
        shared_mix_mode = str(shared_mix_mode).lower()
        if shared_mix_mode not in {"average", "residual"}:
            raise ValueError(
                "shared_mix_mode must be 'average' or 'residual'; "
                f"got {shared_mix_mode!r}"
            )
        self.load_balance_type = load_balance_type
        self.shared_mix_mode = shared_mix_mode
        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_z_loss: Optional[torch.Tensor] = None
        self.last_entropy_loss: Optional[torch.Tensor] = None
        self.last_diagnostics: Dict = {}
        self.last_token_router_probs: Optional[torch.Tensor] = None
        self.last_token_gating_weights: Optional[torch.Tensor] = None
        self.last_valid_token_mask: Optional[torch.Tensor] = None

        self.router = TopKTokenRouter(
            input_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            noise_std=router_noise,
            temperature=router_temperature,
        )
        self.experts = nn.ModuleList(
            [
                TemporalConv(input_size, hidden_size, conv_type=conv_type)
                for _ in range(self.num_experts)
            ]
        )

        if self.use_shared_expert:
            self.shared_expert = TemporalConv(input_size, hidden_size, conv_type=conv_type)
            self.shared_expert_gate_logits = nn.Parameter(
                torch.tensor(float(shared_expert_gate_init), dtype=torch.float32)
            )

    def update_lgt(self, lgt: torch.Tensor) -> torch.Tensor:
        return self.experts[0].update_lgt(lgt)

    def get_aux_loss(self) -> Optional[torch.Tensor]:
        return self.last_aux_loss

    def anneal_router_noise(self, global_step: int) -> None:
        if self.router_noise_anneal_steps <= 0:
            return
        frac = min(1.0, max(float(global_step), 0.0) / float(self.router_noise_anneal_steps))
        self.router.set_noise_std(self.router_noise_init * (1.0 - frac))

    def _run_expert(
        self,
        expert: TemporalConv,
        frame_feat: torch.Tensor,
        lgt: torch.Tensor,
    ) -> torch.Tensor:
        output = expert(frame_feat, lgt)["visual_feat"]
        return output.permute(1, 2, 0)

    def forward(self, frame_feat: torch.Tensor, lgt: torch.Tensor):
        input_lengths = lgt.to(frame_feat.device).long().clamp(
            min=1, max=frame_feat.shape[-1]
        )
        input_mask = (
            torch.arange(frame_feat.shape[-1], device=frame_feat.device).unsqueeze(0)
            < input_lengths.unsqueeze(1)
        )
        frame_feat = frame_feat * input_mask.unsqueeze(1).to(frame_feat.dtype)

        output_lengths = self.update_lgt(lgt).to(frame_feat.device).long()
        max_output_len = int(
            self.update_lgt(
                torch.tensor(
                    [frame_feat.shape[-1]],
                    device=frame_feat.device,
                    dtype=torch.long,
                )
            )
            .long()
            .item()
        )
        token_route_input, valid_mask = _token_route_inputs(
            frame_feat,
            input_lengths=lgt,
            output_lengths=output_lengths,
            max_output_len=max_output_len,
        )
        if self.routing_granularity == "sequence":
            route_input, route_mask = _sequence_route_inputs(
                token_route_input, valid_mask
            )
        elif self.routing_granularity == "segment":
            route_input, route_mask = _segment_route_inputs(
                token_route_input, valid_mask, self.segment_size
            )
        else:
            route_input, route_mask = token_route_input, valid_mask

        route_probs, route_indices, router_logits = self.router(route_input)
        balance_fn = (
            _cv2_load_balancing_loss
            if self.load_balance_type == "cv2"
            else _load_balancing_loss
        )
        self.last_aux_loss = balance_fn(
            router_logits,
            route_probs,
            route_indices,
            self.num_experts,
            token_mask=route_mask,
        )
        valid_route = route_mask.unsqueeze(-1).to(router_logits.dtype)
        z_denom = valid_route.sum().clamp_min(1.0)
        self.last_z_loss = (router_logits.float().pow(2) * valid_route).sum() / z_denom
        full_router_probs = F.softmax(router_logits.float(), dim=-1)
        entropy = -(
            full_router_probs.clamp_min(1e-9)
            * full_router_probs.clamp_min(1e-9).log()
        ).sum(dim=-1)
        mean_entropy = entropy.masked_select(route_mask).mean()
        self.last_entropy_loss = (
            math.log(self.num_experts) - mean_entropy
        ).clamp_min(0.0)
        top_k_probs, expert_indices = _broadcast_route_decisions(
            route_probs,
            route_indices,
            num_tokens=max_output_len,
            routing_granularity=self.routing_granularity,
            segment_size=self.segment_size,
        )
        token_router_probs, _ = _broadcast_route_decisions(
            F.softmax(router_logits.detach(), dim=-1),
            route_indices.detach(),
            num_tokens=max_output_len,
            routing_granularity=self.routing_granularity,
            segment_size=self.segment_size,
        )
        self.last_token_router_probs = token_router_probs
        gating_weights = top_k_probs.new_zeros(
            *top_k_probs.shape[:2], self.num_experts
        )
        gating_weights.scatter_add_(-1, expert_indices, top_k_probs)
        self.last_token_gating_weights = gating_weights.detach()
        self.last_valid_token_mask = valid_mask.detach()

        routed_output = frame_feat.new_zeros(
            frame_feat.shape[0], max_output_len, self.hidden_size
        )

        for expert_idx, expert in enumerate(self.experts):
            selected = ((expert_indices == expert_idx).any(dim=-1) & valid_mask)
            if not selected.any():
                continue

            expert_output = self._run_expert(
                expert,
                frame_feat,
                lgt,
            )
            expert_output = expert_output.transpose(1, 2)
            expert_weight = torch.zeros_like(top_k_probs[..., 0])
            for route_idx in range(self.top_k):
                expert_weight = expert_weight + (
                    top_k_probs[..., route_idx]
                    * (expert_indices[..., route_idx] == expert_idx).to(
                        top_k_probs.dtype
                    )
                )

            routed_output = (
                routed_output
                + expert_output * expert_weight.unsqueeze(-1).to(expert_output.dtype)
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

        shared_gate = None
        if self.use_shared_expert:
            shared_output = self._run_expert(
                self.shared_expert, frame_feat, lgt
            ).transpose(1, 2)
            shared_gate = torch.sigmoid(self.shared_expert_gate_logits).to(routed_output.dtype)
            if self.shared_mix_mode == "residual":
                routed_output = shared_output + shared_gate * routed_output
            else:
                routed_output = (
                    routed_output + shared_gate * shared_output
                ) / (1.0 + shared_gate)

        router_probs = F.softmax(router_logits.detach(), dim=-1)
        dispatch = torch.zeros_like(router_probs)
        dispatch.scatter_add_(dim=-1, index=route_indices.detach(), src=route_probs.detach())
        valid_mask_f = route_mask.unsqueeze(-1).to(router_probs.dtype)
        denom = valid_mask_f[..., 0].sum().clamp_min(1.0)
        masked_router_probs = router_probs * valid_mask_f
        self.last_diagnostics = {
            "expert_mass": (
                (dispatch * valid_mask_f).sum(dim=(0, 1)) / denom
            ).detach().cpu(),
            "expert_prob": (
                masked_router_probs.sum(dim=(0, 1)) / denom
            ).detach().cpu(),
            "router_entropy": (
                -(
                    router_probs.clamp_min(1e-9)
                    * router_probs.clamp_min(1e-9).log()
                )
                .sum(dim=-1)
                .masked_select(route_mask)
                .mean()
                .detach()
                .cpu()
            ),
            "top_k": self.top_k,
            "routing_granularity": self.routing_granularity,
            "segment_size": self.segment_size,
            "load_balance_type": self.load_balance_type,
            "shared_mix_mode": self.shared_mix_mode,
            "shared_gate": None if shared_gate is None else shared_gate.detach().cpu(),
        }

        routed_output = routed_output * valid_mask.unsqueeze(-1).to(
            routed_output.dtype
        )
        return {
            "visual_feat": routed_output.permute(1, 0, 2),
            "conv_logits": None,
            "feat_len": output_lengths.cpu(),
        }
    

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, stride=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out + residual  # Element-wise addition
        out = self.relu(out)
        return out


class GlorTemporalConv(nn.Module):
    def __init__(self, input_channels, output_channels, dilation_rate=1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)
        )
        self.layers.append(ResidualBlock(output_channels))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 2, 1)

