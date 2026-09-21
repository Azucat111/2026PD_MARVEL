"""Paper-faithful AHGNN + Actor-Critic for high-level GPPO task allocation.

This module implements the neural side of the Phase-2 graph:

    G = (T, U, L, E)

with:
    - distinct UAV/task projections;
    - edge-aware adaptive attention when updating UAV nodes;
    - task-node update using predecessor, successor, UAV-neighbor, and self
      information through separate MLP paths;
    - a pairwise actor over (subtask, UAV);
    - a graph-value critic;
    - GPPO action masking.

The implementation intentionally keeps MARVEL's original `utils/model.py`
untouched.  MARVEL remains the low-level exploration controller.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .task_graph import EDGE_FEATURE_NAMES, TASK_FEATURE_NAMES, UAV_FEATURE_NAMES


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    """Mean over valid entries.

    mask convention here:
        True = valid
    """
    if mask is None:
        return x.mean(dim=dim)

    mask_f = mask.to(dtype=x.dtype)
    while mask_f.dim() < x.dim():
        mask_f = mask_f.unsqueeze(-1)
    weighted = x * mask_f
    denom = mask_f.sum(dim=dim).clamp_min(1.0)
    return weighted.sum(dim=dim) / denom


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        activation=nn.ELU,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveUAVAttention(nn.Module):
    """Edge-aware adaptive attention from subtask neighbors to each UAV.

    The GPPO paper updates UAV embeddings first.  Each UAV aggregates adjacent
    subtask information, and the subtask representation is augmented with the
    UAV-task edge feature.  Separate transformations are used for UAV and task
    sides.

    Input shapes:
        uav_h          [B, U, D]
        task_h         [B, T, D]
        edge_h         [B, T, U, D]
        capability     [B, T, U] bool
        task_valid     [B, T] bool (optional)
        uav_valid      [B, U] bool (optional)

    Output:
        updated_uav_h  [B, U, D]
        attention      [B, T, U]
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Distinct transforms for UAV and task/edge sides.
        self.uav_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.task_proj = nn.Linear(embed_dim * 2, embed_dim, bias=False)

        # Attention score vector c in the paper-inspired formulation.
        self.attn_score = nn.Linear(embed_dim * 2, 1, bias=False)

        # Adaptive modulation f(task, uav, edge).
        self.adaptive_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

        self.self_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        uav_h: torch.Tensor,
        task_h: torch.Tensor,
        edge_h: torch.Tensor,
        capability: torch.Tensor,
        task_valid: Optional[torch.Tensor] = None,
        uav_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, U, D = uav_h.shape
        _, T, _ = task_h.shape

        # Expand per UAV-task pair.
        uav_pair = uav_h.unsqueeze(1).expand(B, T, U, D)
        task_pair = task_h.unsqueeze(2).expand(B, T, U, D)

        u_proj = self.uav_proj(uav_pair)
        task_edge = torch.cat([task_pair, edge_h], dim=-1)
        te_proj = self.task_proj(task_edge)

        raw_score = self.attn_score(torch.cat([u_proj, te_proj], dim=-1)).squeeze(-1)
        raw_score = F.rrelu(raw_score, training=self.training)

        adaptive = self.adaptive_gate(
            torch.cat([task_pair, uav_pair, edge_h], dim=-1)
        ).squeeze(-1)

        # Adaptive gate modulates the attention logits while retaining sign.
        logits = raw_score * (0.5 + adaptive)

        valid = capability.bool()
        if task_valid is not None:
            valid = valid & task_valid.bool().unsqueeze(-1)
        if uav_valid is not None:
            valid = valid & uav_valid.bool().unsqueeze(1)

        # For UAVs with no valid neighbor, use zero attention and only the
        # self term.  This avoids NaNs from softmax(all -inf).
        any_neighbor = valid.any(dim=1)  # [B, U]
        safe_logits = logits.masked_fill(~valid, -1e9)
        attention = torch.softmax(safe_logits, dim=1)
        attention = attention * valid.to(attention.dtype)
        attention = torch.where(
            any_neighbor.unsqueeze(1),
            attention,
            torch.zeros_like(attention),
        )

        # Aggregated task contribution.
        task_message = te_proj * adaptive.unsqueeze(-1)
        aggregate = (attention.unsqueeze(-1) * task_message).sum(dim=1)

        updated = self.self_proj(uav_h) + aggregate
        updated = F.elu(updated)
        updated = self.norm(updated + uav_h)

        if uav_valid is not None:
            updated = updated * uav_valid.unsqueeze(-1).to(updated.dtype)

        return updated, attention


class TaskNodeUpdater(nn.Module):
    """Task-node update with separate predecessor/successor/UAV/self paths.

    The GPPO paper describes separate MLP processing for predecessor task,
    successor task, neighboring UAV information, the task itself, and their
    integration.  This module follows that structure.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.pred_mlp = MLP(embed_dim, embed_dim)
        self.succ_mlp = MLP(embed_dim, embed_dim)
        self.uav_mlp = MLP(embed_dim, embed_dim)
        self.self_mlp = MLP(embed_dim, embed_dim)
        self.combine_mlp = MLP(embed_dim * 4, embed_dim, hidden_dim=embed_dim * 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        task_h: torch.Tensor,
        uav_h: torch.Tensor,
        precedence: torch.Tensor,
        capability: torch.Tensor,
        task_valid: Optional[torch.Tensor] = None,
        uav_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = task_h.shape

        # precedence[p, s] == True means p precedes s.
        pred_mask = precedence.bool().transpose(1, 2)  # [B, successor, predecessor]
        succ_mask = precedence.bool()                  # [B, predecessor, successor]

        pred_msg = torch.bmm(
            pred_mask.to(task_h.dtype),
            task_h,
        )
        pred_den = pred_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(task_h.dtype)
        pred_msg = pred_msg / pred_den

        succ_msg = torch.bmm(
            succ_mask.to(task_h.dtype),
            task_h,
        )
        succ_den = succ_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(task_h.dtype)
        succ_msg = succ_msg / succ_den

        # capability [B, T, U] gives neighboring UAVs for each task.
        task_uav_mask = capability.bool()
        if uav_valid is not None:
            task_uav_mask = task_uav_mask & uav_valid.bool().unsqueeze(1)

        uav_sum = torch.bmm(task_uav_mask.to(uav_h.dtype), uav_h)
        uav_den = task_uav_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(uav_h.dtype)
        uav_msg = uav_sum / uav_den

        pred_h = self.pred_mlp(pred_msg)
        succ_h = self.succ_mlp(succ_msg)
        uav_h_agg = self.uav_mlp(uav_msg)
        self_h = self.self_mlp(task_h)

        combined = torch.cat([pred_h, succ_h, uav_h_agg, self_h], dim=-1)
        updated = self.combine_mlp(combined)
        updated = self.norm(F.elu(updated) + task_h)

        if task_valid is not None:
            updated = updated * task_valid.unsqueeze(-1).to(updated.dtype)

        return updated


class AHGNN(nn.Module):
    """Adaptive Heterogeneous Graph Neural Network."""

    def __init__(
        self,
        uav_feature_dim: int = len(UAV_FEATURE_NAMES),
        task_feature_dim: int = len(TASK_FEATURE_NAMES),
        edge_feature_dim: int = len(EDGE_FEATURE_NAMES),
        embed_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.uav_input = nn.Linear(uav_feature_dim, embed_dim)
        self.task_input = nn.Linear(task_feature_dim, embed_dim)
        self.edge_input = nn.Linear(edge_feature_dim, embed_dim)

        self.uav_layers = nn.ModuleList(
            [AdaptiveUAVAttention(embed_dim) for _ in range(n_layers)]
        )
        self.task_layers = nn.ModuleList(
            [TaskNodeUpdater(embed_dim) for _ in range(n_layers)]
        )

    def forward(
        self,
        uav_features: torch.Tensor,
        task_features: torch.Tensor,
        edge_features: torch.Tensor,
        capability_matrix: torch.Tensor,
        precedence_matrix: torch.Tensor,
        uav_valid_mask: Optional[torch.Tensor] = None,
        task_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        uav_h = self.uav_input(uav_features)
        task_h = self.task_input(task_features)
        edge_h = self.edge_input(edge_features)

        all_attention = []

        for uav_layer, task_layer in zip(self.uav_layers, self.task_layers):
            uav_h, attention = uav_layer(
                uav_h=uav_h,
                task_h=task_h,
                edge_h=edge_h,
                capability=capability_matrix,
                task_valid=task_valid_mask,
                uav_valid=uav_valid_mask,
            )
            task_h = task_layer(
                task_h=task_h,
                uav_h=uav_h,
                precedence=precedence_matrix,
                capability=capability_matrix,
                task_valid=task_valid_mask,
                uav_valid=uav_valid_mask,
            )
            all_attention.append(attention)

        return {
            "uav_embeddings": uav_h,
            "task_embeddings": task_h,
            "attention": torch.stack(all_attention, dim=1),
        }


class PairwiseActor(nn.Module):
    """Score every feasible (subtask, UAV) pair."""

    def __init__(self, embed_dim: int, edge_feature_dim: int = len(EDGE_FEATURE_NAMES)):
        super().__init__()
        self.edge_proj = nn.Linear(edge_feature_dim, embed_dim)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.ELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        task_h: torch.Tensor,
        uav_h: torch.Tensor,
        edge_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = task_h.shape
        U = uav_h.shape[1]

        task_pair = task_h.unsqueeze(2).expand(B, T, U, D)
        uav_pair = uav_h.unsqueeze(1).expand(B, T, U, D)
        edge_pair = self.edge_proj(edge_features)

        logits = self.scorer(
            torch.cat([task_pair, uav_pair, edge_pair], dim=-1)
        ).squeeze(-1)

        # TaskGraph convention: True = invalid.
        if action_mask is not None:
            logits = logits.masked_fill(action_mask.bool(), -1e9)

        return logits


class GraphCritic(nn.Module):
    """Graph-level value function for PPO."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        task_h: torch.Tensor,
        uav_h: torch.Tensor,
        task_valid_mask: Optional[torch.Tensor] = None,
        uav_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        task_global = _masked_mean(task_h, task_valid_mask, dim=1)
        uav_global = _masked_mean(uav_h, uav_valid_mask, dim=1)
        return self.value(torch.cat([task_global, uav_global], dim=-1)).squeeze(-1)


class GPPOActorCritic(nn.Module):
    """High-level GPPO model operating on the Phase-2 heterogeneous graph."""

    def __init__(
        self,
        uav_feature_dim: int = len(UAV_FEATURE_NAMES),
        task_feature_dim: int = len(TASK_FEATURE_NAMES),
        edge_feature_dim: int = len(EDGE_FEATURE_NAMES),
        embed_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.ahgnn = AHGNN(
            uav_feature_dim=uav_feature_dim,
            task_feature_dim=task_feature_dim,
            edge_feature_dim=edge_feature_dim,
            embed_dim=embed_dim,
            n_layers=n_layers,
        )
        self.actor = PairwiseActor(embed_dim, edge_feature_dim=edge_feature_dim)
        self.critic = GraphCritic(embed_dim)

    def forward(
        self,
        uav_features: torch.Tensor,
        task_features: torch.Tensor,
        edge_features: torch.Tensor,
        capability_matrix: torch.Tensor,
        precedence_matrix: torch.Tensor,
        action_mask: torch.Tensor,
        uav_valid_mask: Optional[torch.Tensor] = None,
        task_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.ahgnn(
            uav_features=uav_features,
            task_features=task_features,
            edge_features=edge_features,
            capability_matrix=capability_matrix,
            precedence_matrix=precedence_matrix,
            uav_valid_mask=uav_valid_mask,
            task_valid_mask=task_valid_mask,
        )

        uav_h = encoded["uav_embeddings"]
        task_h = encoded["task_embeddings"]

        pair_logits = self.actor(
            task_h=task_h,
            uav_h=uav_h,
            edge_features=edge_features,
            action_mask=action_mask,
        )

        flat_logits = pair_logits.flatten(start_dim=1)
        flat_mask = action_mask.flatten(start_dim=1).bool()

        # Detect states with no feasible high-level action.  The caller should
        # handle these as terminal/no-op states rather than sampling nonsense.
        has_valid_action = (~flat_mask).any(dim=1)

        value = self.critic(
            task_h=task_h,
            uav_h=uav_h,
            task_valid_mask=task_valid_mask,
            uav_valid_mask=uav_valid_mask,
        )

        return {
            "pair_logits": pair_logits,
            "flat_logits": flat_logits,
            "value": value,
            "has_valid_action": has_valid_action,
            "uav_embeddings": uav_h,
            "task_embeddings": task_h,
            "attention": encoded["attention"],
        }

    def distribution(
        self,
        model_out: Dict[str, torch.Tensor],
    ) -> torch.distributions.Categorical:
        if not torch.all(model_out["has_valid_action"]):
            raise RuntimeError(
                "At least one batch element has no valid GPPO action. "
                "Handle terminal/no-op states before sampling."
            )
        return torch.distributions.Categorical(logits=model_out["flat_logits"])

    @torch.no_grad()
    def act(
        self,
        *,
        deterministic: bool = False,
        **forward_kwargs,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(**forward_kwargs)
        dist = self.distribution(out)

        if deterministic:
            action = torch.argmax(out["flat_logits"], dim=-1)
        else:
            action = dist.sample()

        return {
            **out,
            "action": action,
            "log_prob": dist.log_prob(action),
            "entropy": dist.entropy(),
        }


def batch_single_task_graph(graph, device=None) -> Dict[str, torch.Tensor]:
    """Convert one Phase-2 TaskGraph into a batch of size 1."""
    tensors = graph.to_torch(device=device)
    return {
        "uav_features": tensors["uav_features"].unsqueeze(0),
        "task_features": tensors["task_features"].unsqueeze(0),
        "edge_features": tensors["edge_features"].unsqueeze(0),
        "capability_matrix": tensors["capability_matrix"].unsqueeze(0),
        "precedence_matrix": tensors["precedence_matrix"].unsqueeze(0),
        "action_mask": tensors["action_mask"].unsqueeze(0),
        "uav_valid_mask": torch.ones(
            (1, graph.num_uavs), dtype=torch.bool, device=device
        ),
        "task_valid_mask": torch.ones(
            (1, graph.num_subtasks), dtype=torch.bool, device=device
        ),
    }
