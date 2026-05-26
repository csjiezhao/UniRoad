from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class EdgePrototypeNet(nn.Module):
    """Road prototype encoder with fixed z=128 contract (role_dim=64)."""

    def __init__(
        self,
        node_feat_dim: int,
        node_raw_dim: int,
        edge_feat_dim: int,
        edge_attr_dim: int,
        hidden_dim: int = 128,
        role_dim: int = 64,
        role_count: int = 20,
        dropout: float = 0.1,
        edge_context_mode: str = "bi",
        proj_dim: int = 128,
        role_bank_momentum: float = 0.96,
        role_bank_mix: float = 0.3,
    ):
        super().__init__()
        self.node_raw_dim = int(node_raw_dim)
        self.edge_feat_dim = int(edge_feat_dim)
        self.edge_attr_dim = int(edge_attr_dim)
        self.hidden_dim = int(hidden_dim)
        self.role_dim = int(role_dim)
        self.role_count = int(role_count)
        self.dropout = float(dropout)

        if edge_context_mode != "bi":
            raise ValueError("Finalized prototype freezes edge_context_mode='bi'.")
        self.edge_context_mode = edge_context_mode

        self.conv1 = GATv2Conv(
            in_channels=node_feat_dim,
            out_channels=hidden_dim,
            heads=2,
            concat=False,
            dropout=dropout,
            edge_dim=edge_attr_dim,
        )
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=2,
            concat=False,
            dropout=dropout,
            edge_dim=edge_attr_dim,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        readout_in_dim = hidden_dim * 2 + edge_feat_dim + node_raw_dim * 2 + edge_attr_dim * 4
        self.edge_readout = nn.Sequential(
            nn.Linear(readout_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.role_proj = nn.Linear(hidden_dim, role_dim)
        self.role_assign = nn.Linear(role_dim, role_count)
        self.role_embeddings = nn.Parameter(torch.randn(role_count, role_dim) * 0.02)

        self.register_buffer("prototype_bank", torch.randn(role_count, role_dim) * 0.02)
        self.role_bank_momentum = float(role_bank_momentum)
        self.role_bank_mix = float(role_bank_mix)

        self.residual_mlp = nn.Sequential(
            nn.Linear(role_dim * 2, role_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(role_dim, role_dim),
        )

        self.struct_decoder = nn.Sequential(
            nn.Linear(role_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),
        )
        self.mask_edge_decoder = nn.Sequential(
            nn.Linear(role_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_feat_dim),
        )
        self.mask_node_decoder = nn.Sequential(
            nn.Linear(role_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_raw_dim * 2),
        )

        self.proj_head = nn.Sequential(
            nn.Linear(role_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.pred_head = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Compactness head keeps representation tight but avoids collapse via variance floor.
        self.compact_head = nn.Sequential(
            nn.Linear(role_dim * 2, role_dim),
            nn.ReLU(),
            nn.Linear(role_dim, role_dim),
        )

    def _aggregate_edge_context(self, num_nodes: int, node_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((num_nodes, edge_attr.shape[-1]), device=edge_attr.device, dtype=edge_attr.dtype)
        cnt = torch.zeros((num_nodes, 1), device=edge_attr.device, dtype=edge_attr.dtype)
        out.index_add_(0, node_index, edge_attr)
        cnt.index_add_(0, node_index, torch.ones((node_index.shape[0], 1), device=edge_attr.device, dtype=edge_attr.dtype))
        return out / cnt.clamp(min=1.0)

    def _edge_context(self, num_nodes: int, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        outgoing = self._aggregate_edge_context(num_nodes=num_nodes, node_index=edge_index[0], edge_attr=edge_attr)
        incoming = self._aggregate_edge_context(num_nodes=num_nodes, node_index=edge_index[1], edge_attr=edge_attr)
        return outgoing, incoming

    def _apply_conv(self, layer_id: int, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        conv = self.conv1 if layer_id == 1 else self.conv2
        return conv(x, edge_index, edge_attr=edge_attr)

    @torch.no_grad()
    def update_prototype_bank(self, h_role: torch.Tensor, pi: torch.Tensor) -> None:
        assign = pi.sum(dim=0, keepdim=False).unsqueeze(-1).clamp(min=1e-6)
        proto = torch.matmul(pi.transpose(0, 1), h_role) / assign
        self.prototype_bank.mul_(self.role_bank_momentum).add_(proto * (1.0 - self.role_bank_momentum))

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x_in = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        h1 = self._apply_conv(1, x_in, edge_index, edge_attr)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = self._apply_conv(2, h1, edge_index, edge_attr)
        h2 = self.norm2(h2 + h1)
        x = F.relu(h2)

        out_ctx, in_ctx = self._edge_context(num_nodes=x.shape[0], edge_index=edge_index, edge_attr=edge_attr)
        center_edge_index = batch.center_edge_index
        u_idx = center_edge_index[0]
        v_idx = center_edge_index[1]

        h_u = x[u_idx]
        h_v = x[v_idx]
        x_u = batch.x[u_idx, : self.node_raw_dim]
        x_v = batch.x[v_idx, : self.node_raw_dim]

        u_out = out_ctx[u_idx]
        u_in = in_ctx[u_idx]
        v_out = out_ctx[v_idx]
        v_in = in_ctx[v_idx]

        h_e = self.edge_readout(
            torch.cat([h_u, h_v, batch.center_edge_feat, x_u, x_v, u_out, u_in, v_out, v_in], dim=-1)
        )
        h_role = self.role_proj(h_e)

        role_logits = self.role_assign(h_role)
        pi = torch.softmax(role_logits, dim=-1)

        r_soft = torch.matmul(pi, self.role_embeddings)
        r_bank = torch.matmul(pi, self.prototype_bank.detach())
        r = (1.0 - self.role_bank_mix) * r_soft + self.role_bank_mix * r_bank

        eps = self.residual_mlp(torch.cat([h_role, r], dim=-1))
        z = torch.cat([r, eps], dim=-1)

        pred_struct = self.struct_decoder(z)
        pred_sketch_cont = pred_struct[:, :7]
        pred_sketch_hist = pred_struct[:, 7:15]
        pred_deadend_logit = pred_struct[:, 15:16]

        pred_mask_edge = self.mask_edge_decoder(z)
        pred_mask_nodes = self.mask_node_decoder(z)

        proj = self.proj_head(z)
        pred_proj = self.pred_head(proj)
        compact_z = self.compact_head(z)

        return {
            "h": h_e,
            "h_role": h_role,
            "pi": pi,
            "r": r,
            "eps": eps,
            "z": z,
            "role_embeddings": self.role_embeddings,
            "prototype_bank": self.prototype_bank,
            "pred_sketch_cont": pred_sketch_cont,
            "pred_sketch_hist": pred_sketch_hist,
            "pred_deadend_logit": pred_deadend_logit,
            "pred_mask_edge": pred_mask_edge,
            "pred_mask_nodes": pred_mask_nodes,
            "proj": proj,
            "pred_proj": pred_proj,
            "compact_z": compact_z,
        }


def compute_pretrain_core_losses(
    outputs: Dict[str, torch.Tensor],
    batch,
    w_role_ent: float = 0.02,
    w_role_orth: float = 0.02,
) -> Dict[str, torch.Tensor]:
    loss_cont = F.mse_loss(outputs["pred_sketch_cont"], batch.sketch_cont)
    loss_hist = F.mse_loss(outputs["pred_sketch_hist"], batch.sketch_hist)
    loss_deadend = F.binary_cross_entropy_with_logits(outputs["pred_deadend_logit"], batch.sketch_deadend)
    loss_struct = loss_cont + loss_hist + loss_deadend

    p_usage = outputs["pi"].mean(dim=0)
    usage_entropy = -(p_usage * torch.log(p_usage + 1e-12)).sum()

    emb = outputs["role_embeddings"]
    emb = F.normalize(emb, dim=-1)
    gram = torch.matmul(emb, emb.t())
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    role_orth = ((gram - eye) ** 2).mean()

    # Compactness: role-aligned tightness + variance floor to avoid collapse.
    tight = ((outputs["h_role"] - outputs["r"]) ** 2).mean()
    var_mean = outputs["compact_z"].var(dim=0, unbiased=False).mean()
    var_floor = F.relu(0.05 - var_mean)
    compact = tight + 0.2 * var_floor

    role_reg = -w_role_ent * usage_entropy + w_role_orth * role_orth

    return {
        "struct": loss_struct,
        "cont": loss_cont,
        "hist": loss_hist,
        "deadend": loss_deadend,
        "usage_entropy": usage_entropy,
        "role_orth": role_orth,
        "compact": compact,
        "role_reg": role_reg,
    }


def compute_mask_loss(
    outputs_masked: Dict[str, torch.Tensor],
    target_center_edge_feat: torch.Tensor,
    target_center_node_raw: torch.Tensor,
) -> torch.Tensor:
    loss_edge = F.mse_loss(outputs_masked["pred_mask_edge"], target_center_edge_feat)
    loss_nodes = F.mse_loss(outputs_masked["pred_mask_nodes"], target_center_node_raw)
    return loss_edge + loss_nodes


def compute_consistency_loss(
    outputs_view1: Dict[str, torch.Tensor],
    outputs_view2: Dict[str, torch.Tensor],
    consistency_type: str = "none",
) -> torch.Tensor:
    if consistency_type == "none":
        return outputs_view1["z"].new_tensor(0.0)

    if consistency_type not in ("byol_like", "simsiam_like"):
        raise ValueError(f"Unsupported consistency_type: {consistency_type}")

    z1 = F.normalize(outputs_view1["proj"], dim=-1)
    z2 = F.normalize(outputs_view2["proj"], dim=-1)
    p1 = F.normalize(outputs_view1["pred_proj"], dim=-1)
    p2 = F.normalize(outputs_view2["pred_proj"], dim=-1)

    if consistency_type == "byol_like":
        loss12 = F.mse_loss(p1, z2.detach())
        loss21 = F.mse_loss(p2, z1.detach())
    else:
        loss12 = 1.0 - (p1 * z2.detach()).sum(dim=-1).mean()
        loss21 = 1.0 - (p2 * z1.detach()).sum(dim=-1).mean()
    return 0.5 * (loss12 + loss21)
