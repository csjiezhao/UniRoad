from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MobilityBranchNet(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        emb_dim: int = 128,
        ctx_dim: int = 64,
        short_layers: int = 1,
        mid_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        if emb_dim != 128:
            raise ValueError(f"Expected emb_dim=128, got: {emb_dim}")

        self.num_tokens = int(num_tokens)
        self.emb_dim = int(emb_dim)
        self.ctx_dim = int(ctx_dim)
        self.pad_token_id = int(pad_token_id)

        self.token_embedding = nn.Embedding(self.num_tokens, self.emb_dim, padding_idx=self.pad_token_id)

        self.hour_embedding = nn.Embedding(24, 16)
        self.daytype_embedding = nn.Embedding(2, 8)
        self.time_dim = 24

        self.short_time_proj = nn.Linear(self.time_dim, self.emb_dim)
        self.mid_time_proj = nn.Linear(self.time_dim, self.emb_dim)

        self.short_pos = nn.Parameter(torch.randn(1, 5, self.emb_dim) * 0.02)
        self.mid_pos = nn.Parameter(torch.randn(1, 11, self.emb_dim) * 0.02)

        short_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=nhead,
            dim_feedforward=self.emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.short_encoder = nn.TransformerEncoder(short_layer, num_layers=short_layers)

        mid_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=nhead,
            dim_feedforward=self.emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.mid_encoder = nn.TransformerEncoder(mid_layer, num_layers=mid_layers)

        fuse_in = self.emb_dim * 2 + self.time_dim
        fuse_hidden = self.emb_dim + self.ctx_dim
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, fuse_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fuse_hidden, fuse_hidden),
            nn.ReLU(),
        )

        self.shared_head = nn.Linear(fuse_hidden, self.emb_dim)
        self.ctx_head = nn.Linear(fuse_hidden, self.ctx_dim)
        self.pred_head = nn.Linear(self.emb_dim + self.ctx_dim, self.emb_dim)

        self.hour_head = nn.Linear(self.ctx_dim, 24)
        self.daytype_head = nn.Linear(self.ctx_dim, 1)

        self.attn_pool = nn.Linear(self.emb_dim, 1)
        self.unk_mob = nn.Parameter(torch.randn(self.emb_dim) * 0.02)

    def _time_features(self, hour: torch.Tensor, daytype: torch.Tensor) -> torch.Tensor:
        hour_emb = self.hour_embedding(hour.long())
        day_emb = self.daytype_embedding(daytype.long())
        return torch.cat([hour_emb, day_emb], dim=-1)

    def forward(
        self,
        short_tokens: torch.Tensor,
        mid_tokens: torch.Tensor,
        hour: torch.Tensor,
        daytype: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        t_feat = self._time_features(hour=hour, daytype=daytype)

        short_mask = short_tokens.eq(self.pad_token_id)
        mid_mask = mid_tokens.eq(self.pad_token_id)

        short_x = self.token_embedding(short_tokens)
        short_x = short_x + self.short_pos
        short_x = short_x + self.short_time_proj(t_feat).unsqueeze(1)

        mid_x = self.token_embedding(mid_tokens)
        mid_x = mid_x + self.mid_pos
        mid_x = mid_x + self.mid_time_proj(t_feat).unsqueeze(1)

        short_out = self.short_encoder(short_x, src_key_padding_mask=short_mask)
        mid_out = self.mid_encoder(mid_x, src_key_padding_mask=mid_mask)

        h_short = short_out[:, 2, :]
        h_mid = mid_out[:, 5, :]

        fused = self.fuse(torch.cat([h_short, h_mid, t_feat], dim=-1))
        h_shared = self.shared_head(fused)
        h_ctx = self.ctx_head(fused)

        h_pred = self.pred_head(torch.cat([h_shared, h_ctx], dim=-1))

        hour_logits = self.hour_head(h_ctx)
        daytype_logit = self.daytype_head(h_ctx)

        attn_logit = torch.tanh(self.attn_pool(h_shared)).squeeze(-1)

        return {
            "h_short": h_short,
            "h_mid": h_mid,
            "h_shared": h_shared,
            "h_ctx": h_ctx,
            "h_pred": h_pred,
            "hour_logits": hour_logits,
            "daytype_logit": daytype_logit,
            "attn_logit": attn_logit,
        }


def compute_context_center_loss(
    model: MobilityBranchNet,
    h_pred: torch.Tensor,
    center_tokens: torch.Tensor,
    rand_neg_tokens: Optional[torch.Tensor],
    freq_neg_tokens: Optional[torch.Tensor],
    temperature: float = 0.07,
    include_inbatch: bool = True,
) -> Dict[str, torch.Tensor]:
    center_emb = model.token_embedding(center_tokens)
    pos_scores = (h_pred * center_emb).sum(dim=-1, keepdim=True)

    score_parts = [pos_scores]

    if rand_neg_tokens is not None and rand_neg_tokens.numel() > 0:
        rand_emb = model.token_embedding(rand_neg_tokens)
        rand_scores = (rand_emb * h_pred.unsqueeze(1)).sum(dim=-1)
        score_parts.append(rand_scores)

    if freq_neg_tokens is not None and freq_neg_tokens.numel() > 0:
        freq_emb = model.token_embedding(freq_neg_tokens)
        freq_scores = (freq_emb * h_pred.unsqueeze(1)).sum(dim=-1)
        score_parts.append(freq_scores)

    if include_inbatch and center_tokens.shape[0] > 1:
        inbatch_scores = h_pred @ center_emb.t()
        eye = torch.eye(center_tokens.shape[0], device=h_pred.device, dtype=torch.bool)
        mask_val = torch.finfo(inbatch_scores.dtype).min
        inbatch_scores = inbatch_scores.masked_fill(eye, mask_val)
        score_parts.append(inbatch_scores)

    logits = torch.cat(score_parts, dim=1) / float(temperature)
    labels = torch.zeros((center_tokens.shape[0],), dtype=torch.long, device=center_tokens.device)

    loss = F.cross_entropy(logits, labels)
    top1 = (logits.argmax(dim=1) == 0).float().mean()

    return {
        "loss": loss,
        "top1": top1,
    }


def compute_same_road_consistency_loss(h_shared: torch.Tensor, center_tokens: torch.Tensor) -> torch.Tensor:
    if h_shared.shape[0] <= 1:
        return h_shared.new_tensor(0.0)

    z = F.normalize(h_shared, dim=-1)
    uniq, inv, counts = torch.unique(center_tokens, return_inverse=True, return_counts=True)

    terms = []
    for gid in range(uniq.shape[0]):
        if int(counts[gid]) <= 1:
            continue
        mask = inv.eq(gid)
        group = z[mask]
        centroid = F.normalize(group.mean(dim=0, keepdim=True), dim=-1)
        sim = (group * centroid).sum(dim=-1)
        terms.append(1.0 - sim.mean())

    if not terms:
        return h_shared.new_tensor(0.0)
    return torch.stack(terms).mean()


def compute_time_context_loss(
    hour_logits: torch.Tensor,
    daytype_logit: torch.Tensor,
    hour_target: torch.Tensor,
    daytype_target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    loss_hour = F.cross_entropy(hour_logits, hour_target.long())
    loss_day = F.binary_cross_entropy_with_logits(daytype_logit.squeeze(-1), daytype_target.float())

    pred_hour = hour_logits.argmax(dim=-1)
    hour_acc = pred_hour.eq(hour_target.long()).float().mean()

    pred_day = (torch.sigmoid(daytype_logit.squeeze(-1)) >= 0.5).long()
    day_acc = pred_day.eq(daytype_target.long()).float().mean()

    return {
        "loss": loss_hour + loss_day,
        "loss_hour": loss_hour,
        "loss_day": loss_day,
        "hour_acc": hour_acc,
        "day_acc": day_acc,
    }
