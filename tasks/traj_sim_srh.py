from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


def _augment_path(path: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = path
    n = int(len(x))
    if n <= 4:
        return x

    # Random crop: keep 70%~100%
    keep_ratio = float(rng.uniform(0.7, 1.0))
    keep = max(4, int(round(n * keep_ratio)))
    if keep < n:
        st = int(rng.integers(0, n - keep + 1))
        x = x[st : st + keep]

    # Random drop points with low probability
    if len(x) > 6:
        mask = rng.random(len(x)) > 0.10
        if int(mask.sum()) >= 4:
            x = x[mask]
    return x


class _SimDataset(Dataset):
    def __init__(self, paths: List[np.ndarray]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return self.paths[idx], int(idx)


def _collate_two_views(batch, rng: np.random.Generator):
    paths, ids = zip(*batch)
    v1 = [_augment_path(p, rng) for p in paths]
    v2 = [_augment_path(p, rng) for p in paths]

    len1 = np.asarray([len(x) for x in v1], dtype=np.int64)
    len2 = np.asarray([len(x) for x in v2], dtype=np.int64)
    max1 = int(len1.max())
    max2 = int(len2.max())

    tok1 = np.zeros((len(v1), max1), dtype=np.int64)
    tok2 = np.zeros((len(v2), max2), dtype=np.int64)
    for i, p in enumerate(v1):
        tok1[i, : len(p)] = p + 1
    for i, p in enumerate(v2):
        tok2[i, : len(p)] = p + 1

    return (
        torch.from_numpy(tok1).long(),
        torch.from_numpy(len1).long(),
        torch.from_numpy(tok2).long(),
        torch.from_numpy(len2).long(),
        torch.tensor(ids, dtype=torch.long),
    )


def _collate_one_view(batch):
    paths, ids = zip(*batch)
    lens = np.asarray([len(x) for x in paths], dtype=np.int64)
    m = int(lens.max())
    tok = np.zeros((len(paths), m), dtype=np.int64)
    for i, p in enumerate(paths):
        tok[i, : len(p)] = p + 1
    return torch.from_numpy(tok).long(), torch.from_numpy(lens).long(), torch.tensor(ids, dtype=torch.long)


class _TrajEncoder(nn.Module):
    def __init__(self, edge_emb: np.ndarray, hidden_dim: int = 128):
        super().__init__()
        emb = np.zeros((edge_emb.shape[0] + 1, edge_emb.shape[1]), dtype=np.float32)
        emb[1:] = edge_emb
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(emb), freeze=True, padding_idx=0)
        self.rnn = nn.GRU(
            input_size=int(edge_emb.shape[1]),
            hidden_size=int(hidden_dim),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        packed = pack_padded_sequence(x, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        z = self.proj(h)
        z = nn.functional.normalize(z, dim=-1)
        return z


def _info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (z1 @ z2.t()) / float(temperature)
    labels = torch.arange(z1.shape[0], device=z1.device, dtype=torch.long)
    loss12 = nn.functional.cross_entropy(logits, labels)
    loss21 = nn.functional.cross_entropy(logits.t(), labels)
    return 0.5 * (loss12 + loss21)


def _rank_metrics(query: np.ndarray, gallery: np.ndarray, k_list: Tuple[int, ...] = (3, 10)) -> Dict[str, float]:
    q = torch.from_numpy(query.astype(np.float32))
    g = torch.from_numpy(gallery.astype(np.float32))
    # cosine similarity because vectors are normalized
    sim = q @ g.t()
    n = sim.shape[0]

    pos = sim[torch.arange(n), torch.arange(n)]
    rank = 1 + (sim > pos.unsqueeze(1)).sum(dim=1)
    out = {
        "mean_rank": float(rank.float().mean().item()),
    }
    for k in k_list:
        out[f"hr@{int(k)}"] = float((rank <= int(k)).float().mean().item())
    return out


def eval_with_traj_emb(
    paths: List[np.ndarray],
    edge_emb: np.ndarray,
    device: str = "cpu",
    seed: int = 42,
    max_epoch: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    temperature: float = 0.10,
    hidden_dim: int = 128,
) -> Dict[str, object]:
    if edge_emb.ndim != 2:
        raise ValueError(f"edge_emb must be 2D, got {edge_emb.shape}")
    if len(paths) < 32:
        raise ValueError(f"Need at least 32 trajectories for similarity task, got {len(paths)}")

    rng = np.random.default_rng(int(seed))
    dev = torch.device(device)
    model = _TrajEncoder(edge_emb=edge_emb.astype(np.float32), hidden_dim=int(hidden_dim)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    ds = _SimDataset(paths)
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=lambda b: _collate_two_views(b, rng),
    )

    for _epoch in range(1, int(max_epoch) + 1):
        model.train()
        for x1, l1, x2, l2, _ids in loader:
            x1 = x1.to(dev)
            l1 = l1.to(dev)
            x2 = x2.to(dev)
            l2 = l2.to(dev)
            z1 = model(x1, l1)
            z2 = model(x2, l2)
            loss = _info_nce(z1, z2, temperature=float(temperature))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

    # Retrieval evaluation on two augmented views of the same trajectory set.
    model.eval()
    eval_rng_a = np.random.default_rng(int(seed) + 101)
    eval_rng_b = np.random.default_rng(int(seed) + 202)
    view_a = [_augment_path(p, eval_rng_a) for p in paths]
    view_b = [_augment_path(p, eval_rng_b) for p in paths]

    qa_loader = DataLoader(_SimDataset(view_a), batch_size=int(batch_size), shuffle=False, collate_fn=_collate_one_view)
    qb_loader = DataLoader(_SimDataset(view_b), batch_size=int(batch_size), shuffle=False, collate_fn=_collate_one_view)

    def _encode(loader_one):
        out = []
        with torch.no_grad():
            for x, l, _ids in loader_one:
                x = x.to(dev)
                l = l.to(dev)
                out.append(model(x, l).detach().cpu().numpy())
        return np.concatenate(out, axis=0)

    emb_a = _encode(qa_loader)
    emb_b = _encode(qb_loader)
    m = _rank_metrics(query=emb_a, gallery=emb_b, k_list=(3, 10))
    return {
        "hr@3": float(m["hr@3"]),
        "hr@10": float(m["hr@10"]),
        "mean_rank": float(m["mean_rank"]),
        "num_query": int(emb_a.shape[0]),
    }

