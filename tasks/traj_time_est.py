from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


class _TrajTimeDataset(Dataset):
    def __init__(self, paths: List[np.ndarray], targets: np.ndarray):
        self.paths = paths
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return self.paths[idx], float(self.targets[idx])


def _collate(batch):
    paths, y = zip(*batch)
    lengths = np.asarray([len(p) for p in paths], dtype=np.int64)
    max_len = int(lengths.max())
    tokens = np.zeros((len(paths), max_len), dtype=np.int64)
    for i, p in enumerate(paths):
        # +1 because 0 is reserved for padding embedding
        tokens[i, : len(p)] = p + 1
    return (
        torch.from_numpy(tokens).long(),
        torch.from_numpy(lengths).long(),
        torch.tensor(y, dtype=torch.float32),
    )


class _TrajRegressor(nn.Module):
    def __init__(self, edge_emb: np.ndarray, hidden_dim: int = 128):
        super().__init__()
        emb = np.zeros((edge_emb.shape[0] + 1, edge_emb.shape[1]), dtype=np.float32)
        emb[1:] = edge_emb
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(emb), freeze=True, padding_idx=0)
        self.encoder = nn.GRU(
            input_size=int(edge_emb.shape[1]),
            hidden_size=int(hidden_dim),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        packed = pack_padded_sequence(x, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.encoder(packed)
        # bidirectional GRU: h_n shape [2, B, H]
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.head(h).squeeze(-1)


def _split_indices(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_train = int(round(0.6 * n))
    n_val = int(round(0.2 * n))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    return mae, rmse


def eval_with_traj_emb(
    paths: List[np.ndarray],
    travel_time: np.ndarray,
    edge_emb: np.ndarray,
    device: str = "cpu",
    seed: int = 42,
    max_epoch: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dim: int = 128,
) -> Dict[str, object]:
    if len(paths) != int(travel_time.shape[0]):
        raise ValueError("paths and travel_time length mismatch")
    if edge_emb.ndim != 2:
        raise ValueError(f"edge_emb must be 2D, got {edge_emb.shape}")
    if len(paths) < 32:
        raise ValueError(f"Need at least 32 trajectories for TTE, got {len(paths)}")

    train_idx, val_idx, test_idx = _split_indices(len(paths), seed=seed)

    def _subset(ids: np.ndarray):
        return [paths[int(i)] for i in ids], travel_time[ids]

    tr_paths, tr_y = _subset(train_idx)
    va_paths, va_y = _subset(val_idx)
    te_paths, te_y = _subset(test_idx)

    tr_loader = DataLoader(_TrajTimeDataset(tr_paths, tr_y), batch_size=int(batch_size), shuffle=True, collate_fn=_collate)
    va_loader = DataLoader(_TrajTimeDataset(va_paths, va_y), batch_size=int(batch_size), shuffle=False, collate_fn=_collate)
    te_loader = DataLoader(_TrajTimeDataset(te_paths, te_y), batch_size=int(batch_size), shuffle=False, collate_fn=_collate)

    dev = torch.device(device)
    model = _TrajRegressor(edge_emb=edge_emb.astype(np.float32), hidden_dim=int(hidden_dim)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    patience = 8

    for epoch in range(1, int(max_epoch) + 1):
        model.train()
        for x, l, y in tr_loader:
            x = x.to(dev)
            l = l.to(dev)
            y = y.to(dev)
            pred = model(x, l)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for x, l, _y in va_loader:
                x = x.to(dev)
                l = l.to(dev)
                val_preds.append(model(x, l).detach().cpu().numpy())
        val_pred = np.concatenate(val_preds, axis=0)
        _mae, val_rmse = _mae_rmse(va_y.astype(np.float32), val_pred.astype(np.float32))

        if val_rmse < best_val:
            best_val = float(val_rmse)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_preds = []
    with torch.no_grad():
        for x, l, _y in te_loader:
            x = x.to(dev)
            l = l.to(dev)
            test_preds.append(model(x, l).detach().cpu().numpy())
    test_pred = np.concatenate(test_preds, axis=0)
    test_mae, test_rmse = _mae_rmse(te_y.astype(np.float32), test_pred.astype(np.float32))

    return {
        "mae": float(test_mae),
        "rmse": float(test_rmse),
        "best_epoch": int(best_epoch),
        "val_best_rmse": float(best_val),
        "num_train": int(train_idx.shape[0]),
        "num_val": int(val_idx.shape[0]),
        "num_test": int(test_idx.shape[0]),
    }

