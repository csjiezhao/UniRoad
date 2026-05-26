import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold


class Predictor(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.net = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_kfold_indices(n_samples: int, num_fold: int, seed: int):
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for KFold, got {n_samples}")
    n_splits = max(2, min(int(num_fold), int(n_samples)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_samples))), n_splits


def _check_fold_integrity(n_samples: int, splits):
    all_val = []
    for fold_id, (train_idx, val_idx) in enumerate(splits):
        inter = np.intersect1d(train_idx, val_idx, assume_unique=False)
        if inter.size > 0:
            raise ValueError(f"Fold leakage in fold {fold_id}: {inter.size}")
        all_val.append(val_idx)

    merged_val = np.concatenate(all_val, axis=0)
    uniq, cnt = np.unique(merged_val, return_counts=True)
    missing = np.setdiff1d(np.arange(n_samples), uniq, assume_unique=False)
    duplicated = uniq[cnt > 1]
    if merged_val.size != n_samples or missing.size > 0 or duplicated.size > 0:
        raise ValueError(
            f"Fold coverage mismatch: covered={merged_val.size}, expected={n_samples}, "
            f"missing={missing.size}, duplicated={duplicated.size}"
        )

    return {
        "n_samples": int(n_samples),
        "n_folds": int(len(splits)),
        "covered_once": True,
        "leakage": False,
    }


def eval_with_emb(
    city,
    emb_path,
    label_path,
    num_fold,
    epochs,
    device="cpu",
    seed=42,
    return_meta=False,
):
    print(f"--- Road Speed Inference ({city}) ---")

    embeddings = np.load(emb_path).astype(np.float32)
    labels = np.load(label_path)

    valid_mask = np.isfinite(labels) & (labels >= 0.0)
    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask].astype(np.float32)

    n_samples, embed_dim = embeddings.shape

    splits, n_splits = _build_kfold_indices(n_samples=n_samples, num_fold=num_fold, seed=seed)
    fold_check = _check_fold_integrity(n_samples=n_samples, splits=splits)

    preds = np.empty((n_samples,), dtype=np.float32)
    trues = labels.copy()

    for fold_id, (train_idx, val_idx) in enumerate(splits):
        _set_seed(seed + fold_id)

        x_train = torch.tensor(embeddings[train_idx], dtype=torch.float32, device=device)
        y_train = torch.tensor(labels[train_idx], dtype=torch.float32, device=device).unsqueeze(-1)
        x_val = torch.tensor(embeddings[val_idx], dtype=torch.float32, device=device)
        y_val = torch.tensor(labels[val_idx], dtype=torch.float32, device=device).unsqueeze(-1)

        model = Predictor(emb_dim=embed_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss().to(device)

        best_mse = float("inf")
        best_pred = None

        for _ in range(int(epochs)):
            model.train()
            optimizer.zero_grad()
            pred_train = model(x_train)
            loss = criterion(pred_train, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_val = model(x_val)
                mse = mean_squared_error(y_val.detach().cpu().numpy(), pred_val.detach().cpu().numpy())
                if mse < best_mse:
                    best_mse = float(mse)
                    best_pred = pred_val.detach().cpu().numpy().reshape(-1)

        if best_pred is None:
            raise RuntimeError("No best prediction collected in speed fold.")
        preds[val_idx] = best_pred

    mae = float(mean_absolute_error(trues, preds))
    rmse = float(mean_squared_error(trues, preds) ** 0.5)

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    meta = {
        "protocol": "fixed_shuffled_kfold",
        "seed": int(seed),
        "num_samples": int(n_samples),
        "num_folds_effective": int(n_splits),
        "fold_check": fold_check,
    }

    if return_meta:
        return mae, rmse, meta
    return mae, rmse
