import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold


class Classifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Linear(embed_dim, num_classes)

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
    num_classes=None,
    device="cpu",
    seed=42,
    return_meta=False,
):
    print(f"--- Road Type Classification ({city}) ---")

    embeddings = np.load(emb_path).astype(np.float32)
    labels = np.load(label_path)

    valid_mask = np.isfinite(labels) & (labels != -1)
    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask].astype(np.int64)

    unique_labels = sorted(np.unique(labels).tolist())
    label_mapping = {int(v): i for i, v in enumerate(unique_labels)}
    labels = np.asarray([label_mapping[int(v)] for v in labels], dtype=np.int64)

    n_samples, embed_dim = embeddings.shape
    n_classes = len(unique_labels)
    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes, got {n_classes}")

    splits, n_splits = _build_kfold_indices(n_samples=n_samples, num_fold=num_fold, seed=seed)
    fold_check = _check_fold_integrity(n_samples=n_samples, splits=splits)

    preds = np.empty((n_samples,), dtype=np.int64)
    trues = labels.copy()

    for fold_id, (train_idx, val_idx) in enumerate(splits):
        _set_seed(seed + fold_id)

        x_train = torch.tensor(embeddings[train_idx], dtype=torch.float32, device=device)
        y_train = torch.tensor(labels[train_idx], dtype=torch.long, device=device)
        x_val = torch.tensor(embeddings[val_idx], dtype=torch.float32, device=device)
        y_val = torch.tensor(labels[val_idx], dtype=torch.long, device=device)

        model = Classifier(embed_dim=embed_dim, num_classes=n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss().to(device)

        best_acc = -1.0
        best_pred = None

        for _ in range(int(epochs)):
            model.train()
            optimizer.zero_grad()
            logits = model(x_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(x_val)
                val_pred = torch.argmax(val_logits, dim=-1)
                acc = accuracy_score(y_val.detach().cpu().numpy(), val_pred.detach().cpu().numpy())
                if acc > best_acc:
                    best_acc = float(acc)
                    best_pred = val_pred.detach().cpu().numpy()

        if best_pred is None:
            raise RuntimeError("No best prediction collected in classification fold.")
        preds[val_idx] = best_pred

    micro_f1 = float(f1_score(trues, preds, average="micro", zero_division=0))
    macro_f1 = float(f1_score(trues, preds, average="macro", zero_division=0))

    print(f"Mi-F1: {micro_f1:.4f}, Ma-F1: {macro_f1:.4f}")

    meta = {
        "protocol": "fixed_shuffled_kfold",
        "seed": int(seed),
        "num_samples": int(n_samples),
        "num_folds_effective": int(n_splits),
        "num_classes": int(n_classes),
        "raw_classes": [int(v) for v in unique_labels],
        "label_mapping": {str(k): int(v) for k, v in label_mapping.items()},
        "fold_check": fold_check,
    }

    if return_meta:
        return micro_f1, macro_f1, meta
    return micro_f1, macro_f1
