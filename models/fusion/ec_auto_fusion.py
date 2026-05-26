import argparse
import json
import os
from typing import Dict, List, Tuple

import faiss  # type: ignore
import numpy as np


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps)).astype(np.float32, copy=False)


def knn_indices_cosine(z: np.ndarray, k: int) -> np.ndarray:
    n = int(z.shape[0])
    n_neighbors = min(n, int(k) + 1)
    if n_neighbors <= 1:
        return np.zeros((n, 0), dtype=np.int64)

    x = z.astype(np.float32, copy=False)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    index = faiss.IndexFlatIP(int(x.shape[1]))
    index.add(x)
    _, idx = index.search(x, n_neighbors)

    out = np.zeros((n, min(int(k), max(n - 1, 0))), dtype=np.int64)
    for i in range(n):
        neighbors = idx[i][idx[i] != i]
        if neighbors.size == 0:
            out[i] = i
            continue
        if neighbors.size >= out.shape[1]:
            out[i] = neighbors[: out.shape[1]]
        else:
            pad = np.full((out.shape[1] - neighbors.size,), int(neighbors[-1]), dtype=np.int64)
            out[i] = np.concatenate([neighbors, pad], axis=0)
    return out


def local_variance(z: np.ndarray, k: int) -> np.ndarray:
    n = int(z.shape[0])
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    nbr_idx = knn_indices_cosine(z, k=int(k))
    if nbr_idx.shape[1] == 0:
        return np.zeros((n,), dtype=np.float32)
    sigma2 = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        nn = z[nbr_idx[i]]
        mu = np.mean(nn, axis=0, keepdims=True)
        diff = nn - mu
        sigma2[i] = float(np.mean(np.sum(diff * diff, axis=1)))
    return sigma2


def build_ec_auto(
    z_p: np.ndarray,
    z_s: np.ndarray,
    z_m: np.ndarray,
    k: int = 20,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, float]]:
    z_p_t = layer_norm(z_p)
    z_s_t = layer_norm(z_s)
    z_m_t = layer_norm(z_m)

    sigma2_p = local_variance(z_p_t, k=k)
    sigma2_s = local_variance(z_s_t, k=k)
    sigma2_m = local_variance(z_m_t, k=k)

    p_p = 1.0 / np.maximum(sigma2_p + eps, 1e-12)
    p_s = 1.0 / np.maximum(sigma2_s + eps, 1e-12)
    p_m = 1.0 / np.maximum(sigma2_m + eps, 1e-12)
    precision = np.stack([p_p, p_s, p_m], axis=1).astype(np.float32, copy=False)
    alpha = precision / np.maximum(np.sum(precision, axis=1, keepdims=True), 1e-12)

    views = np.stack([z_p_t, z_s_t, z_m_t], axis=1).astype(np.float32, copy=False)
    n = int(views.shape[0])
    anchor_idx = np.argmax(alpha, axis=1)
    anchor = views[np.arange(n), anchor_idx, :]
    delta = np.sum(alpha[:, :, None] * (views - anchor[:, None, :]), axis=1)

    m = 1.0 - alpha[np.arange(n), anchor_idx]
    m_bar = float(np.mean(m))
    e_a = float(np.mean(np.linalg.norm(anchor, axis=1)))
    e_delta = float(np.mean(np.linalg.norm(delta, axis=1)))
    gamma = float((m_bar * e_a) / (e_delta + eps))

    out = layer_norm(anchor + gamma * delta)
    meta = {
        "k": int(k),
        "eps": float(eps),
        "gamma": float(gamma),
        "m_bar": float(m_bar),
        "E_a": float(e_a),
        "E_delta": float(e_delta),
        "mean_alpha_p": float(alpha[:, 0].mean()),
        "mean_alpha_s": float(alpha[:, 1].mean()),
        "mean_alpha_m": float(alpha[:, 2].mean()),
    }
    return out.astype(np.float32, copy=False), meta


def _load_city_views(emb_root: str, city: str, src_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_p = np.load(os.path.join(emb_root, "profile", city, "z.npy")).astype(np.float32, copy=False)
    z_s = np.load(os.path.join(emb_root, "structure", f"{src_id}__to__{city}", "z.npy")).astype(np.float32, copy=False)
    z_m = np.load(os.path.join(emb_root, "mobility", f"{src_id}__to__{city}", "z.npy")).astype(np.float32, copy=False)
    if z_p.shape != z_s.shape or z_p.shape != z_m.shape:
        raise RuntimeError(f"{city}: shape mismatch P={z_p.shape}, S={z_s.shape}, M={z_m.shape}")
    return z_p, z_s, z_m


def main() -> None:
    parser = argparse.ArgumentParser(description="Core EC-auto fusion for joint-training cities.")
    parser.add_argument("--cities", type=str, default="chengdu,porto,rome,sanfran")
    parser.add_argument("--src-id", type=str, default="joint4")
    parser.add_argument("--emb-root", type=str, default="embs")
    parser.add_argument("--out-root", type=str, default="embs/fusion")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    cities: List[str] = [x.strip() for x in args.cities.split(",") if x.strip()]
    if not cities:
        raise ValueError("No valid cities provided.")

    os.makedirs(args.out_root, exist_ok=True)
    summary: List[Dict[str, object]] = []

    for city in cities:
        z_p, z_s, z_m = _load_city_views(emb_root=args.emb_root, city=city, src_id=args.src_id)
        z_fused, meta = build_ec_auto(z_p=z_p, z_s=z_s, z_m=z_m, k=int(args.k), eps=float(args.eps))

        out_dir = os.path.join(args.out_root, f"{args.src_id}__to__{city}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "z.npy")
        np.save(out_path, z_fused)

        row = {
            "city": city,
            "src_id": args.src_id,
            "shape": list(z_fused.shape),
            "out_path": out_path,
            "fusion": meta,
        }
        summary.append(row)
        with open(os.path.join(out_dir, "fusion_meta.json"), "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=True, indent=2)
        print(f"[Fusion] {city}: saved {out_path}")

    with open(os.path.join(args.out_root, "summary_fusion.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    print(f"Saved summary: {os.path.join(args.out_root, 'summary_fusion.json')}")


if __name__ == "__main__":
    main()
