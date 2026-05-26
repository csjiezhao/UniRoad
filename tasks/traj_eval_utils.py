import math
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

MOBILITY_CACHE_TAG_VERSION = 2


def _sampling_tag(
    max_traj_per_city: int,
    seed: int,
    coverage_ratio: float,
    num_bins: int,
) -> str:
    cov = f"{coverage_ratio:.2f}".replace(".", "p")
    return (
        f"mobility_v{int(MOBILITY_CACHE_TAG_VERSION)}_k{int(max_traj_per_city)}_seed{int(seed)}_cov{cov}_bins{int(num_bins)}"
    )


def _cache_file_path(
    city_path: str,
    max_traj_per_city: int,
    seed: int,
    coverage_ratio: float,
    num_bins: int,
) -> str:
    return os.path.join(
        city_path,
        "cache",
        f"{_sampling_tag(max_traj_per_city, seed, coverage_ratio, num_bins)}.pt",
    )


def _parse_path(path_text: object, num_edges: int) -> np.ndarray:
    s = str(path_text).strip()
    if len(s) < 2 or s[0] != "[" or s[-1] != "]":
        return np.zeros((0,), dtype=np.int64)
    body = s[1:-1].strip()
    if not body:
        return np.zeros((0,), dtype=np.int64)
    arr = np.fromstring(body, sep=",", dtype=np.int64)
    if arr.size == 0:
        return arr
    arr = arr[(arr >= 0) & (arr < int(num_edges))]
    return arr.astype(np.int64, copy=False)


def _allocate_quotas(target: int, counts: np.ndarray) -> np.ndarray:
    out = np.zeros_like(counts, dtype=np.int64)
    if target <= 0:
        return out
    total = int(counts.sum())
    if total <= 0:
        return out
    raw = counts.astype(np.float64) * (float(target) / float(total))
    base = np.floor(raw).astype(np.int64)
    rem = int(target - int(base.sum()))
    if rem > 0:
        frac = raw - base.astype(np.float64)
        order = np.argsort(-frac)
        for idx in order[:rem]:
            base[int(idx)] += 1
    out[:] = base
    return out


def build_reserved_trajectory_eval_data(
    city: str,
    data_root: str,
    sample_size: int = 50000,
    sample_seed: int = 42,
    max_traj_per_city: int = 150000,
    cache_seed: int = 42,
    coverage_ratio: float = 0.30,
    num_path_bins: int = 10,
    coverage_quota_ratio: float = 0.20,
) -> Dict[str, object]:
    city_path = os.path.join(data_root, city)
    cache_path = _cache_file_path(
        city_path=city_path,
        max_traj_per_city=max_traj_per_city,
        seed=cache_seed,
        coverage_ratio=coverage_ratio,
        num_bins=num_path_bins,
    )
    if not os.path.exists(cache_path):
        raise FileNotFoundError(cache_path)

    packed = torch.load(cache_path, map_location="cpu")
    meta = dict(packed["meta"])
    num_edges = int(meta["num_edges"])

    selected = packed["selected_indices"].cpu().numpy().astype(np.int64)
    reserved = packed["reserved_indices"].cpu().numpy().astype(np.int64)
    overlap = int(np.intersect1d(selected, reserved, assume_unique=False).size)
    if overlap > 0:
        raise RuntimeError(f"Reserved/selected overlap detected: {overlap}")

    reserved_set = set(reserved.tolist())
    rng = np.random.default_rng(int(sample_seed))

    traj_csv = os.path.join(city_path, "trajectories.csv")
    if not os.path.exists(traj_csv):
        raise FileNotFoundError(traj_csv)

    target_n = int(sample_size)
    if target_n <= 0:
        raise ValueError("sample_size must be positive")

    coverage_quota = max(1, int(round(float(target_n) * float(coverage_quota_ratio))))
    coverage_ids: List[int] = []
    covered_edges = np.zeros((num_edges,), dtype=bool)

    candidate_ids: List[int] = []
    candidate_num_path: List[int] = []
    candidate_time: List[float] = []

    row0 = 0
    for chunk in pd.read_csv(traj_csv, usecols=["num_path", "travel_time", "path"], chunksize=20000):
        m = int(len(chunk))
        row_ids = np.arange(row0, row0 + m, dtype=np.int64)
        row0 += m
        mask = np.fromiter((int(x) in reserved_set for x in row_ids), dtype=bool, count=m)
        if not bool(mask.any()):
            continue

        sub = chunk.loc[mask]
        sub_ids = row_ids[mask]
        sub_num_path = pd.to_numeric(sub["num_path"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        sub_time = pd.to_numeric(sub["travel_time"], errors="coerce").to_numpy(dtype=np.float64)
        sub_path = sub["path"].astype(str).tolist()

        for rid, npath, tval, ptxt in zip(sub_ids, sub_num_path, sub_time, sub_path):
            if not np.isfinite(tval) or float(tval) < 0.0 or int(npath) <= 0:
                continue
            rid_int = int(rid)
            candidate_ids.append(rid_int)
            candidate_num_path.append(int(npath))
            candidate_time.append(float(tval))

            if len(coverage_ids) < coverage_quota:
                arr = _parse_path(ptxt, num_edges=num_edges)
                if arr.size == 0:
                    continue
                gain = bool((~covered_edges[arr]).any())
                if gain:
                    coverage_ids.append(rid_int)
                    covered_edges[arr] = True

    if not candidate_ids:
        raise RuntimeError(f"No valid reserved trajectories for city={city}")

    candidate_ids_np = np.asarray(candidate_ids, dtype=np.int64)
    candidate_num_path_np = np.asarray(candidate_num_path, dtype=np.int64)
    candidate_time_np = np.asarray(candidate_time, dtype=np.float64)

    target_n = int(min(target_n, candidate_ids_np.shape[0]))
    coverage_ids = coverage_ids[: min(len(coverage_ids), target_n)]
    coverage_set = set(coverage_ids)

    left_n = int(target_n - len(coverage_ids))
    keep_mask = np.fromiter((int(x) not in coverage_set for x in candidate_ids_np), dtype=bool, count=candidate_ids_np.shape[0])
    rest_ids = candidate_ids_np[keep_mask]
    rest_num_path = candidate_num_path_np[keep_mask]

    if left_n > 0 and rest_ids.size > 0:
        q = np.linspace(0.0, 1.0, 11, dtype=np.float64)[1:-1]
        edges = np.quantile(rest_num_path.astype(np.float64), q) if q.size > 0 else np.asarray([], dtype=np.float64)
        bucket = np.searchsorted(edges, rest_num_path, side="right").astype(np.int64)
        num_bucket = max(int(bucket.max()) + 1, 1) if bucket.size > 0 else 1
        counts = np.bincount(bucket, minlength=num_bucket).astype(np.int64)
        quotas = _allocate_quotas(target=left_n, counts=counts)

        sampled_rest: List[int] = []
        for b in range(num_bucket):
            idx = np.where(bucket == b)[0]
            if idx.size == 0:
                continue
            qn = int(min(quotas[b], idx.size))
            if qn <= 0:
                continue
            pick = rng.choice(idx, size=qn, replace=False)
            sampled_rest.extend(rest_ids[pick].tolist())
    else:
        sampled_rest = []

    sampled_ids = coverage_ids + sampled_rest
    if len(sampled_ids) < target_n:
        existing = set(sampled_ids)
        for rid in candidate_ids_np.tolist():
            if rid in existing:
                continue
            sampled_ids.append(int(rid))
            existing.add(int(rid))
            if len(sampled_ids) >= target_n:
                break

    sampled_ids = sampled_ids[:target_n]
    sampled_set = set(int(x) for x in sampled_ids)

    sampled_paths: List[np.ndarray] = []
    sampled_time: List[float] = []
    sampled_len: List[int] = []
    sampled_row: List[int] = []
    selected_coverage = np.zeros((num_edges,), dtype=bool)

    row0 = 0
    for chunk in pd.read_csv(traj_csv, usecols=["num_path", "travel_time", "path"], chunksize=20000):
        m = int(len(chunk))
        row_ids = np.arange(row0, row0 + m, dtype=np.int64)
        row0 += m
        mask = np.fromiter((int(x) in sampled_set for x in row_ids), dtype=bool, count=m)
        if not bool(mask.any()):
            continue

        sub = chunk.loc[mask]
        sub_ids = row_ids[mask]
        sub_num_path = pd.to_numeric(sub["num_path"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        sub_time = pd.to_numeric(sub["travel_time"], errors="coerce").to_numpy(dtype=np.float64)
        sub_path = sub["path"].astype(str).tolist()

        for rid, npath, tval, ptxt in zip(sub_ids, sub_num_path, sub_time, sub_path):
            if not np.isfinite(tval) or float(tval) < 0.0 or int(npath) <= 0:
                continue
            arr = _parse_path(ptxt, num_edges=num_edges)
            if arr.size == 0:
                continue
            sampled_row.append(int(rid))
            sampled_paths.append(arr)
            sampled_time.append(float(tval))
            sampled_len.append(int(npath))
            selected_coverage[arr] = True

    if len(sampled_paths) < target_n:
        raise RuntimeError(
            f"Sample extraction shortfall for city={city}: want={target_n}, got={len(sampled_paths)}"
        )

    sampled_paths = sampled_paths[:target_n]
    sampled_time = sampled_time[:target_n]
    sampled_len = sampled_len[:target_n]
    sampled_row = sampled_row[:target_n]

    cand_q = np.quantile(candidate_num_path_np.astype(np.float64), np.linspace(0.0, 1.0, 11, dtype=np.float64))
    cand_edges = np.unique(cand_q)
    if cand_edges.size <= 2:
        cand_bucket = np.zeros_like(candidate_num_path_np)
    else:
        cand_bucket = np.searchsorted(cand_edges[1:-1], candidate_num_path_np, side="right").astype(np.int64)
    cand_bucket_counts = np.bincount(cand_bucket, minlength=max(int(cand_bucket.max()) + 1, 1)).astype(np.int64)

    sampled_len_np = np.asarray(sampled_len, dtype=np.int64)
    if cand_edges.size <= 2:
        sel_bucket = np.zeros_like(sampled_len_np)
    else:
        sel_bucket = np.searchsorted(cand_edges[1:-1], sampled_len_np, side="right").astype(np.int64)
    sel_bucket_counts = np.bincount(sel_bucket, minlength=max(int(cand_bucket_counts.shape[0]), 1)).astype(np.int64)

    sample_info = {
        "city": city,
        "cache_path": cache_path,
        "candidate_count": int(candidate_ids_np.shape[0]),
        "selected_count": int(len(sampled_paths)),
        "target_sample_size": int(sample_size),
        "actual_sample_size": int(target_n),
        "sample_seed": int(sample_seed),
        "zero_overlap_with_mobility_train": bool(overlap == 0),
        "reserved_selected_overlap_count": int(overlap),
        "coverage_quota_ratio": float(coverage_quota_ratio),
        "coverage_preselected_count": int(len(coverage_ids)),
        "num_edges": int(num_edges),
        "selected_road_coverage": float(selected_coverage.mean()),
        "num_path_candidate_bucket_counts": [int(x) for x in cand_bucket_counts.tolist()],
        "num_path_selected_bucket_counts": [int(x) for x in sel_bucket_counts.tolist()],
        "num_path_selected_mean": float(sampled_len_np.mean()),
        "num_path_selected_median": float(np.median(sampled_len_np)),
    }

    return {
        "paths": sampled_paths,
        "travel_time": np.asarray(sampled_time, dtype=np.float32),
        "num_path": sampled_len_np.astype(np.int32),
        "row_ids": np.asarray(sampled_row, dtype=np.int64),
        "sample_info": sample_info,
    }
