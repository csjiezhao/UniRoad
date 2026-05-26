import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


CACHE_VERSION = 2

_CITY_TIMEZONE = {
    "chengdu": "Asia/Shanghai",
    "porto": "Europe/Lisbon",
    "rome": "Europe/Rome",
    "sanfran": "America/Los_Angeles",
}


def _resolve_timezone(city: str):
    tz_name = _CITY_TIMEZONE.get(str(city).strip().lower(), "UTC")
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return timezone.utc


def _sampling_tag(max_traj_per_city: int, seed: int, coverage_ratio: float, num_bins: int) -> str:
    cov = f"{coverage_ratio:.2f}".replace(".", "p")
    return f"mobility_v{CACHE_VERSION}_k{int(max_traj_per_city)}_seed{int(seed)}_cov{cov}_bins{int(num_bins)}"


def _cache_file_path(city_path: str, max_traj_per_city: int, seed: int, coverage_ratio: float, num_bins: int) -> str:
    cache_dir = os.path.join(city_path, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    tag = _sampling_tag(
        max_traj_per_city=max_traj_per_city,
        seed=seed,
        coverage_ratio=coverage_ratio,
        num_bins=num_bins,
    )
    return os.path.join(cache_dir, f"{tag}.pt")


def _cache_meta_path(cache_path: str) -> str:
    return cache_path.replace(".pt", "_meta.json")


def _parse_int_list(text: object) -> Optional[List[int]]:
    if text is None:
        return None
    s = str(text).strip()
    if len(s) < 2 or s[0] != "[" or s[-1] != "]":
        return None
    body = s[1:-1].strip()
    if not body:
        return []
    out: List[int] = []
    for tok in body.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        try:
            out.append(int(tok))
        except Exception:
            return None
    return out


def _parse_first_timestamp(text: object) -> Optional[int]:
    if text is None:
        return None
    s = str(text).strip()
    if len(s) < 2 or s[0] != "[" or s[-1] != "]":
        return None
    body = s[1:-1].strip()
    if not body:
        return None
    tok = body.split(",", 1)[0].strip()
    if tok == "":
        return None
    try:
        return int(tok)
    except Exception:
        return None


def _timestamp_to_hour_daytype(ts: int, city: str) -> Tuple[int, int]:
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(_resolve_timezone(city))
    hour = int(dt.hour)
    daytype = 1 if dt.weekday() >= 5 else 0
    return hour, daytype


def _is_valid_path(path: Sequence[int], num_edges: int) -> bool:
    if not path:
        return False
    for rid in path:
        if rid < 0 or rid >= num_edges:
            return False
    return True


def _path_length_edges(lengths: np.ndarray, num_bins: int) -> np.ndarray:
    if num_bins <= 1:
        return np.asarray([], dtype=np.float64)
    q = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)[1:-1]
    if q.size == 0:
        return np.asarray([], dtype=np.float64)
    return np.quantile(lengths.astype(np.float64), q)


def _bucket_id(length: int, edges: np.ndarray) -> int:
    if edges.size == 0:
        return 0
    return int(np.searchsorted(edges, length, side="right"))


@dataclass
class _TrajectoryRecord:
    orig_idx: int
    path: List[int]
    hour: int
    daytype: int
    length: int


@dataclass
class CityMobilityCache:
    city: str
    city_id: int
    num_edges: int
    selected_indices: np.ndarray
    reserved_indices: np.ndarray
    traj_orig_idx: np.ndarray
    traj_offsets: np.ndarray
    traj_edges: np.ndarray
    traj_hour: np.ndarray
    traj_daytype: np.ndarray
    road_occ_count: np.ndarray
    quantile_edges: np.ndarray
    meta: Dict[str, object]
    cache_path: str

    @property
    def num_selected(self) -> int:
        return int(self.traj_orig_idx.shape[0])

    @property
    def num_reserved(self) -> int:
        return int(self.reserved_indices.shape[0])

    @property
    def num_occurrences(self) -> int:
        return int(self.traj_edges.shape[0])


def _read_num_edges(city_path: str) -> int:
    edges_path = os.path.join(city_path, "G_edges.csv")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(edges_path)
    edge_idx = pd.read_csv(edges_path, usecols=["edge_idx"])["edge_idx"].to_numpy(dtype=np.int64)
    if edge_idx.size == 0:
        return 0
    return int(edge_idx.max()) + 1


def _allocate_fill_quotas(fill_target: int, counts: np.ndarray) -> np.ndarray:
    quotas = np.zeros_like(counts, dtype=np.int64)
    if fill_target <= 0:
        return quotas
    total = int(counts.sum())
    if total <= 0:
        return quotas

    raw = counts.astype(np.float64) * (float(fill_target) / float(total))
    base = np.floor(raw).astype(np.int64)
    rem = int(fill_target - int(base.sum()))
    if rem > 0:
        frac = raw - base.astype(np.float64)
        order = np.argsort(-frac)
        for i in order[:rem]:
            base[int(i)] += 1
    quotas[:] = base
    return quotas


def _iter_traj_rows(trajectories_path: str, chunksize: int = 20000):
    cols = ["path", "tlist", "num_path"]
    row_idx = 0
    for chunk in pd.read_csv(trajectories_path, usecols=cols, chunksize=chunksize):
        paths = chunk["path"].tolist()
        tlists = chunk["tlist"].tolist()
        for path_text, tlist_text in zip(paths, tlists):
            yield row_idx, path_text, tlist_text
            row_idx += 1


def _build_city_cache(
    city_path: str,
    city: str,
    city_id: int,
    max_traj_per_city: int,
    seed: int,
    coverage_ratio: float,
    num_bins: int,
) -> CityMobilityCache:
    trajectories_path = os.path.join(city_path, "trajectories.csv")
    if not os.path.exists(trajectories_path):
        raise FileNotFoundError(trajectories_path)

    num_edges = _read_num_edges(city_path)
    if num_edges <= 0:
        raise ValueError(f"No edges found for city={city}")

    rng = np.random.default_rng(int(seed) + int(city_id) * 10007)

    valid_indices: List[int] = []
    valid_lengths: List[int] = []

    coverage_records: List[_TrajectoryRecord] = []
    coverage_index_set: set = set()
    covered = np.zeros((num_edges,), dtype=bool)

    raw_cov_quota = int(round(float(max_traj_per_city) * float(coverage_ratio)))
    raw_cov_quota = max(0, raw_cov_quota)

    for row_idx, path_text, tlist_text in _iter_traj_rows(trajectories_path):
        path = _parse_int_list(path_text)
        if path is None or not _is_valid_path(path, num_edges=num_edges):
            continue

        ts0 = _parse_first_timestamp(tlist_text)
        if ts0 is None:
            continue

        hour, daytype = _timestamp_to_hour_daytype(ts0, city=city)
        plen = int(len(path))

        valid_indices.append(int(row_idx))
        valid_lengths.append(plen)

        if len(coverage_records) >= raw_cov_quota:
            continue

        add_new = False
        for rid in path:
            if not covered[rid]:
                add_new = True
                break
        if not add_new:
            continue

        coverage_records.append(
            _TrajectoryRecord(
                orig_idx=int(row_idx),
                path=path,
                hour=int(hour),
                daytype=int(daytype),
                length=plen,
            )
        )
        coverage_index_set.add(int(row_idx))
        for rid in path:
            covered[rid] = True

    if not valid_indices:
        raise ValueError(f"No valid trajectories found for city={city}")

    valid_indices_np = np.asarray(valid_indices, dtype=np.int64)
    valid_lengths_np = np.asarray(valid_lengths, dtype=np.int32)

    effective_cap = int(min(max_traj_per_city, valid_indices_np.shape[0]))
    cov_quota = int(min(effective_cap, int(round(float(effective_cap) * float(coverage_ratio)))))
    coverage_records = coverage_records[:cov_quota]
    coverage_index_set = {x.orig_idx for x in coverage_records}

    quantile_edges = _path_length_edges(valid_lengths_np.astype(np.float64), num_bins=max(1, int(num_bins)))

    if quantile_edges.size == 0:
        valid_bucket_ids = np.zeros_like(valid_lengths_np, dtype=np.int64)
    else:
        valid_bucket_ids = np.searchsorted(quantile_edges, valid_lengths_np, side="right").astype(np.int64)

    coverage_mask = np.isin(valid_indices_np, np.asarray(sorted(coverage_index_set), dtype=np.int64), assume_unique=False)

    n_bucket = max(1, int(num_bins))
    counts_excl = np.bincount(valid_bucket_ids[~coverage_mask], minlength=n_bucket).astype(np.int64)
    fill_target = int(max(0, effective_cap - len(coverage_records)))
    fill_quotas = _allocate_fill_quotas(fill_target=fill_target, counts=counts_excl)

    reservoirs: List[List[_TrajectoryRecord]] = [[] for _ in range(n_bucket)]
    seen = np.zeros((n_bucket,), dtype=np.int64)

    for row_idx, path_text, tlist_text in _iter_traj_rows(trajectories_path):
        if row_idx in coverage_index_set:
            continue

        path = _parse_int_list(path_text)
        if path is None or not _is_valid_path(path, num_edges=num_edges):
            continue

        ts0 = _parse_first_timestamp(tlist_text)
        if ts0 is None:
            continue

        plen = int(len(path))
        b = _bucket_id(plen, quantile_edges)
        quota = int(fill_quotas[b])
        if quota <= 0:
            continue

        seen[b] += 1
        hour, daytype = _timestamp_to_hour_daytype(ts0, city=city)
        rec = _TrajectoryRecord(
            orig_idx=int(row_idx),
            path=path,
            hour=int(hour),
            daytype=int(daytype),
            length=plen,
        )

        if len(reservoirs[b]) < quota:
            reservoirs[b].append(rec)
        else:
            j = int(rng.integers(0, int(seen[b])))
            if j < quota:
                reservoirs[b][j] = rec

    fill_records: List[_TrajectoryRecord] = []
    for bucket_records in reservoirs:
        fill_records.extend(bucket_records)

    selected_records = coverage_records + fill_records

    if len(selected_records) < effective_cap:
        need = int(effective_cap - len(selected_records))
        selected_set = {x.orig_idx for x in selected_records}
        topup: List[_TrajectoryRecord] = []
        seen_top = 0

        for row_idx, path_text, tlist_text in _iter_traj_rows(trajectories_path):
            if row_idx in selected_set:
                continue

            path = _parse_int_list(path_text)
            if path is None or not _is_valid_path(path, num_edges=num_edges):
                continue

            ts0 = _parse_first_timestamp(tlist_text)
            if ts0 is None:
                continue

            seen_top += 1
            hour, daytype = _timestamp_to_hour_daytype(ts0, city=city)
            rec = _TrajectoryRecord(
                orig_idx=int(row_idx),
                path=path,
                hour=int(hour),
                daytype=int(daytype),
                length=int(len(path)),
            )
            if len(topup) < need:
                topup.append(rec)
            else:
                j = int(rng.integers(0, int(seen_top)))
                if j < need:
                    topup[j] = rec

        selected_records.extend(topup)

    selected_records.sort(key=lambda x: x.orig_idx)
    if len(selected_records) > effective_cap:
        selected_records = selected_records[:effective_cap]

    selected_indices = np.asarray([x.orig_idx for x in selected_records], dtype=np.int64)
    selected_set_np = np.asarray(selected_indices, dtype=np.int64)

    sel_mask = np.isin(valid_indices_np, selected_set_np, assume_unique=False)
    reserved_indices = valid_indices_np[~sel_mask]

    traj_orig_idx = np.asarray([x.orig_idx for x in selected_records], dtype=np.int64)
    traj_hour = np.asarray([x.hour for x in selected_records], dtype=np.uint8)
    traj_daytype = np.asarray([x.daytype for x in selected_records], dtype=np.uint8)

    offsets = [0]
    flat_edges: List[int] = []
    for rec in selected_records:
        flat_edges.extend(rec.path)
        offsets.append(offsets[-1] + len(rec.path))

    traj_offsets = np.asarray(offsets, dtype=np.int64)
    traj_edges = np.asarray(flat_edges, dtype=np.int32)
    road_occ_count = np.bincount(traj_edges.astype(np.int64), minlength=num_edges).astype(np.int64)

    meta = {
        "cache_version": CACHE_VERSION,
        "city": city,
        "city_id": int(city_id),
        "num_edges": int(num_edges),
        "max_traj_per_city": int(max_traj_per_city),
        "seed": int(seed),
        "coverage_ratio": float(coverage_ratio),
        "num_bins": int(num_bins),
        "n_valid": int(valid_indices_np.shape[0]),
        "n_selected": int(traj_orig_idx.shape[0]),
        "n_reserved": int(reserved_indices.shape[0]),
        "n_occurrences": int(traj_edges.shape[0]),
        "quantile_edges": quantile_edges.astype(np.float64).tolist(),
        "selected_road_coverage": float((road_occ_count > 0).mean()),
        "timezone": _CITY_TIMEZONE.get(str(city).strip().lower(), "UTC"),
    }

    cache_path = _cache_file_path(
        city_path=city_path,
        max_traj_per_city=max_traj_per_city,
        seed=seed,
        coverage_ratio=coverage_ratio,
        num_bins=num_bins,
    )

    packed = {
        "meta": meta,
        "selected_indices": torch.from_numpy(selected_indices),
        "reserved_indices": torch.from_numpy(reserved_indices),
        "traj_orig_idx": torch.from_numpy(traj_orig_idx),
        "traj_offsets": torch.from_numpy(traj_offsets),
        "traj_edges": torch.from_numpy(traj_edges),
        "traj_hour": torch.from_numpy(traj_hour),
        "traj_daytype": torch.from_numpy(traj_daytype),
        "road_occ_count": torch.from_numpy(road_occ_count),
        "quantile_edges": torch.from_numpy(quantile_edges.astype(np.float64)),
    }
    torch.save(packed, cache_path)

    with open(_cache_meta_path(cache_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    return CityMobilityCache(
        city=city,
        city_id=int(city_id),
        num_edges=num_edges,
        selected_indices=selected_indices,
        reserved_indices=reserved_indices,
        traj_orig_idx=traj_orig_idx,
        traj_offsets=traj_offsets,
        traj_edges=traj_edges,
        traj_hour=traj_hour,
        traj_daytype=traj_daytype,
        road_occ_count=road_occ_count,
        quantile_edges=quantile_edges.astype(np.float64),
        meta=meta,
        cache_path=cache_path,
    )


def load_city_mobility_cache(
    city_path: str,
    city: str,
    city_id: int,
    max_traj_per_city: int,
    seed: int,
    coverage_ratio: float,
    num_bins: int,
) -> CityMobilityCache:
    cache_path = _cache_file_path(
        city_path=city_path,
        max_traj_per_city=max_traj_per_city,
        seed=seed,
        coverage_ratio=coverage_ratio,
        num_bins=num_bins,
    )
    if not os.path.exists(cache_path):
        raise FileNotFoundError(cache_path)

    packed = torch.load(cache_path, map_location="cpu")
    meta = dict(packed["meta"])

    if int(meta.get("cache_version", -1)) != CACHE_VERSION:
        raise RuntimeError(
            f"Mobility cache version mismatch: found={meta.get('cache_version')} expected={CACHE_VERSION}"
        )
    if int(meta.get("max_traj_per_city", -1)) != int(max_traj_per_city):
        raise RuntimeError("Mobility cache max_traj_per_city mismatch")
    if int(meta.get("seed", -1)) != int(seed):
        raise RuntimeError("Mobility cache seed mismatch")
    if abs(float(meta.get("coverage_ratio", -1.0)) - float(coverage_ratio)) > 1e-9:
        raise RuntimeError("Mobility cache coverage_ratio mismatch")
    if int(meta.get("num_bins", -1)) != int(num_bins):
        raise RuntimeError("Mobility cache num_bins mismatch")

    return CityMobilityCache(
        city=city,
        city_id=int(city_id),
        num_edges=int(meta["num_edges"]),
        selected_indices=packed["selected_indices"].cpu().numpy().astype(np.int64),
        reserved_indices=packed["reserved_indices"].cpu().numpy().astype(np.int64),
        traj_orig_idx=packed["traj_orig_idx"].cpu().numpy().astype(np.int64),
        traj_offsets=packed["traj_offsets"].cpu().numpy().astype(np.int64),
        traj_edges=packed["traj_edges"].cpu().numpy().astype(np.int32),
        traj_hour=packed["traj_hour"].cpu().numpy().astype(np.uint8),
        traj_daytype=packed["traj_daytype"].cpu().numpy().astype(np.uint8),
        road_occ_count=packed["road_occ_count"].cpu().numpy().astype(np.int64),
        quantile_edges=packed["quantile_edges"].cpu().numpy().astype(np.float64),
        meta=meta,
        cache_path=cache_path,
    )


def get_or_build_city_mobility_cache(
    city_path: str,
    city: str,
    city_id: int,
    max_traj_per_city: int,
    seed: int,
    coverage_ratio: float = 0.30,
    num_bins: int = 10,
    force_rebuild: bool = False,
) -> CityMobilityCache:
    if not force_rebuild:
        try:
            return load_city_mobility_cache(
                city_path=city_path,
                city=city,
                city_id=city_id,
                max_traj_per_city=max_traj_per_city,
                seed=seed,
                coverage_ratio=coverage_ratio,
                num_bins=num_bins,
            )
        except Exception:
            pass

    return _build_city_cache(
        city_path=city_path,
        city=city,
        city_id=city_id,
        max_traj_per_city=max_traj_per_city,
        seed=seed,
        coverage_ratio=coverage_ratio,
        num_bins=num_bins,
    )


def split_trajectory_positions(
    num_selected: int,
    seed: int,
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
) -> Dict[str, np.ndarray]:
    order = np.arange(int(num_selected), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(order)

    n_train = int(round(float(num_selected) * float(train_ratio)))
    n_train = min(max(n_train, 1), max(num_selected - 1, 1)) if num_selected >= 2 else num_selected

    n_val = int(round(float(num_selected) * float(val_ratio)))
    if num_selected >= 2:
        n_val = max(n_val, 1)
    n_val = min(n_val, max(num_selected - n_train, 0))

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]

    if val_idx.size == 0 and train_idx.size > 1:
        val_idx = train_idx[-1:].copy()
        train_idx = train_idx[:-1]

    return {
        "train": train_idx.astype(np.int64),
        "val": val_idx.astype(np.int64),
    }


@dataclass
class CityOccurrenceSource:
    city: str
    city_id: int
    token_offset: int
    num_edges: int
    traj_offsets: np.ndarray
    traj_edges: np.ndarray
    traj_hour: np.ndarray
    traj_daytype: np.ndarray
    traj_positions: np.ndarray


class OccurrenceDataset(Dataset):
    def __init__(
        self,
        sources: Sequence[CityOccurrenceSource],
        pad_token_id: int = 0,
        mask_center: bool = False,
        mask_token_id: Optional[int] = None,
    ):
        self.sources = list(sources)
        self.pad_token_id = int(pad_token_id)
        self.mask_center = bool(mask_center)
        self.mask_token_id = int(mask_token_id) if mask_token_id is not None else None
        if self.mask_center and self.mask_token_id is None:
            raise ValueError("mask_center=True requires mask_token_id")
        if self.mask_center and self.mask_token_id == self.pad_token_id:
            raise ValueError("mask_token_id must be different from pad_token_id")

        traj_source: List[np.ndarray] = []
        traj_pos: List[np.ndarray] = []
        traj_lengths: List[np.ndarray] = []

        for src_id, src in enumerate(self.sources):
            pos = np.asarray(src.traj_positions, dtype=np.int64)
            if pos.size == 0:
                continue
            lens = src.traj_offsets[pos + 1] - src.traj_offsets[pos]
            keep = lens > 0
            pos = pos[keep]
            lens = lens[keep]
            if pos.size == 0:
                continue

            traj_source.append(np.full((pos.shape[0],), src_id, dtype=np.int32))
            traj_pos.append(pos.astype(np.int64))
            traj_lengths.append(lens.astype(np.int64))

        if traj_source:
            self.traj_source = np.concatenate(traj_source, axis=0)
            self.traj_pos = np.concatenate(traj_pos, axis=0)
            self.traj_lengths = np.concatenate(traj_lengths, axis=0)
        else:
            self.traj_source = np.zeros((0,), dtype=np.int32)
            self.traj_pos = np.zeros((0,), dtype=np.int64)
            self.traj_lengths = np.zeros((0,), dtype=np.int64)

        self.occ_offsets = np.zeros((self.traj_lengths.shape[0] + 1,), dtype=np.int64)
        if self.traj_lengths.size > 0:
            self.occ_offsets[1:] = np.cumsum(self.traj_lengths, dtype=np.int64)

        self.short_radius_max = 2
        self.mid_radius_max = 5

    def __len__(self) -> int:
        return int(self.occ_offsets[-1])

    @staticmethod
    def _build_window(path: np.ndarray, center_pos: int, radius: int, max_radius: int, pad_token_id: int) -> np.ndarray:
        out = np.full((2 * max_radius + 1,), int(pad_token_id), dtype=np.int64)
        for d in range(-radius, radius + 1):
            src_pos = center_pos + d
            dst_pos = d + max_radius
            if 0 <= src_pos < path.shape[0]:
                out[dst_pos] = int(path[src_pos])
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        idx = int(idx)
        traj_i = int(np.searchsorted(self.occ_offsets, idx, side="right") - 1)
        occ_i = int(idx - self.occ_offsets[traj_i])

        src_id = int(self.traj_source[traj_i])
        src = self.sources[src_id]
        traj_pos = int(self.traj_pos[traj_i])

        start = int(src.traj_offsets[traj_pos])
        end = int(src.traj_offsets[traj_pos + 1])
        path = src.traj_edges[start:end].astype(np.int64)
        m = int(path.shape[0])

        r_short = max(1, min(2, m // 10))
        r_mid = max(1, min(5, m // 3))

        short_local = self._build_window(
            path=path,
            center_pos=occ_i,
            radius=r_short,
            max_radius=self.short_radius_max,
            pad_token_id=-1,
        )
        mid_local = self._build_window(
            path=path,
            center_pos=occ_i,
            radius=r_mid,
            max_radius=self.mid_radius_max,
            pad_token_id=-1,
        )

        center_road = int(path[occ_i])

        short_tokens = np.where(short_local >= 0, short_local + int(src.token_offset) + 1, self.pad_token_id)
        mid_tokens = np.where(mid_local >= 0, mid_local + int(src.token_offset) + 1, self.pad_token_id)
        if self.mask_center:
            short_tokens[self.short_radius_max] = int(self.mask_token_id)
            mid_tokens[self.mid_radius_max] = int(self.mask_token_id)
        center_token = int(src.token_offset) + center_road + 1

        return {
            "short_tokens": torch.as_tensor(short_tokens, dtype=torch.long),
            "mid_tokens": torch.as_tensor(mid_tokens, dtype=torch.long),
            "hour": torch.tensor(int(src.traj_hour[traj_pos]), dtype=torch.long),
            "daytype": torch.tensor(int(src.traj_daytype[traj_pos]), dtype=torch.float32),
            "center_token": torch.tensor(center_token, dtype=torch.long),
            "center_road": torch.tensor(center_road, dtype=torch.long),
            "city_id": torch.tensor(int(src.city_id), dtype=torch.long),
        }


def make_occurrence_source(
    cache: CityMobilityCache,
    token_offset: int,
    traj_positions: np.ndarray,
) -> CityOccurrenceSource:
    return CityOccurrenceSource(
        city=cache.city,
        city_id=int(cache.city_id),
        token_offset=int(token_offset),
        num_edges=int(cache.num_edges),
        traj_offsets=cache.traj_offsets,
        traj_edges=cache.traj_edges,
        traj_hour=cache.traj_hour,
        traj_daytype=cache.traj_daytype,
        traj_positions=np.asarray(traj_positions, dtype=np.int64),
    )


def build_city_token_offsets(caches: Sequence[CityMobilityCache]) -> Dict[str, int]:
    offsets: Dict[str, int] = {}
    cur = 0
    for c in caches:
        offsets[c.city] = int(cur)
        cur += int(c.num_edges)
    return offsets


def build_joint_city_caches(
    data_root: str,
    cities: Sequence[str],
    max_traj_per_city: int,
    seed: int,
    coverage_ratio: float = 0.30,
    num_bins: int = 10,
    force_rebuild: bool = False,
) -> List[CityMobilityCache]:
    out: List[CityMobilityCache] = []
    for city_id, city in enumerate(cities):
        city_path = os.path.join(data_root, city)
        cache = get_or_build_city_mobility_cache(
            city_path=city_path,
            city=city,
            city_id=city_id,
            max_traj_per_city=max_traj_per_city,
            seed=seed,
            coverage_ratio=coverage_ratio,
            num_bins=num_bins,
            force_rebuild=force_rebuild,
        )
        out.append(cache)
    return out
