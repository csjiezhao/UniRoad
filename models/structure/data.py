import json
import math
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from utils.graph_io import load_graph_from_files


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, np.number)):
        if np.isnan(value):
            return default
        return float(value)
    try:
        parsed = float(str(value))
        if np.isnan(parsed):
            return default
        return parsed
    except Exception:
        return default


def _parse_lanes(value: object) -> Tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    text = str(value).strip().lower()
    if text in ("", "nan", "none"):
        return 0.0, 0.0
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return 0.0, 0.0
    lanes = float(np.mean(nums))
    return lanes, 1.0


def _has_truthy_tag(value: object, expected_tokens: Sequence[str] = ("yes", "true", "1")) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text in ("", "nan", "none"):
        return 0.0
    for tok in expected_tokens:
        if tok in text:
            return 1.0
    return 0.0


def _bearing_deg(linestring) -> float:
    coords = list(linestring.coords)
    if len(coords) < 2:
        return 0.0
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if ang < 0:
        ang += 360.0
    return ang


def _sinuosity(linestring) -> float:
    coords = list(linestring.coords)
    if len(coords) < 2:
        return 1.0
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    straight = math.hypot(x2 - x1, y2 - y1)
    if straight < 1e-8:
        return 1.0
    return float(linestring.length / straight)


def _angle_diff_180(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d


def _normalize_columns(values: np.ndarray, indices: Sequence[int]) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    out = values.copy().astype(np.float32)
    mean = np.zeros(values.shape[1], dtype=np.float32)
    std = np.ones(values.shape[1], dtype=np.float32)
    for idx in indices:
        col = out[:, idx]
        m = float(col.mean())
        s = float(col.std())
        if s < 1e-6:
            s = 1.0
        out[:, idx] = (col - m) / s
        mean[idx] = m
        std[idx] = s
    return out, {"mean": mean.tolist(), "std": std.tolist(), "normalized_indices": list(indices)}


def _build_line_adjacency(edge_idx_to_uvk: Dict[int, Tuple[int, int, int]]) -> Dict[int, set]:
    incident: Dict[int, set] = defaultdict(set)
    for edge_idx, (u, v, _key) in edge_idx_to_uvk.items():
        incident[int(u)].add(edge_idx)
        incident[int(v)].add(edge_idx)

    line_adj: Dict[int, set] = {edge_idx: set() for edge_idx in edge_idx_to_uvk}
    for edge_set in incident.values():
        edge_list = list(edge_set)
        for i, e_i in enumerate(edge_list):
            for j in range(i + 1, len(edge_list)):
                e_j = edge_list[j]
                line_adj[e_i].add(e_j)
                line_adj[e_j].add(e_i)
    return line_adj


def _k_hop_edge_distances(line_adj: Dict[int, set], center_edge_idx: int, k_hop: int) -> Dict[int, int]:
    dist = {center_edge_idx: 0}
    q = deque([center_edge_idx])
    while q:
        cur = q.popleft()
        if dist[cur] >= k_hop:
            continue
        for nxt in line_adj[cur]:
            if nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            q.append(nxt)
    return dist


def _to_bidirected_edges(
    local_edges: Iterable[Tuple[int, int]],
    local_edge_attrs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if local_edge_attrs.ndim == 1:
        local_edge_attrs = local_edge_attrs.reshape(0, 0)
    pairs = []
    attrs = []
    local_edges = list(local_edges)
    for i, (a, b) in enumerate(local_edges):
        if a == b:
            continue
        attr = local_edge_attrs[i]
        pairs.append((a, b))
        attrs.append(attr)
        pairs.append((b, a))
        attrs.append(attr)
    if not pairs:
        edge_attr_dim = int(local_edge_attrs.shape[1]) if local_edge_attrs.ndim == 2 else 0
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, edge_attr_dim), dtype=np.float32)
    edge_index = np.asarray(pairs, dtype=np.int64).T
    edge_attr = np.asarray(attrs, dtype=np.float32)
    return edge_index, edge_attr


def _turn_hist_8(center_bearing: float, neighbor_bearings: Sequence[float]) -> np.ndarray:
    hist = np.zeros(8, dtype=np.float32)
    if not neighbor_bearings:
        return hist
    for b in neighbor_bearings:
        diff = _angle_diff_180(center_bearing, b)
        bucket = min(int(diff // 22.5), 7)
        hist[bucket] += 1.0
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist


@dataclass
class CacheBundle:
    cache_path: str
    meta: Dict[str, object]
    normalizers: Dict[str, Dict[str, List[float]]]
    samples: List[Dict[str, torch.Tensor]]


class EdgeCenteredDataset(Dataset):
    def __init__(self, samples: Sequence[Dict[str, torch.Tensor]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        s = self.samples[idx]
        return Data(
            x=s["x"],
            edge_index=s["edge_index"],
            edge_attr=s["edge_attr"],
            center_edge_index=s["center_edge_index"],
            center_edge_feat=s["center_edge_feat"],
            sketch_cont=s["sketch_cont"],
            sketch_hist=s["sketch_hist"],
            sketch_deadend=s["sketch_deadend"],
            edge_idx=s["edge_idx"],
        )


class CacheMismatchError(RuntimeError):
    pass


def _cache_file_path(city_path: str, k_hop: int) -> str:
    cache_dir = os.path.join(city_path, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"prototype_v3_k{k_hop}.pt")


def _cache_meta_path(cache_path: str) -> str:
    return cache_path.replace(".pt", "_meta.json")


def load_cache(city_path: str, k_hop: int) -> CacheBundle:
    cache_path = _cache_file_path(city_path, k_hop)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(cache_path)
    try:
        packed = torch.load(cache_path, map_location="cpu", weights_only=True)
    except Exception:
        packed = torch.load(cache_path, map_location="cpu", weights_only=False)
    meta = packed["meta"]
    if int(meta.get("k_hop", -1)) != int(k_hop):
        raise CacheMismatchError(f"Cached k_hop={meta.get('k_hop')} != requested k_hop={k_hop}")
    if int(meta.get("cache_version", 0)) < 3:
        raise CacheMismatchError("Cache schema mismatch: expected v3 cache.")
    samples = packed["samples"]
    if not samples:
        raise CacheMismatchError("Cache has no samples")
    first = samples[0]
    required_keys = {"edge_attr", "center_edge_feat"}
    if any(k not in first for k in required_keys):
        raise CacheMismatchError("Cache schema mismatch: missing v3 sample keys")
    if "edge_attr_dim" not in meta:
        raise CacheMismatchError("Cache schema mismatch: missing v3 dims in meta")
    return CacheBundle(
        cache_path=cache_path,
        meta=meta,
        normalizers=packed["normalizers"],
        samples=samples,
    )


def build_cache(city_path: str, city: str, k_hop: int = 3, force_rebuild: bool = False) -> CacheBundle:
    cache_path = _cache_file_path(city_path, k_hop)
    if os.path.exists(cache_path) and not force_rebuild:
        return load_cache(city_path=city_path, k_hop=k_hop)

    graph, nodes_gdf, edges_gdf = load_graph_from_files(city_path)

    # Stable mappings between edge_idx and primal edge tuple.
    edge_idx_to_uvk: Dict[int, Tuple[int, int, int]] = {}
    for (u, v, key), row in edges_gdf.iterrows():
        edge_idx_to_uvk[int(row["edge_idx"])] = (int(u), int(v), int(key))
    num_edges = len(edge_idx_to_uvk)

    # Node raw features: [street_count, degree, in_degree, out_degree]
    undirected_graph = nx.Graph()
    undirected_graph.add_nodes_from(graph.nodes())
    for u, v, _key in graph.edges(keys=True):
        undirected_graph.add_edge(int(u), int(v))

    node_ids = sorted(nodes_gdf.index.astype(int).tolist())
    node_id_to_pos = {nid: i for i, nid in enumerate(node_ids)}

    node_raw = np.zeros((len(node_ids), 4), dtype=np.float32)
    for nid in node_ids:
        row = nodes_gdf.loc[nid]
        sc = _safe_float(row.get("street_count", 0.0), 0.0)
        deg = float(undirected_graph.degree(nid))
        indeg = float(graph.in_degree(nid))
        outdeg = float(graph.out_degree(nid))
        node_raw[node_id_to_pos[nid], :] = np.asarray([sc, deg, indeg, outdeg], dtype=np.float32)
    node_raw_norm, node_norm_stats = _normalize_columns(node_raw, [0, 1, 2, 3])
    node_raw_map = {nid: node_raw_norm[node_id_to_pos[nid]] for nid in node_ids}

    # Edge raw features (v2):
    # [log_length, sin_bearing, cos_bearing, curvature, sinuosity, oneway,
    #  bridge, tunnel, roundabout, lanes_parsed, lanes_known_mask]
    sorted_edge_idxs = sorted(edge_idx_to_uvk.keys())
    edge_raw = np.zeros((len(sorted_edge_idxs), 11), dtype=np.float32)
    edge_bearing_map: Dict[int, float] = {}
    edge_raw_map: Dict[int, np.ndarray] = {}

    for i, edge_idx in enumerate(sorted_edge_idxs):
        u, v, key = edge_idx_to_uvk[edge_idx]
        row = edges_gdf.loc[(u, v, key)]
        geom = row["geometry"]

        length = _safe_float(row.get("length", 0.0), 0.0)
        log_len = math.log1p(max(length, 0.0))
        bearing = _bearing_deg(geom)
        sinuosity = _sinuosity(geom)
        curvature = max(sinuosity - 1.0, 0.0)
        oneway = float(bool(row.get("oneway", False)))
        bridge = _has_truthy_tag(row.get("bridge", None))
        tunnel = _has_truthy_tag(row.get("tunnel", None))
        junction_text = str(row.get("junction", "")).lower()
        roundabout = 1.0 if ("roundabout" in junction_text or "circular" in junction_text) else 0.0
        lanes_val, lanes_known = _parse_lanes(row.get("lanes", None))

        feat = np.asarray(
            [
                log_len,
                math.sin(math.radians(bearing)),
                math.cos(math.radians(bearing)),
                curvature,
                sinuosity,
                oneway,
                bridge,
                tunnel,
                roundabout,
                lanes_val,
                lanes_known,
            ],
            dtype=np.float32,
        )
        edge_raw[i, :] = feat
        edge_bearing_map[edge_idx] = bearing

    # Normalize continuous-like columns only; keep binary masks/tags untouched.
    edge_norm_cols = [0, 3, 4, 9]
    edge_raw_norm, edge_norm_stats = _normalize_columns(edge_raw, edge_norm_cols)
    for i, edge_idx in enumerate(sorted_edge_idxs):
        edge_raw_map[edge_idx] = edge_raw_norm[i]

    line_adj = _build_line_adjacency(edge_idx_to_uvk)

    intermediates: List[Dict[str, object]] = []
    sketch_cont_all = np.zeros((num_edges, 7), dtype=np.float32)

    for row_idx, center_edge_idx in enumerate(sorted_edge_idxs):
        u, v, _key = edge_idx_to_uvk[center_edge_idx]
        dist = _k_hop_edge_distances(line_adj, center_edge_idx, k_hop)
        selected_edge_idxs = sorted(dist.keys())

        neighbor_edge_count = float(sum(1 for d in dist.values() if d == 1))
        two_hop_edge_count = float(sum(1 for d in dist.values() if d == 2))

        deg_u = float(undirected_graph.degree(u))
        deg_v = float(undirected_graph.degree(v))
        branch_u = max(deg_u - 1.0, 0.0)
        branch_v = max(deg_v - 1.0, 0.0)
        dead_end_flag = 1.0 if (deg_u <= 1.0 or deg_v <= 1.0) else 0.0

        local_nx = nx.Graph()
        local_nodes = set()
        for edge_idx in selected_edge_idxs:
            a, b, _k = edge_idx_to_uvk[edge_idx]
            local_nodes.add(a)
            local_nodes.add(b)
            local_nx.add_edge(a, b)

        local_density = float(nx.density(local_nx)) if local_nx.number_of_nodes() > 1 else 0.0

        neighbor_bearings = [edge_bearing_map[eid] for eid, hop in dist.items() if hop == 1]
        turn_hist = _turn_hist_8(edge_bearing_map[center_edge_idx], neighbor_bearings)

        sketch_cont = np.asarray(
            [deg_u, deg_v, neighbor_edge_count, two_hop_edge_count, branch_u, branch_v, local_density],
            dtype=np.float32,
        )
        sketch_cont_all[row_idx] = sketch_cont

        intermediates.append(
            {
                "center_edge_idx": center_edge_idx,
                "center_u": u,
                "center_v": v,
                "selected_edge_idxs": selected_edge_idxs,
                "dist": dist,
                "turn_hist": turn_hist,
                "dead_end_flag": dead_end_flag,
            }
        )

    sketch_cont_norm, sketch_norm_stats = _normalize_columns(sketch_cont_all, [0, 1, 2, 3, 4, 5, 6])

    sketch_cont_map = {int(intermediates[i]["center_edge_idx"]): sketch_cont_norm[i] for i in range(len(intermediates))}

    samples: List[Dict[str, torch.Tensor]] = []
    edge_attr_dim = int(edge_raw.shape[1] + 4)

    for item in intermediates:
        center_edge_idx = int(item["center_edge_idx"])
        center_u = int(item["center_u"])
        center_v = int(item["center_v"])
        selected_edge_idxs = list(item["selected_edge_idxs"])
        dist = item["dist"]

        local_nodes = set()
        local_edges_info: List[Tuple[int, int, int]] = []
        for edge_idx in selected_edge_idxs:
            a, b, _k = edge_idx_to_uvk[edge_idx]
            local_nodes.add(a)
            local_nodes.add(b)
            local_edges_info.append((a, b, edge_idx))

        local_nodes_sorted = sorted(local_nodes)
        local_node_to_idx = {nid: i for i, nid in enumerate(local_nodes_sorted)}

        local_nx = nx.Graph()
        local_nx.add_nodes_from(local_nodes_sorted)
        for a, b, _ in local_edges_info:
            local_nx.add_edge(a, b)

        dist_u = nx.single_source_shortest_path_length(local_nx, center_u, cutoff=k_hop + 1)
        dist_v = nx.single_source_shortest_path_length(local_nx, center_v, cutoff=k_hop + 1)

        node_feats = np.zeros((len(local_nodes_sorted), 7), dtype=np.float32)
        for i, nid in enumerate(local_nodes_sorted):
            raw = node_raw_map[nid]
            du = float(dist_u.get(nid, k_hop + 1)) / float(k_hop + 1)
            dv = float(dist_v.get(nid, k_hop + 1)) / float(k_hop + 1)
            center_flag = 1.0 if (nid == center_u or nid == center_v) else 0.0
            node_feats[i] = np.concatenate([raw, np.asarray([du, dv, center_flag], dtype=np.float32)], axis=0)

        local_edges_idx = []
        local_edge_attr = []
        for a, b, edge_idx in local_edges_info:
            local_edges_idx.append((local_node_to_idx[a], local_node_to_idx[b]))
            base = edge_raw_map[edge_idx]
            hop_norm = float(dist[edge_idx]) / float(max(k_hop, 1))
            share_u = 1.0 if (a == center_u or b == center_u) else 0.0
            share_v = 1.0 if (a == center_v or b == center_v) else 0.0
            is_center = 1.0 if edge_idx == center_edge_idx else 0.0
            local_edge_attr.append(
                np.concatenate(
                    [
                        base,
                        np.asarray([hop_norm, share_u, share_v, is_center], dtype=np.float32),
                    ],
                    axis=0,
                )
            )

        if local_edge_attr:
            local_edge_attr_arr = np.asarray(local_edge_attr, dtype=np.float32)
        else:
            local_edge_attr_arr = np.zeros((0, edge_attr_dim), dtype=np.float32)

        edge_index, edge_attr = _to_bidirected_edges(local_edges_idx, local_edge_attr_arr)
        center_edge_index = np.asarray(
            [[local_node_to_idx[center_u]], [local_node_to_idx[center_v]]],
            dtype=np.int64,
        )

        sample = {
            "x": torch.from_numpy(node_feats).float(),
            "edge_index": torch.from_numpy(edge_index).long(),
            "edge_attr": torch.from_numpy(edge_attr).float(),
            "center_edge_index": torch.from_numpy(center_edge_index).long(),
            "center_edge_feat": torch.from_numpy(edge_raw_map[center_edge_idx]).float().unsqueeze(0),
            "sketch_cont": torch.from_numpy(sketch_cont_map[center_edge_idx]).float().unsqueeze(0),
            "sketch_hist": torch.from_numpy(np.asarray(item["turn_hist"], dtype=np.float32)).float().unsqueeze(0),
            "sketch_deadend": torch.tensor([[float(item["dead_end_flag"])]], dtype=torch.float32),
            "edge_idx": torch.tensor([int(center_edge_idx)], dtype=torch.long),
        }
        samples.append(sample)

    meta = {
        "cache_version": 3,
        "city": city,
        "k_hop": int(k_hop),
        "num_samples": int(len(samples)),
        "num_edges": int(num_edges),
        "node_feat_dim": 7,
        "node_raw_dim": 4,
        "edge_feat_dim": 11,
        "edge_attr_dim": int(edge_attr_dim),
        "sketch_cont_dim": 7,
        "sketch_hist_dim": 8,
        "sketch_deadend_dim": 1,
    }
    normalizers = {
        "node_raw": node_norm_stats,
        "edge_feat": edge_norm_stats,
        "sketch_cont": sketch_norm_stats,
    }

    packed = {
        "meta": meta,
        "normalizers": normalizers,
        "samples": samples,
    }
    torch.save(packed, cache_path)

    with open(_cache_meta_path(cache_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    return CacheBundle(
        cache_path=cache_path,
        meta=meta,
        normalizers=normalizers,
        samples=samples,
    )


def get_or_build_cache(city_path: str, city: str, k_hop: int, force_rebuild: bool = False) -> CacheBundle:
    if force_rebuild:
        return build_cache(city_path=city_path, city=city, k_hop=k_hop, force_rebuild=True)
    try:
        return load_cache(city_path=city_path, k_hop=k_hop)
    except (FileNotFoundError, CacheMismatchError):
        return build_cache(city_path=city_path, city=city, k_hop=k_hop, force_rebuild=True)


def split_indices(num_samples: int, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.arange(num_samples)
    rng.shuffle(order)

    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    n_test = num_samples - n_train - n_val

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val : n_train + n_val + n_test]

    return {
        "train": np.asarray(train_idx, dtype=np.int64),
        "val": np.asarray(val_idx, dtype=np.int64),
        "test": np.asarray(test_idx, dtype=np.int64),
    }



