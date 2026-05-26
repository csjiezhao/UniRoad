import argparse
import json
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.mobility.data import (
    OccurrenceDataset,
    build_city_token_offsets,
    build_joint_city_caches,
    make_occurrence_source,
    split_trajectory_positions,
)
from models.mobility.net import (
    MobilityBranchNet,
    compute_context_center_loss,
    compute_same_road_consistency_loss,
    compute_time_context_loss,
)
from utils.graph_io import list_available_cities, parse_city_list, set_seed, stable_city_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mobility branch training and readout")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train mobility encoder")
    p_train.add_argument("--cities", type=str, default="chengdu,porto,rome,sanfran")
    p_train.add_argument("--src-id", type=str, default="joint4")

    p_train.add_argument("--data-root", type=str, default="data")
    p_train.add_argument("--ckpt-root", type=str, default="ckpts")

    p_train.add_argument("--max-traj-per-city", type=int, default=150000)
    p_train.add_argument("--coverage-ratio", type=float, default=0.30)
    p_train.add_argument("--num-path-bins", type=int, default=10)
    p_train.add_argument("--force-rebuild-cache", action="store_true")
    p_train.add_argument("--mask-center", action=argparse.BooleanOptionalAction, default=True)

    p_train.add_argument("--emb-dim", type=int, default=128)
    p_train.add_argument("--ctx-dim", type=int, default=64)
    p_train.add_argument("--short-layers", type=int, default=1)
    p_train.add_argument("--mid-layers", type=int, default=2)
    p_train.add_argument("--nhead", type=int, default=4)
    p_train.add_argument("--dropout", type=float, default=0.1)

    p_train.add_argument("--max-epoch", type=int, default=200)
    p_train.add_argument("--patience", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--num-workers", type=int, default=4)
    p_train.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p_train.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p_train.add_argument("--prefetch-factor", type=int, default=2)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight-decay", type=float, default=1e-5)
    p_train.add_argument("--grad-clip", type=float, default=2.0)
    p_train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p_train.add_argument("--save-every-epoch", action=argparse.BooleanOptionalAction, default=False)
    p_train.add_argument("--save-best-history", action=argparse.BooleanOptionalAction, default=True)

    p_train.add_argument("--train-ratio", type=float, default=0.9)
    p_train.add_argument("--val-ratio", type=float, default=0.1)

    p_train.add_argument("--temperature", type=float, default=0.07)
    p_train.add_argument("--num-rand-neg", type=int, default=8)
    p_train.add_argument("--num-freq-neg", type=int, default=8)
    p_train.add_argument("--num-freq-buckets", type=int, default=8)
    p_train.add_argument("--include-inbatch-neg", action=argparse.BooleanOptionalAction, default=True)
    p_train.add_argument("--lambda-consistency", type=float, default=0.1)
    p_train.add_argument("--lambda-time", type=float, default=0.1)

    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--device", type=str, default="cpu")
    p_train.add_argument("--train-tag", type=str, default="")
    p_train.add_argument("--max-train-steps-per-epoch", type=int, default=0)
    p_train.add_argument("--max-val-steps-per-epoch", type=int, default=0)

    p_readout = sub.add_parser("readout", help="Export mobility embedding")
    p_readout.add_argument("--model-pt", type=str, required=True)
    p_readout.add_argument("--trg-city", type=str, required=True)
    p_readout.add_argument("--src-id", type=str, default="")
    p_readout.add_argument("--data-root", type=str, default="data")
    p_readout.add_argument("--emb-root", type=str, default="embs")
    p_readout.add_argument("--batch-size", type=int, default=512)
    p_readout.add_argument("--device", type=str, default="cpu")
    p_readout.add_argument("--readout-split", type=str, default="train_only", choices=["train_only"])
    p_readout.add_argument("--train-tag", type=str, default="")

    return parser


def _pick_args(args: argparse.Namespace, keys: Sequence[str]) -> Dict[str, Any]:
    return {k: getattr(args, k) for k in keys}


def _resolve_train_cities(cities_raw: str, data_root: str) -> List[str]:
    cities = parse_city_list(cities_raw)
    if not cities:
        cities = list_available_cities(data_root)
    if not cities:
        raise ValueError("No valid cities found for mobility training.")
    return cities


class NegativeSampler:
    def __init__(
        self,
        caches,
        city_offsets: Dict[str, int],
        num_tokens: int,
        num_freq_buckets: int,
        seed: int,
    ):
        self.num_tokens = int(num_tokens)
        self.num_freq_buckets = int(num_freq_buckets)
        self.rng = np.random.default_rng(int(seed) + 7103)

        self.city_tokens: Dict[int, np.ndarray] = {}
        self.city_bucket_tokens: Dict[int, List[np.ndarray]] = {}
        self.token_bucket = np.full((self.num_tokens,), -1, dtype=np.int16)

        for cache in caches:
            city_id = int(cache.city_id)
            offset = int(city_offsets[cache.city])
            road_count = int(cache.num_edges)

            tokens = (np.arange(road_count, dtype=np.int64) + offset + 1).astype(np.int64)
            self.city_tokens[city_id] = tokens

            freq = cache.road_occ_count.astype(np.float64)
            if freq.size == 0:
                bucket_ids = np.zeros((0,), dtype=np.int16)
            else:
                transformed = np.log1p(freq)
                if self.num_freq_buckets <= 1:
                    edges = np.asarray([], dtype=np.float64)
                else:
                    q = np.linspace(0.0, 1.0, self.num_freq_buckets + 1, dtype=np.float64)[1:-1]
                    edges = np.quantile(transformed, q) if q.size > 0 else np.asarray([], dtype=np.float64)
                bucket_ids = np.searchsorted(edges, transformed, side="right").astype(np.int16)

            bucket_lists: List[np.ndarray] = []
            for b in range(self.num_freq_buckets):
                idx = np.where(bucket_ids == b)[0]
                if idx.size == 0:
                    bucket_lists.append(tokens)
                else:
                    bucket_lists.append(tokens[idx])

            self.city_bucket_tokens[city_id] = bucket_lists
            self.token_bucket[tokens] = bucket_ids

    def _sample_one(self, candidates: np.ndarray, avoid: int, k: int) -> np.ndarray:
        if candidates.size == 0:
            return np.full((k,), avoid, dtype=np.int64)

        out = self.rng.choice(candidates, size=int(k), replace=True).astype(np.int64)
        if candidates.size > 1:
            mask = out == avoid
            tries = 0
            while bool(mask.any()) and tries < 5:
                out[mask] = self.rng.choice(candidates, size=int(mask.sum()), replace=True)
                mask = out == avoid
                tries += 1
            if bool(mask.any()):
                fallback = candidates[candidates != avoid]
                if fallback.size > 0:
                    out[mask] = self.rng.choice(fallback, size=int(mask.sum()), replace=True)
        return out

    def sample(self, city_ids: torch.Tensor, center_tokens: torch.Tensor, num_rand: int, num_freq: int):
        city_np = city_ids.detach().cpu().numpy().astype(np.int64)
        center_np = center_tokens.detach().cpu().numpy().astype(np.int64)
        bsz = center_np.shape[0]

        rand = np.zeros((bsz, int(num_rand)), dtype=np.int64) if num_rand > 0 else None
        freq = np.zeros((bsz, int(num_freq)), dtype=np.int64) if num_freq > 0 else None

        for i in range(bsz):
            city_id = int(city_np[i])
            center_tok = int(center_np[i])
            city_tokens = self.city_tokens[city_id]

            if rand is not None:
                rand[i] = self._sample_one(city_tokens, avoid=center_tok, k=int(num_rand))

            if freq is not None:
                b = int(self.token_bucket[center_tok])
                if b < 0 or b >= self.num_freq_buckets:
                    cand = city_tokens
                else:
                    cand = self.city_bucket_tokens[city_id][b]
                freq[i] = self._sample_one(cand, avoid=center_tok, k=int(num_freq))

        rand_t = torch.from_numpy(rand).to(center_tokens.device) if rand is not None else None
        freq_t = torch.from_numpy(freq).to(center_tokens.device) if freq is not None else None
        return rand_t, freq_t


def _mean_metric(logs: Dict[str, float], n: int) -> Dict[str, float]:
    if n <= 0:
        return {k: 0.0 for k in logs}
    return {k: v / n for k, v in logs.items()}


def _run_epoch(
    model: MobilityBranchNet,
    loader: DataLoader,
    device: torch.device,
    optimizer,
    neg_sampler: NegativeSampler,
    train_cfg: Dict[str, Any],
    max_steps: int,
    scaler=None,
    use_amp: bool = False,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    logs = {
        "total": 0.0,
        "ctx": 0.0,
        "ctx_top1": 0.0,
        "cons": 0.0,
        "time": 0.0,
        "time_hour": 0.0,
        "time_day": 0.0,
        "time_hour_acc": 0.0,
        "time_day_acc": 0.0,
    }
    count = 0

    for step, batch in enumerate(loader, start=1):
        if max_steps > 0 and step > max_steps:
            break

        short_tokens = batch["short_tokens"].to(device)
        mid_tokens = batch["mid_tokens"].to(device)
        hour = batch["hour"].to(device)
        daytype = batch["daytype"].to(device)
        center_tokens = batch["center_token"].to(device)
        city_ids = batch["city_id"].to(device)

        rand_neg, freq_neg = neg_sampler.sample(
            city_ids=city_ids,
            center_tokens=center_tokens,
            num_rand=int(train_cfg["num_rand_neg"]),
            num_freq=int(train_cfg["num_freq_neg"]),
        )

        with torch.cuda.amp.autocast(enabled=bool(use_amp and device.type == "cuda")):
            out = model(
                short_tokens=short_tokens,
                mid_tokens=mid_tokens,
                hour=hour,
                daytype=daytype.long(),
            )

            ctx = compute_context_center_loss(
                model=model,
                h_pred=out["h_pred"],
                center_tokens=center_tokens,
                rand_neg_tokens=rand_neg,
                freq_neg_tokens=freq_neg,
                temperature=float(train_cfg["temperature"]),
                include_inbatch=bool(train_cfg.get("include_inbatch_neg", True)),
            )
            cons = compute_same_road_consistency_loss(out["h_shared"], center_tokens=center_tokens)
            time_obj = compute_time_context_loss(
                hour_logits=out["hour_logits"],
                daytype_logit=out["daytype_logit"],
                hour_target=hour,
                daytype_target=daytype,
            )

            total = (
                ctx["loss"]
                + float(train_cfg["lambda_consistency"]) * cons
                + float(train_cfg["lambda_time"]) * time_obj["loss"]
            )

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and bool(scaler.is_enabled()):
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg["grad_clip"]))
                optimizer.step()

        bs = int(short_tokens.shape[0])
        count += bs
        logs["total"] += float(total.detach().cpu()) * bs
        logs["ctx"] += float(ctx["loss"].detach().cpu()) * bs
        logs["ctx_top1"] += float(ctx["top1"].detach().cpu()) * bs
        logs["cons"] += float(cons.detach().cpu()) * bs
        logs["time"] += float(time_obj["loss"].detach().cpu()) * bs
        logs["time_hour"] += float(time_obj["loss_hour"].detach().cpu()) * bs
        logs["time_day"] += float(time_obj["loss_day"].detach().cpu()) * bs
        logs["time_hour_acc"] += float(time_obj["hour_acc"].detach().cpu()) * bs
        logs["time_day_acc"] += float(time_obj["day_acc"].detach().cpu()) * bs

    return _mean_metric(logs, count)


def train_command(network_cfg: Dict[str, Any], train_cfg: Dict[str, Any]) -> str:
    if int(network_cfg["emb_dim"]) != 128:
        raise ValueError("Mobility embedding dimension must be fixed to 128")

    set_seed(int(train_cfg["seed"]))

    cities = _resolve_train_cities(train_cfg["cities"], train_cfg["data_root"])

    caches = build_joint_city_caches(
        data_root=train_cfg["data_root"],
        cities=cities,
        max_traj_per_city=int(train_cfg["max_traj_per_city"]),
        seed=int(train_cfg["seed"]),
        coverage_ratio=float(train_cfg["coverage_ratio"]),
        num_bins=int(train_cfg["num_path_bins"]),
        force_rebuild=bool(train_cfg["force_rebuild_cache"]),
    )
    city_offsets = build_city_token_offsets(caches)
    mask_center = bool(train_cfg.get("mask_center", True))
    num_road_tokens = sum(int(c.num_edges) for c in caches)
    num_tokens = 1 + num_road_tokens + (1 if mask_center else 0)
    mask_token_id = (num_tokens - 1) if mask_center else None

    train_sources = []
    val_sources = []
    city_split_meta = {}

    for cache in caches:
        split = split_trajectory_positions(
            num_selected=cache.num_selected,
            seed=stable_city_seed(int(train_cfg["seed"]), cache.city),
            train_ratio=float(train_cfg["train_ratio"]),
            val_ratio=float(train_cfg["val_ratio"]),
        )
        train_sources.append(
            make_occurrence_source(
                cache=cache,
                token_offset=int(city_offsets[cache.city]),
                traj_positions=split["train"],
            )
        )
        val_sources.append(
            make_occurrence_source(
                cache=cache,
                token_offset=int(city_offsets[cache.city]),
                traj_positions=split["val"],
            )
        )
        city_split_meta[cache.city] = {
            "num_selected": int(cache.num_selected),
            "num_train_traj": int(split["train"].shape[0]),
            "num_val_traj": int(split["val"].shape[0]),
            "num_reserved": int(cache.num_reserved),
            "num_occurrences": int(cache.num_occurrences),
            "cache_path": cache.cache_path,
        }

    train_ds = OccurrenceDataset(
        train_sources,
        pad_token_id=0,
        mask_center=mask_center,
        mask_token_id=mask_token_id,
    )
    val_ds = OccurrenceDataset(
        val_sources,
        pad_token_id=0,
        mask_center=mask_center,
        mask_token_id=mask_token_id,
    )

    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", False)) and str(train_cfg["device"]).startswith("cuda")
    persistent_workers = bool(train_cfg.get("persistent_workers", False)) and num_workers > 0
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        **loader_kwargs,
    )

    device = torch.device(train_cfg["device"])
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model = MobilityBranchNet(
        num_tokens=int(num_tokens),
        emb_dim=int(network_cfg["emb_dim"]),
        ctx_dim=int(network_cfg["ctx_dim"]),
        short_layers=int(network_cfg["short_layers"]),
        mid_layers=int(network_cfg["mid_layers"]),
        nhead=int(network_cfg["nhead"]),
        dropout=float(network_cfg["dropout"]),
        pad_token_id=0,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    neg_sampler = NegativeSampler(
        caches=caches,
        city_offsets=city_offsets,
        num_tokens=num_tokens,
        num_freq_buckets=int(train_cfg["num_freq_buckets"]),
        seed=int(train_cfg["seed"]),
    )

    train_tag = str(train_cfg.get("train_tag", "")).strip()
    if train_tag:
        out_dir = os.path.join(train_cfg["ckpt_root"], "mobility_debug", train_tag)
    else:
        out_dir = os.path.join(train_cfg["ckpt_root"], "mobility", train_cfg["src_id"])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best.pt")

    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(1, int(train_cfg["max_epoch"]) + 1):
        tr = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            neg_sampler=neg_sampler,
            train_cfg=train_cfg,
            max_steps=int(train_cfg.get("max_train_steps_per_epoch", 0)),
            scaler=scaler,
            use_amp=amp_enabled,
        )
        va = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            neg_sampler=neg_sampler,
            train_cfg=train_cfg,
            max_steps=int(train_cfg.get("max_val_steps_per_epoch", 0)),
            scaler=None,
            use_amp=amp_enabled,
        )

        history.append({"epoch": epoch, "train": tr, "val": va})

        improved = va["total"] < best_val
        if improved:
            best_val = va["total"]
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        payload = {
            "model_state": model.state_dict(),
            "network_config": dict(network_cfg),
            "train_config": dict(train_cfg),
            "src_id": str(train_cfg["src_id"]),
            "train_cities": list(cities),
            "city_offsets": {k: int(v) for k, v in city_offsets.items()},
            "city_num_edges": {c.city: int(c.num_edges) for c in caches},
            "city_id_map": {c.city: int(c.city_id) for c in caches},
            "num_tokens": int(num_tokens),
            "tokenizer_config": {
                "pad_token_id": 0,
                "mask_center": bool(mask_center),
                "mask_token_id": int(mask_token_id) if mask_token_id is not None else None,
            },
            "sampling_config": {
                "max_traj_per_city": int(train_cfg["max_traj_per_city"]),
                "coverage_ratio": float(train_cfg["coverage_ratio"]),
                "num_path_bins": int(train_cfg["num_path_bins"]),
                "seed": int(train_cfg["seed"]),
            },
            "split_meta": city_split_meta,
            "best_epoch": int(best_epoch),
            "best_val_total": float(best_val),
            "history": history,
        }

        if bool(train_cfg.get("save_every_epoch", False)):
            epoch_path = os.path.join(out_dir, f"epoch_{epoch:03d}.pt")
            torch.save(payload, epoch_path)
        if improved:
            torch.save(payload, ckpt_path)
            if bool(train_cfg.get("save_best_history", True)):
                best_hist_path = os.path.join(out_dir, f"best_{epoch:03d}.pt")
                torch.save(payload, best_hist_path)

        print(
            f"[Epoch {epoch:03d}] train_total={tr['total']:.4f} val_total={va['total']:.4f} "
            f"val_ctx={va['ctx']:.4f} val_cons={va['cons']:.4f} val_time={va['time']:.4f} "
            f"val_ctx_top1={va['ctx_top1']:.4f} val_hour_acc={va['time_hour_acc']:.4f}"
        )

        if stale >= int(train_cfg["patience"]):
            print(f"Early stop at epoch {epoch}, best_epoch={best_epoch}, best_val_total={best_val:.4f}")
            break

    print(f"Saved checkpoint: {ckpt_path}")
    return ckpt_path


def readout_command(readout_cfg: Dict[str, Any]) -> str:
    model_pt = readout_cfg["model_pt"]
    if not os.path.exists(model_pt):
        raise FileNotFoundError(model_pt)

    ckpt = torch.load(model_pt, map_location="cpu")
    network_cfg = dict(ckpt["network_config"])
    sampling_cfg = dict(ckpt["sampling_config"])

    trg_city = str(readout_cfg["trg_city"])
    city_offsets = {k: int(v) for k, v in ckpt["city_offsets"].items()}
    city_id_map = {k: int(v) for k, v in ckpt["city_id_map"].items()}
    city_num_edges = {k: int(v) for k, v in ckpt["city_num_edges"].items()}
    tokenizer_cfg = dict(ckpt.get("tokenizer_config", {}))

    if trg_city not in city_offsets:
        raise ValueError(f"Target city {trg_city} not in trained city set: {sorted(city_offsets.keys())}")

    city_id = int(city_id_map[trg_city])
    token_offset = int(city_offsets[trg_city])

    from models.mobility.data import get_or_build_city_mobility_cache  # local import to avoid cycles

    city_path = os.path.join(readout_cfg["data_root"], trg_city)
    cache = get_or_build_city_mobility_cache(
        city_path=city_path,
        city=trg_city,
        city_id=city_id,
        max_traj_per_city=int(sampling_cfg["max_traj_per_city"]),
        seed=int(sampling_cfg["seed"]),
        coverage_ratio=float(sampling_cfg["coverage_ratio"]),
        num_bins=int(sampling_cfg["num_path_bins"]),
        force_rebuild=False,
    )

    if int(cache.num_edges) != int(city_num_edges[trg_city]):
        raise RuntimeError(
            f"Target city edge count mismatch: cache={cache.num_edges} ckpt={city_num_edges[trg_city]}"
        )

    source = make_occurrence_source(
        cache=cache,
        token_offset=token_offset,
        traj_positions=np.arange(cache.num_selected, dtype=np.int64),
    )
    num_tokens = int(ckpt.get("num_tokens", 1 + sum(city_num_edges.values())))
    default_mask_id = num_tokens - 1 if num_tokens > (1 + sum(city_num_edges.values())) else None
    mask_center = bool(tokenizer_cfg.get("mask_center", False))
    mask_token_id_raw = tokenizer_cfg.get("mask_token_id", default_mask_id)
    mask_token_id = int(mask_token_id_raw) if mask_center and (mask_token_id_raw is not None) else None

    dataset = OccurrenceDataset(
        [source],
        pad_token_id=int(tokenizer_cfg.get("pad_token_id", 0)),
        mask_center=mask_center,
        mask_token_id=mask_token_id,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(readout_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    model = MobilityBranchNet(
        num_tokens=num_tokens,
        emb_dim=int(network_cfg["emb_dim"]),
        ctx_dim=int(network_cfg["ctx_dim"]),
        short_layers=int(network_cfg["short_layers"]),
        mid_layers=int(network_cfg["mid_layers"]),
        nhead=int(network_cfg["nhead"]),
        dropout=float(network_cfg["dropout"]),
        pad_token_id=0,
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(torch.device(readout_cfg["device"]))
    model.eval()

    num_edges = int(cache.num_edges)
    emb_dim = int(network_cfg["emb_dim"])

    sum_exp = np.zeros((num_edges,), dtype=np.float64)
    sum_vec = np.zeros((num_edges, emb_dim), dtype=np.float64)
    counts = np.zeros((num_edges,), dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            short_tokens = batch["short_tokens"].to(readout_cfg["device"])
            mid_tokens = batch["mid_tokens"].to(readout_cfg["device"])
            hour = batch["hour"].to(readout_cfg["device"])
            daytype = batch["daytype"].to(readout_cfg["device"])

            out = model(
                short_tokens=short_tokens,
                mid_tokens=mid_tokens,
                hour=hour,
                daytype=daytype.long(),
            )

            roads = batch["center_road"].detach().cpu().numpy().astype(np.int64)
            h_shared = out["h_shared"].detach().cpu().numpy().astype(np.float64)
            weights = np.exp(out["attn_logit"].detach().cpu().numpy().astype(np.float64))

            np.add.at(sum_exp, roads, weights)
            np.add.at(sum_vec, roads, h_shared * weights[:, None])
            np.add.at(counts, roads, 1)

    z_mob = np.zeros((num_edges, emb_dim), dtype=np.float32)
    covered = sum_exp > 0

    if bool(covered.any()):
        z_mob[covered] = (sum_vec[covered] / sum_exp[covered, None]).astype(np.float32)

    unk = model.unk_mob.detach().cpu().numpy().astype(np.float32)
    z_mob[~covered] = unk

    confidence = np.zeros((num_edges,), dtype=np.float32)
    if bool((counts > 0).any()):
        denom = float(np.percentile(counts[counts > 0], 95))
        denom = max(denom, 1.0)
        confidence[covered] = np.clip(
            np.log1p(counts[covered].astype(np.float64)) / np.log1p(denom),
            0.0,
            1.0,
        ).astype(np.float32)
    confidence[~covered] = 0.0

    src_id = readout_cfg["src_id"] or ckpt.get("src_id", "mobility_src")
    train_tag = str(readout_cfg.get("train_tag", "")).strip()
    if train_tag:
        out_dir = os.path.join(readout_cfg["emb_root"], "mobility_debug", train_tag)
    else:
        out_dir = os.path.join(readout_cfg["emb_root"], "mobility", f"{src_id}__to__{trg_city}")
    os.makedirs(out_dir, exist_ok=True)

    z_mob_path = os.path.join(out_dir, "z_mob.npy")
    z_path = os.path.join(out_dir, "z.npy")
    coverage_path = os.path.join(out_dir, "coverage_count.npy")
    conf_path = os.path.join(out_dir, "confidence.npy")

    np.save(z_mob_path, z_mob)
    np.save(z_path, z_mob)
    np.save(coverage_path, counts.astype(np.int64))
    np.save(conf_path, confidence)

    meta = {
        "model_pt": model_pt,
        "src_id": src_id,
        "trg_city": trg_city,
        "readout_split": str(readout_cfg["readout_split"]),
        "network_config": dict(network_cfg),
        "sampling_config": dict(sampling_cfg),
        "train_tag": train_tag,
        "embedding": "z_mob",
        "shape": list(z_mob.shape),
        "tokenizer_config": {
            "pad_token_id": int(tokenizer_cfg.get("pad_token_id", 0)),
            "mask_center": bool(mask_center),
            "mask_token_id": int(mask_token_id) if mask_token_id is not None else None,
        },
        "num_covered_roads": int(covered.sum()),
        "num_zero_coverage_roads": int((~covered).sum()),
        "cache_path": cache.cache_path,
    }

    meta_path = os.path.join(out_dir, "readout_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    print(f"Saved readout: {z_mob_path}")
    print(f"Saved compatibility copy: {z_path}")
    print(f"Saved coverage: {coverage_path}")
    print(f"Saved confidence: {conf_path}")
    print(f"Saved meta: {meta_path}")

    return z_mob_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        network_cfg = _pick_args(
            args,
            ["emb_dim", "ctx_dim", "short_layers", "mid_layers", "nhead", "dropout"],
        )
        train_cfg = _pick_args(
            args,
            [
                "cities",
                "src_id",
                "data_root",
                "ckpt_root",
                "max_traj_per_city",
                "coverage_ratio",
                "num_path_bins",
                "force_rebuild_cache",
                "mask_center",
                "max_epoch",
                "patience",
                "batch_size",
                "num_workers",
                "pin_memory",
                "persistent_workers",
                "prefetch_factor",
                "lr",
                "weight_decay",
                "grad_clip",
                "amp",
                "save_every_epoch",
                "save_best_history",
                "train_ratio",
                "val_ratio",
                "temperature",
                "num_rand_neg",
                "num_freq_neg",
                "num_freq_buckets",
                "include_inbatch_neg",
                "lambda_consistency",
                "lambda_time",
                "seed",
                "device",
                "train_tag",
                "max_train_steps_per_epoch",
                "max_val_steps_per_epoch",
            ],
        )
        train_command(network_cfg=network_cfg, train_cfg=train_cfg)
    else:
        readout_cfg = _pick_args(
            args,
            [
                "model_pt",
                "trg_city",
                "src_id",
                "data_root",
                "emb_root",
                "batch_size",
                "device",
                "readout_split",
                "train_tag",
            ],
        )
        readout_command(readout_cfg=readout_cfg)


if __name__ == "__main__":
    main()


