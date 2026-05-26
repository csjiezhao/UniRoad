import argparse
import json
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset
from torch_geometric.loader import DataLoader

from models.structure.data import EdgeCenteredDataset, get_or_build_cache, split_indices
from models.structure.net import (
    EdgePrototypeNet,
    compute_consistency_loss,
    compute_mask_loss,
    compute_pretrain_core_losses,
)
from utils.graph_io import list_available_cities, parse_city_list, set_seed, stable_city_seed


FINAL_ENCODER_TYPE = "gatv2_edge"
FINAL_EDGE_CONTEXT_MODE = "bi"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UniRoad structure branch train/readout")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train encoder and save checkpoint")
    p_train.add_argument("--scope", type=str, default="joint", choices=["single", "joint"])
    p_train.add_argument("--src-city", type=str, default="chengdu")
    p_train.add_argument("--joint-cities", type=str, default="chengdu,porto,rome,sanfran")
    p_train.add_argument("--joint-id", type=str, default="joint4")

    p_train.add_argument("--data-root", type=str, default="data")
    p_train.add_argument("--ckpt-root", type=str, default="ckpts")

    p_train.add_argument("--k-hop", type=int, default=3)
    p_train.add_argument("--hidden-dim", type=int, default=128)
    p_train.add_argument("--role-dim", type=int, default=64)
    p_train.add_argument("--role-count", type=int, default=16)
    p_train.add_argument("--dropout", type=float, default=0.1)

    p_train.add_argument("--max-epoch", type=int, default=100)
    p_train.add_argument("--patience", type=int, default=12)
    p_train.add_argument("--batch-size", type=int, default=16384)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight-decay", type=float, default=1e-5)
    p_train.add_argument("--grad-clip", type=float, default=2.0)

    p_train.add_argument("--train-ratio", type=float, default=0.8)
    p_train.add_argument("--val-ratio", type=float, default=0.1)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--device", type=str, default="cpu")
    p_train.add_argument("--build-cache", action="store_true")

    p_train.add_argument("--mask-ratio-start", type=float, default=0.05)
    p_train.add_argument("--mask-ratio-end", type=float, default=0.35)

    p_train.add_argument("--w-struct", type=float, default=1.0)
    p_train.add_argument("--w-mask", type=float, default=0.8)
    p_train.add_argument("--w-consistency", type=float, default=0.0)
    p_train.add_argument("--w-role-ent", type=float, default=0.02)
    p_train.add_argument("--w-role-orth", type=float, default=0.02)
    p_train.add_argument("--w-compact", type=float, default=0.1)

    p_train.add_argument("--w-cont", type=float, default=1.0)
    p_train.add_argument("--w-hist", type=float, default=1.0)
    p_train.add_argument("--w-deadend", type=float, default=1.0)

    p_train.add_argument("--consistency-type", type=str, default="none", choices=["none", "byol_like", "simsiam_like"])
    p_train.add_argument("--consistency-aug", type=str, default="mask", choices=["mask", "jitter", "dropedge"])
    p_train.add_argument("--consistency-aug-ratio", type=float, default=0.15)

    p_train.add_argument("--save-epoch-ckpt", action="store_true")
    p_train.add_argument("--train-tag", type=str, default="")
    p_train.add_argument("--train-tag-root", type=str, default="structure")

    p_train.add_argument(
        "--ckpt-select-metric",
        type=str,
        default="custom",
        choices=["val_total", "val_struct", "val_mask", "custom"],
    )
    p_train.add_argument("--ckpt-select-alpha", type=float, default=0.5)

    p_readout = sub.add_parser("readout", help="Load checkpoint and export embedding")
    p_readout.add_argument("--model-pt", type=str, required=True)
    p_readout.add_argument("--trg-city", type=str, required=True)
    p_readout.add_argument("--src-id", type=str, default="")
    p_readout.add_argument("--data-root", type=str, default="data")
    p_readout.add_argument("--emb-root", type=str, default="embs")
    p_readout.add_argument("--batch-size", type=int, default=512)
    p_readout.add_argument("--device", type=str, default="cpu")
    p_readout.add_argument("--train-tag", type=str, default="")
    p_readout.add_argument("--train-tag-root", type=str, default="structure_final")

    return parser


def _pick_args(args: argparse.Namespace, keys: Sequence[str]) -> Dict[str, Any]:
    return {key: getattr(args, key) for key in keys}


def _normalize_network_cfg(network_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(network_cfg)
    cfg["encoder_type"] = FINAL_ENCODER_TYPE
    cfg["edge_context_mode"] = FINAL_EDGE_CONTEXT_MODE
    return cfg


def _normalize_train_cfg(train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(train_cfg)
    cfg["w_struct"] = float(cfg.get("w_struct", 1.0))
    cfg["w_mask"] = float(cfg.get("w_mask", 0.8))
    cfg["w_consistency"] = float(cfg.get("w_consistency", 0.0))
    cfg["w_role_ent"] = float(cfg.get("w_role_ent", 0.02))
    cfg["w_role_orth"] = float(cfg.get("w_role_orth", 0.02))
    cfg["w_compact"] = float(cfg.get("w_compact", 0.1))

    cfg["w_cont"] = float(cfg.get("w_cont", 1.0))
    cfg["w_hist"] = float(cfg.get("w_hist", 1.0))
    cfg["w_deadend"] = float(cfg.get("w_deadend", 1.0))

    cfg["consistency_type"] = str(cfg.get("consistency_type", "none"))
    cfg["consistency_aug"] = str(cfg.get("consistency_aug", "mask"))
    cfg["consistency_aug_ratio"] = float(cfg.get("consistency_aug_ratio", 0.15))
    cfg["train_tag_root"] = str(cfg.get("train_tag_root", "structure_final"))
    cfg["ckpt_select_metric"] = str(cfg.get("ckpt_select_metric", "custom"))
    cfg["ckpt_select_alpha"] = float(cfg.get("ckpt_select_alpha", 0.5))
    return cfg


def _resolve_train_cities(train_cfg: Dict[str, Any]) -> List[str]:
    if train_cfg["scope"] == "single":
        return [train_cfg["src_city"]]
    cities = parse_city_list(train_cfg["joint_cities"])
    if not cities:
        cities = list_available_cities(train_cfg["data_root"])
    if not cities:
        raise ValueError("No cities found for joint training")
    return sorted(set(cities))


def _resolve_src_id(train_cfg: Dict[str, Any], train_cities: Sequence[str]) -> str:
    if train_cfg["scope"] == "single":
        return train_cfg["src_city"]
    if train_cfg["joint_id"]:
        return train_cfg["joint_id"]
    return "joint_" + "-".join(train_cities)


def _build_city_datasets(network_cfg: Dict[str, Any], train_cfg: Dict[str, Any], train_cities: Sequence[str]):
    city_cache: Dict[str, object] = {}
    city_dataset: Dict[str, Dataset] = {}
    city_splits: Dict[str, Dict[str, np.ndarray]] = {}

    for city in train_cities:
        city_path = os.path.join(train_cfg["data_root"], city)
        cache = get_or_build_cache(
            city_path=city_path,
            city=city,
            k_hop=network_cfg["k_hop"],
            force_rebuild=train_cfg["build_cache"],
        )
        dataset = EdgeCenteredDataset(cache.samples)
        splits = split_indices(
            num_samples=len(dataset),
            seed=stable_city_seed(train_cfg["seed"], city),
            train_ratio=train_cfg["train_ratio"],
            val_ratio=train_cfg["val_ratio"],
        )
        city_cache[city] = cache
        city_dataset[city] = dataset
        city_splits[city] = splits
    return city_cache, city_dataset, city_splits


def _build_train_val_sets(
    train_cfg: Dict[str, Any],
    train_cities: Sequence[str],
    city_dataset: Dict[str, Dataset],
    city_splits: Dict[str, Dict[str, np.ndarray]],
):
    if train_cfg["scope"] == "single":
        city = train_cfg["src_city"]
        train_ds = Subset(city_dataset[city], city_splits[city]["train"].tolist())
        val_ds = Subset(city_dataset[city], city_splits[city]["val"].tolist())
    else:
        train_parts = []
        val_parts = []
        for city in train_cities:
            train_parts.append(Subset(city_dataset[city], city_splits[city]["train"].tolist()))
            val_parts.append(Subset(city_dataset[city], city_splits[city]["val"].tolist()))
        train_ds = ConcatDataset(train_parts)
        val_ds = ConcatDataset(val_parts)
    return train_ds, val_ds


def _curriculum_mask_ratio(epoch: int, max_epoch: int, start: float, end: float) -> float:
    if max_epoch <= 1:
        return float(end)
    p = float(epoch - 1) / float(max_epoch - 1)
    p = min(max(p, 0.0), 1.0)
    return float(start + (end - start) * p)


def _build_masked_view(batch, node_raw_dim: int, edge_feat_dim: int, mask_ratio: float):
    masked = batch.clone()
    target_center_edge_feat = batch.center_edge_feat.clone()
    u_idx = batch.center_edge_index[0]
    v_idx = batch.center_edge_index[1]
    target_center_node_raw = torch.cat([batch.x[u_idx, :node_raw_dim], batch.x[v_idx, :node_raw_dim]], dim=-1)

    if mask_ratio > 0:
        edge_mask = torch.rand_like(masked.center_edge_feat) < mask_ratio
        masked.center_edge_feat = masked.center_edge_feat.masked_fill(edge_mask, 0.0)

        x = masked.x.clone()
        node_mask = torch.rand_like(x[:, :node_raw_dim]) < mask_ratio
        x[:, :node_raw_dim] = x[:, :node_raw_dim].masked_fill(node_mask, 0.0)
        masked.x = x

        edge_attr = masked.edge_attr.clone()
        if edge_attr.shape[0] > 0:
            attr_mask = torch.rand_like(edge_attr[:, :edge_feat_dim]) < mask_ratio
            edge_attr[:, :edge_feat_dim] = edge_attr[:, :edge_feat_dim].masked_fill(attr_mask, 0.0)
            masked.edge_attr = edge_attr

    return masked, target_center_edge_feat, target_center_node_raw


def _apply_consistency_aug(batch, node_raw_dim: int, edge_feat_dim: int, aug_type: str, aug_ratio: float):
    out = batch.clone()
    if aug_ratio <= 0 or aug_type == "none":
        return out

    if aug_type == "mask":
        edge_mask = torch.rand_like(out.center_edge_feat) < aug_ratio
        out.center_edge_feat = out.center_edge_feat.masked_fill(edge_mask, 0.0)

        x = out.x.clone()
        node_mask = torch.rand_like(x[:, :node_raw_dim]) < aug_ratio
        x[:, :node_raw_dim] = x[:, :node_raw_dim].masked_fill(node_mask, 0.0)
        out.x = x

        edge_attr = out.edge_attr.clone()
        if edge_attr.shape[0] > 0:
            attr_mask = torch.rand_like(edge_attr[:, :edge_feat_dim]) < aug_ratio
            edge_attr[:, :edge_feat_dim] = edge_attr[:, :edge_feat_dim].masked_fill(attr_mask, 0.0)
            out.edge_attr = edge_attr

    elif aug_type == "jitter":
        sigma = max(1e-3, 0.05 * aug_ratio)
        out.center_edge_feat = out.center_edge_feat + sigma * torch.randn_like(out.center_edge_feat)

        x = out.x.clone()
        x[:, :node_raw_dim] = x[:, :node_raw_dim] + sigma * torch.randn_like(x[:, :node_raw_dim])
        out.x = x

        edge_attr = out.edge_attr.clone()
        if edge_attr.shape[0] > 0:
            edge_attr[:, :edge_feat_dim] = edge_attr[:, :edge_feat_dim] + sigma * torch.randn_like(edge_attr[:, :edge_feat_dim])
            out.edge_attr = edge_attr

    elif aug_type == "dropedge":
        num_e = int(out.edge_index.shape[1])
        if num_e > 1:
            keep = torch.rand(num_e, device=out.edge_index.device) > aug_ratio
            if int(keep.sum().item()) == 0:
                keep[0] = True
            out.edge_index = out.edge_index[:, keep]
            out.edge_attr = out.edge_attr[keep]
    else:
        raise ValueError(f"Unsupported consistency augmentation: {aug_type}")

    return out


def _mean_metric(logs: Dict[str, float], n: int) -> Dict[str, float]:
    if n <= 0:
        return {k: 0.0 for k in logs}
    return {k: v / n for k, v in logs.items()}


def _checkpoint_selection_score(val_metrics: Dict[str, float], metric: str, alpha: float) -> float:
    if metric == "val_total":
        return float(val_metrics["total"])
    if metric == "val_struct":
        return float(val_metrics["struct"])
    if metric == "val_mask":
        return float(val_metrics["mask"])
    if metric == "custom":
        # Favor structural reconstruction while keeping mask recoverability.
        return float(val_metrics["struct"] + alpha * val_metrics["mask"])
    raise ValueError(f"Unsupported ckpt selection metric: {metric}")


def _build_model_from_meta(meta: Dict[str, Any], network_cfg: Dict[str, Any], device: torch.device):
    model = EdgePrototypeNet(
        node_feat_dim=int(meta["node_feat_dim"]),
        node_raw_dim=int(meta["node_raw_dim"]),
        edge_feat_dim=int(meta["edge_feat_dim"]),
        edge_attr_dim=int(meta["edge_attr_dim"]),
        hidden_dim=int(network_cfg["hidden_dim"]),
        role_dim=int(network_cfg["role_dim"]),
        role_count=int(network_cfg["role_count"]),
        dropout=float(network_cfg["dropout"]),
        edge_context_mode=str(network_cfg["edge_context_mode"]),
    )
    return model.to(device)


def _run_epoch(model, loader, device, train_cfg: Dict[str, Any], epoch: int, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    mask_ratio = _curriculum_mask_ratio(
        epoch=epoch,
        max_epoch=train_cfg["max_epoch"],
        start=train_cfg["mask_ratio_start"],
        end=train_cfg["mask_ratio_end"],
    )

    logs = {
        "total": 0.0,
        "struct": 0.0,
        "cont": 0.0,
        "hist": 0.0,
        "deadend": 0.0,
        "mask": 0.0,
        "consistency": 0.0,
        "usage_entropy": 0.0,
        "role_orth": 0.0,
        "compact": 0.0,
        "role_reg": 0.0,
    }
    count = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)

        core = compute_pretrain_core_losses(
            outputs=out,
            batch=batch,
            w_role_ent=train_cfg["w_role_ent"],
            w_role_orth=train_cfg["w_role_orth"],
        )
        struct_weighted = (
            train_cfg["w_cont"] * core["cont"] + train_cfg["w_hist"] * core["hist"] + train_cfg["w_deadend"] * core["deadend"]
        )

        masked_batch, tgt_edge, tgt_nodes = _build_masked_view(
            batch=batch,
            node_raw_dim=int(batch.x.shape[1] - 3),
            edge_feat_dim=int(batch.center_edge_feat.shape[1]),
            mask_ratio=mask_ratio,
        )
        out_masked = model(masked_batch)
        loss_mask = compute_mask_loss(out_masked, tgt_edge, tgt_nodes)

        loss_consistency = out["z"].new_tensor(0.0)
        if train_cfg["w_consistency"] > 0 and train_cfg["consistency_type"] != "none":
            v1 = _apply_consistency_aug(
                batch=batch,
                node_raw_dim=int(batch.x.shape[1] - 3),
                edge_feat_dim=int(batch.center_edge_feat.shape[1]),
                aug_type=train_cfg["consistency_aug"],
                aug_ratio=train_cfg["consistency_aug_ratio"],
            )
            v2 = _apply_consistency_aug(
                batch=batch,
                node_raw_dim=int(batch.x.shape[1] - 3),
                edge_feat_dim=int(batch.center_edge_feat.shape[1]),
                aug_type=train_cfg["consistency_aug"],
                aug_ratio=train_cfg["consistency_aug_ratio"],
            )
            out_v1 = model(v1)
            out_v2 = model(v2)
            loss_consistency = compute_consistency_loss(
                outputs_view1=out_v1,
                outputs_view2=out_v2,
                consistency_type=train_cfg["consistency_type"],
            )

        total = (
            train_cfg["w_struct"] * struct_weighted
            + train_cfg["w_mask"] * loss_mask
            + train_cfg["w_consistency"] * loss_consistency
            + core["role_reg"]
            + train_cfg["w_compact"] * core["compact"]
        )

        if train_mode:
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg["grad_clip"])
            optimizer.step()
            model.update_prototype_bank(out["h_role"].detach(), out["pi"].detach())

        bs = int(batch.num_graphs)
        count += bs
        logs["total"] += float(total.detach().cpu()) * bs
        logs["struct"] += float(struct_weighted.detach().cpu()) * bs
        logs["cont"] += float(core["cont"].detach().cpu()) * bs
        logs["hist"] += float(core["hist"].detach().cpu()) * bs
        logs["deadend"] += float(core["deadend"].detach().cpu()) * bs
        logs["mask"] += float(loss_mask.detach().cpu()) * bs
        logs["consistency"] += float(loss_consistency.detach().cpu()) * bs
        logs["usage_entropy"] += float(core["usage_entropy"].detach().cpu()) * bs
        logs["role_orth"] += float(core["role_orth"].detach().cpu()) * bs
        logs["compact"] += float(core["compact"].detach().cpu()) * bs
        logs["role_reg"] += float(core["role_reg"].detach().cpu()) * bs

    return _mean_metric(logs, count)


def train_command(network_cfg: Dict[str, Any], train_cfg: Dict[str, Any]) -> str:
    network_cfg = _normalize_network_cfg(network_cfg)
    if int(network_cfg["role_dim"]) * 2 != 128:
        raise ValueError(f"Expected z dimension 128, got role_dim={network_cfg['role_dim']} -> z={2 * int(network_cfg['role_dim'])}")
    train_cfg = _normalize_train_cfg(train_cfg)

    set_seed(train_cfg["seed"])
    train_cities = _resolve_train_cities(train_cfg)
    src_id = _resolve_src_id(train_cfg, train_cities)

    city_cache, city_dataset, city_splits = _build_city_datasets(network_cfg, train_cfg, train_cities)
    train_ds, val_ds = _build_train_val_sets(train_cfg, train_cities, city_dataset, city_splits)

    ref_city = train_cfg["src_city"] if train_cfg["src_city"] in city_cache else train_cities[0]
    ref_meta = city_cache[ref_city].meta

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False)

    device = torch.device(train_cfg["device"])
    model = _build_model_from_meta(ref_meta, network_cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    train_tag = str(train_cfg.get("train_tag", "")).strip()
    if train_tag:
        out_dir = os.path.join(train_cfg["ckpt_root"], train_cfg["train_tag_root"], train_tag)
    else:
        out_dir = os.path.join(train_cfg["ckpt_root"], "structure", src_id)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best.pt")

    best_score = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    select_metric = str(train_cfg.get("ckpt_select_metric", "custom"))
    select_alpha = float(train_cfg.get("ckpt_select_alpha", 0.5))

    print("[Train] strict unsupervised pretraining: road_type/speed labels are NOT used in encoder losses.")
    print(f"[Train] frozen backbone: encoder={FINAL_ENCODER_TYPE}, edge_context={FINAL_EDGE_CONTEXT_MODE}")

    for epoch in range(1, int(train_cfg["max_epoch"]) + 1):
        tr = _run_epoch(model, train_loader, device, train_cfg, epoch=epoch, optimizer=optimizer)
        va = _run_epoch(model, val_loader, device, train_cfg, epoch=epoch, optimizer=None)
        history.append({"epoch": epoch, "train": tr, "val": va})

        current_score = _checkpoint_selection_score(
            va,
            metric=select_metric,
            alpha=select_alpha,
        )

        if bool(train_cfg.get("save_epoch_ckpt", False)):
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "network_config": dict(network_cfg),
                    "train_config": dict(train_cfg),
                    "src_id": src_id,
                    "train_cities": list(train_cities),
                    "ref_meta": ref_meta,
                    "epoch": epoch,
                    "train_metrics": tr,
                    "val_metrics": va,
                    "history": history,
                },
                os.path.join(out_dir, f"epoch_{epoch:03d}.pt"),
            )

        if current_score < best_score:
            best_score = current_score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "network_config": dict(network_cfg),
                    "train_config": dict(train_cfg),
                    "src_id": src_id,
                    "train_cities": list(train_cities),
                    "ref_meta": ref_meta,
                    "best_epoch": best_epoch,
                    "best_val_total": float(va["total"]),
                    "best_select_metric": select_metric,
                    "best_select_alpha": float(select_alpha),
                    "best_select_score": float(best_score),
                    "history": history,
                },
                ckpt_path,
            )
        else:
            stale += 1

        print(
            f"[Epoch {epoch:03d}] train_total={tr['total']:.4f} val_total={va['total']:.4f} "
            f"val_struct={va['struct']:.4f} val_mask={va['mask']:.4f} "
            f"val_cons={va['consistency']:.4f} val_compact={va['compact']:.4f} "
            f"sel({select_metric})={current_score:.4f}"
        )

        if stale >= int(train_cfg["patience"]):
            print(f"Early stop at epoch {epoch}, best_epoch={best_epoch}, best_score={best_score:.4f}")
            break

    print(f"Saved checkpoint: {ckpt_path}")
    return ckpt_path


def readout_command(readout_cfg: Dict[str, Any]) -> str:
    if not os.path.exists(readout_cfg["model_pt"]):
        raise FileNotFoundError(readout_cfg["model_pt"])

    ckpt = torch.load(readout_cfg["model_pt"], map_location="cpu")
    network_cfg = ckpt.get("network_config")
    if not network_cfg:
        raise ValueError("Checkpoint missing 'network_config'")
    network_cfg = _normalize_network_cfg(network_cfg)

    if int(network_cfg["role_dim"]) * 2 != 128:
        raise ValueError(f"Expected z dimension 128, got role_dim={network_cfg['role_dim']} -> z={2 * int(network_cfg['role_dim'])}")

    src_id = readout_cfg["src_id"] or ckpt.get("src_id", "unknown_src")
    city_path = os.path.join(readout_cfg["data_root"], readout_cfg["trg_city"])
    cache = get_or_build_cache(
        city_path=city_path,
        city=readout_cfg["trg_city"],
        k_hop=int(network_cfg["k_hop"]),
        force_rebuild=False,
    )
    dataset = EdgeCenteredDataset(cache.samples)
    loader = DataLoader(dataset, batch_size=int(readout_cfg["batch_size"]), shuffle=False)

    model = _build_model_from_meta(cache.meta, network_cfg, torch.device(readout_cfg["device"]))
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print(f"[Readout] non-strict ckpt load: missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()

    num_edges = int(cache.meta["num_edges"])
    z = np.zeros((num_edges, int(network_cfg["role_dim"]) * 2), dtype=np.float32)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(readout_cfg["device"])
            out = model(batch)
            edge_idx = batch.edge_idx.view(-1).detach().cpu().numpy().astype(np.int64)
            z[edge_idx] = out["z"].detach().cpu().numpy()

    train_tag = str(readout_cfg.get("train_tag", "")).strip()
    train_tag_root = str(readout_cfg.get("train_tag_root", "structure_final"))
    if train_tag:
        out_dir = os.path.join(readout_cfg["emb_root"], train_tag_root, train_tag)
    else:
        out_dir = os.path.join(readout_cfg["emb_root"], "structure", f"{src_id}__to__{readout_cfg['trg_city']}")
    os.makedirs(out_dir, exist_ok=True)

    z_path = os.path.join(out_dir, "z.npy")
    np.save(z_path, z)

    meta = {
        "model_pt": readout_cfg["model_pt"],
        "src_id": src_id,
        "trg_city": readout_cfg["trg_city"],
        "network_config": dict(network_cfg),
        "train_tag": train_tag,
        "train_tag_root": train_tag_root,
        "embedding": "z",
        "shape": list(z.shape),
    }
    with open(os.path.join(out_dir, "readout_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    print(f"Saved readout: {z_path}")
    return z_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        network_cfg = _pick_args(
            args,
            ["k_hop", "hidden_dim", "role_dim", "role_count", "dropout"],
        )
        train_cfg = _pick_args(
            args,
            [
                "scope",
                "src_city",
                "joint_cities",
                "joint_id",
                "data_root",
                "ckpt_root",
                "max_epoch",
                "patience",
                "batch_size",
                "lr",
                "weight_decay",
                "grad_clip",
                "train_ratio",
                "val_ratio",
                "seed",
                "device",
                "build_cache",
                "mask_ratio_start",
                "mask_ratio_end",
                "w_struct",
                "w_mask",
                "w_consistency",
                "w_role_ent",
                "w_role_orth",
                "w_compact",
                "w_cont",
                "w_hist",
                "w_deadend",
                "consistency_type",
                "consistency_aug",
                "consistency_aug_ratio",
                "save_epoch_ckpt",
                "train_tag",
                "train_tag_root",
                "ckpt_select_metric",
                "ckpt_select_alpha",
            ],
        )
        train_command(network_cfg, train_cfg)
        return

    readout_cfg = _pick_args(
        args,
        ["model_pt", "trg_city", "src_id", "data_root", "emb_root", "batch_size", "device", "train_tag", "train_tag_root"],
    )
    readout_command(readout_cfg)


if __name__ == "__main__":
    main()


