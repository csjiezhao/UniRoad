import argparse
import json
import os

import numpy as np

from tasks.road_speed_inf import eval_with_emb as eval_speed
from tasks.road_type_cls import eval_with_emb as eval_type
from tasks.traj_eval_utils import build_reserved_trajectory_eval_data
from tasks.traj_sim_srh import eval_with_traj_emb as eval_traj_sim
from tasks.traj_time_est import eval_with_traj_emb as eval_traj_time


def evaluate(
    city: str,
    emb_path: str,
    data_root: str,
    num_fold: int,
    cls_epochs: int,
    reg_epochs: int,
    device: str,
    seed: int,
    traj_task: str = "none",
    traj_sample_size: int = 50000,
    traj_seed: int = 42,
    traj_max_epoch: int = 20,
    traj_max_traj_per_city: int = 150000,
    traj_cache_seed: int = 42,
    traj_coverage_ratio: float = 0.30,
    traj_num_path_bins: int = 10,
):
    city_path = os.path.join(data_root, city)
    type_path = os.path.join(city_path, "road_types.npy")
    speed_path = os.path.join(city_path, "road_speeds.npy")

    micro, macro, cls_meta = eval_type(
        city=city,
        emb_path=emb_path,
        label_path=type_path,
        num_fold=num_fold,
        epochs=cls_epochs,
        device=device,
        seed=seed,
        return_meta=True,
    )

    mae, rmse, reg_meta = eval_speed(
        city=city,
        emb_path=emb_path,
        label_path=speed_path,
        num_fold=num_fold,
        epochs=reg_epochs,
        device=device,
        seed=seed + 997,
        return_meta=True,
    )

    result = {
        "city": city,
        "emb_path": emb_path,
        "fixed": {
            "type_micro_f1": float(micro),
            "type_macro_f1": float(macro),
            "speed_mae": float(mae),
            "speed_rmse": float(rmse),
            "classification": cls_meta,
            "regression": reg_meta,
        },
        "type_micro_f1": float(micro),
        "type_macro_f1": float(macro),
        "speed_mae": float(mae),
        "speed_rmse": float(rmse),
    }

    if traj_task != "none":
        edge_emb = np.load(emb_path).astype(np.float32)
        traj_data = build_reserved_trajectory_eval_data(
            city=city,
            data_root=data_root,
            sample_size=int(traj_sample_size),
            sample_seed=int(traj_seed),
            max_traj_per_city=int(traj_max_traj_per_city),
            cache_seed=int(traj_cache_seed),
            coverage_ratio=float(traj_coverage_ratio),
            num_path_bins=int(traj_num_path_bins),
        )
        traj_out = {
            "sample_info": traj_data["sample_info"],
        }
        if traj_task in ("tte", "both"):
            traj_out["tte"] = eval_traj_time(
                paths=traj_data["paths"],
                travel_time=traj_data["travel_time"],
                edge_emb=edge_emb,
                device=device,
                seed=int(traj_seed),
                max_epoch=int(traj_max_epoch),
            )
        if traj_task in ("sim", "both"):
            traj_out["sim"] = eval_traj_sim(
                paths=traj_data["paths"],
                edge_emb=edge_emb,
                device=device,
                seed=int(traj_seed),
                max_epoch=int(traj_max_epoch),
            )
        result["trajectory"] = traj_out

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate profile road embedding on downstream tasks")
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--emb-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--out-json", type=str, default="")

    parser.add_argument("--num-fold", type=int, default=5)
    parser.add_argument("--cls-epochs", type=int, default=200)
    parser.add_argument("--reg-epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--traj-task", type=str, default="none", choices=["none", "tte", "sim", "both"])
    parser.add_argument("--traj-sample-size", type=int, default=50000)
    parser.add_argument("--traj-seed", type=int, default=42)
    parser.add_argument("--traj-max-epoch", type=int, default=20)
    parser.add_argument("--traj-max-traj-per-city", type=int, default=150000)
    parser.add_argument("--traj-cache-seed", type=int, default=42)
    parser.add_argument("--traj-coverage-ratio", type=float, default=0.30)
    parser.add_argument("--traj-num-path-bins", type=int, default=10)
    args = parser.parse_args()

    result = evaluate(
        city=args.city,
        emb_path=args.emb_path,
        data_root=args.data_root,
        num_fold=args.num_fold,
        cls_epochs=args.cls_epochs,
        reg_epochs=args.reg_epochs,
        device=args.device,
        seed=args.seed,
        traj_task=args.traj_task,
        traj_sample_size=args.traj_sample_size,
        traj_seed=args.traj_seed,
        traj_max_epoch=args.traj_max_epoch,
        traj_max_traj_per_city=args.traj_max_traj_per_city,
        traj_cache_seed=args.traj_cache_seed,
        traj_coverage_ratio=args.traj_coverage_ratio,
        traj_num_path_bins=args.traj_num_path_bins,
    )

    out_json = args.out_json.strip()
    if not out_json:
        out_json = os.path.join(os.path.dirname(args.emb_path), f"metrics_{args.city}.json")

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True, indent=2)

    print(json.dumps(result, ensure_ascii=True, indent=2))
    print(f"Saved metrics: {out_json}")


if __name__ == "__main__":
    main()


