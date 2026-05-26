import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .llm_api import LLMCaller


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_profile_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def get_embedding_safe(edge_idx: int, text: str, llm: LLMCaller) -> Tuple[int, List[float]]:
    if not isinstance(text, str) or text.strip() == "":
        return edge_idx, []

    for attempt in range(1, 4):
        try:
            return edge_idx, llm.get_embedding(text)
        except Exception as exc:
            if attempt == 3:
                print(f"[Embedding] failed edge={edge_idx}: {exc}")
                return edge_idx, []
            time.sleep(attempt)
    return edge_idx, []


def process_embeddings_parallel(rows: List[Dict[str, object]], llm: LLMCaller, max_workers: int) -> Dict[int, List[float]]:
    results_map: Dict[int, List[float]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for row in rows:
            edge_idx = int(row["edge_idx"])
            profile = row.get("profile", row.get("road_profile", ""))
            futures[executor.submit(get_embedding_safe, edge_idx, str(profile), llm)] = edge_idx

        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding profiles"):
            edge_idx, vector = future.result()
            if vector:
                results_map[int(edge_idx)] = vector
    return results_map


def save_embedding_outputs(city: str, vectors: Dict[int, List[float]], out_json: str, out_npy: str) -> None:
    out_dir = _project_root() / "embs" / "profile" / city
    out_dir.mkdir(parents=True, exist_ok=True)

    if not vectors:
        raise RuntimeError("No valid embeddings generated.")

    dim = len(next(iter(vectors.values())))
    max_edge_idx = max(int(k) for k in vectors.keys())
    z = np.zeros((max_edge_idx + 1, dim), dtype=np.float32)
    mask = np.zeros((max_edge_idx + 1,), dtype=np.bool_)

    rows = []
    for edge_idx, vec in sorted(vectors.items(), key=lambda x: int(x[0])):
        z[int(edge_idx)] = np.asarray(vec, dtype=np.float32)
        mask[int(edge_idx)] = True
        rows.append({"edge_idx": int(edge_idx), "embedding": vec})

    with (out_dir / out_json).open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    np.save(out_dir / out_npy, z)
    np.save(out_dir / "valid_mask.npy", mask.astype(np.uint8))

    # Compatibility filename for downstream scripts.
    np.save(out_dir / "z.npy", z)
    print(f"Saved profile embeddings: {(out_dir / out_npy)} shape={z.shape}")


def main(
    city: str = "chengdu",
    platform: str = "OpenRouter",
    model: str = "emb-3s",
    max_workers: int = 64,
    input_filename: str = "road_profiles_generated.jsonl",
    output_json: str = "road_embeddings_all.json",
    output_npy: str = "road_embeddings_all.npy",
) -> None:
    in_path = _project_root() / "data" / city / input_filename
    if not in_path.exists():
        raise FileNotFoundError(f"Profile file not found: {in_path}")

    rows = _load_profile_jsonl(in_path)
    if not rows:
        raise RuntimeError("Empty profile input.")

    llm = LLMCaller(platform, model)
    vectors = process_embeddings_parallel(rows=rows, llm=llm, max_workers=max_workers)
    save_embedding_outputs(city=city, vectors=vectors, out_json=output_json, out_npy=output_npy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed LLM-generated road profiles.")
    parser.add_argument("--city", type=str, default="chengdu")
    parser.add_argument("--platform", type=str, default="OpenRouter")
    parser.add_argument("--model", type=str, default="emb-3s")
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--input", type=str, default="road_profiles_generated.jsonl", help="Input jsonl under data/<city>/")
    parser.add_argument("--out-json", type=str, default="road_embeddings_all.json")
    parser.add_argument("--out-npy", type=str, default="road_embeddings_all.npy")
    args = parser.parse_args()

    start = time.time()
    main(
        city=args.city,
        platform=args.platform,
        model=args.model,
        max_workers=args.max_workers,
        input_filename=args.input,
        output_json=args.out_json,
        output_npy=args.out_npy,
    )
    print(f"Elapsed: {time.time() - start:.2f}s")
