import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import pandas as pd
from json_repair import repair_json
from tqdm import tqdm

from .llm_api import LLMCaller


ROAD_PROFILE_PROMPT = """
You are an urban data expert. Your task is to transform structured road attributes into a high-density semantic feature description, which will be used to generate high-quality text embeddings.

### Road Attribute Data
- Road Name: {name}
- Land Use: {landuse}
- Number of Lanes: {lanes}
- Max Speed: {maxspeed}
- One-way: {oneway}
- Length: {length} meters
- Structural Features: Bridge={bridge}, Tunnel={tunnel}
- Nearby POIs: {poi_features}

### Critical Instructions
1. Denoising Principle (Most Important): If any data point (e.g., Number of Lanes, Max Speed, Land Use) is empty, 'nan', 'None', or '0', absolutely do not mention "unknown", "unclear", "missing", or "unable to evaluate" in the output. Simply skip that dimension and pretend it does not exist.
2. Inferential Imputation:
   - If Landuse is missing but POI contains shops/restaurants, describe it as having a "commercial atmosphere"; if POI is schools/libraries, describe it as a "cultural and educational area".
   - If Highway is residential and there is no lane information, infer it as a "community street suitable for low-speed traffic".
3. High-Density Description:
   - Remove meaningless filler words such as "This road segment is" or "According to the data".
   - Focus on describing: Functional Hierarchy (arterial/branch road), Physical Form (one-way/multi-lane/bridge), and Environmental Characteristics (specific commercial/residential/educational atmosphere).
4. Output Style: Objective, compact, similar to an encyclopedia summary. Keep the word count between 50-80 words.

### Please output strictly according to the following JSON format, without any extra content:
{
    "road_profile": "Generated road profile text content"
}
""".strip()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_cell(value):
    if pd.isna(value) or value == "":
        return "None"
    return value


def get_profile_from_llm(row_data: Dict[str, object], llm: LLMCaller) -> str:
    fmt_data = {k: _safe_cell(v) for k, v in row_data.items()}
    prompt_content = ROAD_PROFILE_PROMPT.format(**fmt_data)
    messages = [{"role": "user", "content": prompt_content}]

    for attempt in range(1, 4):
        try:
            response_text = llm.get_response(messages)
            parsed_json = json.loads(repair_json(response_text))
            profile_text = str(parsed_json.get("road_profile", "")).strip()
            if not profile_text:
                raise ValueError("Empty profile returned")
            return profile_text
        except Exception as exc:
            if attempt == 3:
                print(f"[Profile] failed on edge={row_data.get('edge_idx', 'NA')}: {exc}")
                return ""
            time.sleep(2**attempt)
    return ""


def process_profiles_parallel(df: pd.DataFrame, llm: LLMCaller, max_workers: int) -> Dict[int, str]:
    results: Dict[int, str] = {}
    required_cols = ["edge_idx", "name", "landuse", "poi_features", "lanes", "maxspeed", "oneway", "length", "bridge", "tunnel"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_profile_from_llm, row[required_cols].to_dict(), llm): int(row["edge_idx"])
            for _, row in df.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating profiles"):
            eid = futures[future]
            try:
                results[eid] = future.result()
            except Exception:
                results[eid] = ""
    return results


def save_profiles_jsonl(city: str, profiles: Dict[int, str], output_filename: str) -> Path:
    out_dir = _project_root() / "data" / city
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_filename

    rows: List[Dict[str, object]] = []
    for edge_idx, text in sorted(profiles.items(), key=lambda x: int(x[0])):
        rows.append({"edge_idx": int(edge_idx), "profile": text})

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved profile jsonl: {out_path}")
    return out_path


def main(
    city: str = "chengdu",
    platform: str = "OpenRouter",
    model: str = "gpt-4o-mini",
    max_workers: int = 32,
    input_filename: str = "G_edges.csv",
    output_filename: str = "road_profiles_generated.jsonl",
    limit_rows: int = 0,
) -> None:
    input_path = _project_root() / "data" / city / input_filename
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "edge_idx" not in df.columns:
        df["edge_idx"] = df.index.astype(int)
    if limit_rows > 0:
        df = df.head(limit_rows)

    llm = LLMCaller(platform, model)
    profiles = process_profiles_parallel(df=df, llm=llm, max_workers=max_workers)
    save_profiles_jsonl(city=city, profiles=profiles, output_filename=output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM road profiles (jsonl).")
    parser.add_argument("--city", type=str, default="chengdu")
    parser.add_argument("--platform", type=str, default="OpenRouter")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--input", type=str, default="G_edges.csv", help="Input file under data/<city>/")
    parser.add_argument("--out", type=str, default="road_profiles_generated.jsonl", help="Output jsonl under data/<city>/")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for debugging.")
    args = parser.parse_args()

    start = time.time()
    main(
        city=args.city,
        platform=args.platform,
        model=args.model,
        max_workers=args.max_workers,
        input_filename=args.input,
        output_filename=args.out,
        limit_rows=args.limit,
    )
    print(f"Elapsed: {time.time() - start:.2f}s")
