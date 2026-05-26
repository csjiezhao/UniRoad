from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render transparent road basemaps from G_edges.csv files.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root directory containing city subfolders.")
    parser.add_argument(
        "--cities",
        type=str,
        default="chengdu,porto,rome,sanfran",
        help="Comma-separated city names, e.g. chengdu,porto,rome,sanfran",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis_vis_mpl/road_basemaps"),
        help="Output directory for PNG basemaps.",
    )
    parser.add_argument("--size", type=int, default=2048, help="Output image size in pixels (square).")
    parser.add_argument("--line-color", type=str, default="black", help="Matplotlib line color.")
    parser.add_argument("--line-width", type=float, default=0.35, help="Road line width in points.")
    parser.add_argument(
        "--auto-crs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-project likely lon/lat data to local UTM per city before plotting.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.02,
        help="Extra margin as fraction of the square extent (0.02 means 2%% on each side).",
    )
    return parser.parse_args()


def iter_line_strings(geom: BaseGeometry) -> Iterator[LineString]:
    if geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
        return
    if isinstance(geom, MultiLineString):
        for part in geom.geoms:
            if not part.is_empty:
                yield part
        return
    if hasattr(geom, "geoms"):
        for sub in geom.geoms:
            yield from iter_line_strings(sub)


def load_segments(edges_path: Path) -> tuple[list[tuple[list[float], list[float]]], tuple[float, float, float, float], int]:
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edges file: {edges_path}")

    df = pd.read_csv(edges_path, usecols=["geometry"], dtype={"geometry": "string"})
    segments: list[tuple[list[float], list[float]]] = []
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    skipped = 0

    for geo_text in df["geometry"].dropna():
        text = str(geo_text).strip()
        if not text:
            skipped += 1
            continue
        try:
            geom = wkt.loads(text)
        except Exception:
            skipped += 1
            continue

        found_line = False
        for line in iter_line_strings(geom):
            coords = list(line.coords)
            if len(coords) < 2:
                skipped += 1
                continue
            xs = [xy[0] for xy in coords]
            ys = [xy[1] for xy in coords]
            segments.append((xs, ys))

            lx0, ly0, lx1, ly1 = line.bounds
            minx = min(minx, lx0)
            miny = min(miny, ly0)
            maxx = max(maxx, lx1)
            maxy = max(maxy, ly1)
            found_line = True

        if not found_line:
            skipped += 1

    if not segments:
        raise ValueError(f"No valid line geometries found in {edges_path}")

    return segments, (minx, miny, maxx, maxy), skipped


def is_likely_lonlat(bounds: tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    return (
        -180.0 <= minx <= 180.0
        and -180.0 <= maxx <= 180.0
        and -90.0 <= miny <= 90.0
        and -90.0 <= maxy <= 90.0
        and 0.0 < width <= 20.0
        and 0.0 < height <= 20.0
    )


def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def compute_bounds(segments: list[tuple[list[float], list[float]]]) -> tuple[float, float, float, float]:
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for xs, ys in segments:
        minx = min(minx, min(xs))
        maxx = max(maxx, max(xs))
        miny = min(miny, min(ys))
        maxy = max(maxy, max(ys))
    return minx, miny, maxx, maxy


def auto_project_segments(
    segments: list[tuple[list[float], list[float]]],
    raw_bounds: tuple[float, float, float, float],
    auto_crs: bool,
) -> tuple[list[tuple[list[float], list[float]]], tuple[float, float, float, float], str]:
    if not auto_crs:
        return segments, raw_bounds, "raw (auto-crs disabled)"
    if not is_likely_lonlat(raw_bounds):
        return segments, raw_bounds, "raw (non-lonlat detected)"

    minx, miny, maxx, maxy = raw_bounds
    center_lon = (minx + maxx) * 0.5
    center_lat = (miny + maxy) * 0.5
    target_epsg = utm_epsg_from_lonlat(center_lon, center_lat)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)

    projected: list[tuple[list[float], list[float]]] = []
    for xs, ys in segments:
        tx, ty = transformer.transform(xs, ys)
        projected.append((list(tx), list(ty)))
    projected_bounds = compute_bounds(projected)
    return projected, projected_bounds, f"EPSG:4326 -> EPSG:{target_epsg}"


def square_bounds(bounds: tuple[float, float, float, float], padding: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 1e-12)
    height = max(maxy - miny, 1e-12)
    side = max(width, height)
    side *= 1.0 + 2.0 * padding

    cx = (minx + maxx) * 0.5
    cy = (miny + maxy) * 0.5
    half = side * 0.5
    return cx - half, cy - half, cx + half, cy + half


def render_city(
    segments: list[tuple[list[float], list[float]]],
    bounds: tuple[float, float, float, float],
    out_path: Path,
    size: int,
    line_color: str,
    line_width: float,
) -> None:
    dpi = 256
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor=(0, 0, 0, 0))
    ax = fig.add_axes([0, 0, 1, 1], facecolor=(0, 0, 0, 0))

    for xs, ys in segments:
        ax.plot(
            xs,
            ys,
            color=line_color,
            linewidth=line_width,
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
        )

    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=dpi,
        transparent=True,
        facecolor=(0, 0, 0, 0),
        edgecolor="none",
        pad_inches=0,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be positive.")
    if args.padding < 0:
        raise ValueError("--padding must be >= 0.")
    if args.line_width <= 0:
        raise ValueError("--line-width must be positive.")

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    if not cities:
        raise ValueError("No valid cities parsed from --cities.")

    print(f"Rendering {len(cities)} city basemap(s) to {args.out_dir} at {args.size}x{args.size}...")
    for city in cities:
        edges_path = args.data_root / city / "G_edges.csv"
        segments, raw_bounds, skipped = load_segments(edges_path)
        plot_segments, plot_bounds, crs_note = auto_project_segments(segments, raw_bounds, args.auto_crs)
        padded_bounds = square_bounds(plot_bounds, args.padding)
        out_path = args.out_dir / f"{city}.png"
        render_city(
            segments=plot_segments,
            bounds=padded_bounds,
            out_path=out_path,
            size=args.size,
            line_color=args.line_color,
            line_width=args.line_width,
        )
        print(f"[OK] {city}: {len(plot_segments)} segments, skipped={skipped}, crs={crs_note}, saved={out_path}")


if __name__ == "__main__":
    main()
