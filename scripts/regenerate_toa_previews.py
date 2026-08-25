"""Regenerate TOA PNG previews from existing NPZ map bundles."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREVIEW_HELPER_PATH = (
    PROJECT_ROOT
    / "source/tutorial/tutorial/tasks/manager_based/toa_preview.py"
)
spec = importlib.util.spec_from_file_location("toa_preview", PREVIEW_HELPER_PATH)
toa_preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toa_preview)


def _as_text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def regenerate_preview(npz_path: Path) -> Path:
    """Regenerate the PNG next to one NPZ bundle."""

    with np.load(npz_path) as data:
        maps = np.asarray(data["maps"], dtype=np.float32)
        goals = np.asarray(data["goals"], dtype=np.float32)
        extent = np.asarray(data["extent"], dtype=np.float32)
        titles = [_as_text(value) for value in data["variant_names"]]

    map_count, height, width = maps.shape
    columns = 2 if map_count > 1 else 1
    rows = (map_count + columns - 1) // columns
    preview = Image.new("RGB", (columns * width, rows * height), color=(255, 255, 255))
    x_min, x_max, y_min, y_max = extent

    for map_id, toa_map in enumerate(maps):
        rgb, display_max = toa_preview.toa_to_rgb(toa_map)
        panel = Image.fromarray(np.flipud(rgb), mode="RGB")
        draw = ImageDraw.Draw(panel)
        goal_x = int(round((goals[map_id, 0] - x_min) / (x_max - x_min) * (width - 1)))
        goal_y = (height - 1) - int(
            round((goals[map_id, 1] - y_min) / (y_max - y_min) * (height - 1))
        )
        draw.ellipse((goal_x - 4, goal_y - 4, goal_x + 4, goal_y + 4), fill=(255, 0, 0))
        label = f"{titles[map_id]} max={display_max:.1f}"
        draw.rectangle((0, 0, 148, 13), fill=(255, 255, 255))
        draw.text((2, 1), label, fill=(0, 0, 0))
        preview.paste(panel, ((map_id % columns) * width, (map_id // columns) * height))

    output_path = npz_path.with_suffix(".png")
    preview.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            PROJECT_ROOT / "outputs/training_maze_toa",
            PROJECT_ROOT / "outputs/test_maze_toa",
        ],
        help="NPZ files or directories containing NPZ bundles.",
    )
    args = parser.parse_args()

    npz_paths: list[Path] = []
    for path in args.paths:
        npz_paths.extend(sorted(path.glob("*.npz")) if path.is_dir() else [path])
    if not npz_paths:
        raise FileNotFoundError("No TOA NPZ files were found.")

    for npz_path in npz_paths:
        print(regenerate_preview(npz_path))


if __name__ == "__main__":
    main()
