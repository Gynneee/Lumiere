import argparse
import json
import random
from pathlib import Path

from PIL import Image

import prepare_diverse_training as diverse
import prepare_synthetic_training as synthetic


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "dataset" / "scene_expansion" / "train.jsonl"


def expand_manifest(manifest_path, output_path, variants, seed):
    rng = random.Random(seed)
    rows = []
    with manifest_path.open(encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_path = (manifest_path.parent / row["source"]).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"{manifest_path}:{line_number}: source photo missing: {source_path}"
                )
            category = row.get("category")
            if category not in ("portrait", "landscape", "architecture", "urban"):
                raise ValueError(
                    f"{manifest_path}:{line_number}: unsupported scene category {category!r}"
                )
            stem = Path(row["source"]).stem
            reference_dir = manifest_path.parent / "images" / "reference"
            reference_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(source_path) as source:
                source_image = source.convert("RGB")
            for variant in range(variants):
                settings = diverse.random_settings(rng, variant)
                reference_path = reference_dir / f"{stem}_variant_{variant:02}.jpg"
                if not reference_path.is_file():
                    synthetic.apply_settings(source_image, settings).save(
                        reference_path, "JPEG", quality=90
                    )
                variant_row = dict(row)
                variant_row["reference"] = reference_path.relative_to(
                    manifest_path.parent
                ).as_posix()
                variant_row["settings"] = settings
                variant_row["group"] = row.get("group", stem)
                variant_row["split"] = row["split"]
                rows.append(variant_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(
        f"Created {len(rows)} labeled variants from "
        f"{len({row['group'] for row in rows})} unique scene photos: {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create known-setting synthetic scene edits for acquired photos."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variants-per-photo", type=int, default=3)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    if args.variants_per_photo < 1:
        parser.error("--variants-per-photo must be positive.")
    manifest_path = args.manifest.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else manifest_path.with_name("variants_manifest.jsonl")
    )
    expand_manifest(
        manifest_path, output_path, args.variants_per_photo, args.seed
    )


if __name__ == "__main__":
    main()
