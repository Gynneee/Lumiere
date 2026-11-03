import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_MANIFEST_PATHS = (
    ROOT / "dataset" / "flower_match" / "train.jsonl",
    ROOT / "dataset" / "flower_match" / "js_teacher_manifest.jsonl",
    ROOT / "dataset" / "diverse_match" / "train.jsonl",
    ROOT / "dataset" / "diverse_match" / "js_teacher_manifest.jsonl",
    ROOT / "dataset" / "portrait_match" / "train.jsonl",
    ROOT / "dataset" / "portrait_match" / "js_teacher_manifest.jsonl",
)


def prepare_manifest(
    output_path,
    seed,
    additional_manifests,
    max_source_groups_per_category,
    pairs_per_source,
    balance_categories,
):
    rows = []
    source_hashes = {}
    manifest_paths = (*BASE_MANIFEST_PATHS, *additional_manifests)
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Training manifest not found: {manifest_path}")
        with manifest_path.open(encoding="utf-8") as manifest:
            for line_number, line in enumerate(manifest, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    source_path = (manifest_path.parent / row["source"]).resolve()
                    reference_path = (manifest_path.parent / row["reference"]).resolve()
                    if not source_path.is_file() or not reference_path.is_file():
                        raise FileNotFoundError("source or reference image does not exist")
                    row["_source_path"] = str(source_path)
                    row["_reference_path"] = str(reference_path)
                    if source_path not in source_hashes:
                        source_hashes[source_path] = hashlib.sha256(
                            source_path.read_bytes()
                        ).hexdigest()
                    row["_source_hash"] = source_hashes[source_path]
                    row["_origin"] = f"{manifest_path.parent.name}/{manifest_path.name}"
                    row["_category"] = row.get("category") or (
                        "portrait"
                        if manifest_path.parent.name == "portrait_match"
                        else "flower"
                        if manifest_path.parent.name == "flower_match"
                        else Path(row["source"]).stem.split("_", 1)[0]
                    )
                    if row["_category"] not in (
                        "flower",
                        "portrait",
                        "landscape",
                        "architecture",
                        "urban",
                    ):
                        raise ValueError(f"unsupported category {row['_category']!r}")
                    rows.append(row)
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    OSError,
                ) as error:
                    raise ValueError(
                        f"{manifest_path}:{line_number}: invalid training row: {error}"
                    ) from error

    categories_by_group = {}
    for row in rows:
        category_counts = categories_by_group.setdefault(row["_source_hash"], Counter())
        category_counts[row["_category"]] += 1

    priority = {
        "flower": 0,
        "portrait": 1,
        "landscape": 2,
        "architecture": 3,
        "urban": 4,
    }
    category_by_group = {
        group_hash: max(
            category_counts,
            key=lambda category: (category_counts[category], priority[category]),
        )
        for group_hash, category_counts in categories_by_group.items()
    }
    groups_by_category = {}
    for row in rows:
        category = category_by_group[row["_source_hash"]]
        row["_category"] = category
        groups_by_category.setdefault(category, {}).setdefault(
            row["_source_hash"], []
        ).append(row)

    required_categories = {
        "flower",
        "portrait",
        "landscape",
        "architecture",
        "urban",
    }
    missing_categories = required_categories - groups_by_category.keys()
    if missing_categories:
        raise ValueError(
            "Cannot build a balanced manifest; missing categories: "
            + ", ".join(sorted(missing_categories))
        )

    balanced_group_limit = (
        min(
            max_source_groups_per_category,
            *(len(groups) for groups in groups_by_category.values()),
        )
        if balance_categories
        else None
    )
    rng = random.Random(seed)
    selected_rows = []
    selected_groups = {}
    for category, groups in sorted(groups_by_category.items()):
        group_hashes = sorted(groups)
        rng.shuffle(group_hashes)
        if balanced_group_limit is not None:
            group_hashes = group_hashes[:balanced_group_limit]
        selected_groups[category] = group_hashes
        for group_hash in group_hashes:
            group_rows = groups[group_hash]
            if pairs_per_source is not None:
                rng.shuffle(group_rows)
                group_rows = group_rows[:pairs_per_source]
            selected_rows.extend(group_rows)

    all_selected_groups = sorted(
        {
            group_hash
            for group_hashes in selected_groups.values()
            for group_hash in group_hashes
        }
    )
    validation_groups = set()
    for group_hashes in selected_groups.values():
        validation_count = max(1, round(len(group_hashes) * 0.2))
        if validation_count < len(group_hashes):
            validation_groups.update(rng.sample(group_hashes, validation_count))
        else:
            validation_groups.update(group_hashes)
    split_by_group = {
        group_hash: (
            "validation" if group_hash in validation_groups else "train"
        )
        for group_hash in all_selected_groups
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in selected_rows:
            split = split_by_group[row["_source_hash"]]
            example = {
                key: value
                for key, value in row.items()
                if not key.startswith("_")
            }
            example["source"] = row["_source_path"]
            example["reference"] = row["_reference_path"]
            example["group"] = row["_source_hash"]
            example["category"] = row["_category"]
            example["split"] = split
            example["origin"] = row["_origin"]
            output.write(json.dumps(example, separators=(",", ":")) + "\n")
            counts[(row["_category"], split)] += 1

    print(f"Combined manifest: {output_path}")
    print(
        f"Pairs: {len(selected_rows)}; unique source photos: "
        f"{len(all_selected_groups)}"
    )
    for (category, split), count in sorted(counts.items()):
        print(f"{category}: {split}={count}")
    print(
        "Source photos by category: "
        + ", ".join(
            f"{category}={len(group_hashes)}"
            for category, group_hashes in sorted(selected_groups.items())
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Combine the available local datasets with source-photo-safe splits."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dataset" / "expanded_run" / "combined_manifest.jsonl",
    )
    parser.add_argument(
        "--additional-manifest",
        type=Path,
        action="append",
        default=[],
        help="Add another labeled manifest (repeatable), such as scene variants.",
    )
    parser.add_argument(
        "--max-source-groups-per-category",
        type=int,
        default=50,
        help="Cap source-photo groups per category to keep category proportions balanced.",
    )
    parser.add_argument(
        "--pairs-per-source",
        type=int,
        default=None,
        help="Maximum number of labeled variants retained for each source photo.",
    )
    parser.add_argument(
        "--balance-categories",
        action="store_true",
        help="Downsample to equal unique-source counts per category.",
    )
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    if args.max_source_groups_per_category < 1 or (
        args.pairs_per_source is not None and args.pairs_per_source < 1
    ):
        parser.error("group and optional pair limits must be positive.")
    prepare_manifest(
        args.output.resolve(),
        args.seed,
        [path.resolve() for path in args.additional_manifest],
        args.max_source_groups_per_category,
        args.pairs_per_source,
        args.balance_categories,
    )


if __name__ == "__main__":
    main()
