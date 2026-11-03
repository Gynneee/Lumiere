import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn


FEATURES_PER_IMAGE = 107
INPUT_SIZE = FEATURES_PER_IMAGE * 2
OUTPUT_SIZE = 50
CURVE_X = (0, 32, 64, 128, 192, 224, 255)
HUE_BANDS = (0, 30, 60, 120, 180, 240, 270, 315)
HIST_BINS = 512
QUANTILES = tuple(0.025 + index * 0.05 for index in range(20))


def srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_oklab(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack(
        (
            0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
        ),
        axis=-1,
    )


def hsv_hue_to_oklab_angle(hue):
    h = (hue % 360) / 60
    c = 1.0
    x = 1 - abs(h % 2 - 1)
    sector = int(h)
    rgb = ((c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x))[sector]
    lab = linear_to_oklab(srgb_to_linear(np.array(rgb, dtype=np.float64)))
    return math.degrees(math.atan2(lab[2], lab[1])) % 360


BAND_ANGLES = tuple(hsv_hue_to_oklab_angle(hue) for hue in HUE_BANDS)


def histogram(values, low, high):
    indices = ((values - low) / (high - low) * HIST_BINS).astype(np.int64)
    return np.bincount(np.clip(indices, 0, HIST_BINS - 1), minlength=HIST_BINS).astype(
        np.float64
    )


def histogram_quantile(counts, low, high, total, quantile):
    target = quantile * total
    accumulated = 0.0
    for index, count in enumerate(counts):
        next_accumulated = accumulated + count
        if next_accumulated >= target:
            fraction = (target - accumulated) / count if count > 0 else 0.5
            return low + ((index + fraction) / HIST_BINS) * (high - low)
        accumulated = next_accumulated
    return high


def analyze_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((256, 256), Image.Resampling.LANCZOS)
        rgb = np.asarray(image, dtype=np.float64) / 255.0

    linear = srgb_to_linear(rgb)
    lab = linear_to_oklab(linear.reshape(-1, 3))
    lightness, a, b = lab.T
    chroma = np.sqrt(a * a + b * b)
    total = len(lightness)
    encoded = rgb.reshape(-1, 3)
    red, green, blue = encoded.T
    cb = -0.168736 * red - 0.331264 * green + 0.5 * blue + 0.5
    cr = 0.5 * red - 0.418688 * green - 0.081312 * blue + 0.5
    in_range = lambda values, low, high, feather: np.clip(
        np.minimum((values - low) / feather, (high - values) / feather), 0, 1
    )
    skin_weights = (
        in_range(cb, 0.2, 0.55, 0.08)
        * in_range(cr, 0.45, 0.82, 0.08)
        * np.clip((red - 0.025) / 0.12, 0, 1)
        * np.clip((red / np.maximum(green, 0.01) - 0.85) / 0.2, 0, 1)
        * np.clip((green / np.maximum(blue, 0.015) - 0.65) / 0.3, 0, 1)
    )
    skin_count = float(skin_weights.sum())
    skin_features = [skin_count / total, 0.0, 0.0, 0.0, 0.0, 0.0]
    skin_pixels = skin_weights > 0
    if skin_count > 0:
        skin_lightness_hist = np.bincount(
            np.clip(
                (lightness[skin_pixels] * HIST_BINS).astype(np.int64),
                0,
                HIST_BINS - 1,
            ),
            weights=skin_weights[skin_pixels],
            minlength=HIST_BINS,
        )
        skin_features[1:] = (
            histogram_quantile(skin_lightness_hist, 0, 1, skin_count, 0.25),
            histogram_quantile(skin_lightness_hist, 0, 1, skin_count, 0.5),
            histogram_quantile(skin_lightness_hist, 0, 1, skin_count, 0.75),
            float(np.dot(a[skin_pixels], skin_weights[skin_pixels]) / skin_count)
            / 0.4,
            float(np.dot(b[skin_pixels], skin_weights[skin_pixels]) / skin_count)
            / 0.4,
        )

    y = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    y = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    value_hist = histogram(y.reshape(-1), 0, 1)
    light_hist = histogram(lightness, 0, 1)
    a_hist = histogram(a, -0.4, 0.4)
    b_hist = histogram(b, -0.4, 0.4)
    chroma_hist = histogram(chroma, 0, 0.4)

    luminance_quantiles = [
        histogram_quantile(value_hist, 0, 1, total, q) for q in QUANTILES
    ]
    a_quantiles = [histogram_quantile(a_hist, -0.4, 0.4, total, q) for q in QUANTILES]
    b_quantiles = [histogram_quantile(b_hist, -0.4, 0.4, total, q) for q in QUANTILES]

    zone_weights = np.stack(
        (
            1 - smooth_array(lightness, 0.25, 0.65),
            np.exp(-((lightness - 0.55) / 0.2) ** 2),
            smooth_array(lightness, 0.45, 0.85),
        ),
        axis=1,
    )
    zone = []
    for weights in zone_weights.T:
        weight_sum = float(weights.sum())
        zone.extend(
            (
                weight_sum / total,
                float(np.dot(weights, a) / weight_sum) if weight_sum else 0.0,
                float(np.dot(weights, b) / weight_sum) if weight_sum else 0.0,
            )
        )

    band_share = np.zeros(8)
    band_chroma = np.zeros(8)
    band_hue_delta = np.zeros(8)
    band_lightness = np.zeros(8)
    angles = np.degrees(np.arctan2(b, a)) % 360
    valid = np.flatnonzero(chroma > 0.02)
    valid_angles = (angles[valid] - BAND_ANGLES[0]) % 360 + BAND_ANGLES[0]
    valid_angles = np.minimum(
        valid_angles, np.nextafter(BAND_ANGLES[0] + 360, BAND_ANGLES[0])
    )
    boundaries = np.asarray(BAND_ANGLES[1:] + (BAND_ANGLES[0] + 360,))
    band_index = np.searchsorted(boundaries, valid_angles, side="right")
    next_index = (band_index + 1) % 8
    start = np.asarray(BAND_ANGLES)[band_index]
    end = boundaries[band_index]
    t = (valid_angles - start) / (end - start)
    t = t * t * (3 - 2 * t)
    selected_chroma = chroma[valid]
    selected_lightness = lightness[valid]
    for band, weights, delta in (
        (band_index, 1 - t, valid_angles - start),
        (next_index, t, valid_angles - end),
    ):
        weighted_chroma = weights * selected_chroma
        band_share += np.bincount(band, weights=weights, minlength=8)
        band_chroma += np.bincount(band, weights=weighted_chroma, minlength=8)
        band_hue_delta += np.bincount(
            band, weights=weighted_chroma * delta, minlength=8
        )
        band_lightness += np.bincount(
            band, weights=weighted_chroma * selected_lightness, minlength=8
        )

    bands = []
    for index in range(8):
        share = band_share[index]
        chroma_sum = band_chroma[index]
        bands.extend(
            (
                share / total,
                chroma_sum / share if share else 0.0,
                band_hue_delta[index] / chroma_sum if chroma_sum else 0.0,
                band_lightness[index] / chroma_sum if chroma_sum else 0.0,
            )
        )

    stats = {
        "dq": luminance_quantiles + a_quantiles + b_quantiles,
        "band": bands,
        "zone": zone,
        "skin": skin_features,
        "value_hist": value_hist,
        "light_hist": light_hist,
        "a_hist": a_hist,
        "b_hist": b_hist,
        "chroma_hist": chroma_hist,
        "total": total,
    }
    return stats


def smooth_array(values, low, high):
    t = np.clip((values - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def features_from_stats(stats):
    quantiles = stats["dq"]
    values = (
        quantiles[:20]
        + [value / 0.4 for value in quantiles[20:40]]
        + [value / 0.4 for value in quantiles[40:60]]
    )
    for index in range(8):
        share, chroma, hue_delta, lightness = stats["band"][index * 4 : index * 4 + 4]
        values.extend((share, chroma / 0.4, hue_delta / 45, lightness))
    for index in range(3):
        share, a, b = stats["zone"][index * 3 : index * 3 + 3]
        values.extend((share, a / 0.4, b / 0.4))
    values.extend(stats["skin"])
    if len(values) != FEATURES_PER_IMAGE:
        raise ValueError(f"Expected {FEATURES_PER_IMAGE} image features; got {len(values)}")
    return values


def validate_range(value, low, high, label):
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{label} must be a finite number from {low} to {high}.")
    return number


def target_vector(settings, sample_number):
    label = f"sample {sample_number} settings"
    values = []
    for key, limit in (
        ("temp", 100),
        ("tint", 100),
        ("exposure", 5),
        ("contrast", 100),
        ("highlights", 100),
        ("shadows", 100),
        ("whites", 100),
        ("blacks", 100),
        ("vibrance", 100),
        ("saturation", 100),
    ):
        values.append(validate_range(settings[key], -limit, limit, f"{label}.{key}") / limit)

    curve = settings["curve"]
    if len(curve) != len(CURVE_X):
        raise ValueError(f"{label}.curve must contain {len(CURVE_X)} output values.")
    values.extend(
        validate_range(value, 0, 255, f"{label}.curve[{index}]") / 127.5 - 1
        for index, value in enumerate(curve)
    )

    hsl = settings["hsl"]
    if len(hsl) != 8:
        raise ValueError(f"{label}.hsl must contain exactly 8 color-band objects.")
    for index, band in enumerate(hsl):
        for key in ("h", "s", "l"):
            values.append(
                validate_range(band[key], -100, 100, f"{label}.hsl[{index}].{key}") / 100
            )

    grade = settings["grade"]
    for key in ("sh", "mid", "hi"):
        hue = validate_range(grade[key]["hue"], 0, 360, f"{label}.grade.{key}.hue")
        saturation = validate_range(
            grade[key]["sat"], 0, 100, f"{label}.grade.{key}.sat"
        )
        radians = math.radians(hue)
        values.extend((math.cos(radians), math.sin(radians), saturation / 50 - 1))

    if len(values) != OUTPUT_SIZE:
        raise ValueError(f"Expected {OUTPUT_SIZE} model targets; got {len(values)}")
    return values


def load_dataset(manifest_paths):
    features, targets, groups, splits, categories = [], [], [], [], []
    image_features = {}
    total_rows = 0
    for manifest_path in manifest_paths:
        with manifest_path.open(encoding="utf-8") as manifest:
            total_rows += sum(1 for line in manifest if line.strip())
    for manifest_path in manifest_paths:
        base = manifest_path.parent
        group_prefix = str(base.resolve())
        with manifest_path.open(encoding="utf-8") as manifest:
            for line_number, line in enumerate(manifest, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    source_path = (base / row["source"]).resolve()
                    reference_path = (base / row["reference"]).resolve()
                    for image_path in (source_path, reference_path):
                        if image_path not in image_features:
                            image_features[image_path] = features_from_stats(
                                analyze_image(image_path)
                            )
                    source_features = image_features[source_path]
                    reference_features = image_features[reference_path]
                    features.append(source_features + reference_features)
                    targets.append(target_vector(row["settings"], len(targets) + 1))
                    groups.append(
                        f"{group_prefix}::{row.get('group', row['source'])}"
                    )
                    splits.append(str(row.get("split", "")))
                    category = row.get("category")
                    if category is None:
                        category = (
                            "flower"
                            if base.name == "flower_match"
                            else "portrait"
                            if base.name == "portrait_match"
                            else Path(row["source"]).stem.split("_", 1)[0]
                        )
                    categories.append(str(category))
                    if len(features) % 500 == 0:
                        print(
                            f"Extracted image features for {len(features)}/{total_rows} "
                            "training pairs.",
                            flush=True,
                        )
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    OSError,
                    json.JSONDecodeError,
                ) as error:
                    raise ValueError(
                        f"{manifest_path}:{line_number}: invalid training row: {error}"
                    ) from error
    if len(features) < 5:
        raise ValueError(
            f"At least 5 labeled examples are required; found {len(features)}."
        )
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(groups),
        np.asarray(splits),
        np.asarray(categories),
    )


def write_training_graph(output_dir, history):
    width, height = 900, 480
    left, top, right, bottom = 82, 32, 28, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    largest_loss = max(
        max(row["trainingLoss"], row["validationLoss"]) for row in history
    )
    y_max = max(0.01, math.ceil(largest_loss * 100) / 100)

    def point(index, loss):
        x = left + (index / max(1, len(history) - 1)) * plot_width
        y = top + (1 - loss / y_max) * plot_height
        return f"{x:.1f},{y:.1f}"

    training_points = " ".join(
        point(index, row["trainingLoss"]) for index, row in enumerate(history)
    )
    validation_points = " ".join(
        point(index, row["validationLoss"]) for index, row in enumerate(history)
    )
    grid = []
    for step in range(6):
        fraction = step / 5
        y = top + fraction * plot_height
        value = y_max * (1 - fraction)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" '
            f'y2="{y:.1f}" stroke="#d7dee8"/><text x="{left - 10}" '
            f'y="{y + 4:.1f}" text-anchor="end" font-size="12">{value:.3f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="18">Training and validation loss</text>
{''.join(grid)}
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#263445"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#263445"/>
<polyline fill="none" stroke="#2463eb" stroke-width="2.5" points="{training_points}"/>
<polyline fill="none" stroke="#e05a36" stroke-width="2.5" points="{validation_points}"/>
<text x="{left}" y="{height - 42}" font-family="sans-serif" font-size="13">Epoch 1</text>
<text x="{left + plot_width}" y="{height - 42}" text-anchor="end" font-family="sans-serif" font-size="13">Epoch {len(history)}</text>
<line x1="{width - 250}" y1="{height - 38}" x2="{width - 222}" y2="{height - 38}" stroke="#2463eb" stroke-width="3"/>
<text x="{width - 214}" y="{height - 34}" font-family="sans-serif" font-size="13">Training MSE</text>
<line x1="{width - 130}" y1="{height - 38}" x2="{width - 102}" y2="{height - 38}" stroke="#e05a36" stroke-width="3"/>
<text x="{width - 94}" y="{height - 34}" font-family="sans-serif" font-size="13">Validation MSE</text>
<text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">Mean squared error</text>
</svg>
"""
    graph_path = output_dir / "training_curves.svg"
    graph_path.write_text(svg, encoding="utf-8")
    mae_graph_path = output_dir / "training_mae_curves.svg"
    if all("validationMae" in row and "trainingMae" in row for row in history):
        largest_mae = max(
            max(row["trainingMae"], row["validationMae"]) for row in history
        )
        mae_max = max(0.01, math.ceil(largest_mae * 100) / 100)

        def mae_point(index, mae):
            x = left + (index / max(1, len(history) - 1)) * plot_width
            y = top + (1 - mae / mae_max) * plot_height
            return f"{x:.1f},{y:.1f}"

        training_mae_points = " ".join(
            mae_point(index, row["trainingMae"])
            for index, row in enumerate(history)
        )
        validation_mae_points = " ".join(
            mae_point(index, row["validationMae"])
            for index, row in enumerate(history)
        )
        mae_grid = []
        for step in range(6):
            fraction = step / 5
            y = top + fraction * plot_height
            value = mae_max * (1 - fraction)
            mae_grid.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" '
                f'y2="{y:.1f}" stroke="#d7dee8"/><text x="{left - 10}" '
                f'y="{y + 4:.1f}" text-anchor="end" font-size="12">{value:.3f}</text>'
            )
        mae_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="18">Training and validation normalized MAE</text>
{''.join(mae_grid)}
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#263445"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#263445"/>
<polyline fill="none" stroke="#2463eb" stroke-width="2.5" points="{training_mae_points}"/>
<polyline fill="none" stroke="#e05a36" stroke-width="2.5" points="{validation_mae_points}"/>
<text x="{left}" y="{height - 42}" font-family="sans-serif" font-size="13">Epoch 1</text>
<text x="{left + plot_width}" y="{height - 42}" text-anchor="end" font-family="sans-serif" font-size="13">Epoch {len(history)}</text>
<line x1="{width - 250}" y1="{height - 38}" x2="{width - 222}" y2="{height - 38}" stroke="#2463eb" stroke-width="3"/>
<text x="{width - 214}" y="{height - 34}" font-family="sans-serif" font-size="13">Training MAE</text>
<line x1="{width - 130}" y1="{height - 38}" x2="{width - 102}" y2="{height - 38}" stroke="#e05a36" stroke-width="3"/>
<text x="{width - 94}" y="{height - 34}" font-family="sans-serif" font-size="13">Validation MAE</text>
<text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">Mean absolute error (normalized)</text>
</svg>
"""
        mae_graph_path.write_text(mae_svg, encoding="utf-8")
    return graph_path, mae_graph_path


class ColorMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, OUTPUT_SIZE),
            nn.Tanh(),
        )

    def forward(self, features):
        return self.layers(features)


def train(manifest_paths, output_path, epochs, seed, balance_category_loss):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    manifest_path = manifest_paths[0]
    x, y, groups, splits, categories = load_dataset(manifest_paths)

    if np.any(splits):
        if not np.all(np.isin(splits, ("train", "validation"))):
            raise ValueError(
                "When a manifest uses explicit splits, every row must use "
                "'train' or 'validation'."
            )
        training_indices = np.flatnonzero(splits == "train")
        validation_indices = np.flatnonzero(splits == "validation")
        if not len(training_indices) or not len(validation_indices):
            raise ValueError("Explicit data splits must both contain examples.")
        training_groups = np.unique(groups[training_indices])
        validation_groups = np.unique(groups[validation_indices])
        if np.intersect1d(training_groups, validation_groups).size:
            raise ValueError("Source-image groups must not appear in both data splits.")
    else:
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError("Training requires at least two distinct source-image groups.")
        shuffled_groups = np.random.permutation(unique_groups)
        validation_group_count = min(
            len(unique_groups) - 1, max(1, round(len(unique_groups) * 0.2))
        )
        validation_groups = shuffled_groups[:validation_group_count]
        validation_mask = np.isin(groups, validation_groups)
        validation_indices = np.flatnonzero(validation_mask)
        training_indices = np.flatnonzero(~validation_mask)
    train_x = torch.from_numpy(x[training_indices])
    train_y = torch.from_numpy(y[training_indices])
    validation_x = torch.from_numpy(x[validation_indices])
    validation_y = torch.from_numpy(y[validation_indices])
    training_categories = categories[training_indices]
    validation_categories = categories[validation_indices]
    category_weights = {}

    def row_weights(category_values):
        counts = {
            str(category): int(np.count_nonzero(category_values == category))
            for category in np.unique(category_values)
        }
        category_count = len(counts)
        return torch.as_tensor(
            [
                len(category_values) / (category_count * counts[str(category)])
                for category in category_values
            ],
            dtype=torch.float32,
        )

    if balance_category_loss:
        train_weights = row_weights(training_categories)
        validation_weights = row_weights(validation_categories)
        category_weights = {
            category: len(training_categories)
            / (
                len(np.unique(training_categories))
                * int(np.count_nonzero(training_categories == category))
            )
            for category in sorted(set(training_categories))
        }
    else:
        train_weights = torch.ones(len(training_indices), dtype=torch.float32)
        validation_weights = torch.ones(len(validation_indices), dtype=torch.float32)

    model = ColorMatcher()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience = 40
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        per_example_loss = torch.mean((model(train_x) - train_y) ** 2, dim=1)
        loss = torch.mean(per_example_loss * train_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            epoch_training_predictions = model(train_x)
            epoch_validation_predictions = model(validation_x)
            training_loss = (
                torch.mean(
                    torch.mean((epoch_training_predictions - train_y) ** 2, dim=1)
                    * train_weights
                ).item()
            )
            validation_loss = torch.mean(
                torch.mean((epoch_validation_predictions - validation_y) ** 2, dim=1)
                * validation_weights
            ).item()
            training_mae = torch.mean(
                torch.mean(torch.abs(epoch_training_predictions - train_y), dim=1)
                * train_weights
            ).item()
            validation_mae = torch.mean(
                torch.mean(torch.abs(epoch_validation_predictions - validation_y), dim=1)
                * validation_weights
            ).item()
        history.append(
            {
                "epoch": epoch,
                "trainingLoss": float(training_loss),
                "validationLoss": float(validation_loss),
                "trainingMae": float(training_mae),
                "validationMae": float(validation_mae),
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch}/{epochs}: training MSE={training_loss:.6f}, "
                f"validation MSE={validation_loss:.6f}, "
                f"validation MAE={validation_mae:.6f}.",
                flush=True,
            )
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        training_predictions = model(train_x).numpy()
        predictions = model(validation_x).numpy()
    training_per_example_mse = np.mean(
        (training_predictions - train_y.numpy()) ** 2, axis=1
    )
    training_per_example_mae = np.mean(
        np.abs(training_predictions - train_y.numpy()), axis=1
    )
    training_mse = float(
        np.mean(training_per_example_mse * train_weights.numpy())
    )
    training_normalized_mae = float(
        np.mean(training_per_example_mae * train_weights.numpy())
    )
    mae_by_output = np.abs(predictions - validation_y.numpy()).mean(axis=0)
    training_mae_by_output = np.abs(training_predictions - train_y.numpy()).mean(axis=0)
    zero_baseline_mae_by_output = np.abs(validation_y.numpy()).mean(axis=0)
    category_metrics = {}
    for category in sorted(set(validation_categories)):
        category = str(category)
        category_mask = validation_categories == category
        category_predictions = predictions[category_mask]
        category_targets = validation_y.numpy()[category_mask]
        category_groups = groups[validation_indices][category_mask]
        category_metrics[category] = {
            "examples": int(category_mask.sum()),
            "sourceImages": int(len(np.unique(category_groups))),
            "mse": float(np.mean((category_predictions - category_targets) ** 2)),
            "normalizedMae": float(
                np.abs(category_predictions - category_targets).mean()
            ),
            "zeroBaselineNormalizedMae": float(
                np.abs(category_targets).mean()
            ),
        }
    macro_normalized_mae = float(
        np.mean([metrics["normalizedMae"] for metrics in category_metrics.values()])
    )
    macro_zero_baseline_mae = float(
        np.mean(
            [
                metrics["zeroBaselineNormalizedMae"]
                for metrics in category_metrics.values()
            ]
        )
    )
    basic_setting_limits = np.asarray((100, 100, 5, 100, 100, 100, 100, 100, 100, 100))
    basic_setting_names = (
        "temp",
        "tint",
        "exposure",
        "contrast",
        "highlights",
        "shadows",
        "whites",
        "blacks",
        "vibrance",
        "saturation",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path, mae_graph_path = write_training_graph(
        manifest_path.parent, history
    )
    history_path = manifest_path.parent / "training_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as history_file:
        history_file.write(
            "epoch,training_mse,validation_mse,training_normalized_mae,"
            "validation_normalized_mae\n"
        )
        for row in history:
            history_file.write(
                f"{row['epoch']},{row['trainingLoss']:.9f},"
                f"{row['validationLoss']:.9f},{row['trainingMae']:.9f},"
                f"{row['validationMae']:.9f}\n"
            )
    example = torch.zeros((1, INPUT_SIZE), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        output_path,
        input_names=["features"],
        output_names=["settings"],
        dynamic_axes={"features": {0: "batch"}, "settings": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    report = {
        "examples": len(x),
        "seed": seed,
        "epochsRequested": epochs,
        "categoryMetrics": category_metrics,
        "categoryLossBalanced": balance_category_loss,
        "trainingCategoryWeights": category_weights,
        "meanCategoryNormalizedAbsoluteError": float(
            np.mean(
                [
                    metrics["normalizedMae"]
                    for metrics in category_metrics.values()
                ]
            )
        ),
        "zeroBaselineMeanCategoryNormalizedAbsoluteError": float(
            macro_zero_baseline_mae
        ),
        "microMeanNormalizedAbsoluteErrorOnAllOutputs": float(
            mae_by_output.mean()
        ),
        "microZeroBaselineMeanNormalizedAbsoluteErrorOnAllOutputs": float(
            zero_baseline_mae_by_output.mean()
        ),
        "trainingExamples": len(training_indices),
        "validationExamples": len(validation_indices),
        "trainingSourceImages": len(np.unique(groups[training_indices])),
        "validationSourceImages": len(np.unique(groups[validation_indices])),
        "bestEpoch": best_epoch,
        "epochsCompleted": len(history),
        "validationMse": best_loss,
        "trainingMse": training_mse,
        "trainingMeanNormalizedAbsoluteErrorOnAllOutputs": float(
            training_normalized_mae
        ),
        "meanNormalizedAbsoluteErrorOnAllOutputs": macro_normalized_mae,
        "zeroBaselineMeanNormalizedAbsoluteErrorOnAllOutputs": float(
            macro_zero_baseline_mae
        ),
        "history": history,
        "basicSettingMae": {
            name: float(error * limit)
            for name, error, limit in zip(
                basic_setting_names, mae_by_output[:10], basic_setting_limits
            )
        },
        "zeroBaselineBasicSettingMae": {
            name: float(error * limit)
            for name, error, limit in zip(
                basic_setting_names,
                zero_baseline_mae_by_output[:10],
                basic_setting_limits,
            )
        },
        "model": str(output_path),
        "inputFeatures": INPUT_SIZE,
        "outputSettings": OUTPUT_SIZE,
    }
    report_path = manifest_path.parent / "training_report.json"
    report["report"] = str(report_path)
    report["trainingHistory"] = str(history_path)
    report["trainingGraph"] = str(graph_path)
    report["trainingMaeGraph"] = str(mae_graph_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Train the local Grade Match model and export it for browser inference."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Labeled JSONL dataset; repeat to combine multiple datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public/models/color_matcher.onnx"),
        help="ONNX output path (default: public/models/color_matcher.onnx).",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--balance-category-loss",
        action="store_true",
        help="Weight each scene category equally in training and validation loss.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    train(
        [path.resolve() for path in args.manifest],
        args.output.resolve(),
        args.epochs,
        args.seed,
        args.balance_category_loss,
    )


if __name__ == "__main__":
    main()
