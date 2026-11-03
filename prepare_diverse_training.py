import argparse
import csv
import html
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

import prepare_synthetic_training as synthetic


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "dataset" / "diverse_match"
PHOTO_DIR = OUTPUT_DIR / "photos"
SOURCE_DIR = OUTPUT_DIR / "images" / "source"
REFERENCE_DIR = OUTPUT_DIR / "images" / "reference"
MANIFEST_PATH = OUTPUT_DIR / "train.jsonl"
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "CineMatchLocalTraining/1.0 (educational use)"
REQUEST_DELAY_SECONDS = 5.0
LAST_REQUEST_AT = 0.0
CATEGORIES = {
    "portrait": "portrait photograph person",
    "landscape": "landscape nature photograph",
    "architecture": "architecture building photograph",
    "urban": "city street photograph",
}
ALLOWED_LICENSES = {
    "cc0",
    "public domain",
    "cc by 2.0",
    "cc by 3.0",
    "cc by 4.0",
    "cc by-sa 2.0",
    "cc by-sa 3.0",
    "cc by-sa 4.0",
}
ARTWORK_TERMS = re.compile(
    r"\b(painting|painted|drawing|engraving|illustration|sculpture|statue|"
    r"poster|cartoon|mural|artwork|oil on canvas)\b",
    re.IGNORECASE,
)


def read_url(request, timeout=60):
    global LAST_REQUEST_AT
    for attempt in range(5):
        wait = REQUEST_DELAY_SECONDS - (time.monotonic() - LAST_REQUEST_AT)
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            LAST_REQUEST_AT = time.monotonic()
            return data
        except urllib.error.HTTPError as error:
            LAST_REQUEST_AT = time.monotonic()
            if error.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            retry_after = error.headers.get("Retry-After", "")
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 2 ** (attempt + 1)
            delay = min(max(delay, REQUEST_DELAY_SECONDS), 60)
            print(
                f"Commons requested a pause (HTTP {error.code}); retrying in "
                f"{delay:g}s.",
                flush=True,
            )
            time.sleep(delay)


def strip_html(value):
    text = re.sub(r"<[^>]*>", " ", html.unescape(value or ""))
    return " ".join(text.split())


def write_attribution(output_dir, credits):
    with (output_dir / "ATTRIBUTION.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as attribution:
        attribution.write("\n".join(credits[:4]) + "\n")
        writer = csv.writer(attribution, delimiter="\t", lineterminator="\n")
        writer.writerows(credits[4:])


def commons_search(query, result_count):
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{query} filemime:image/jpeg",
        "gsrnamespace": "6",
        "gsrlimit": str(min(50, max(result_count * 3, 25))),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": "640",
        "format": "json",
    }
    request = urllib.request.Request(
        API_URL + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": USER_AGENT},
    )
    payload = json.loads(read_url(request, timeout=45))
    pages = list(payload.get("query", {}).get("pages", {}).values())
    return sorted(pages, key=lambda page: page.get("index", 10_000))


def portrait_photo_pages(result_count):
    titles = []
    for category in (
        "Portrait photographs of women",
        "Portrait photographs of men",
        "Portrait photographs of people",
    ):
        continuation = {}
        category_titles = []
        while len(category_titles) < result_count * 2:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmtype": "file",
                "cmnamespace": "6",
                "cmlimit": "50",
                "format": "json",
                **continuation,
            }
            request = urllib.request.Request(
                API_URL + "?" + urllib.parse.urlencode(params),
                headers={"User-Agent": USER_AGENT},
            )
            payload = json.loads(read_url(request, timeout=45))
            category_titles.extend(
                page["title"]
                for page in payload.get("query", {}).get("categorymembers", [])
            )
            continuation = payload.get("continue", {})
            if not continuation:
                break
            time.sleep(REQUEST_DELAY_SECONDS)
        titles.extend(category_titles)

    titles = list(dict.fromkeys(titles))
    pages = []
    for start in range(0, len(titles), 40):
        params = {
            "action": "query",
            "titles": "|".join(titles[start : start + 40]),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": "640",
            "format": "json",
        }
        request = urllib.request.Request(
            API_URL + "?" + urllib.parse.urlencode(params),
            headers={"User-Agent": USER_AGENT},
        )
        payload = json.loads(read_url(request, timeout=45))
        pages.extend(payload.get("query", {}).get("pages", {}).values())
        time.sleep(REQUEST_DELAY_SECONDS)
    return pages


def valid_license(metadata):
    label = strip_html(metadata.get("LicenseShortName", {}).get("value", "")).lower()
    return label in ALLOWED_LICENSES


def get_photo_candidates(category, query, target_count):
    selected = []
    seen = set()
    for attempt in range(3):
        pages = (
            portrait_photo_pages(target_count)
            if category == "portrait"
            else commons_search(query, target_count)
        )
        for page in pages:
            title = page.get("title", "")
            page_id = str(page.get("pageid", ""))
            if not page_id or page_id in seen or ARTWORK_TERMS.search(title):
                continue
            info = (page.get("imageinfo") or [{}])[0]
            metadata = info.get("extmetadata", {})
            thumb_url = info.get("thumburl")
            mime = strip_html(metadata.get("MimeType", {}).get("value", ""))
            if (
                not thumb_url
                or not valid_license(metadata)
                or (mime and not mime.startswith("image/"))
            ):
                continue
            seen.add(page_id)
            selected.append(
                {
                    "id": page_id,
                    "title": title,
                    "url": thumb_url,
                    "pageUrl": info.get("descriptionurl", ""),
                    "license": strip_html(
                        metadata.get("LicenseShortName", {}).get("value", "")
                    ),
                    "licenseUrl": strip_html(
                        metadata.get("LicenseUrl", {}).get("value", "")
                    ),
                    "artist": strip_html(metadata.get("Artist", {}).get("value", "")),
                    "credit": strip_html(metadata.get("Credit", {}).get("value", "")),
                }
            )
            if len(selected) >= target_count:
                return selected
        if attempt < 2:
            time.sleep(1)
    if len(selected) < target_count:
        print(
            f"Found {len(selected)}/{target_count} licensed {category} images.",
            flush=True,
        )
    return selected


def download_photo(candidate, destination):
    request = urllib.request.Request(
        candidate["url"], headers={"User-Agent": USER_AGENT}
    )
    data = read_url(request)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    try:
        with Image.open(destination) as image:
            image.verify()
        with Image.open(destination) as image:
            image.load()
            return image.convert("RGB")
    except Exception as error:
        destination.unlink(missing_ok=True)
        raise ValueError(f"Downloaded thumbnail is not a readable image: {error}") from error


def random_settings(rng, index):
    settings = synthetic.random_settings(rng)
    settings.update(
        {
            "temp": rng.uniform(-28, 28),
            "tint": rng.uniform(-18, 18),
            "exposure": rng.uniform(-0.9, 0.9),
            "contrast": rng.uniform(-35, 40),
            "highlights": rng.uniform(-50, 35),
            "shadows": rng.uniform(-25, 50),
            "whites": rng.uniform(-35, 30),
            "blacks": rng.uniform(-30, 35),
            "vibrance": rng.uniform(-40, 50),
            "saturation": rng.uniform(-30, 25),
        }
    )
    return settings


def prepare(count_per_category, seed, categories=None, output_dir=OUTPUT_DIR):
    categories = categories or CATEGORIES
    photo_dir = output_dir / "photos"
    source_dir = output_dir / "images" / "source"
    reference_dir = output_dir / "images" / "reference"
    manifest_path = output_dir / "train.jsonl"
    rng = random.Random(seed)
    for path in (photo_dir, source_dir, reference_dir):
        path.mkdir(parents=True, exist_ok=True)

    candidates = []
    for category, query in categories.items():
        found = get_photo_candidates(category, query, count_per_category)
        candidates.extend((category, item) for item in found)
        print(f"{category}: found {len(found)} openly licensed photo candidates.", flush=True)

    if len(candidates) < 10:
        raise RuntimeError(
            "Commons returned fewer than 10 suitable photos. Try again later or "
            "provide a local image dataset."
        )

    split_by_group = {}
    per_category_index = {name: 0 for name in categories}
    for category, _ in candidates:
        index = per_category_index[category]
        per_category_index[category] += 1
        split_by_group[(category, index)] = (
            "validation" if index % 5 == 0 else "train"
        )

    rows = []
    credits = [
        "Images are downloaded from Wikimedia Commons thumbnails for local training.",
        "Only Public Domain, CC0, CC BY 2.0/3.0/4.0, and CC BY-SA 2.0/3.0/4.0 "
        "items are selected; attribution is recorded below.",
        "Generated reference images are synthetic derivatives. Keep this file "
        "with the local dataset and check each source page before redistributing.",
        "",
    ]
    counts = {name: 0 for name in categories}
    for source_index, (category, candidate) in enumerate(candidates):
        stem = f"{category}_{candidate['id']}"
        group_index = counts[category]
        counts[category] += 1
        source_relative = Path("images") / "source" / f"{stem}.jpg"
        reference_relative = Path("images") / "reference" / f"{stem}_edit.jpg"
        source_path = output_dir / source_relative
        reference_path = output_dir / reference_relative
        photo_path = photo_dir / category / f"{stem}.jpg"

        if source_path.is_file():
            with Image.open(source_path) as existing:
                source_image = existing.convert("RGB")
        else:
            source_image = download_photo(candidate, photo_path)
            source_image.thumbnail((640, 640), Image.Resampling.LANCZOS)
            source_image.save(source_path, "JPEG", quality=90)

        settings = random_settings(rng, source_index)
        if not reference_path.is_file():
            synthetic.apply_settings(source_image, settings).save(
                reference_path, "JPEG", quality=90
            )
        split = "validation" if group_index % 5 == 0 else "train"
        rows.append(
            {
                "source": source_relative.as_posix(),
                "reference": reference_relative.as_posix(),
                "settings": settings,
                "group": stem,
                "split": split,
                "category": category,
            }
        )
        credits.append(
            (
                candidate["title"],
                candidate["artist"],
                candidate["credit"],
                candidate["license"],
                candidate["licenseUrl"],
                candidate["pageUrl"],
                str(source_relative),
            )
        )
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for row in rows:
                manifest.write(json.dumps(row, separators=(",", ":")) + "\n")
        write_attribution(output_dir, credits)
        print(
            f"Prepared {category} photo {group_index + 1}/{per_category_index[category]}.",
            flush=True,
        )

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in rows:
            manifest.write(json.dumps(row, separators=(",", ":")) + "\n")
    attribution_path = output_dir / "ATTRIBUTION.tsv"
    write_attribution(output_dir, credits)
    print(
        json.dumps(
            {
                "examples": len(rows),
                "categories": counts,
                "trainingExamples": sum(row["split"] == "train" for row in rows),
                "validationExamples": sum(
                    row["split"] == "validation" for row in rows
                ),
                "manifest": str(manifest_path),
                "attribution": str(attribution_path),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download openly licensed portrait, landscape, architecture, and "
            "urban photos and create same-photo synthetic training pairs."
        )
    )
    parser.add_argument("--count-per-category", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--portrait-only",
        action="store_true",
        help="Prepare only portrait photographs in a separate dataset folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output dataset folder.",
    )
    args = parser.parse_args()
    maximum_count = 120 if args.portrait_only else 50
    if not 5 <= args.count_per_category <= maximum_count:
        parser.error(
            f"--count-per-category must be from 5 to {maximum_count} "
            f"{'portraits' if args.portrait_only else 'images per category'}."
        )
    categories = {"portrait": CATEGORIES["portrait"]} if args.portrait_only else None
    prepare(args.count_per_category, args.seed, categories, args.output_dir.resolve())


if __name__ == "__main__":
    main()
