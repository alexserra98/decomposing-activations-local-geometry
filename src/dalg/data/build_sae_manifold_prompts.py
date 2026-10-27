"""Build the prompt datasets from "Do Sparse Autoencoders Capture Concept Manifolds?".

The build_* functions adapt Goodfire's prompt-generation logic.
The years dataset uses 20 neutral templates for each year from 1000 through 2000.
WikiText uses its canonical Salesforce/wikitext ID for Hugging Face compatibility.
Upstream data.py:
https://github.com/goodfire-ai/sae-manifold/blob/f2632ddbb25f4c670254ceea999058b9ba4b0450/data.py

Run from the repository root:
    PYTHONPATH=src python -m dalg.data.build_sae_manifold_prompts
    PYTHONPATH=src python -m dalg.data.build_sae_manifold_prompts --manifold age days --out dalg-cache/sae_manifold_prompts_preview

Writes prompts.jsonl and metadata.json, without tokenization or model loading.
Both output files must be absent. Hugging Face downloads default to scratch
through dalg-cache/huggingface; --cache-dir overrides that location.
"""

# Upstream dataset builders are distributed under the following license:
# MIT License
#
# Copyright (c) 2026 Usha Bhalla and the authors of "Do Sparse Autoencoders Capture Concept Manifolds?"
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import argparse
import colorsys
import json
import os
from pathlib import Path


UPSTREAM_REVISION = "f2632ddbb25f4c670254ceea999058b9ba4b0450"
UPSTREAM_URL = (
    f"https://github.com/goodfire-ai/sae-manifold/blob/{UPSTREAM_REVISION}/data.py"
)
DEFAULT_OUT = Path("dalg-cache/sae_manifold_prompts")
DEFAULT_CACHE_DIR = Path("dalg-cache/huggingface")


def build_colors():
    from datasets import load_dataset
    ds = load_dataset("burkelibbey/colors", split="train")
    prompts, labels = [], []
    for i, row in enumerate(ds):
        if i % 10 != 0:
            continue
        h_code = row['color']
        r, g, b = (int(h_code.lstrip('#')[j:j+2], 16) / 255 for j in (0, 2, 4))
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        if l < 0.2 or l > 0.8 or s < 0.2 or s > 0.8:
            continue
        prompts.append(f"The hex code {h_code} is for the color")
        labels.append(dict(hex=h_code, hue=h, lightness=l, saturation=s))
    return prompts, labels


def build_age():
    prompts, labels = [], []
    for age in range(1, 100):
        prompts.append(f"They are {age} years old.")
        labels.append(dict(age=float(age)))
    return prompts, labels


def build_temperature():
    """Append .01–.99 to each integer temperature, retaining integer prompts."""
    prompts, labels = [], []
    for whole in range(-30, 120):
        for fraction in range(100):
            value = str(whole) if fraction == 0 else f"{whole}.{fraction:02d}"
            f = float(value)
            prompts.append(f"Today it's {value} degrees Fahrenheit outside")
            labels.append(dict(fahrenheit=f, celsius=(f - 32) * 5 / 9))
    return prompts, labels


def build_geography():
    from datasets import load_dataset
    ds = load_dataset("jamescalam/world-cities-geo", split="train")
    prompts, labels = [], []
    for i, row in enumerate(ds):
        if i % 3 != 0:
            continue
        prompts.append(
            f"The geographical coordinates latitude {row['latitude']} "
            f"and longitude {row['longitude']} are in the country of"
        )
        labels.append(dict(
            latitude=row['latitude'], longitude=row['longitude'],
            country=row['country'], region=row['region'],
            continent=row['continent'],
        ))
    return prompts, labels


def build_days():
    """Day × time-of-day grid — 7 days × 8 times = 56 prompts.

    Single template ``"It's {time} on {day}"`` so the resulting manifold is a
    clean 2D discrete grid: ``day_idx`` (cyclic) × ``time_idx`` (ordinal).
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]
    times = [
        "the crack of dawn", "early morning", "late morning", "noon",
        "afternoon", "evening", "night", "midnight",
    ]
    prompts, labels = [], []
    for di, day in enumerate(days):
        for ti, time in enumerate(times):
            prompts.append(f"It's {time} on {day}")
            labels.append(dict(day=day, day_idx=di,
                               time=time, time_idx=ti))
    return prompts, labels


YEAR_TEMPLATES = (
    "The year is {year}",
    "According to this record, the year is {year}",
    "For this entry, the year is {year}",
    "In this account, the year is {year}",
    "As noted here, the year is {year}",
    "The calendar indicates that the year is {year}",
    "As recorded here, the year is {year}",
    "According to this entry, the year is {year}",
    "In this record, the year is {year}",
    "For the record, the year is {year}",
    "As indicated here, the year is {year}",
    "As stated in this entry, the year is {year}",
    "The record indicates that the year is {year}",
    "The entry indicates that the year is {year}",
    "The notation indicates that the year is {year}",
    "For this record, the year is {year}",
    "According to the calendar, the year is {year}",
    "In the calendar shown here, the year is {year}",
    "As specified in this record, the year is {year}",
    "Based on this entry, the year is {year}",
)


def build_years():
    """Years 1000–2000 inclusive crossed with neutral year-ending templates."""
    prompts, labels = [], []
    for year in range(1000, 2001):
        for template_id, template in enumerate(YEAR_TEMPLATES):
            prompts.append(template.format(year=year))
            labels.append(dict(
                year=float(year),
                decade_digit=float(year % 10),
                decade=float((year % 100) // 10),
                template_id=template_id,
            ))
    return prompts, labels


def build_formality():
    from datasets import load_dataset
    ds = load_dataset("osyvokon/pavlick-formality-scores", split="train")
    prompts, labels = [], []
    for row in ds:
        text = row['sentence']
        score = row['avg_score']
        if not text or not text.strip() or len(text.strip()) < 3:
            continue
        prompts.append(text.strip())
        labels.append(dict(formality=float(score)))
        if len(prompts) >= 1000:
            break
    return prompts, labels


def build_sent_length():
    """WikiText sentences labeled by whitespace token count (3-80)."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    prompts, labels = [], []
    seen = set()
    for row in ds:
        text = row['text'].strip()
        if not text or len(text) < 10 or text.startswith('='):
            continue
        for sent in text.replace('? ', '.\n').replace('! ', '.\n').split('.\n'):
            sent = sent.strip()
            if not sent or len(sent) < 10:
                continue
            if not sent.endswith(('.', '?', '!')):
                sent = sent + '.'
            n_tokens = len(sent.split())
            if n_tokens < 3 or n_tokens > 80 or sent in seen:
                continue
            seen.add(sent)
            prompts.append(sent)
            labels.append(dict(n_tokens=float(n_tokens), n_chars=float(len(sent))))
            if len(prompts) >= 5000:
                break
        if len(prompts) >= 5000:
            break
    return prompts, labels


MANIFOLDS = dict(
    colors=build_colors,
    age=build_age,
    temperature=build_temperature,
    geography=build_geography,
    days=build_days,
    years=build_years,
    formality=build_formality,
    sent_length=build_sent_length,
)

# Selection settings describe the builders above, not CLI overrides.
SOURCE_SETTINGS = {
    "colors": {
        "dataset": "burkelibbey/colors", "config": None, "split": "train",
        "selection": {"stride": 10, "lightness": [0.2, 0.8], "saturation": [0.2, 0.8]},
    },
    "age": {
        "dataset": None, "config": None, "split": None,
        "selection": {"age_range": [1, 100], "stop_exclusive": True},
    },
    "temperature": {
        "dataset": None, "config": None, "split": None,
        "selection": {
            "integer_part_range": [-30, 120], "stop_exclusive": True,
            "fractional_suffix_range": [0, 100], "decimal_places": 2,
            "retain_integer_prompts": True, "order": "integer_then_fractional_suffix",
        },
    },
    "geography": {
        "dataset": "jamescalam/world-cities-geo", "config": None, "split": "train",
        "selection": {"stride": 3},
    },
    "days": {
        "dataset": None, "config": None, "split": None,
        "selection": {"n_days": 7, "n_times": 8, "order": "day_then_time"},
    },
    "years": {
        "dataset": None, "config": None, "split": None,
        "selection": {
            "year_range": [1000, 2001], "stop_exclusive": True,
            "granularity": "year", "order": "year_then_template",
            "templates": YEAR_TEMPLATES,
        },
    },
    "formality": {
        "dataset": "osyvokon/pavlick-formality-scores", "config": None, "split": "train",
        "selection": {"limit": 1000, "min_stripped_chars": 3},
    },
    "sent_length": {
        "dataset": "Salesforce/wikitext", "config": "wikitext-103-raw-v1", "split": "train",
        "selection": {
            "limit": 5000, "whitespace_word_count": [3, 80],
            "min_chars": 10, "deduplicate": True,
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output directory; existing output files are never overwritten.")
    parser.add_argument("--manifold", nargs="+", choices=list(MANIFOLDS),
                        help="Manifolds to build (default: all, in upstream registry order).")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Hugging Face download/cache root (default: dalg-cache/huggingface).")
    args = parser.parse_args()

    prompts_path = args.out / "prompts.jsonl"
    metadata_path = args.out / "metadata.json"
    for path in (prompts_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new --out directory.")

    # Set these before the builders import datasets/huggingface_hub.
    cache_dir = args.cache_dir.resolve()
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    names = [name for name in MANIFOLDS if args.manifold is None or name in args.manifold]

    rows, counts = [], {}
    for name in names:
        print(f"Building {name}...", flush=True)
        prompts, labels = MANIFOLDS[name]()
        if not prompts:
            raise ValueError(f"{name} produced no prompts.")
        for prompt, label in zip(prompts, labels, strict=True):
            rows.append({
                "row_id": len(rows), "subset": name, "prompt": prompt, "labels": label,
            })
        counts[name] = len(prompts)
        print(f"  {name}: {len(prompts):,} prompts", flush=True)

    import datasets

    metadata = {
        "upstream": {"url": UPSTREAM_URL, "revision": UPSTREAM_REVISION},
        "manifolds": names,
        "sources": {name: SOURCE_SETTINGS[name] for name in names},
        "generation": {
            "builder_module": "dalg.data.build_sae_manifold_prompts",
            "cache_dir": str(cache_dir),
            "datasets_version": datasets.__version__,
            "order": "upstream registry order, then builder output order",
            "postprocessing": "none; preserve prompt whitespace, labels, and duplicates",
        },
        "counts": counts,
        "total_rows": len(rows),
        "data_file": prompts_path.name,
        "row_fields": {
            "row_id": "Zero-based contiguous row index across all selected manifolds.",
            "subset": "Upstream manifold name.",
            "prompt": "Exact prompt returned by the builder.",
            "labels": "Manifold-specific label dictionary.",
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with prompts_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    print(f"Saved {len(rows):,} prompts to {prompts_path}", flush=True)
    print(f"Metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
