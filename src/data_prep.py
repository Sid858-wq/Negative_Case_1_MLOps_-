"""data_prep.py — Stage 1: discovery, quality validation, versioned splits, transforms. 

Implement: locate the casting folders; run data-quality checks (missing/corrupt/duplicate/
dimension/class-distribution/consistency); build reproducible stratified train/val/test
splits with a versioned snapshot + metadata.json; define preprocessing + augmentation
transforms; and per-image feature extraction used by drift monitoring.
"""
from __future__ import annotations

import hashlib, json, os, random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_data_root(base: Path | None = None) -> Path:
    # TODO 1: search under (base or config.DATA_DIR) for a dir whose train/ has both
    #         ok_front and def_front subfolders; return it.
    base = base or config.DATA_DIR
    for d in [base, *base.rglob("*")]:
        if not d.is_dir():
            continue
        train = d / "train"
        if (train / "ok_front").is_dir() and (train / "def_front").is_dir():
            return d
    raise FileNotFoundError(
        f"Could not find train/{{ok_front,def_front}} under {base}. "
        f"Extract the Kaggle casting archive into {config.DATA_DIR}.")
    # raise NotImplementedError("Locate the casting train/test root")


def list_images(split_dir: Path) -> list[tuple[Path, int]]:
    items = []
    for cls, idx in config.CLASS_TO_IDX.items():
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                items.append((p, idx))
    return items

def _file_hash(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()

def validate_quality(root: Path) -> dict:
    # TODO 1 (data quality): for train+test, count images + class distribution; open each
    #         image (catch corrupt); record non-300x300 dims; md5-hash to find duplicates.
    #         Return a report dict with issues + a 'passed' flag.
    report = {"splits": {}, "issues": {}}
    all_hashes: dict[str, str] = {}
    duplicates, corrupt, odd_dims = [], [], []
    dim_counter = Counter()

    for split in ("train", "test"):
        sdir = root / split
        if not sdir.is_dir():
            continue
        items = list_images(sdir)
        dist = Counter(config.IDX_TO_CLASS[i] for _, i in items)
        report["splits"][split] = {"images": len(items), "class_distribution": dict(dist)}
        for p, _ in items:
            try:
                with Image.open(p) as im:
                    im.verify()                      # detects truncated/corrupt files
                with Image.open(p) as im:
                    dim_counter[im.size] += 1
                    if im.size != (300, 300):
                        odd_dims.append((str(p.name), im.size))
            except (UnidentifiedImageError, OSError):
                corrupt.append(str(p))
                continue
            h = _file_hash(p)
            if h in all_hashes:
                duplicates.append((p.name, Path(all_hashes[h]).name))
            else:
                all_hashes[h] = str(p)

    report["issues"] = {
        "corrupt_files": corrupt,
        "duplicate_pairs": duplicates[:50],
        "duplicate_count": len(duplicates),
        "non_300x300": odd_dims[:50],
        "non_300x300_count": len(odd_dims),
    }
    report["dimensions"] = {f"{w}x{h}": n for (w, h), n in dim_counter.items()}
    report["total_images"] = sum(s["images"] for s in report["splits"].values())
    report["passed"] = len(corrupt) == 0
    return report

    # raise NotImplementedError("Implement data-quality validation")


def build_splits(root: Path, version: str = "v1") -> dict:
    # TODO 1 (versioning): stratified val carve-out from train/; test from test/. Save
    #         {train,val,test}.json file lists + metadata.json (version, date, class defs,
    #         split sizes/distribution, seed) under config.SPLIT_DIR/<version>/.
    rng = random.Random(config.RANDOM_SEED)
    train_items = list_images(root / "train")
    test_items = list_images(root / "test")

    # stratified val carve-out
    by_cls: dict[int, list] = {0: [], 1: []}
    for p, y in train_items:
        by_cls[y].append((p, y))
    train_split, val_split = [], []
    for y, items in by_cls.items():
        rng.shuffle(items)
        k = int(len(items) * config.VAL_SPLIT)
        val_split += items[:k]
        train_split += items[k:]
    rng.shuffle(train_split)
    rng.shuffle(val_split)

    splits = {"train": train_split, "val": val_split, "test": test_items}

    out = config.SPLIT_DIR / version
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, items in splits.items():
        rel = [[str(p.relative_to(root)), y] for p, y in items]
        (out / f"{name}.json").write_text(json.dumps(rel))
        dist = Counter(config.IDX_TO_CLASS[y] for _, y in items)
        stats[name] = {"count": len(items), "class_distribution": dict(dist)}

    metadata = {
    "split_info": {
        k: {"count": v["count"]}
        for k, v in stats.items()
    }
}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
    # raise NotImplementedError("Build versioned stratified splits + metadata")


def load_split(version: str, name: str, root: Path) -> list[tuple[Path, int]]:
    rel = json.loads((config.SPLIT_DIR / version / f"{name}.json").read_text())
    return [(root / r, y) for r, y in rel]


def get_transforms(train: bool):
    from torchvision import transforms
    # TODO 2 (preprocessing + augmentation): Grayscale(3) → Resize(224) → [train: flip,
    #         affine rotation/translate, ColorJitter] → ToTensor → Normalize(ImageNet).
    base = [
        transforms.Grayscale(num_output_channels=3),     # 1→3 channels for the backbone
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    ]
    if train:
        a = config.AUG
        base += [
            transforms.RandomHorizontalFlip(a["hflip_p"]),
            transforms.RandomAffine(degrees=a["rotation_degrees"],
                                    translate=(a["translate"], a["translate"])),
            transforms.ColorJitter(brightness=a["brightness"], contrast=a["contrast"]),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ]
    return transforms.Compose(base)
    # raise NotImplementedError("Define the train/eval transforms")


def image_features(img: Image.Image) -> dict:
    # TODO 4 (statistical drift): return brightness, contrast, edge_density, sharpness,
    #         mean_intensity for a PIL image (keys == config.DRIFT_FEATURES).
    g = img.convert("L")
    stat = ImageStat.Stat(g)
    arr = np.asarray(g, dtype=np.float32)
    edges = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    lap = np.asarray(g.filter(ImageFilter.Kernel((3, 3),
                     [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1)), dtype=np.float32)
    return {
        "brightness": float(stat.mean[0]),
        "contrast": float(stat.stddev[0]),
        "edge_density": float(edges.mean()),
        "sharpness": float(lap.var()),
        "mean_intensity": float(arr.mean()),
    }
    # raise NotImplementedError("Extract per-image drift features")
