"""monitoring.py — Stage 4: statistical + embedding drift + confidence monitoring.

Implement the three drift signals against the clean reference baseline:
  1. statistical drift  — Evidently DataDriftPreset + PSI on image features
  2. embedding drift    — PSI on ResNet-embedding distance-to-centroid distribution
  3. confidence         — mean predicted confidence reference vs current
Use a corrupted copy of clean images as the simulated "current" production batch.
Outputs drift_report.html + drift_summary.json.   Run: python -m src.monitoring

Embedding drift (TODO 4) — see the conceptual walkthrough in
Operations_Monitoring_and_Evidence.ipynb (Stage 4.3):
  1. feature extraction  — penultimate 512-dim ResNet embedding (model.EmbeddingExtractor)
  2. embedding generation — embeddings for reference + current batches
  3. feature-space compare — reduce each to distance-to-reference-centroid (one distribution per batch)
  4. drift calculation   — PSI between the two distance distributions (> ~0.10 => drifted)
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep
from src.model import load_model, EmbeddingExtractor


def psi(reference, current, bins: int = 10) -> float:
    # TODO 4: Population Stability Index between two 1-D distributions (quantile bins).
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / max(len(ref), 1)
    c = np.histogram(cur, edges)[0] / max(len(cur), 1)
    r = np.clip(r, 1e-6, None); c = np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))
    # raise NotImplementedError


def corrupt(img: Image.Image) -> Image.Image:
    # TODO 4: simulate camera/lighting drift (brightness/blur/rotate/noise via config.DRIFT_SIM).
    s = config.DRIFT_SIM
    g = img.convert("L")
    g = ImageEnhance.Brightness(g).enhance(s["brightness"])
    g = g.filter(ImageFilter.GaussianBlur(s["blur_radius"]))
    g = g.rotate(s["rotate"])
    arr = np.asarray(g, float) + np.random.normal(0, s["noise_std"], np.asarray(g).shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    # raise NotImplementedError

def _batch(items, apply_corruption: bool):
    import random as _random
    net = load_model()
    extractor = EmbeddingExtractor(net).eval()
    tf = data_prep.get_transforms(train=False)
    rng = _random.Random(config.RANDOM_SEED)
    drift_fraction = getattr(config, "CURRENT_DRIFT_FRACTION", 0.5)
    import torch
    import torch.nn.functional as F
    feats, embs, confs = [], [], []
    with torch.no_grad():
        for path, _ in items:
            with Image.open(path) as im:
                g = im.convert("L")
            if apply_corruption and rng.random() < drift_fraction:
                g = corrupt(g)
            feats.append(data_prep.image_features(g))
            x = tf(g).unsqueeze(0)
            embs.append(extractor(x).numpy()[0])
            confs.append(float(F.softmax(net(x), dim=1).max().item()))
    return pd.DataFrame(feats), np.array(embs), float(np.mean(confs))

def run() -> dict:
    # TODO 4: build reference (clean) + current (corrupted) batches → features, embeddings,
    #         mean confidence. Run Evidently DataDriftPreset + PSI; embedding PSI on
    #         distance-to-centroid; confidence drop. Write drift_summary.json + drift_report.html
    #         and set retrain_recommended from the configured thresholds.
    np.random.seed(config.RANDOM_SEED)
    root = data_prep.find_data_root()
    # reference = clean validation sample; current = corrupted copy of a test sample
    from src.train import _subsample
    ref_items = _subsample(data_prep.load_split("v1", "val", root), 300)
    cur_items = _subsample(data_prep.load_split("v1", "test", root), 300)

    ref_df, ref_emb, ref_conf = _batch(ref_items, apply_corruption=False)
    cur_df, cur_emb, cur_conf = _batch(cur_items, apply_corruption=True)

    # 1. statistical drift — Evidently + PSI
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df[config.DRIFT_FEATURES],
               current_data=cur_df[config.DRIFT_FEATURES])
    report.save_html(str(config.ARTIFACT_DIR / "drift_report.html"))
    ev = report.as_dict()["metrics"][0]["result"]
    n_drifted = int(ev["number_of_drifted_columns"])
    feat_psi = {f: round(psi(ref_df[f].values, cur_df[f].values), 4)
                for f in config.DRIFT_FEATURES}

    # 2. embedding drift — PSI on distance-to-reference-centroid
    centroid = ref_emb.mean(axis=0)
    ref_dist = np.linalg.norm(ref_emb - centroid, axis=1)
    cur_dist = np.linalg.norm(cur_emb - centroid, axis=1)
    embed_psi = round(psi(ref_dist, cur_dist), 4)

    # 3. confidence monitoring
    conf_drop = round(ref_conf - cur_conf, 4)

    drift_share = n_drifted / len(config.DRIFT_FEATURES)
    summary = {
        "statistical_drift": {
            "n_features": len(config.DRIFT_FEATURES),
            "n_drifted": n_drifted,
            "drift_share": round(drift_share, 3),
            "dataset_drift": bool(ev["dataset_drift"]),
            "feature_psi": feat_psi,
        },
        "embedding_drift": {
            "psi": embed_psi,
            "threshold": config.EMBEDDING_DRIFT_THRESHOLD,
            "drifted": embed_psi > config.EMBEDDING_DRIFT_THRESHOLD,
        },
        "confidence": {
            "reference_mean": round(ref_conf, 4),
            "current_mean": round(cur_conf, 4),
            "drop": conf_drop,
            "alert": conf_drop > config.CONFIDENCE_DROP_THRESHOLD,
        },
        "retrain_recommended": bool(
            drift_share >= config.DRIFT_SHARE_THRESHOLD
            or embed_psi > config.EMBEDDING_DRIFT_THRESHOLD
            or conf_drop > config.CONFIDENCE_DROP_THRESHOLD),
    }
    (config.ARTIFACT_DIR / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[drift] statistical: {n_drifted}/{len(config.DRIFT_FEATURES)} features drifted "
          f"(share {drift_share:.2f}) | embedding PSI {embed_psi} | "
          f"confidence {ref_conf:.3f}→{cur_conf:.3f} (drop {conf_drop}) | "
          f"retrain={summary['retrain_recommended']}")
    return summary
    # raise NotImplementedError


if __name__ == "__main__":
    run()
