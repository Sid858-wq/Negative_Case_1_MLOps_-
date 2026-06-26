"""train.py — Stage 2/3: transfer-learning training + MLflow tracking + registry. 

Implement: build splits, train the ResNet18 head (CrossEntropy, Adam on trainable params,
early stop on val F1), log params/metrics/model to MLflow, register + promote to the
@production alias, evaluate on test, and save a clean-data reference baseline (image
features + embeddings) for drift monitoring.   Run: python -m src.train
"""
from __future__ import annotations

import json, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep, evaluate
from src.dataset import CastingDataset
from src.model import build_model, trainable_parameters, save_model, EmbeddingExtractor
from torch.utils.data import DataLoader


def set_seed(seed: int = config.RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _subsample(items, cap):
    # TODO: stratified subsample to `cap` (or return items if cap falsy/too small).
    def _subsample(items, cap: int):
        if not cap or len(items) <= cap:
            return items
    rng = random.Random(config.RANDOM_SEED)
    by = {0: [], 1: []}
    for it in items:
        by[it[1]].append(it)
    per = cap // 2
    out = []
    for y, lst in by.items():
        rng.shuffle(lst); out += lst[:per]
    rng.shuffle(out)
    return out
    # raise NotImplementedError


def class_weights(items) -> torch.Tensor:
    # TODO: inverse-frequency class weights for CrossEntropyLoss.
    from collections import Counter
    c = Counter(y for _, y in items)
    n = sum(c.values())
    w = [n / (config.NUM_CLASSES * c.get(i, 1)) for i in range(config.NUM_CLASSES)]
    return torch.tensor(w, dtype=torch.float32, device=config.DEVICE)

    # raise NotImplementedError

def evaluate_loader(net, loader):
    yt, yp, pr = evaluate.predict(net, loader)
    return evaluate.compute_metrics(yt, yp, pr)

def save_reference_baseline(net, ref_items) -> dict:
    # TODO 4: save reference_features.csv + reference_embeddings.npz for clean ref images.
    from PIL import Image
    tf = data_prep.get_transforms(train=False)
    extractor = EmbeddingExtractor(net).to(config.DEVICE).eval()
    feats, embs = [], []
    with torch.no_grad():
        for path, _ in ref_items:
            with Image.open(path) as im:
                g = im.convert("L")
                feats.append(data_prep.image_features(g))
                x = tf(g).unsqueeze(0).to(config.DEVICE)
                embs.append(extractor(x).cpu().numpy()[0])
    import csv
    with open(config.ARTIFACT_DIR / "reference_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=config.DRIFT_FEATURES)
        w.writeheader(); w.writerows(feats)
    np.savez_compressed(config.REFERENCE_EMBED, embeddings=np.array(embs))
    return {"reference_size": len(ref_items)}

    # raise NotImplementedError


# def main() -> int:
#     import mlflow, mlflow.pytorch
#     from mlflow import MlflowClient
#     set_seed()
#     root = data_prep.find_data_root()
#     qc = data_prep.validate_quality(root)
#     (config.ARTIFACT_DIR / "data_quality_report.json").write_text(json.dumps(qc, indent=2))
#     data_prep.build_splits(root, "v1")
#     # TODO 2/3: load splits, subsample train, build loaders, build model + optimiser + loss.
#     # TODO 3 (MLflow): set_experiment; start_run; log_params; per-epoch log_metrics; early stop.
#     # TODO 3 (eval + registry): test metrics; plot_eval; save_model; log_model + register +
#     #         set @production alias; write model_meta.json + metrics.json.
#     # TODO 4: save_reference_baseline on a clean val sample.

#     raise NotImplementedError("Implement the training + MLflow + registry workflow")


def main() -> int:
    import mlflow
    import mlflow.pytorch
    from mlflow import MlflowClient

    set_seed()
    log = print
    root = data_prep.find_data_root()
    log(f"[data] root = {root}")

    # quality validation (saved report)
    qc = data_prep.validate_quality(root)
    (config.ARTIFACT_DIR / "data_quality_report.json").write_text(json.dumps(qc, indent=2))
    log(f"[data] {qc['total_images']} images | quality passed = {qc['passed']} | "
        f"duplicates = {qc['issues']['duplicate_count']}")

    meta = data_prep.build_splits(root, version="v1")
    log(f"[data] splits: " + ", ".join(f"{k}={v['count']}" for k, v in meta['split_info'].items()))

    train_items = _subsample(data_prep.load_split("v1", "train", root), config.MAX_TRAIN_IMAGES)
    val_items = data_prep.load_split("v1", "val", root)
    test_items = data_prep.load_split("v1", "test", root)
    log(f"[train] using {len(train_items)} train images (cap={config.MAX_TRAIN_IMAGES})")

    g = torch.Generator(); g.manual_seed(config.RANDOM_SEED)
    train_loader = DataLoader(CastingDataset(train_items, True), batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS, generator=g)
    val_loader = DataLoader(CastingDataset(val_items, False), batch_size=config.BATCH_SIZE,
                            num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(CastingDataset(test_items, False), batch_size=config.BATCH_SIZE,
                             num_workers=config.NUM_WORKERS)

    net = build_model().to(config.DEVICE)
    opt = torch.optim.Adam(trainable_parameters(net), lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss(weight=class_weights(train_items))

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    best_f1, best_state, patience = -1.0, None, 0

    with mlflow.start_run() as run:
        mlflow.log_params({
            "backbone": config.BACKBONE, "freeze_backbone": config.FREEZE_BACKBONE,
            "img_size": config.IMG_SIZE, "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE, "lr": config.LEARNING_RATE,
            "train_images": len(train_items), "seed": config.RANDOM_SEED,
        })
        for epoch in range(1, config.EPOCHS + 1):
            net.train(); running = 0.0
            for x, y in train_loader:
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                opt.zero_grad()
                loss = crit(net(x), y)
                loss.backward(); opt.step()
                running += loss.item() * x.size(0)
            tr_loss = running / len(train_items)
            vm = evaluate_loader(net, val_loader)
            mlflow.log_metrics({"train_loss": tr_loss, "val_f1_defect": vm["f1_defect"],
                                "val_accuracy": vm["accuracy"], "val_recall_defect": vm["recall_defect"]},
                               step=epoch)
            log(f"[epoch {epoch}] loss={tr_loss:.4f} val_acc={vm['accuracy']} "
                f"val_f1={vm['f1_defect']} val_recall={vm['recall_defect']}")
            if vm["f1_defect"] > best_f1:
                best_f1, best_state, patience = vm["f1_defect"], {k: v.cpu().clone() for k, v in net.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= config.EARLY_STOP_PATIENCE:
                    log(f"[train] early stop at epoch {epoch}"); break

        if best_state is not None:
            net.load_state_dict(best_state)

        test_m = evaluate_loader(net, test_loader)
        log(f"[test] {test_m}")
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()
                            if isinstance(v, (int, float))})
        evaluate.plot_eval(*evaluate.predict(net, test_loader))
        mlflow.log_artifact(str(config.ARTIFACT_DIR / "model_eval.png"))

        # persist + register + promote
        save_model(net)
        meta_out = {"registered_model": config.REGISTERED_MODEL, "run_id": run.info.run_id,
                    "test_metrics": test_m, "val_f1_defect": best_f1,
                    "classes": config.CLASS_TO_IDX, "img_size": config.IMG_SIZE}
        config.MODEL_META_PATH.write_text(json.dumps(meta_out, indent=2))
        config.METRICS_PATH.write_text(json.dumps(test_m, indent=2))

        try:
            mlflow.pytorch.log_model(
                net,
                name="model",
                registered_model_name=config.REGISTERED_MODEL)

            client = MlflowClient()
            versions = client.search_model_versions(
                f"name='{config.REGISTERED_MODEL}'")
            latest = max(int(v.version) for v in versions)

            log(f"[mlflow] registered {config.REGISTERED_MODEL} v{latest}")
        except Exception as e:                       # registry is best-effort, training still valid
            log(f"[mlflow] registry step skipped: {e}")

    # reference baseline for drift monitoring (clean validation sample)
    ref = _subsample(val_items, 400)
    save_reference_baseline(net, ref)
    log(f"[drift] saved reference baseline ({len(ref)} clean images)")
    log("[done] training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
