"""retrain.py — Stage 4: drift-triggered retraining, version compare, rollback.

Implement: read drift_summary.json; if retraining is recommended, measure the production
model on a drifted batch, train a drift-augmented candidate, compare, then PROMOTE the
candidate to @production only if it improves (within PROMOTE_EPSILON) else ROLL BACK.
Record retraining_decision.json + manage MLflow registry versions.   Run: python -m src.retrain
"""
from __future__ import annotations

import json, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep, evaluate
from src.model import build_model, trainable_parameters, load_model, save_model
from src.monitoring import corrupt
from src.train import _subsample, set_seed, class_weights


# def main() -> int:
#     drift = json.loads((config.ARTIFACT_DIR / "drift_summary.json").read_text())
#     if not drift.get("retrain_recommended"):
#         # TODO 4: record a 'no_retrain' decision and return.
#         raise NotImplementedError
#     # TODO 4: evaluate current production on a fully-drifted eval batch (corrupt_frac=1.0).
#     # TODO 4: train a drift-augmented candidate (corrupt_frac~0.4, few epochs, capped).
#     # TODO 4: compare candidate vs production F1 on the drifted batch.
#     # TODO 4: promote candidate→@production iff cand_f1 >= prod_f1 + PROMOTE_EPSILON, else
#     #         rollback (keep incumbent). Register the candidate version; move/keep the alias.
#     # TODO 4: write retraining_decision.json (scores, action, version history).
#     raise NotImplementedError("Implement retraining + version compare + rollback")

class _MaybeCorruptDataset(Dataset):
    """Casting dataset that corrupts a fraction of images (drift-aware training)."""
    def __init__(self, items, train, corrupt_frac=0.0, seed=config.RANDOM_SEED):
        self.items = items
        self.tf = data_prep.get_transforms(train=train)
        self.corrupt_frac = corrupt_frac
        self.rng = random.Random(seed)
        self.flags = [self.rng.random() < corrupt_frac for _ in items]

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as im:
            g = im.convert("L")
        if self.flags[i]:
            g = corrupt(g)
        return self.tf(g), label


def _eval_on(net, items):
    ds = _MaybeCorruptDataset(items, train=False, corrupt_frac=1.0)   # fully drifted batch
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE)
    yt, yp, pr = evaluate.predict(net, loader)
    return evaluate.compute_metrics(yt, yp, pr)


def _train_candidate(train_items, epochs=3, cap=1200):
    set_seed()
    items = _subsample(train_items, cap)
    ds = _MaybeCorruptDataset(items, train=True, corrupt_frac=0.4)    # drift-augmented
    g = torch.Generator(); g.manual_seed(config.RANDOM_SEED)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, generator=g)
    net = build_model().to(config.DEVICE)
    opt = torch.optim.Adam(trainable_parameters(net), lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss(weight=class_weights(items))
    for ep in range(1, epochs + 1):
        net.train()
        for x, y in loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            opt.zero_grad(); crit(net(x), y).backward(); opt.step()
        print(f"[retrain] candidate epoch {ep}/{epochs} done")
    return net


def main() -> int:
    drift = json.loads((config.ARTIFACT_DIR / "drift_summary.json").read_text())
    psi_trigger = (drift["statistical_drift"]["drift_share"]>= config.DRIFT_SHARE_THRESHOLD)
    decision = {"trigger": psi_trigger,"drift_share": drift["statistical_drift"]["drift_share"],
                "embedding_psi": drift["embedding_drift"]["psi"],
                "confidence_drop": drift["confidence"]["drop"],
}
    if not psi_trigger:
        decision["action"] = "no_retrain"
        (config.ARTIFACT_DIR / "retraining_decision.json").write_text(json.dumps(decision, indent=2))
        print("[retrain] no drift trigger — skipping retraining"); return 0

    root = data_prep.find_data_root()
    train_items = data_prep.load_split("v1", "train", root)
    eval_items = _subsample(data_prep.load_split("v1", "test", root), 300)

    production = load_model()
    prod_m = _eval_on(production, eval_items)
    print(f"[retrain] production on drifted batch: f1={prod_m['f1_defect']} acc={prod_m['accuracy']}")

    candidate = _train_candidate(train_items)
    cand_m = _eval_on(candidate, eval_items)
    print(f"[retrain] candidate on drifted batch:  f1={cand_m['f1_defect']} acc={cand_m['accuracy']}")

    improved = cand_m["f1_defect"] >= prod_m["f1_defect"] + config.PROMOTE_EPSILON
    decision.update({
        "production_drifted_f1": prod_m["f1_defect"],
        "candidate_drifted_f1": cand_m["f1_defect"],
        "promote_epsilon": config.PROMOTE_EPSILON,
        "action": "promote_candidate" if improved else "rollback_keep_production",
    })

    # registry version management + alias move (best-effort)
    try:
        import mlflow, mlflow.pytorch
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        client = MlflowClient()
        with mlflow.start_run(run_name="retrain_candidate"):
            mlflow.log_metrics({"production_drifted_f1": prod_m["f1_defect"],
                                "candidate_drifted_f1": cand_m["f1_defect"]})
            mlflow.pytorch.log_model(candidate, name="model",
                                     registered_model_name=config.REGISTERED_MODEL)
        versions = client.search_model_versions(f"name='{config.REGISTERED_MODEL}'")
        cand_version = max(int(v.version) for v in versions)
        decision["candidate_version"] = cand_version
        if improved:
            client.set_registered_model_alias(config.REGISTERED_MODEL,
                                               config.PRODUCTION_ALIAS, cand_version)
            save_model(candidate)                       # promote local weights too
            decision["production_version"] = cand_version
        else:
            prev = [int(v.version) for v in versions if int(v.version) < cand_version]
            decision["production_version"] = max(prev) if prev else cand_version
        print(f"[retrain] {decision['action']} → production = v{decision['production_version']}")
    except Exception as e:
        decision["registry_note"] = f"registry step skipped: {e}"
        if improved:
            save_model(candidate)
        print(f"[retrain] {decision['action']} (registry skipped: {e})")

    (config.ARTIFACT_DIR / "retraining_decision.json").write_text(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
