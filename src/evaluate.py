"""evaluate.py — Stage 3: evaluation metrics, plots, failure-case analysis. 

Positive class = DEFECT (recall on defects is the headline QC metric). Implement
prediction, imbalance-aware metrics, confusion/ROC plots, and a misclassified-sample list.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


@torch.no_grad()
def predict(net, loader, return_embeddings: bool = False):
    # TODO 3: return (y_true, y_pred, y_prob_defect[, embeddings]) over the loader.
    net.eval()
    ys, preds, probs, embs = [], [], [], []
    extractor = None
    if return_embeddings:
        from src.model import EmbeddingExtractor
        extractor = EmbeddingExtractor(net).to(config.DEVICE).eval()
    for x, y in loader:
        x = x.to(config.DEVICE)
        logits = net(x)
        p = F.softmax(logits, dim=1)[:, config.POSITIVE_IDX]
        preds.append(logits.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
        ys.append(np.asarray(y))
        if extractor is not None:
            embs.append(extractor(x).cpu().numpy())
    out = [np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)]
    if return_embeddings:
        out.append(np.concatenate(embs))
    return tuple(out)
    # raise NotImplementedError


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    # TODO 3: accuracy, precision/recall/f1 for the DEFECT class (pos_label=config.POSITIVE_IDX),
    #         macro_f1, roc_auc, confusion_matrix. Return a dict.
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    pos = config.POSITIVE_IDX
    m = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_defect": round(float(precision_score(y_true, y_pred, pos_label=pos, zero_division=0)), 4),
        "recall_defect": round(float(recall_score(y_true, y_pred, pos_label=pos, zero_division=0)), 4),
        "f1_defect": round(float(f1_score(y_true, y_pred, pos_label=pos, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }
    try:
        m["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    except ValueError:
        m["roc_auc"] = None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    m["confusion_matrix"] = cm.tolist()           # [[TN,FP],[FN,TP]] with defect=1
    m["n"] = int(len(y_true))
    return m
    # raise NotImplementedError


def plot_eval(y_true, y_pred, y_prob, out: Path | None = None) -> Path:
    # TODO 3: save a confusion-matrix + ROC figure to artifacts/model_eval.png.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    out = out or (config.ARTIFACT_DIR / "model_eval.png")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ax[0].imshow(cm, cmap="Blues")
    ax[0].set_xticks([0, 1]); ax[0].set_yticks([0, 1])
    ax[0].set_xticklabels(config.CLASSES); ax[0].set_yticklabels(config.CLASSES)
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual"); ax[0].set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, cm[i, j], ha="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax[1].plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
        ax[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].set_title("ROC curve"); ax[1].legend()
    except Exception:
        pass
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
    return out
    # raise NotImplementedError


def failure_cases(items, y_true, y_pred, y_prob, limit: int = 20) -> list[dict]:
    # TODO 3: list misclassified samples with predicted p_defect (error analysis).
    return []
    # raise NotImplementedError
