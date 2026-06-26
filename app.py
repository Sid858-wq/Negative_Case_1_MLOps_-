"""app.py — Stage 3: FastAPI inference service.

Implement /health (liveness + loaded model info) and POST /predict (multipart image upload
→ {label, prob_defect, confidence}). Load the model once at startup; log every prediction
to artifacts/predictions.log.   Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io, json, time
from contextlib import asynccontextmanager
from pathlib import Path

import torch, torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

import config
from src import data_prep
from src.model import load_model

_state = {"model": None, "tf": None, "meta": {}}


def _load():
    # TODO 3: if config.MODEL_PATH exists, load model + eval transforms + model_meta.json.
    if config.MODEL_PATH.exists():
        _state["model"] = load_model()
        _state["tf"] = data_prep.get_transforms(train=False)
        if config.MODEL_META_PATH.exists():
            _state["meta"] = json.loads(config.MODEL_META_PATH.read_text())

    # raise NotImplementedError


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load(); yield


app = FastAPI(title="Casting Defect Detection API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    # TODO 3: return status + model_loaded + classes + positive_class + test_metrics.
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "registered_model": config.REGISTERED_MODEL,
        "classes": config.CLASS_TO_IDX,
        "positive_class": config.POSITIVE_CLASS,
        "test_metrics": _state["meta"].get("test_metrics", {}),
    }
    # raise NotImplementedError


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    # TODO 3: validate model loaded (503 else); read image (400 if invalid); preprocess;
    #         softmax; return {label, is_defective, prob_defect, confidence}; log prediction.
    if _state["model"] is None:
        raise HTTPException(503, "Model not loaded — train first (python -m src.train).")
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("L")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Uploaded file is not a valid image.")

    x = _state["tf"](img).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        probs = F.softmax(_state["model"](x), dim=1)[0]
    p_defect = float(probs[config.POSITIVE_IDX])
    idx = int(probs.argmax())
    result = {
        "filename": file.filename,
        "label": config.IDX_TO_CLASS[idx],
        "is_defective": bool(idx == config.POSITIVE_IDX),
        "prob_defect": round(p_defect, 4),
        "confidence": round(float(probs.max()), 4),
    }
    _log_prediction(result)
    return result


def _log_prediction(result: dict) -> None:
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **result}
    with open(config.PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")

    # raise NotImplementedError
