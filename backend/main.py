"""
SkinGuard AI — FastAPI Backend
Production-ready REST API for skin cancer detection.
"""

import os
import sys
import io
import base64
import logging
import time
import tempfile
from datetime import datetime

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import yaml

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models import build_model
from src.inference import SkinCancerPredictor
from src.dataset import LABEL_NAMES, LABEL_DESCRIPTIONS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---- Configuration ----
CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.join(PROJECT_ROOT, "configs/config.yaml"))
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", None)

# ---- App ----
app = FastAPI(
    title="SkinGuard AI API",
    description="Production-grade skin cancer detection API powered by a 3-model ensemble (EVA-02 + ConvNeXt-V2 + Swin-V2)",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Global predictor ----
predictor = None


def get_predictor():
    """Lazy-load the model predictor."""
    global predictor
    if predictor is None:
        logger.info("Loading SkinGuard AI model...")
        predictor = SkinCancerPredictor(
            config_path=CONFIG_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            use_tta=False,  # Disable TTA for faster API responses
        )
        logger.info("Model loaded successfully!")
    return predictor


# ---- Endpoints ----

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/model-info")
async def model_info():
    """Return model architecture and training info."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
    except Exception:
        cfg = {}

    return {
        "name": "SkinGuard AI",
        "version": "1.0.0",
        "architecture": "EVA-02-Large + ConvNeXt-V2-Large + Swin-V2-Base Ensemble",
        "parameters": "589.3M",
        "accuracy": 98.68,
        "auroc": 0.9995,
        "classes": cfg.get("data", {}).get("class_names", LABEL_NAMES[:7]),
        "num_classes": 7,
        "image_size": cfg.get("data", {}).get("image_size", 384),
        "training": {
            "dataset": "marmal88/skin_cancer (HAM10000)",
            "epochs": 40,
            "training_time": "4.26 hours",
            "gpu": "NVIDIA H100 80GB",
        },
        "class_descriptions": LABEL_DESCRIPTIONS,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict skin cancer type from an uploaded image.

    Returns prediction, confidence scores, risk assessment, and Grad-CAM visualization.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

    start_time = time.time()

    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Save to temp file (predictor expects a file path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        # Run prediction
        pred = get_predictor()
        result = pred.predict(temp_path, return_gradcam=True)

        # Encode Grad-CAM as base64
        gradcam_b64 = None
        if "gradcam" in result and result["gradcam"] is not None:
            gradcam_img = result["gradcam"]
            if isinstance(gradcam_img, np.ndarray):
                pil_img = Image.fromarray(gradcam_img)
            else:
                pil_img = gradcam_img
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Build response
        response = {
            "prediction": result["prediction"],
            "risk_assessment": result["risk_assessment"],
            "probabilities": result["probabilities"],
            "top_3": result["top_3"],
            "gradcam": gradcam_b64,
            "metadata": {
                "inference_time_ms": round((time.time() - start_time) * 1000, 1),
                "filename": file.filename,
                "model_version": "1.0.0",
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        # Cleanup temp file
        if "temp_path" in locals():
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@app.on_event("startup")
async def startup_event():
    """Pre-load model on startup and mount frontend."""
    logger.info("SkinGuard AI API starting up...")

    # Mount the React frontend build (if it exists)
    frontend_dist = os.path.join(PROJECT_ROOT, "frontend", "dist")
    if os.path.isdir(frontend_dist):
        # Serve static assets (JS, CSS, images)
        app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")
        logger.info(f"Frontend mounted from: {frontend_dist}")

        # Catch-all: serve index.html for any non-API route (SPA fallback)
        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            file_path = os.path.join(frontend_dist, full_path)
            if os.path.isfile(file_path):
                return FileResponse(file_path)
            return FileResponse(os.path.join(frontend_dist, "index.html"))
    else:
        logger.warning(f"Frontend not found at {frontend_dist}. Run 'cd frontend && npm run build' first.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
