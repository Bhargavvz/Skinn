"""
SkinGuard AI — Gradio Web Demo
Production-grade skin cancer detection with Grad-CAM visualization.
"""

import os
import sys
import logging
import tempfile
from datetime import datetime

import gradio as gr
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import build_model
from src.inference import SkinCancerPredictor
from src.gradcam import generate_gradcam, preprocess_image
from src.dataset import LABEL_NAMES, LABEL_DESCRIPTIONS

logger = logging.getLogger(__name__)

# ---- Config ----
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/config.yaml")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", None)

# ---- Initialize predictor ----
predictor = None


def initialize():
    """Lazy initialization of the predictor."""
    global predictor
    if predictor is None:
        predictor = SkinCancerPredictor(
            config_path=CONFIG_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            use_tta=True,
        )
    return predictor


# ---- Risk level badge colors ----
RISK_COLORS = {
    "HIGH": "#FF4444",
    "MEDIUM": "#FF8C00",
    "LOW": "#00C853",
}


def predict(image):
    """
    Main prediction function for the Gradio interface.
    
    Args:
        image: PIL Image or numpy array from Gradio
    
    Returns:
        confidences: dict for gr.Label
        gradcam_image: numpy array for gr.Image
        risk_html: HTML string for risk assessment
        details_html: HTML for detailed results
    """
    if image is None:
        return None, None, "<p>Please upload an image.</p>", "<p>No image provided.</p>"

    pred = initialize()

    # Save temp image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(f.name)
        else:
            image.save(f.name)
        temp_path = f.name

    try:
        # Get prediction
        result = pred.predict(temp_path, return_gradcam=True)

        # Confidence scores for gr.Label
        confidences = result["probabilities"]

        # Grad-CAM image
        gradcam_img = result.get("gradcam", None)

        # Risk assessment HTML
        risk = result["risk_assessment"]
        prediction = result["prediction"]
        risk_color = RISK_COLORS.get(risk["level"], "#666")

        risk_html = f"""
        <div style="padding: 20px; border-radius: 12px; background: linear-gradient(135deg, #1a1a2e, #16213e); 
             border-left: 5px solid {risk_color}; margin: 10px 0; font-family: 'Inter', sans-serif;">
            <h2 style="margin:0 0 10px 0; color: white; font-size: 24px;">
                🔬 {prediction['class_name']}
            </h2>
            <p style="color: #ccc; margin: 5px 0; font-size: 14px;">
                {prediction['description']}
            </p>
            <div style="margin: 15px 0;">
                <span style="background: {risk_color}; color: white; padding: 6px 16px; 
                      border-radius: 20px; font-weight: bold; font-size: 14px;">
                    ⚠️ {risk['level']} RISK
                </span>
                <span style="color: white; margin-left: 10px; font-size: 16px; font-weight: 600;">
                    {prediction['confidence']*100:.1f}% Confidence
                </span>
            </div>
            <p style="color: #aaa; margin: 10px 0 0 0; font-size: 13px;">
                <strong>Recommended Action:</strong> {risk['action']}
            </p>
        </div>
        """

        # Detailed results HTML
        top3 = result["top_3"]
        details_rows = ""
        for i, t in enumerate(top3, 1):
            bar_width = t["confidence"] * 100
            color = "#4CAF50" if i == 1 else "#2196F3" if i == 2 else "#FF9800"
            desc = LABEL_DESCRIPTIONS.get(t["class_name"], "")
            details_rows += f"""
            <div style="margin: 8px 0; padding: 10px; background: rgba(255,255,255,0.05); 
                 border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: white; font-weight: 600; font-size: 15px;">
                        {i}. {t['class_name']}
                    </span>
                    <span style="color: {color}; font-weight: bold; font-size: 15px;">
                        {t['confidence']*100:.1f}%
                    </span>
                </div>
                <div style="background: rgba(255,255,255,0.1); border-radius: 4px; 
                     height: 6px; margin-top: 6px; overflow: hidden;">
                    <div style="width: {bar_width}%; height: 100%; background: {color}; 
                         border-radius: 4px;"></div>
                </div>
                <p style="color: #888; font-size: 11px; margin: 4px 0 0 0;">{desc}</p>
            </div>
            """

        details_html = f"""
        <div style="padding: 15px; border-radius: 12px; background: linear-gradient(135deg, #0f0f1a, #1a1a2e); 
             font-family: 'Inter', sans-serif;">
            <h3 style="color: white; margin: 0 0 15px 0;">📊 Top Predictions</h3>
            {details_rows}
            <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
            <p style="color: #666; font-size: 11px; margin: 0;">
                Inference time: {result['metadata']['inference_time_ms']:.0f}ms | 
                TTA: {'ON' if result['metadata']['tta_enabled'] else 'OFF'} | 
                Model: SkinGuard AI v1.0
            </p>
        </div>
        """

        return confidences, gradcam_img, risk_html, details_html

    finally:
        os.unlink(temp_path)


# ---- Build Gradio UI ----
def create_app():
    """Create the Gradio application."""
    
    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        max-width: 1200px !important;
        margin: auto !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border: none !important;
    }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="SkinGuard AI — Skin Cancer Detection",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
    ) as app:
        gr.Markdown(
            """
            # 🏥 SkinGuard AI — Skin Cancer Detection
            ### Production-grade AI for dermoscopic skin lesion classification
            
            Upload a dermoscopic image to get instant classification across **8 skin lesion types** 
            with explainable Grad-CAM visualization showing which regions the model focuses on.
            
            > ⚠️ **Disclaimer**: This tool is for educational/research purposes only. 
            > It is NOT a substitute for professional medical diagnosis. Always consult a dermatologist.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📷 Upload Dermoscopic Image",
                    type="pil",
                    height=350,
                )
                predict_btn = gr.Button(
                    "🔍 Analyze Lesion",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown("### 📋 Supported Lesion Types")
                gr.Markdown(
                    """
                    | Code | Type | Risk |
                    |------|------|------|
                    | MEL | Melanoma | 🔴 High |
                    | BCC | Basal Cell Carcinoma | 🔴 High |
                    | SCC | Squamous Cell Carcinoma | 🔴 High |
                    | AK | Actinic Keratosis | 🟠 Medium |
                    | BKL | Benign Keratosis | 🟢 Low |
                    | NV | Melanocytic Nevus | 🟢 Low |
                    | DF | Dermatofibroma | 🟢 Low |
                    | VASC | Vascular Lesion | 🟢 Low |
                    """
                )

            with gr.Column(scale=1):
                risk_output = gr.HTML(label="Risk Assessment")
                gradcam_output = gr.Image(label="🔥 Grad-CAM Heatmap", height=300)
                details_output = gr.HTML(label="Detailed Results")
                confidence_output = gr.Label(
                    label="📊 All Class Probabilities",
                    num_top_classes=8,
                )

        # Connect the button
        predict_btn.click(
            fn=predict,
            inputs=[image_input],
            outputs=[confidence_output, gradcam_output, risk_output, details_output],
        )

        gr.Markdown(
            """
            ---
            **Architecture**: EVA-02-Large + ConvNeXt-V2-Large + Swin-V2-Base Ensemble | 
            **Dataset**: ISIC 2019 (25K images) | **TTA**: 8× augmentation averaging | 
            **Made with** ❤️ using PyTorch, timm, Hugging Face
            """
        )

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    app = create_app()
    app.launch(
        server_port=cfg.get("app", {}).get("server_port", 7860),
        share=cfg.get("app", {}).get("share", False),
        show_error=True,
    )
