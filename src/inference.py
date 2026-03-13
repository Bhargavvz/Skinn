"""
Skin Cancer Detection — Inference Pipeline
Single-image and batch inference with TTA and Grad-CAM.
"""

import os
import sys
import json
import logging
import time

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml

from src.models import build_model
from src.dataset import LABEL_NAMES, LABEL_DESCRIPTIONS
from src.tta import TTAPredictor
from src.gradcam import generate_gradcam, preprocess_image

logger = logging.getLogger(__name__)


class SkinCancerPredictor:
    """
    Production-ready inference pipeline for skin cancer detection.
    
    Features:
        - Single image and batch prediction
        - Test-Time Augmentation (TTA) for robust predictions
        - Grad-CAM explainability
        - Risk assessment
        - JSON output for API integration
    """

    # Risk levels based on diagnosis
    RISK_LEVELS = {
        "MEL": {"level": "HIGH", "color": "red", "action": "Immediate dermatologist referral required"},
        "BCC": {"level": "HIGH", "color": "red", "action": "Dermatologist referral for biopsy"},
        "SCC": {"level": "HIGH", "color": "red", "action": "Urgent dermatologist referral"},
        "AK": {"level": "MEDIUM", "color": "orange", "action": "Monitor and consult dermatologist"},
        "BKL": {"level": "LOW", "color": "green", "action": "Routine monitoring suggested"},
        "NV": {"level": "LOW", "color": "green", "action": "Benign — routine follow-up"},
        "DF": {"level": "LOW", "color": "green", "action": "Benign — no immediate action needed"},
        "VASC": {"level": "LOW", "color": "green", "action": "Benign vascular lesion"},
    }

    def __init__(self, config_path="configs/config.yaml", checkpoint_path=None,
                 use_tta=True, device=None):
        """
        Initialize the predictor.
        
        Args:
            config_path: path to config YAML
            checkpoint_path: path to model checkpoint (.pth)
            use_tta: whether to use Test-Time Augmentation
            device: torch device (auto-detected if None)
        """
        # Load config
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.cfg["project"]["checkpoint_dir"], "best.pth")

        logger.info(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model = build_model(self.cfg, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Class names
        self.class_names = self.cfg["data"].get("class_names", LABEL_NAMES)
        self.num_classes = len(self.class_names)

        # TTA
        self.use_tta = use_tta
        if use_tta:
            self.tta = TTAPredictor(
                self.model,
                image_size=self.cfg["data"]["image_size"],
                device=self.device,
                use_bf16=self.cfg["h100"]["bf16"],
            )

        # Preprocessing
        img_size = self.cfg["data"]["image_size"]
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        logger.info(f"Predictor ready — {self.num_classes} classes, TTA={'ON' if use_tta else 'OFF'}")

    def predict(self, image_path, return_gradcam=False):
        """
        Predict on a single image.
        
        Args:
            image_path: path to the image file
            return_gradcam: whether to generate Grad-CAM overlay
        
        Returns:
            dict with prediction, confidence, risk level, and optional Grad-CAM
        """
        start_time = time.time()

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Predict
        if self.use_tta:
            avg_probs, all_probs, pred_class, confidence = self.tta.predict_single(image_np)
            probs = avg_probs
        else:
            # Simple prediction
            processed = self.transform(image=image_np)["image"].unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits, attn_weights = self.model(processed)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_class = probs.argmax()
                confidence = probs[pred_class]
                attn_weights = attn_weights.cpu().numpy()[0]

        # Build result
        pred_name = self.class_names[pred_class]
        risk = self.RISK_LEVELS.get(pred_name, {"level": "UNKNOWN", "color": "gray", "action": "Consult specialist"})

        result = {
            "prediction": {
                "class_name": pred_name,
                "class_index": int(pred_class),
                "confidence": float(confidence),
                "description": LABEL_DESCRIPTIONS.get(pred_name, ""),
            },
            "risk_assessment": {
                "level": risk["level"],
                "action": risk["action"],
            },
            "probabilities": {
                name: float(probs[i]) for i, name in enumerate(self.class_names)
            },
            "top_3": [
                {
                    "class_name": self.class_names[idx],
                    "confidence": float(probs[idx]),
                }
                for idx in np.argsort(probs)[::-1][:3]
            ],
            "metadata": {
                "image_path": image_path,
                "tta_enabled": self.use_tta,
                "inference_time_ms": (time.time() - start_time) * 1000,
            },
        }

        # Grad-CAM
        if return_gradcam:
            tensor, original = preprocess_image(image_path, self.cfg["data"]["image_size"])
            cam_image, _, _ = generate_gradcam(
                self.model, tensor, original,
                target_class=pred_class, device=str(self.device)
            )
            result["gradcam"] = cam_image

        return result

    def predict_batch(self, image_paths):
        """Predict on multiple images."""
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed for {path}: {e}")
                results.append({"error": str(e), "image_path": path})
        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Skin Cancer Detection — Inference")
    parser.add_argument("--image", type=str, required=True, help="Image path")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if not args.json else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    predictor = SkinCancerPredictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        use_tta=not args.no_tta,
    )

    result = predictor.predict(args.image, return_gradcam=args.gradcam)

    if args.json:
        # Remove non-serializable gradcam array
        output = {k: v for k, v in result.items() if k != "gradcam"}
        print(json.dumps(output, indent=2))
    else:
        pred = result["prediction"]
        risk = result["risk_assessment"]

        print(f"\n{'='*60}")
        print(f"  🔬 SKIN CANCER DETECTION RESULT")
        print(f"{'='*60}")
        print(f"  Image:       {args.image}")
        print(f"  Prediction:  {pred['class_name']} ({pred['confidence']*100:.1f}%)")
        print(f"  Description: {pred['description']}")
        print(f"  Risk Level:  {risk['level']}")
        print(f"  Action:      {risk['action']}")
        print(f"\n  Top 3 Predictions:")
        for i, top in enumerate(result["top_3"], 1):
            print(f"    {i}. {top['class_name']}: {top['confidence']*100:.1f}%")
        print(f"\n  Inference: {result['metadata']['inference_time_ms']:.0f}ms")
        print(f"  TTA: {'ON' if result['metadata']['tta_enabled'] else 'OFF'}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
