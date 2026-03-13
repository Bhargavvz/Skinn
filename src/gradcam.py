"""
Skin Cancer Detection — Grad-CAM Interpretability
Generates heatmap overlays showing which regions the model attends to.
"""

import os
import logging

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml

from src.models import build_model
from src.dataset import LABEL_NAMES, LABEL_DESCRIPTIONS

logger = logging.getLogger(__name__)


def get_target_layers(model):
    """
    Get the last convolutional/attention layer from each backbone
    for Grad-CAM visualization.
    """
    target_layers = []
    
    # Access the original model if compiled
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    for i, backbone in enumerate(base_model.backbones):
        # Try to find the last feature layer
        try:
            if hasattr(backbone, "stages"):
                # ConvNeXt-style
                target_layers.append(backbone.stages[-1])
            elif hasattr(backbone, "layers"):
                # Swin-style
                target_layers.append(backbone.layers[-1])
            elif hasattr(backbone, "blocks"):
                # ViT/EVA-style
                target_layers.append(backbone.blocks[-1])
            elif hasattr(backbone, "features"):
                # EfficientNet-style
                target_layers.append(backbone.features[-1])
            else:
                # Fallback: get last named module
                modules = list(backbone.children())
                if modules:
                    target_layers.append(modules[-2] if len(modules) > 1 else modules[-1])
        except Exception as e:
            logger.warning(f"Could not find target layer for backbone {i}: {e}")

    return target_layers


class EnsembleWrapper(torch.nn.Module):
    """
    Wrapper to make the ensemble model compatible with Grad-CAM.
    Extracts only logits (no attention weights).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model._orig_mod if hasattr(model, "_orig_mod") else model

    def forward(self, x):
        logits, _ = self.model(x)
        return logits


def preprocess_image(image_path, image_size=384):
    """Load and preprocess a single image for Grad-CAM."""
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    result = transform(image=image_np)
    tensor = result["image"].unsqueeze(0)

    # Also get the resized original for overlay
    resize_transform = A.Compose([A.Resize(image_size, image_size)])
    resized = resize_transform(image=image_np)["image"]
    original_resized = resized.astype(np.float32) / 255.0

    return tensor, original_resized


def generate_gradcam(model, image_tensor, original_image, target_class=None, device="cuda"):
    """
    Generate Grad-CAM heatmap for a single image.
    
    Args:
        model: trained ensemble model
        image_tensor: [1, 3, H, W] preprocessed tensor
        original_image: [H, W, 3] float32 normalized to [0, 1]
        target_class: class index (None = use predicted class)
        device: torch device
    
    Returns:
        cam_image: [H, W, 3] overlay image
        prediction: predicted class index
        confidence: prediction confidence
    """
    wrapper = EnsembleWrapper(model)
    wrapper = wrapper.to(device)
    wrapper.eval()

    target_layers = get_target_layers(model)

    if not target_layers:
        logger.warning("No target layers found for Grad-CAM")
        return original_image, 0, 0.0

    image_tensor = image_tensor.to(device)

    # Get prediction
    with torch.no_grad():
        logits, attn_weights = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        prediction = probs.argmax(dim=1).item()
        confidence = probs[0, prediction].item()

    if target_class is None:
        target_class = prediction

    # Generate Grad-CAM for each target layer and average
    cam_images = []
    for target_layer in target_layers:
        try:
            cam = GradCAM(model=wrapper, target_layers=[target_layer])
            targets = [ClassifierOutputTarget(target_class)]
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
            cam_image = show_cam_on_image(original_image, grayscale_cam[0], use_rgb=True)
            cam_images.append(cam_image)
        except Exception as e:
            logger.warning(f"Grad-CAM failed for layer: {e}")

    if cam_images:
        # Average all backbone CAMs
        avg_cam = np.mean([img.astype(np.float32) for img in cam_images], axis=0).astype(np.uint8)
    else:
        avg_cam = (original_image * 255).astype(np.uint8)

    return avg_cam, prediction, confidence


def generate_gradcam_grid(model, image_paths, save_dir, cfg, device="cuda", num_samples=20):
    """
    Generate Grad-CAM visualizations for multiple images.
    Creates a grid showing original and heatmap side by side.
    """
    os.makedirs(save_dir, exist_ok=True)
    image_size = cfg["data"]["image_size"]
    class_names = cfg["data"].get("class_names", LABEL_NAMES)

    results = []

    for i, img_path in enumerate(image_paths[:num_samples]):
        try:
            tensor, original = preprocess_image(img_path, image_size)
            cam_image, pred, conf = generate_gradcam(
                model, tensor, original, device=device
            )

            # Create side-by-side plot
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(original)
            axes[0].set_title("Original", fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(cam_image)
            pred_name = class_names[pred] if pred < len(class_names) else f"Class {pred}"
            axes[1].set_title(
                f"Grad-CAM: {pred_name} ({conf * 100:.1f}%)",
                fontsize=12, fontweight="bold"
            )
            axes[1].axis("off")

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"gradcam_{i:03d}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            results.append({
                "image": img_path,
                "prediction": pred_name,
                "confidence": conf,
                "save_path": save_path,
            })

            logger.info(f"[{i + 1}/{min(num_samples, len(image_paths))}] "
                        f"{os.path.basename(img_path)} → {pred_name} ({conf * 100:.1f}%)")

        except Exception as e:
            logger.error(f"Failed for {img_path}: {e}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Skin Cancer Detection — Grad-CAM")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image-dir", type=str, help="Directory of images")
    parser.add_argument("--output-dir", type=str, default="outputs/gradcam")
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint_path = args.checkpoint or os.path.join(cfg["project"]["checkpoint_dir"], "best.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_model(cfg, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    if args.image:
        # Single image
        tensor, original = preprocess_image(args.image, cfg["data"]["image_size"])
        cam_image, pred, conf = generate_gradcam(model, tensor, original, device=device)

        class_names = cfg["data"].get("class_names", LABEL_NAMES)
        pred_name = class_names[pred] if pred < len(class_names) else f"Class {pred}"

        print(f"\nPrediction: {pred_name} ({conf * 100:.1f}% confidence)")
        print(f"Description: {LABEL_DESCRIPTIONS.get(pred_name, 'N/A')}")

        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, "gradcam_single.png")
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(cam_image)
        axes[1].set_title(f"Grad-CAM: {pred_name} ({conf * 100:.1f}%)")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

    elif args.image_dir:
        # Directory of images
        import glob
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_paths = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))

        generate_gradcam_grid(
            model, image_paths, args.output_dir, cfg,
            device=device, num_samples=args.num_samples
        )


if __name__ == "__main__":
    main()
