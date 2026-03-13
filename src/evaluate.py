"""
Skin Cancer Detection — Evaluation Pipeline
Comprehensive metrics, confusion matrix, ROC curves, classification report.
"""

import os
import logging
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
from torch.amp import autocast
import yaml
from tqdm import tqdm

from src.models import build_model
from src.dataset import get_dataloaders, LABEL_NAMES

logger = logging.getLogger(__name__)


def evaluate_model(model, loader, device, num_classes=8, use_bf16=True):
    """
    Run inference on a dataloader and collect predictions.
    Returns all predictions, probabilities, and ground truth labels.
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)

            with autocast("cuda", dtype=amp_dtype, enabled=use_bf16):
                logits, _ = model(images)

            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    return all_preds, all_probs, all_labels


def compute_metrics(preds, probs, labels, class_names=None, num_classes=8):
    """Compute comprehensive evaluation metrics."""
    if class_names is None:
        class_names = LABEL_NAMES[:num_classes]

    # Basic metrics
    accuracy = accuracy_score(labels, preds) * 100
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(num_classes)), zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    # AUROC
    try:
        auroc = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
        macro_auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auroc = 0.0
        macro_auroc = 0.0

    # Per-class AUROC
    per_class_auroc = []
    for i in range(num_classes):
        try:
            binary_labels = (labels == i).astype(int)
            class_auroc = roc_auc_score(binary_labels, probs[:, i])
            per_class_auroc.append(class_auroc)
        except ValueError:
            per_class_auroc.append(0.0)

    # Per-class accuracy
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1).clip(min=1) * 100

    metrics = {
        "overall": {
            "accuracy": accuracy,
            "macro_precision": macro_precision * 100,
            "macro_recall": macro_recall * 100,
            "macro_f1": macro_f1 * 100,
            "weighted_precision": weighted_precision * 100,
            "weighted_recall": weighted_recall * 100,
            "weighted_f1": weighted_f1 * 100,
            "weighted_auroc": auroc,
            "macro_auroc": macro_auroc,
        },
        "per_class": {},
    }

    for i, name in enumerate(class_names):
        metrics["per_class"][name] = {
            "accuracy": per_class_accuracy[i] if i < len(per_class_accuracy) else 0.0,
            "precision": precision[i] * 100 if i < len(precision) else 0.0,
            "recall": recall[i] * 100 if i < len(recall) else 0.0,
            "f1": f1[i] * 100 if i < len(f1) else 0.0,
            "auroc": per_class_auroc[i] if i < len(per_class_auroc) else 0.0,
            "support": int(support[i]) if i < len(support) else 0,
        }

    return metrics


def plot_confusion_matrix(labels, preds, class_names, save_path):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0], cbar_kws={"shrink": 0.8}
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("True", fontsize=12)

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2%", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1], cbar_kws={"shrink": 0.8}
    )
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("True", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved: {save_path}")


def plot_roc_curves(labels, probs, class_names, save_path):
    """Generate and save per-class ROC curves."""
    num_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        binary_labels = (labels == i).astype(int)
        try:
            fpr, tpr, _ = roc_curve(binary_labels, probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
        except ValueError:
            continue

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Per Class", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved: {save_path}")


def plot_metrics_summary(metrics, class_names, save_path):
    """Generate a per-class metrics bar chart."""
    per_class = metrics["per_class"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metric_names = ["accuracy", "precision", "recall", "f1"]
    titles = ["Per-Class Accuracy (%)", "Per-Class Precision (%)", 
              "Per-Class Recall (%)", "Per-Class F1-Score (%)"]

    for ax, metric, title in zip(axes.flat, metric_names, titles):
        values = [per_class[name][metric] for name in class_names]
        bars = ax.bar(class_names, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(class_names))))
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_ylabel("%")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", fontsize=9)

    plt.suptitle(
        f"Overall Accuracy: {metrics['overall']['accuracy']:.2f}% | "
        f"Weighted AUROC: {metrics['overall']['weighted_auroc']:.4f}",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Metrics summary saved: {save_path}")


def run_full_evaluation(cfg, checkpoint_path=None, split="test"):
    """
    Run complete evaluation pipeline.
    Generates metrics, confusion matrix, ROC curves, and classification report.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg["project"]["checkpoint_dir"], "best.pth")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Build model
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Load data
    data = get_dataloaders(cfg)
    loader = data[split]
    num_classes = data["num_classes"]
    class_names = cfg["data"].get("class_names", LABEL_NAMES[:num_classes])

    # Run evaluation
    preds, probs, labels = evaluate_model(
        model, loader, device, num_classes, use_bf16=cfg["h100"]["bf16"]
    )

    # Compute metrics
    metrics = compute_metrics(preds, probs, labels, class_names, num_classes)

    # Print classification report
    report = classification_report(
        labels, preds, target_names=class_names,
        labels=list(range(num_classes)), digits=4, zero_division=0
    )
    logger.info(f"\n📊 Classification Report ({split} set):\n{report}")

    # Print overall metrics
    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL METRICS ({split} set)")
    logger.info(f"{'='*60}")
    for key, val in metrics["overall"].items():
        logger.info(f"  {key}: {val:.4f}")

    # Save plots
    eval_dir = os.path.join(cfg["project"]["output_dir"], "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    plot_confusion_matrix(
        labels, preds, class_names,
        os.path.join(eval_dir, f"confusion_matrix_{split}.png")
    )
    plot_roc_curves(
        labels, probs, class_names,
        os.path.join(eval_dir, f"roc_curves_{split}.png")
    )
    plot_metrics_summary(
        metrics, class_names,
        os.path.join(eval_dir, f"metrics_summary_{split}.png")
    )

    # Save metrics JSON
    metrics_path = os.path.join(eval_dir, f"metrics_{split}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Skin Cancer Detection — Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_full_evaluation(cfg, args.checkpoint, args.split)


if __name__ == "__main__":
    main()
