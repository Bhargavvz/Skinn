"""
Skin Cancer Detection — H100-Optimized Training Pipeline
BF16, torch.compile, progressive unfreezing, time-budget tracking.
"""

import os
import time
import logging
import json
import csv
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import numpy as np
import yaml
from tqdm import tqdm

from src.models import build_model, count_parameters
from src.dataset import get_dataloaders
from src.losses import FocalLoss, apply_mixup_cutmix, compute_mixed_loss

logger = logging.getLogger(__name__)


class Trainer:
    """
    Production training loop for Skin Cancer Detection.
    
    Features:
        - BF16 mixed precision for H100
        - torch.compile for graph optimization
        - Progressive unfreezing (freeze → unfreeze backbones)
        - Cosine annealing + warmup LR schedule
        - Mixup + CutMix augmentation
        - Class-weighted Focal Loss
        - Gradient accumulation
        - Time-budget watchdog (7h max)
        - TensorBoard + CSV logging
        - Best/last checkpoint saving
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time = time.time()
        self.time_budget = cfg["training"]["time_budget_hours"] * 3600  # seconds

        # Setup directories
        os.makedirs(cfg["project"]["output_dir"], exist_ok=True)
        os.makedirs(cfg["project"]["log_dir"], exist_ok=True)
        os.makedirs(cfg["project"]["checkpoint_dir"], exist_ok=True)

        # Set seeds
        seed = cfg["project"]["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # H100 optimizations
        self._setup_h100_opts()

        # Load data
        logger.info("Loading datasets...")
        data = get_dataloaders(cfg)
        self.train_loader = data["train"]
        self.val_loader = data["val"]
        self.test_loader = data["test"]
        self.class_weights = data["class_weights"].to(self.device)
        self.num_classes = data["num_classes"]

        # Build model
        logger.info("Building ensemble model...")
        self.model = build_model(cfg, pretrained=cfg["model"]["pretrained"])
        self.model = self.model.to(self.device)

        # Compile model for H100
        if cfg["h100"]["torch_compile"]:
            logger.info(f"Compiling model with mode={cfg['h100']['compile_mode']}...")
            self.model = torch.compile(self.model, mode=cfg["h100"]["compile_mode"])

        # Channels last memory format
        if cfg["h100"]["channels_last"]:
            self.model = self.model.to(memory_format=torch.channels_last)

        # Loss function
        loss_cfg = cfg["training"]["loss"]
        self.criterion = FocalLoss(
            gamma=loss_cfg["gamma"],
            alpha=self.class_weights if loss_cfg["use_class_weights"] else None,
            label_smoothing=cfg["training"]["label_smoothing"],
        )

        # Setup optimizer (will be recreated on backbone unfreeze)
        self._setup_optimizer(phase=1)

        # GradScaler for mixed precision
        self.scaler = GradScaler("cuda", enabled=cfg["h100"]["bf16"])

        # Logging
        self.writer = SummaryWriter(cfg["project"]["log_dir"])
        self.csv_log_path = os.path.join(cfg["project"]["log_dir"], "training_log.csv")
        self._init_csv_log()

        # Tracking
        self.best_val_auroc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.global_step = 0

        # Log configuration
        total, trainable = count_parameters(self.model)
        self._log_header(total, trainable)

    def _setup_h100_opts(self):
        """Configure H100 GPU optimizations."""
        cfg = self.cfg["h100"]
        
        if torch.cuda.is_available():
            # TF32 for faster matmul
            if cfg["tf32"]:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
                logger.info("TF32 enabled")

            # cuDNN benchmark
            if cfg["cudnn_benchmark"]:
                torch.backends.cudnn.benchmark = True
                logger.info("cuDNN benchmark enabled")

    def _setup_optimizer(self, phase=1):
        """Setup optimizer with appropriate param groups."""
        cfg = self.cfg["training"]
        opt_cfg = cfg["optimizer"]
        
        if phase == 1:
            # Phase 1: freeze backbones
            self.model.freeze_backbones() if hasattr(self.model, "freeze_backbones") else None
            # If model is compiled, access _orig_mod
            model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            if hasattr(model, "freeze_backbones"):
                model.freeze_backbones()
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt_cfg["lr"],
                weight_decay=opt_cfg["weight_decay"],
                betas=tuple(opt_cfg["betas"]),
            )
        else:
            # Phase 2: unfreeze everything with layerwise LR
            model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            if hasattr(model, "unfreeze_backbones"):
                model.unfreeze_backbones()
            if hasattr(model, "get_param_groups"):
                param_groups = model.get_param_groups(
                    head_lr=opt_cfg["lr"],
                    backbone_lr_multiplier=cfg["freeze"]["backbone_lr_multiplier"],
                )
            else:
                param_groups = [{"params": self.model.parameters(), "lr": opt_cfg["lr"]}]
            
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=opt_cfg["weight_decay"],
                betas=tuple(opt_cfg["betas"]),
            )

        # Scheduler
        sch_cfg = cfg["scheduler"]
        total_epochs = cfg["epochs"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_epochs - sch_cfg["warmup_epochs"],
            T_mult=1,
            eta_min=sch_cfg["eta_min"],
        )

    def _log_header(self, total_params, trainable_params):
        """Log training configuration."""
        logger.info("=" * 80)
        logger.info("SKIN CANCER DETECTION — TRAINING CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"Total Parameters: {total_params / 1e6:.1f}M")
        logger.info(f"Trainable Parameters: {trainable_params / 1e6:.1f}M")
        logger.info(f"Training Samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation Samples: {len(self.val_loader.dataset)}")
        logger.info(f"Batch Size: {self.cfg['training']['batch_size']}")
        logger.info(f"Effective Batch: {self.cfg['training']['batch_size'] * self.cfg['training']['gradient_accumulation_steps']}")
        logger.info(f"Time Budget: {self.cfg['training']['time_budget_hours']}h")
        logger.info(f"BF16: {self.cfg['h100']['bf16']}")
        logger.info(f"torch.compile: {self.cfg['h100']['torch_compile']}")
        logger.info("=" * 80)

    def _init_csv_log(self):
        """Initialize CSV training log."""
        with open(self.csv_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                "val_auroc", "lr", "elapsed_hours", "best_auroc"
            ])

    def _check_time_budget(self, epoch):
        """Check if we're approaching the time budget."""
        elapsed = time.time() - self.start_time
        remaining = self.time_budget - elapsed
        elapsed_hours = elapsed / 3600

        if remaining < 600:  # Less than 10 minutes left
            logger.warning(f"⚠️ TIME BUDGET: Only {remaining / 60:.1f} min remaining! Stopping training.")
            return True

        # Estimate time per epoch
        if epoch > 0:
            time_per_epoch = elapsed / epoch
            epochs_remaining = self.cfg["training"]["epochs"] - epoch
            estimated_time = time_per_epoch * epochs_remaining

            if estimated_time > remaining - 600:
                safe_epochs = int((remaining - 600) / time_per_epoch)
                logger.info(f"⏱️ Budget allows ~{safe_epochs} more epochs (est. {time_per_epoch / 60:.1f} min/epoch)")

        return False

    def _warmup_lr(self, epoch, step, total_steps):
        """Linear warmup for the first few epochs."""
        warmup_epochs = self.cfg["training"]["scheduler"]["warmup_epochs"]
        if epoch < warmup_epochs:
            warmup_steps = warmup_epochs * total_steps
            current_step = epoch * total_steps + step
            lr_scale = min(1.0, current_step / max(1, warmup_steps))
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", self.cfg["training"]["optimizer"]["lr"]) * lr_scale

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        grad_accum = self.cfg["training"]["gradient_accumulation_steps"]
        use_bf16 = self.cfg["h100"]["bf16"]

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg['training']['epochs']} [Train]",
                     leave=False, dynamic_ncols=True)

        self.optimizer.zero_grad()

        for step, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Channels last
            if self.cfg["h100"]["channels_last"]:
                images = images.to(memory_format=torch.channels_last)

            # Warmup LR
            self._warmup_lr(epoch, step, len(self.train_loader))

            # Apply Mixup / CutMix
            mixed_images, y_a, y_b, lam, mixed = apply_mixup_cutmix(
                images, labels, self.cfg, device=self.device
            )

            # Forward pass with BF16
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
            with autocast("cuda", dtype=amp_dtype, enabled=use_bf16):
                logits, attn_weights = self.model(mixed_images)
                loss = compute_mixed_loss(self.criterion, logits, y_a, y_b, lam, mixed)
                loss = loss / grad_accum  # Scale for accumulation

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (step + 1) % grad_accum == 0 or (step + 1) == len(self.train_loader):
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Track metrics
            total_loss += loss.item() * grad_accum
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            current_acc = 100.0 * correct / total
            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{total_loss / (step + 1):.4f}",
                "acc": f"{current_acc:.2f}%",
                "lr": f"{current_lr:.2e}",
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, loader=None):
        """Validate on val/test set. Returns loss, accuracy, AUROC."""
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        use_bf16 = self.cfg["h100"]["bf16"]

        pbar = tqdm(loader, desc="Validating", leave=False, dynamic_ncols=True)

        for images, labels, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.cfg["h100"]["channels_last"]:
                images = images.to(memory_format=torch.channels_last)

            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
            with autocast("cuda", dtype=amp_dtype, enabled=use_bf16):
                logits, _ = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            probs = F.softmax(logits.float(), dim=1)  # float32 for precise probabilities
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total

        # AUROC computation
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        try:
            # One-vs-rest AUROC — need labels param for classes not in y_true
            unique_labels = np.unique(all_labels)
            if len(unique_labels) >= 2:
                auroc = roc_auc_score(
                    all_labels, all_probs,
                    multi_class="ovr",
                    average="weighted",
                    labels=list(range(all_probs.shape[1])),
                )
            else:
                auroc = 0.0
                logger.warning(f"AUROC: only {len(unique_labels)} class(es) in val labels, skipping")
        except Exception as e:
            logger.warning(f"AUROC computation failed: {e}")
            auroc = 0.0

        return avg_loss, accuracy, auroc

    def save_checkpoint(self, epoch, val_auroc, is_best=False):
        """Save model checkpoint."""
        model_state = self.model._orig_mod.state_dict() if hasattr(self.model, "_orig_mod") else self.model.state_dict()
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_auroc": val_auroc,
            "best_val_auroc": self.best_val_auroc,
            "config": self.cfg,
            "class_weights": self.class_weights.cpu(),
            "global_step": self.global_step,
        }

        # Save last checkpoint
        last_path = os.path.join(self.cfg["project"]["checkpoint_dir"], "last.pth")
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = os.path.join(self.cfg["project"]["checkpoint_dir"], "best.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"💾 Best model saved (AUROC: {val_auroc:.4f})")

        # Save periodic
        if (epoch + 1) % self.cfg["training"].get("save_every_n_epochs", 5) == 0:
            periodic_path = os.path.join(
                self.cfg["project"]["checkpoint_dir"], f"epoch_{epoch + 1}.pth"
            )
            torch.save(checkpoint, periodic_path)

    def train(self):
        """Full training loop."""
        epochs = self.cfg["training"]["epochs"]
        phase1_epochs = self.cfg["training"]["freeze"]["phase1_epochs"]
        patience = self.cfg["training"]["early_stopping"]["patience"]

        logger.info(f"\n🚀 Starting training for {epochs} epochs...\n")
        logger.info(f"Phase 1 (epochs 1-{phase1_epochs}): Backbone FROZEN — training head only")
        logger.info(f"Phase 2 (epochs {phase1_epochs + 1}+): Backbone UNFROZEN — fine-tuning all\n")

        for epoch in range(epochs):
            epoch_start = time.time()

            # ---- Time budget check ----
            if self._check_time_budget(epoch):
                logger.info("⏰ Time budget reached — saving final checkpoint and stopping.")
                break

            # ---- Phase transition: unfreeze backbones ----
            if epoch == phase1_epochs:
                logger.info("\n🔓 Phase 2: Unfreezing backbone parameters...")
                self._setup_optimizer(phase=2)

            # ---- Train ----
            train_loss, train_acc = self.train_epoch(epoch)

            # ---- Validate ----
            val_loss, val_acc, val_auroc = self.validate()

            # ---- Learning rate step ----
            if epoch >= self.cfg["training"]["scheduler"]["warmup_epochs"]:
                self.scheduler.step()

            # ---- Elapsed time ----
            elapsed = time.time() - self.start_time
            elapsed_hours = elapsed / 3600
            epoch_time = time.time() - epoch_start

            # ---- Best model ----
            is_best = val_auroc > self.best_val_auroc
            if is_best:
                self.best_val_auroc = val_auroc
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # ---- Save checkpoint ----
            self.save_checkpoint(epoch, val_auroc, is_best=is_best)

            # ---- Logging ----
            current_lr = self.optimizer.param_groups[0]["lr"]
            phase = "Phase1-FROZEN" if epoch < phase1_epochs else "Phase2-FINETUNE"

            logger.info(
                f"Epoch {epoch + 1:3d}/{epochs} [{phase}] | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% AUROC: {val_auroc:.4f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_time / 60:.1f}m | "
                f"Total: {elapsed_hours:.2f}h | "
                f"Best: {self.best_val_auroc:.4f} (ep{self.best_epoch + 1})"
                + (" ★" if is_best else "")
            )

            # TensorBoard
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/accuracy", train_acc, epoch)
            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.writer.add_scalar("val/accuracy", val_acc, epoch)
            self.writer.add_scalar("val/auroc", val_auroc, epoch)
            self.writer.add_scalar("learning_rate", current_lr, epoch)
            self.writer.add_scalar("time/elapsed_hours", elapsed_hours, epoch)

            # CSV log
            with open(self.csv_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, f"{train_loss:.4f}", f"{train_acc:.2f}",
                    f"{val_loss:.4f}", f"{val_acc:.2f}", f"{val_auroc:.4f}",
                    f"{current_lr:.2e}", f"{elapsed_hours:.2f}", f"{self.best_val_auroc:.4f}"
                ])

            # ---- Early stopping ----
            if self.patience_counter >= patience:
                logger.info(f"\n⏹️ Early stopping triggered (no improvement for {patience} epochs)")
                break

        # ---- Final summary ----
        total_time = (time.time() - self.start_time) / 3600
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best Validation AUROC: {self.best_val_auroc:.4f} (epoch {self.best_epoch + 1})")
        logger.info(f"Total Training Time: {total_time:.2f} hours")
        logger.info(f"Checkpoint: {self.cfg['project']['checkpoint_dir']}/best.pth")
        logger.info("=" * 80)

        self.writer.close()

        # Run test evaluation
        logger.info("\n📊 Evaluating on test set...")
        test_loss, test_acc, test_auroc = self.validate(self.test_loader)
        logger.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}% | Test AUROC: {test_auroc:.4f}")

        return {
            "best_val_auroc": self.best_val_auroc,
            "best_epoch": self.best_epoch,
            "test_accuracy": test_acc,
            "test_auroc": test_auroc,
            "total_time_hours": total_time,
        }


def main():
    """Entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Skin Cancer Detection Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config")
    parser.add_argument("--smoke-test", action="store_true", help="Quick test with tiny dataset")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("outputs/training.log"),
        ]
    )

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.smoke_test:
        cfg["training"]["epochs"] = 3
        cfg["training"]["batch_size"] = 4
        cfg["data"]["num_workers"] = 0
        cfg["h100"]["torch_compile"] = False

    # Create trainer and run
    trainer = Trainer(cfg)
    results = trainer.train()

    # Save results
    results_path = os.path.join(cfg["project"]["output_dir"], "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
