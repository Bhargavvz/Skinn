"""
Skin Cancer Detection — Loss Functions
Focal Loss with class weights + Mixup/CutMix support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in skin lesion classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    When gamma > 0, reduces the relative loss for well-classified examples,
    focusing training on hard/misclassified samples (e.g., rare cancer types).
    """

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0, reduction="mean"):
        """
        Args:
            gamma: focusing parameter (0 = standard CE, 2 = strong focus on hard examples)
            alpha: class weights tensor [num_classes] or None
            label_smoothing: label smoothing factor (0.0 to 0.1 typical)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C] raw model outputs
            targets: [B] class indices OR [B, C] soft labels (for mixup)
        """
        num_classes = logits.size(1)

        # Handle soft labels (mixup/cutmix)
        if targets.dim() == 2:
            # Soft labels — use KL divergence style focal loss
            log_probs = F.log_softmax(logits, dim=1)
            probs = torch.exp(log_probs)

            # Apply label smoothing to soft labels
            if self.label_smoothing > 0:
                targets = targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes

            # Focal weight
            focal_weight = (1 - probs) ** self.gamma

            # Weighted cross entropy
            loss = -targets * focal_weight * log_probs

            # Apply class weights
            if self.alpha is not None:
                alpha = self.alpha.to(logits.device)
                loss = loss * alpha.unsqueeze(0)

            loss = loss.sum(dim=1)

        else:
            # Hard labels — standard focal loss
            # Apply label smoothing
            if self.label_smoothing > 0:
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / num_classes)
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + self.label_smoothing / num_classes)
            
            log_probs = F.log_softmax(logits, dim=1)
            probs = torch.exp(log_probs)

            if self.label_smoothing > 0:
                focal_weight = (1 - probs) ** self.gamma
                loss = -(smooth_targets * focal_weight * log_probs).sum(dim=1)
            else:
                # Gather the probability for the true class
                p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

                focal_weight = (1 - p_t) ** self.gamma
                loss = -focal_weight * log_p_t

                # Apply class weights
                if self.alpha is not None:
                    alpha = self.alpha.to(logits.device)
                    alpha_t = alpha.gather(0, targets)
                    loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    Mixup: x = lambda * x_i + (1-lambda) * x_j
    Proven to improve generalization and calibration.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0, device="cuda"):
    """
    CutMix: paste a random patch from another image.
    Superior to Cutout as it preserves all pixels.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    # Generate random bounding box
    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = np.random.randint(0, W)
    cy = np.random.randint(0, H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda based on actual area
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def apply_mixup_cutmix(x, y, cfg, device="cuda"):
    """
    Randomly apply Mixup or CutMix based on config probabilities.
    Returns mixed data and targets for loss computation.
    """
    mixup_prob = cfg["training"].get("mixup_prob", 0.5)
    cutmix_prob = cfg["training"].get("cutmix_prob", 0.5)
    mixup_alpha = cfg["training"].get("mixup_alpha", 0.4)
    cutmix_alpha = cfg["training"].get("cutmix_alpha", 1.0)

    r = np.random.random()
    
    if r < mixup_prob:
        return mixup_data(x, y, alpha=mixup_alpha, device=device) + (True,)
    elif r < mixup_prob + cutmix_prob:
        return cutmix_data(x, y, alpha=cutmix_alpha, device=device) + (True,)
    else:
        return x, y, y, 1.0, False


def compute_mixed_loss(criterion, logits, y_a, y_b, lam, mixed=False):
    """Compute loss for mixed or normal samples."""
    if mixed and lam < 1.0:
        loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
    else:
        loss = criterion(logits, y_a)
    return loss
