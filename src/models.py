"""
Skin Cancer Detection — Ensemble Model Architecture
EVA-02-Large + ConvNeXt-V2-Large + Swin-V2-Base with attention fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging

logger = logging.getLogger(__name__)


class AttentionFusion(nn.Module):
    """
    Learnable attention-weighted fusion of multi-backbone features.
    Learns which backbone to trust more for each input.
    """

    def __init__(self, feature_dims, hidden_dim=512):
        """
        Args:
            feature_dims: list of feature dimensions from each backbone
            hidden_dim: dimension of fused representation
        """
        super().__init__()
        self.num_backbones = len(feature_dims)

        # Project each backbone's features to the same dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            for dim in feature_dims
        ])

        # Attention weights — learns which backbone to trust
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.num_backbones, self.num_backbones),
        )

        # Temperature for attention softmax
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, features_list):
        """
        Args:
            features_list: list of [B, D_i] tensors from each backbone
        Returns:
            fused: [B, hidden_dim] tensor
            attention_weights: [B, num_backbones] attention distribution
        """
        # Project all features to same dim
        projected = [proj(feat) for proj, feat in zip(self.projections, features_list)]

        # Stack for attention computation
        stacked = torch.stack(projected, dim=1)  # [B, N, hidden_dim]

        # Compute attention weights
        concat = torch.cat(projected, dim=-1)  # [B, N * hidden_dim]
        attn_logits = self.attention(concat)  # [B, N]
        attn_weights = F.softmax(attn_logits / self.temperature, dim=-1)  # [B, N]

        # Weighted sum
        fused = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, hidden_dim]

        return fused, attn_weights


class SkinCancerEnsemble(nn.Module):
    """
    Production-grade ensemble model for skin cancer classification.
    
    Architecture:
        1. EVA-02-Large (448px, 304M params) — best ImageNet accuracy
        2. ConvNeXt-V2-Large (384px, 198M params) — best CNN
        3. Swin-V2-Base (384px, 87M params) — efficient transformer
        
    Features are fused via learnable attention mechanism and classified
    through a shared head with dropout regularization.
    """

    def __init__(self, num_classes=8, cfg=None, pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        # Default backbone configs
        backbone_configs = [
            {"name": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", "input_size": 448},
            {"name": "convnextv2_large.fcmae_ft_in22k_in1k", "input_size": 384},
            {"name": "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k", "input_size": 384},
        ]

        if cfg is not None and "model" in cfg:
            backbone_configs = cfg["model"]["backbones"]

        # Load backbones from timm (Hugging Face)
        self.backbones = nn.ModuleList()
        self.backbone_names = []
        self.backbone_input_sizes = []
        feature_dims = []

        for bc in backbone_configs:
            name = bc["name"]
            logger.info(f"Loading backbone: {name} (pretrained={pretrained})")

            try:
                backbone = timm.create_model(
                    name,
                    pretrained=pretrained,
                    num_classes=0,  # Remove classification head → feature extractor
                    global_pool="avg",
                )
                feat_dim = backbone.num_features
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}. Using fallback.")
                # Fallback to a simpler model
                backbone = timm.create_model(
                    "convnext_base.fb_in22k_ft_in1k",
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg",
                )
                feat_dim = backbone.num_features

            self.backbones.append(backbone)
            self.backbone_names.append(name)
            self.backbone_input_sizes.append(bc.get("input_size", 384))
            feature_dims.append(feat_dim)
            logger.info(f"  → {name}: input={bc.get('input_size', 384)}px, feature_dim={feat_dim}, params={sum(p.numel() for p in backbone.parameters()) / 1e6:.1f}M")

        # Attention-based fusion
        hidden_dim = 512
        if cfg is not None and "model" in cfg:
            hidden_dim = cfg["model"]["fusion"].get("hidden_dim", 512)

        self.fusion = AttentionFusion(feature_dims, hidden_dim=hidden_dim)

        # Classification head
        dropout = 0.4
        if cfg is not None and "model" in cfg:
            dropout = cfg["model"]["fusion"].get("dropout", 0.4)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Store total params
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Ensemble — Total: {total_params / 1e6:.1f}M, Trainable: {trainable_params / 1e6:.1f}M")

    def freeze_backbones(self):
        """Freeze all backbone parameters (Phase 1: train only fusion + head)."""
        for backbone in self.backbones:
            for param in backbone.parameters():
                param.requires_grad = False
        logger.info("Backbones FROZEN — training fusion + classifier only")

    def unfreeze_backbones(self):
        """Unfreeze backbone parameters (Phase 2: fine-tune everything)."""
        for backbone in self.backbones:
            for param in backbone.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Backbones UNFROZEN — all {trainable / 1e6:.1f}M params trainable")

    def get_backbone_features(self, x):
        """Extract features from each backbone, resizing input to match expected size."""
        features = []
        _, _, H, W = x.shape
        for backbone, expected_size in zip(self.backbones, self.backbone_input_sizes):
            if H != expected_size or W != expected_size:
                resized = F.interpolate(x, size=(expected_size, expected_size), mode='bilinear', align_corners=False)
            else:
                resized = x
            feat = backbone(resized)
            features.append(feat)
        return features

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: [B, 3, H, W] input tensor
        Returns:
            logits: [B, num_classes]
            attention_weights: [B, num_backbones] — for interpretability
        """
        # Extract features from each backbone
        features = self.get_backbone_features(x)

        # Fuse features via attention
        fused, attn_weights = self.fusion(features)

        # Classify
        logits = self.classifier(fused)

        return logits, attn_weights

    def get_param_groups(self, head_lr, backbone_lr_multiplier=0.1):
        """
        Get parameter groups with different learning rates.
        Backbone params get lower lr for stable fine-tuning.
        """
        backbone_params = []
        for backbone in self.backbones:
            backbone_params.extend(list(backbone.parameters()))

        head_params = list(self.fusion.parameters()) + list(self.classifier.parameters())

        return [
            {"params": backbone_params, "lr": head_lr * backbone_lr_multiplier},
            {"params": head_params, "lr": head_lr},
        ]


def build_model(cfg, pretrained=True):
    """Build the ensemble model from config."""
    num_classes = cfg["data"]["num_classes"]
    model = SkinCancerEnsemble(
        num_classes=num_classes,
        cfg=cfg,
        pretrained=pretrained,
    )
    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
