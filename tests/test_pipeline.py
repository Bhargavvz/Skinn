"""
SkinGuard AI — Smoke Tests
Validates config, model, dataset, and loss components.
"""

import os
import sys
import pytest
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def config():
    """Load the project config."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


class TestConfig:
    """Test configuration loading and validation."""

    def test_config_loads(self, config):
        """Config YAML loads without errors."""
        assert config is not None
        assert "project" in config
        assert "data" in config
        assert "model" in config
        assert "training" in config

    def test_config_data_section(self, config):
        """Data config has required fields."""
        data = config["data"]
        assert "dataset_name" in data
        assert "image_size" in data
        assert "num_classes" in data
        assert data["num_classes"] == 8
        assert data["image_size"] > 0

    def test_config_model_section(self, config):
        """Model config has backbone definitions."""
        model = config["model"]
        assert "backbones" in model
        assert len(model["backbones"]) == 3
        for bb in model["backbones"]:
            assert "name" in bb
            assert "input_size" in bb

    def test_config_training_section(self, config):
        """Training config has required hyperparameters."""
        training = config["training"]
        assert "epochs" in training
        assert "batch_size" in training
        assert "optimizer" in training
        assert "scheduler" in training
        assert training["time_budget_hours"] == 7.0

    def test_config_h100_section(self, config):
        """H100 optimization config is present."""
        h100 = config["h100"]
        assert "bf16" in h100
        assert "tf32" in h100
        assert "torch_compile" in h100


class TestLoss:
    """Test loss function components."""

    def test_focal_loss_basic(self):
        """FocalLoss computes on hard labels."""
        from src.losses import FocalLoss
        
        criterion = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 8)
        labels = torch.randint(0, 8, (4,))
        
        loss = criterion(logits, labels)
        assert loss.shape == ()  # scalar
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_focal_loss_with_weights(self):
        """FocalLoss with class weights."""
        from src.losses import FocalLoss
        
        weights = torch.ones(8)
        weights[0] = 5.0  # extra weight for melanoma
        criterion = FocalLoss(gamma=2.0, alpha=weights)
        
        logits = torch.randn(4, 8)
        labels = torch.randint(0, 8, (4,))
        
        loss = criterion(logits, labels)
        assert not torch.isnan(loss)

    def test_focal_loss_label_smoothing(self):
        """FocalLoss with label smoothing."""
        from src.losses import FocalLoss
        
        criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
        logits = torch.randn(4, 8)
        labels = torch.randint(0, 8, (4,))
        
        loss = criterion(logits, labels)
        assert not torch.isnan(loss)

    def test_mixup_data(self):
        """Mixup produces valid mixed inputs."""
        from src.losses import mixup_data
        
        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, 8, (4,))
        
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4, device="cpu")
        
        assert mixed_x.shape == x.shape
        assert 0 <= lam <= 1

    def test_cutmix_data(self):
        """CutMix produces valid mixed inputs."""
        from src.losses import cutmix_data
        
        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, 8, (4,))
        
        mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0, device="cpu")
        
        assert mixed_x.shape == x.shape
        assert 0 <= lam <= 1


class TestDataset:
    """Test dataset components."""

    def test_label_constants(self):
        """Label mappings are consistent."""
        from src.dataset import LABEL_MAP, LABEL_NAMES, LABEL_DESCRIPTIONS
        
        assert len(LABEL_MAP) == 8
        assert len(LABEL_NAMES) == 8
        assert len(LABEL_DESCRIPTIONS) == 8
        assert LABEL_MAP["MEL"] == 0
        assert LABEL_MAP["SCC"] == 7

    def test_transforms(self, config):
        """Augmentation transforms produce correct shapes."""
        from src.dataset import get_train_transforms, get_val_transforms
        
        img_size = config["data"]["image_size"]
        
        train_t = get_train_transforms(config)
        val_t = get_val_transforms(config)
        
        # Test with dummy image
        dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        train_result = train_t(image=dummy)["image"]
        val_result = val_t(image=dummy)["image"]
        
        assert train_result.shape == (3, img_size, img_size)
        assert val_result.shape == (3, img_size, img_size)

    def test_skin_lesion_dataset(self):
        """SkinLesionDataset returns correct format."""
        from src.dataset import SkinLesionDataset
        
        # Create dummy data
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(10)]
        labels = [i % 8 for i in range(10)]
        
        dataset = SkinLesionDataset(images, labels, transform=None)
        
        assert len(dataset) == 10
        img, label, meta = dataset[0]
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long


class TestModuleImports:
    """Test that all modules import without errors."""

    def test_import_dataset(self):
        from src import dataset

    def test_import_models(self):
        from src import models

    def test_import_losses(self):
        from src import losses

    def test_import_tta(self):
        from src import tta

    def test_import_evaluate(self):
        from src import evaluate

    def test_import_gradcam(self):
        from src import gradcam

    def test_import_inference(self):
        from src import inference

    def test_import_export(self):
        from src import export


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
