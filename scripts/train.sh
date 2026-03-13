#!/bin/bash
# ============================================================================
# SkinGuard AI — Training Launcher
# Optimized for NVIDIA H100 GPU
# ============================================================================
set -e

echo "=================================================="
echo "  🏥 SkinGuard AI — Skin Cancer Detection"
echo "  Training Pipeline (H100 Optimized)"
echo "=================================================="

# ---- Environment Setup ----
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Enable BF16 and TF32 for H100
export NVIDIA_TF32_OVERRIDE=1

# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Check GPU ----
echo ""
echo "🔍 Checking GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'  CUDA: {torch.version.cuda}')
    print(f'  BF16 Support: {torch.cuda.is_bf16_supported()}')
else:
    print('  ⚠️  No GPU found! Training will be very slow.')
"

echo ""
echo "🚀 Starting training..."
echo "  Config: configs/config.yaml"
echo "  Time Budget: 7 hours"
echo ""

# ---- Create output dirs ----
mkdir -p outputs/logs outputs/checkpoints outputs/evaluation outputs/gradcam

# ---- Launch Training ----
python -m src.trainer \
    --config configs/config.yaml \
    "$@"

echo ""
echo "✅ Training complete!"
echo "  Best checkpoint: outputs/checkpoints/best.pth"
echo "  Logs: outputs/logs/"
echo ""
echo "Next steps:"
echo "  1. Evaluate: bash scripts/evaluate.sh"
echo "  2. Demo:     python app/gradio_app.py"
echo "  3. Export:    python -m src.export --config configs/config.yaml --benchmark"
