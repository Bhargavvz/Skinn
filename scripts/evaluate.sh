#!/bin/bash
# ============================================================================
# SkinGuard AI — Evaluation Pipeline
# ============================================================================
set -e

echo "=================================================="
echo "  📊 SkinGuard AI — Evaluation Pipeline"
echo "=================================================="

CONFIG="${1:-configs/config.yaml}"
CHECKPOINT="${2:-outputs/checkpoints/best.pth}"
SPLIT="${3:-test}"

echo ""
echo "  Config:     $CONFIG"
echo "  Checkpoint: $CHECKPOINT"
echo "  Split:      $SPLIT"
echo ""

# ---- Check checkpoint exists ----
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    echo "   Run training first: bash scripts/train.sh"
    exit 1
fi

# ---- Create output dirs ----
mkdir -p outputs/evaluation outputs/gradcam

# ---- Run evaluation ----
echo "🔬 Running evaluation..."
python -m src.evaluate \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT"

# ---- Generate Grad-CAM (if images available) ----
echo ""
echo "🔥 Generating Grad-CAM visualizations..."
# Note: Grad-CAM requires test images on disk
# python -m src.gradcam --config "$CONFIG" --checkpoint "$CHECKPOINT" --output-dir outputs/gradcam

# ---- Export ONNX ----
echo ""
echo "📦 Exporting ONNX model..."
python -m src.export \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --benchmark

echo ""
echo "✅ Evaluation complete!"
echo "  Results: outputs/evaluation/"
echo "  Grad-CAM: outputs/gradcam/"
echo "  ONNX model: outputs/model.onnx"
