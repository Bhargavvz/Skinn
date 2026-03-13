"""
Skin Cancer Detection — Model Export Pipeline
ONNX and TensorRT export for production deployment.
"""

import os
import time
import logging

import torch
import torch.nn as nn
import numpy as np
import yaml

from src.models import build_model

logger = logging.getLogger(__name__)


class ExportWrapper(nn.Module):
    """Wrapper that outputs only logits (no attention weights) for export."""

    def __init__(self, model):
        super().__init__()
        self.model = model._orig_mod if hasattr(model, "_orig_mod") else model

    def forward(self, x):
        logits, _ = self.model(x)
        return logits


def export_onnx(cfg, checkpoint_path=None, output_path=None):
    """
    Export model to ONNX format.
    
    Features:
        - Dynamic batch size support
        - Opset 17 for modern ops
        - Input/output naming for serving
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg["data"]["image_size"]
    num_classes = cfg["data"]["num_classes"]

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg["project"]["checkpoint_dir"], "best.pth")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Build model
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Wrap for export (logits only)
    export_model = ExportWrapper(model)
    export_model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, image_size, image_size).to(device)

    # Output path
    if output_path is None:
        output_path = cfg["export"]["onnx"].get("output_path", "outputs/model.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export
    opset = cfg["export"]["onnx"].get("opset_version", 17)
    dynamic_batch = cfg["export"]["onnx"].get("dynamic_batch", True)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    logger.info(f"Exporting to ONNX (opset={opset}, dynamic_batch={dynamic_batch})...")

    torch.onnx.export(
        export_model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"✅ ONNX model saved: {output_path} ({file_size:.1f} MB)")

    # Verify with ONNX Runtime
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Test inference
        test_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
        result = session.run([output_name], {input_name: test_input})

        logger.info(f"✅ ONNX verification passed — output shape: {result[0].shape}")
    except ImportError:
        logger.warning("onnxruntime not installed, skipping verification")
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")

    return output_path


def benchmark_inference(cfg, checkpoint_path=None, onnx_path=None, num_runs=100):
    """
    Benchmark inference latency: PyTorch vs ONNX.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg["data"]["image_size"]
    dummy_np = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    dummy_torch = torch.from_numpy(dummy_np).to(device)

    results = {}

    # PyTorch benchmark
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg["project"]["checkpoint_dir"], "best.pth")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model(cfg, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                model(dummy_torch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                model(dummy_torch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        results["pytorch"] = {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "p50_ms": np.percentile(times, 50),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
        }
        logger.info(f"PyTorch: {results['pytorch']['mean_ms']:.2f} ± {results['pytorch']['std_ms']:.2f} ms")

    # ONNX Runtime benchmark
    if onnx_path is None:
        onnx_path = cfg["export"]["onnx"].get("output_path", "outputs/model.onnx")

    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            session = ort.InferenceSession(onnx_path, providers=providers)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            # Warmup
            for _ in range(10):
                session.run([output_name], {input_name: dummy_np})

            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                session.run([output_name], {input_name: dummy_np})
                times.append((time.perf_counter() - start) * 1000)

            results["onnx"] = {
                "mean_ms": np.mean(times),
                "std_ms": np.std(times),
                "p50_ms": np.percentile(times, 50),
                "p95_ms": np.percentile(times, 95),
                "p99_ms": np.percentile(times, 99),
            }
            logger.info(f"ONNX: {results['onnx']['mean_ms']:.2f} ± {results['onnx']['std_ms']:.2f} ms")
        except ImportError:
            logger.warning("onnxruntime not installed")

    # Speedup
    if "pytorch" in results and "onnx" in results:
        speedup = results["pytorch"]["mean_ms"] / results["onnx"]["mean_ms"]
        logger.info(f"ONNX speedup: {speedup:.2f}x")
        results["speedup"] = speedup

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Skin Cancer Detection — Export")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--num-runs", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Export ONNX
    onnx_path = export_onnx(cfg, args.checkpoint, args.output)

    # Benchmark
    if args.benchmark:
        benchmark_inference(cfg, args.checkpoint, onnx_path, args.num_runs)


if __name__ == "__main__":
    main()
