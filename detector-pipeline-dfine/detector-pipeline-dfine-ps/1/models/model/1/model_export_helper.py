"""Export D-FINE checkpoint to ONNX — same as scripts/export_onnx.py but importable."""

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify
from transformers import DFineForObjectDetection, AutoImageProcessor

try:
    from clarifai.utils.logging import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class DFineONNXWrapper(nn.Module):
    """Wrapper to export only logits and pred_boxes from D-FINE model."""

    def __init__(self, model: DFineForObjectDetection):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> tuple:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits, outputs.pred_boxes


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 17,
    simplify_model: bool = True,
    dynamic_batch: bool = True,
):
    """Export D-FINE model to ONNX format.

    Args:
        checkpoint_path: Path to D-FINE HuggingFace checkpoint directory
        output_path: Output ONNX file path
        opset_version: ONNX opset version (17 recommended for D-FINE)
        simplify_model: Whether to simplify the ONNX model
        dynamic_batch: Whether to support dynamic batch size
    """
    logger.info(f"Loading model from {checkpoint_path}...")
    model = DFineForObjectDetection.from_pretrained(checkpoint_path)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(checkpoint_path)
    if isinstance(processor.size, dict):
        height = processor.size.get("height", 640)
        width = processor.size.get("width", 640)
    else:
        height = width = processor.size

    logger.info(f"Model loaded. Input size: {height}x{width}")

    wrapper = DFineONNXWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, height, width)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "pred_boxes": {0: "batch_size"},
        }

    logger.info(f"Exporting to ONNX with opset {opset_version}...")

    # Use dynamo=False to force legacy TorchScript exporter (avoids aten._is_all_true issue in torch>=2.6)
    export_kwargs = dict(
        f=output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes=dynamic_axes,
    )
    try:
        torch.onnx.export(wrapper, dummy_input, dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch doesn't have dynamo param
        torch.onnx.export(wrapper, dummy_input, **export_kwargs)

    logger.info(f"ONNX model exported to {output_path}")

    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verification passed!")

    if simplify_model:
        logger.info("Simplifying ONNX model...")
        simplified_model, check = simplify(onnx_model)
        if check:
            onnx.save(simplified_model, output_path)
            logger.info("Model simplified successfully!")
        else:
            logger.warning("Simplification check failed, keeping original model")

    logger.info(f"Export complete: {output_path}")
    return output_path
