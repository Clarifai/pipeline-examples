#!/usr/bin/env python
"""
Chart Analysis Pipeline Step — Entry Point

A reusable Clarifai pipeline step that analyzes charts, graphs, and data
visualizations using any Clarifai-hosted VLM (Vision Language Model).

Plug into any pipeline via templateRef:
    templateRef:
        name: users/<USER>/apps/<APP>/pipeline_steps/chart-analysis-ps
        template: users/<USER>/apps/<APP>/pipeline_steps/chart-analysis-ps

Inputs (via argparse, mapped from pipeline_step_input_params):
    --image_url            URL of chart/graph image
    --image_base64         Base64-encoded chart image
    --image_path           Local path to chart image
    --analysis_type        general | data_extraction | trend_analysis | comparison | anomaly_detection
    --output_format        summary | detailed | json_table | markdown
    --vlm_model_url        Clarifai VLM model URL for chart analysis
    --vlm_inference_params JSON string with temperature, max_tokens, etc.
    --additional_context   Extra context about the chart domain/metrics
    --user_id              Clarifai user ID
    --app_id               Clarifai app ID
    --save_results         Whether to persist results to Clarifai
"""

import argparse
import inspect
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("chart-analysis-step")

# Local imports (co-located modules)
sys.path.insert(0, str(Path(__file__).parent))
from image_preprocessor import ImagePreprocessor
from vlm_analyzer import VLMAnalyzer


class ChartAnalysisStep:
    """
    Reusable Clarifai pipeline step: Chart/graph analysis via VLM.

    Follows the same introspection pattern used by Clarifai's training pipelines:
    - to_pipeline_parser() builds argparse from the analyze() signature.
    - The entry point parses CLI args and calls analyze(**vars(args)).

    Works with ANY Clarifai-hosted VLM — pass the model URL as a parameter.
    """

    @classmethod
    def _get_argparse_type(cls, annotation):
        """Convert Python type annotations to argparse-compatible types."""
        if annotation in (bool, "bool"):
            return lambda x: str(x).lower() in ("true", "1", "yes")
        elif annotation == int:
            return int
        elif annotation == float:
            return float
        return str

    @classmethod
    def to_pipeline_parser(cls):
        """Auto-generate argparse from the analyze() method signature."""
        parser = argparse.ArgumentParser(
            description="Chart Analysis Pipeline Step"
        )
        sig = inspect.signature(cls.analyze)
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            arg_type = cls._get_argparse_type(param.annotation)
            kwargs = {"type": arg_type, "help": param_name}
            if param.default != inspect.Parameter.empty:
                kwargs["default"] = param.default
            else:
                kwargs["required"] = True
            parser.add_argument(f"--{param_name}", **kwargs)
        return parser

    # ------------------------------------------------------------------
    # Core analysis method — its signature defines the pipeline interface.
    # ------------------------------------------------------------------

    def analyze(
        self,
        image_url: str = "",
        image_base64: str = "",
        image_path: str = "",
        analysis_type: str = "general",
        output_format: str = "detailed",
        vlm_model_url: str = "",
        vlm_inference_params: str = '{"temperature": 0.2, "max_tokens": 4096}',
        additional_context: str = "",
        user_id: str = "",
        app_id: str = "",
        save_results: bool = False,
    ) -> str:
        """
        Run a full chart/graph analysis using a VLM.

        Args:
            image_url: URL of the chart/graph image.
            image_base64: Base64-encoded chart image.
            image_path: Local file path to the chart image.
            analysis_type: Type of analysis to perform. One of:
                general, data_extraction, trend_analysis, comparison,
                anomaly_detection.
            output_format: Output format — summary, detailed, json_table,
                or markdown.
            vlm_model_url: Clarifai VLM model URL
                (e.g. https://clarifai.com/openai/chat-completion/models/gpt-4o).
            vlm_inference_params: JSON string with temperature, max_tokens, etc.
            additional_context: Extra domain context about the chart.
            user_id: Clarifai user ID (for saving results).
            app_id: Clarifai app ID (for saving results).
            save_results: Persist analysis results to Clarifai.

        Returns:
            JSON string with complete analysis results.
        """
        pat = os.environ.get("CLARIFAI_PAT", "")

        logger.info("=" * 60)
        logger.info("CHART ANALYSIS PIPELINE STEP")
        logger.info("=" * 60)

        # ── Phase 1: Load & validate image ────────────────────────────
        logger.info("[Phase 1/3] Loading chart image...")

        preprocessor = ImagePreprocessor()
        image_bytes, mime_type = preprocessor.load_image(
            image_url=image_url,
            image_base64=image_base64,
            image_path=image_path,
        )

        # Get dimensions for metadata
        dimensions = preprocessor.get_image_dimensions(image_bytes)
        dim_str = f"{dimensions[0]}x{dimensions[1]}" if dimensions else "unknown"
        logger.info(
            "Image loaded: %d bytes, %s, dimensions: %s",
            len(image_bytes), mime_type, dim_str,
        )

        # Resize if image is very large (VLMs have input limits)
        image_bytes = preprocessor.resize_if_needed(image_bytes, max_dim=2048)

        # ── Phase 2: Image metadata extraction ────────────────────────
        logger.info("[Phase 2/3] Preparing image metadata...")

        image_meta = {
            "source": "url" if image_url else ("base64" if image_base64 else "file"),
            "source_value": image_url or image_path or "(base64 input)",
            "mime_type": mime_type,
            "size_bytes": len(image_bytes),
            "dimensions": dim_str,
        }

        logger.info("Image metadata: %s", json.dumps(image_meta))

        # ── Phase 3: VLM Analysis ─────────────────────────────────────
        if vlm_model_url:
            logger.info("[Phase 3/3] Analyzing chart with VLM...")

            try:
                inference_params = json.loads(vlm_inference_params)
            except json.JSONDecodeError:
                logger.warning("Invalid vlm_inference_params JSON, using defaults.")
                inference_params = {"temperature": 0.2, "max_tokens": 4096}

            analyzer = VLMAnalyzer(
                model_url=vlm_model_url,
                pat=pat,
                inference_params=inference_params,
            )

            vlm_result = analyzer.analyze(
                image_bytes=image_bytes,
                analysis_type=analysis_type,
                additional_context=additional_context,
                output_format=output_format,
            )

            analysis_text = vlm_result["analysis_text"]
            logger.info("VLM analysis complete.")
        else:
            logger.info("[Phase 3/3] No vlm_model_url provided — generating placeholder.")
            analysis_text = self._no_vlm_placeholder(analysis_type, image_meta)

        # ── Compile Results ───────────────────────────────────────────
        results = {
            "status": "completed",
            "image_metadata": image_meta,
            "analysis_type": analysis_type,
            "output_format": output_format,
            "analysis": analysis_text,
            "model_used": vlm_model_url or "none",
        }

        # ── Optional: Save to Clarifai ────────────────────────────────
        if save_results and user_id and app_id and pat:
            logger.info("Saving results to Clarifai...")
            self._save_to_clarifai(results, user_id, app_id, pat)

        # ── Output ────────────────────────────────────────────────────
        results_json = json.dumps(results, indent=2)
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 60)
        logger.info("\n%s", analysis_text)

        # Print structured JSON to stdout (capturable by Argo workflows)
        print(results_json)

        return results_json

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _no_vlm_placeholder(analysis_type: str, image_meta: dict) -> str:
        """Generate a placeholder when no VLM is configured."""
        return (
            f"## Chart Analysis — VLM Required\n\n"
            f"Image loaded successfully:\n"
            f"- **Source:** {image_meta['source']} ({image_meta['source_value']})\n"
            f"- **Format:** {image_meta['mime_type']}\n"
            f"- **Size:** {image_meta['size_bytes']:,} bytes\n"
            f"- **Dimensions:** {image_meta['dimensions']}\n\n"
            f"**Analysis type requested:** {analysis_type}\n\n"
            f"To perform the actual analysis, provide a `vlm_model_url` "
            f"pointing to any Clarifai-hosted VLM:\n"
            f"- `https://clarifai.com/openai/chat-completion/models/gpt-4o`\n"
            f"- `https://clarifai.com/anthropic/completion/models/claude-sonnet-4`\n"
            f"- `https://clarifai.com/google/gemini/models/gemini-2.0-flash`\n"
            f"- `https://clarifai.com/meta/llama/models/llama-3.2-90b-vision`\n"
        )

    @staticmethod
    def _save_to_clarifai(
        results: dict, user_id: str, app_id: str, pat: str
    ):
        """Persist analysis results to Clarifai as a text input."""
        try:
            from clarifai.client.user import User

            user = User(user_id=user_id, pat=pat)
            app = user.app(app_id=app_id)
            dataset = app.dataset(dataset_id="chart-analyses")
            dataset.upload_from_bytes(
                input_id=f"chart-analysis-{os.urandom(8).hex()}",
                text_bytes=json.dumps(results, indent=2).encode("utf-8"),
                labels=["chart-analysis"],
            )
            logger.info("Results saved to Clarifai successfully.")
        except Exception as e:
            logger.warning("Failed to save results to Clarifai: %s", e)
            logger.info(
                "Results are still available in the pipeline run logs above."
            )


# ======================================================================
# Entry point — invoked by Clarifai's container runtime
# ======================================================================

def main():
    args = ChartAnalysisStep.to_pipeline_parser().parse_args()
    ChartAnalysisStep().analyze(**vars(args))


if __name__ == "__main__":
    main()
