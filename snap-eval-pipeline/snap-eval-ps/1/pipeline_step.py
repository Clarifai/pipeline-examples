#!/usr/bin/env python
"""
Snap-Moment Eval Pipeline Step — Entry Point

A Clarifai pipeline step that analyses evaluation results for a
snap-detection vision model trained on American football videos.

Inputs (via argparse, mapped from pipeline_step_input_params):
    --csv_path              Local path to eval-results CSV
    --csv_url               URL to download the CSV
    --csv_text              Raw CSV content as a string
    --failure_threshold     Seconds; abs error above this = FAIL (default 0.5)
    --top_n_worst           Number of worst predictions to list (default 20)
    --llm_model_url         Clarifai LLM URL for narrative summary
    --inference_params      JSON string with temperature, max_tokens
    --user_id               Clarifai user ID
    --app_id                Clarifai app ID
    --save_results          Whether to persist results to Clarifai
"""

import argparse
import inspect
import io
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("snap-eval-step")

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from eval_analyzer import EvalAnalyzer


class SnapEvalStep:
    """
    Clarifai pipeline step: analyse snap-detection eval CSVs.

    Follows the same introspection pattern used by all pipeline steps:
    - to_pipeline_parser() builds argparse from the analyze() signature.
    - The entry point parses CLI args and calls analyze(**vars(args)).
    """

    @classmethod
    def _get_argparse_type(cls, annotation):
        if annotation in (bool, "bool"):
            return lambda x: str(x).lower() in ("true", "1", "yes")
        elif annotation == int:
            return int
        elif annotation == float:
            return float
        return str

    @classmethod
    def to_pipeline_parser(cls):
        parser = argparse.ArgumentParser(
            description="Snap-Moment Eval Pipeline Step",
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
    # Core analyze method
    # ------------------------------------------------------------------

    def analyze(
        self,
        csv_path: str = "",
        csv_url: str = "",
        csv_text: str = "",
        failure_threshold: float = 0.5,
        top_n_worst: int = 20,
        llm_model_url: str = "",
        inference_params: str = '{"temperature": 0.3, "max_tokens": 4096}',
        user_id: str = "",
        app_id: str = "",
        save_results: bool = False,
    ) -> str:
        """
        Analyse snap-detection evaluation results.

        Args:
            csv_path: Local path to the evaluation CSV.
            csv_url: URL to download the CSV.
            csv_text: Raw CSV content as a string.
            failure_threshold: Seconds; abs error above this = FAIL.
            top_n_worst: Number of worst predictions to include.
            llm_model_url: Clarifai model URL for optional LLM narrative.
            inference_params: JSON string with inference parameters.
            user_id: Clarifai user ID.
            app_id: Clarifai app ID.
            save_results: Persist results to Clarifai.

        Returns:
            JSON string with full analysis results.
        """
        import pandas as pd

        pat = os.environ.get("CLARIFAI_PAT", "")

        logger.info("=" * 60)
        logger.info("SNAP-MOMENT EVAL PIPELINE STEP")
        logger.info("=" * 60)

        # ── Phase 1: Load CSV ─────────────────────────────────────────
        logger.info("[Phase 1/4] Loading evaluation CSV...")

        df = None
        source = "unknown"

        if csv_text:
            logger.info("Mode: inline CSV text (%d chars)", len(csv_text))
            df = pd.read_csv(io.StringIO(csv_text))
            source = "inline"
        elif csv_path:
            logger.info("Mode: local file — %s", csv_path)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            df = pd.read_csv(csv_path)
            source = csv_path
        elif csv_url:
            logger.info("Mode: URL download — %s", csv_url)
            import requests
            resp = requests.get(csv_url, timeout=60)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            source = csv_url
        else:
            raise ValueError(
                "Provide at least one of: csv_path, csv_url, or csv_text."
            )

        logger.info(
            "Loaded %d rows × %d columns from %s",
            len(df), len(df.columns), source,
        )
        logger.info("Columns: %s", list(df.columns))

        # ── Phase 2: Analyse ──────────────────────────────────────────
        logger.info("[Phase 2/4] Running evaluation analysis...")

        analyzer = EvalAnalyzer(
            failure_threshold=failure_threshold,
            top_n_worst=top_n_worst,
        )
        results = analyzer.analyze(df)

        if results.get("status") == "no_data":
            logger.warning("No valid data for analysis.")
            return json.dumps(results, indent=2)

        logger.info("Analysis complete.")

        # ── Phase 3: Generate Report ──────────────────────────────────
        logger.info("[Phase 3/4] Generating markdown report...")

        report_text = analyzer.format_report(results)
        results["report"] = report_text

        # ── Phase 4: Optional LLM Summary ─────────────────────────────
        if llm_model_url:
            logger.info("[Phase 4/4] Generating LLM narrative summary...")
            try:
                inf = json.loads(inference_params)
            except (json.JSONDecodeError, TypeError):
                inf = {"temperature": 0.3, "max_tokens": 4096}

            llm_summary = analyzer.generate_llm_summary(
                results=results,
                model_url=llm_model_url,
                pat=pat,
                inference_params=inf,
            )
            results["llm_summary"] = llm_summary
            logger.info("LLM summary generated.")
        else:
            logger.info("[Phase 4/4] Skipped — no llm_model_url provided.")

        # ── Compile ───────────────────────────────────────────────────
        results["source"] = source

        result_json = json.dumps(results, indent=2, default=str)

        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 60)

        # Print the report for visibility
        logger.info("\n%s", report_text)

        return result_json


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    step = SnapEvalStep()
    parser = step.to_pipeline_parser()
    args = parser.parse_args()
    print(step.analyze(**vars(args)))
