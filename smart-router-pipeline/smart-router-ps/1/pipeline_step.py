#!/usr/bin/env python
"""
Smart Router Pipeline Step — Entry Point

A unified Clarifai pipeline step that acts as an intelligent router:
it classifies the user's intent from a natural-language prompt and
delegates to the appropriate downstream skill:

    • code_review    → Python Code Review (static analysis + LLM)
    • chart_analysis  → Chart/Graph Analysis (image + VLM)
    • snap_eval       → Snap-Moment Eval Analysis (CSV + stats + LLM)

The intent classifier uses a two-tier strategy:
    Tier 1: Fast keyword + regex pattern matching (zero-latency).
    Tier 2: LLM-based classification via any Clarifai model (fallback).

All parameters from both sub-skills are accepted as a flat superset,
so callers never need to know which skill will handle the request.

Plug into any pipeline via templateRef:
    templateRef:
        name: users/<USER>/apps/<APP>/pipeline_steps/smart-router-ps
        template: users/<USER>/apps/<APP>/pipeline_steps/smart-router-ps
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
logger = logging.getLogger("smart-router")

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from intent_classifier import IntentClassifier


class SmartRouterStep:
    """
    Unified router step: classifies intent, then delegates to the
    correct downstream skill (code review or chart analysis).
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
            description="Smart Router Pipeline Step"
        )
        sig = inspect.signature(cls.route)
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
    # Unified route method — superset of both skill interfaces.
    # The user_prompt is the key input; everything else is optional
    # and gets forwarded to the matched skill.
    # ------------------------------------------------------------------

    def route(
        self,
        # ── Router params ──
        user_prompt: str = "",
        model_url: str = "",
        # ── Code Review params ──
        repo_url: str = "",
        branch: str = "main",
        file_patterns: str = "**/*.py",
        code_text: str = "",
        review_strictness: str = "medium",
        # ── Chart Analysis params ──
        image_url: str = "",
        image_base64: str = "",
        image_path: str = "",
        analysis_type: str = "general",
        output_format: str = "detailed",
        additional_context: str = "",
        # ── Snap-Eval params ──
        csv_path: str = "",
        csv_url: str = "",
        csv_text: str = "",
        failure_threshold: float = 0.5,
        top_n_worst: int = 20,
        # ── Shared params ──
        inference_params: str = '{"temperature": 0.3, "max_tokens": 4096}',
        user_id: str = "",
        app_id: str = "",
        save_results: bool = False,
    ) -> str:
        """
        Classify user intent and route to the correct skill.

        Args:
            user_prompt: Natural-language description of what the user wants.
                Used for intent classification. Can also serve as code_text
                or additional_context depending on the matched skill.
            model_url: Clarifai model URL — used as the LLM for code review
                OR the VLM for chart analysis, AND for LLM-based intent
                classification (Tier 2).
            repo_url: Git repository URL (→ code review).
            branch: Git branch (→ code review).
            file_patterns: Glob patterns (→ code review).
            code_text: Raw Python code (→ code review).
            review_strictness: low | medium | high (→ code review).
            image_url: Chart image URL (→ chart analysis).
            image_base64: Base64 chart image (→ chart analysis).
            image_path: Local chart image path (→ chart analysis).
            analysis_type: general | data_extraction | ... (→ chart analysis).
            output_format: summary | detailed | ... (→ chart analysis).
            additional_context: Extra context (→ chart analysis).
            csv_path: Local path to eval CSV file (→ snap eval).
            csv_url: URL to eval CSV file (→ snap eval).
            csv_text: Raw CSV string (→ snap eval).
            failure_threshold: Error threshold in seconds for pass/fail (→ snap eval).
            top_n_worst: Number of worst predictions to list (→ snap eval).
            inference_params: JSON inference params for the model.
            user_id: Clarifai user ID.
            app_id: Clarifai app ID.
            save_results: Persist results to Clarifai.

        Returns:
            JSON string with classification result + skill output.
        """
        pat = os.environ.get("CLARIFAI_PAT", "")

        logger.info("=" * 70)
        logger.info("SMART ROUTER PIPELINE STEP")
        logger.info("=" * 70)

        # ── Phase 1: Intent Classification ────────────────────────────
        logger.info("[Phase 1/3] Classifying intent...")

        # Detect implicit signals from provided inputs
        has_image = bool(image_url or image_base64 or image_path)
        has_code = bool(code_text or repo_url)
        has_csv = bool(csv_path or csv_url or csv_text)

        classifier = IntentClassifier(
            llm_model_url=model_url,
            pat=pat,
        )

        classification = classifier.classify(
            user_prompt=user_prompt,
            has_image=has_image,
            has_code=has_code,
            has_csv=has_csv,
        )

        logger.info(
            "Intent classified: %s (confidence=%.2f, method=%s)",
            classification.intent, classification.confidence, classification.method,
        )
        logger.info("Reasoning: %s", classification.reasoning)

        # Retrieve execution steps for the matched skill (from SKILL.md)
        execution_steps = classifier.get_execution_steps(classification.intent)
        if execution_steps:
            logger.info(
                "Loaded %d execution planning steps for '%s'.",
                len(execution_steps), classification.intent,
            )
        else:
            logger.info("No execution steps defined for '%s'.", classification.intent)

        # ── Phase 2: Route to Skill ───────────────────────────────────
        logger.info("[Phase 2/3] Routing to skill: %s", classification.intent)

        skill_output = {}
        if classification.intent == "code_review":
            skill_output = self._run_code_review(
                repo_url=repo_url,
                branch=branch,
                file_patterns=file_patterns,
                code_text=code_text or user_prompt,  # Fall back to prompt as code
                review_strictness=review_strictness,
                llm_model_url=model_url,
                inference_params=inference_params,
                user_id=user_id,
                app_id=app_id,
                save_results=save_results,
                pat=pat,
                execution_steps=execution_steps,
                user_prompt=user_prompt,
            )
        elif classification.intent == "chart_analysis":
            skill_output = self._run_chart_analysis(
                image_url=image_url,
                image_base64=image_base64,
                image_path=image_path,
                analysis_type=analysis_type,
                output_format=output_format,
                vlm_model_url=model_url,
                inference_params=inference_params,
                additional_context=additional_context or user_prompt,
                user_id=user_id,
                app_id=app_id,
                save_results=save_results,
                pat=pat,
                execution_steps=execution_steps,
                user_prompt=user_prompt,
            )
        elif classification.intent == "snap_eval":
            skill_output = self._run_snap_eval(
                csv_path=csv_path,
                csv_url=csv_url,
                csv_text=csv_text,
                failure_threshold=failure_threshold,
                top_n_worst=top_n_worst,
                llm_model_url=model_url,
                inference_params=inference_params,
                user_id=user_id,
                app_id=app_id,
                save_results=save_results,
                pat=pat,
                execution_steps=execution_steps,
                user_prompt=user_prompt,
            )
        else:
            logger.warning(
                "No skill matched — passing prompt directly to model (no skill prompts)."
            )
            skill_output = self._run_passthrough(
                user_prompt=user_prompt,
                model_url=model_url,
                inference_params=inference_params,
                pat=pat,
            )

        # ── Phase 3: Compile Final Response ───────────────────────────
        logger.info("[Phase 3/3] Compiling final response...")

        result = {
            "router": {
                "intent": classification.intent,
                "confidence": round(classification.confidence, 3),
                "method": classification.method,
                "reasoning": classification.reasoning,
                "all_scores": {
                    k: round(v, 3) for k, v in classification.all_scores.items()
                },
                "execution_steps_used": len(execution_steps),
            },
            "skill_output": skill_output,
        }

        result_json = json.dumps(result, indent=2, default=str)

        logger.info("=" * 70)
        logger.info("ROUTING COMPLETE")
        logger.info("=" * 70)
        logger.info("\n%s", result_json)

        return result_json

    # ------------------------------------------------------------------
    # Skill Runners
    # ------------------------------------------------------------------

    def _run_code_review(
        self, *, repo_url, branch, file_patterns, code_text,
        review_strictness, llm_model_url, inference_params,
        user_id, app_id, save_results, pat,
        execution_steps=None, user_prompt="",
    ) -> dict:
        """Import and run the Code Review skill.

        If execution_steps are provided (from SKILL.md), they are
        prepended to the user prompt so the LLM follows a deterministic
        plan when generating its review.
        """
        logger.info("Executing: Python Code Review skill")

        # Build an enhanced prompt that includes execution planning steps
        enhanced_code_text = code_text
        if execution_steps and llm_model_url:
            plan = self._build_planning_prefix(execution_steps, user_prompt)
            # Prepend the planning steps as a comment block so the LLM
            # sees them alongside the code.  The actual code_text is
            # passed through unchanged — the plan is appended as context.
            logger.info(
                "Injecting %d execution steps into LLM context.",
                len(execution_steps),
            )
            # We inject via the code_text because that's what gets sent
            # to the LLM. Wrap the plan so it doesn't break code parsing.
            enhanced_code_text = (
                f"{code_text}\n\n"
                f"# ── EXECUTION PLAN (from SKILL.md) ──\n"
                f"# The following steps guide this review:\n"
                + "".join(f"# Step {i+1}: {s}\n" for i, s in enumerate(execution_steps))
                + f"#\n# User request: {user_prompt}\n"
            )

        # Dynamically import from the code-review pipeline step
        code_review_dir = str(
            Path(__file__).parent.parent.parent.parent
            / "code-review-pipeline" / "python-code-review-ps" / "1"
        )

        # Save and manipulate sys.path + modules to avoid name collisions
        import importlib
        saved_module = sys.modules.pop("pipeline_step", None)
        sys.path.insert(0, code_review_dir)
        try:
            cr_module = importlib.import_module("pipeline_step")
            CodeReviewStep = cr_module.CodeReviewStep
        finally:
            # Restore original module
            sys.path.remove(code_review_dir)
            if saved_module is not None:
                sys.modules["pipeline_step"] = saved_module

        step = CodeReviewStep()
        result_json = step.review(
            repo_url=repo_url,
            branch=branch,
            file_patterns=file_patterns,
            code_text=enhanced_code_text,
            review_strictness=review_strictness,
            llm_model_url=llm_model_url,
            llm_inference_params=inference_params,
            user_id=user_id,
            app_id=app_id,
            save_results=save_results,
        )

        try:
            return json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            return {"raw_output": str(result_json)}

    def _run_chart_analysis(
        self, *, image_url, image_base64, image_path, analysis_type,
        output_format, vlm_model_url, inference_params, additional_context,
        user_id, app_id, save_results, pat,
        execution_steps=None, user_prompt="",
    ) -> dict:
        """Import and run the Chart Analysis skill.

        If execution_steps are provided (from SKILL.md), they are
        injected into the additional_context so the VLM follows a
        deterministic plan when analyzing the chart.
        """
        logger.info("Executing: Chart Analysis skill")

        # Enhance additional_context with execution planning steps
        enhanced_context = additional_context
        if execution_steps and vlm_model_url:
            plan = self._build_planning_prefix(execution_steps, user_prompt)
            logger.info(
                "Injecting %d execution steps into VLM context.",
                len(execution_steps),
            )
            enhanced_context = (
                f"{plan}\n\n"
                f"User request: {user_prompt}\n\n"
                f"{additional_context}"
            ).strip()

        chart_analysis_dir = str(
            Path(__file__).parent.parent.parent.parent
            / "chart-analysis-pipeline" / "chart-analysis-ps" / "1"
        )

        import importlib
        saved_module = sys.modules.pop("pipeline_step", None)
        sys.path.insert(0, chart_analysis_dir)
        try:
            ca_module = importlib.import_module("pipeline_step")
            ChartAnalysisStep = ca_module.ChartAnalysisStep
        finally:
            sys.path.remove(chart_analysis_dir)
            if saved_module is not None:
                sys.modules["pipeline_step"] = saved_module

        step = ChartAnalysisStep()

        # Guard: chart analysis requires at least one image input
        if not any([image_url, image_base64, image_path]):
            return {
                "status": "error",
                "message": (
                    "Chart analysis was selected but no image was provided. "
                    "Please supply one of: image_url, image_base64, or image_path."
                ),
            }

        result_json = step.analyze(
            image_url=image_url,
            image_base64=image_base64,
            image_path=image_path,
            analysis_type=analysis_type,
            output_format=output_format,
            vlm_model_url=vlm_model_url,
            vlm_inference_params=inference_params,
            additional_context=enhanced_context,
            user_id=user_id,
            app_id=app_id,
            save_results=save_results,
        )

        try:
            return json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            return {"raw_output": str(result_json)}

    def _run_snap_eval(
        self, *, csv_path, csv_url, csv_text, failure_threshold,
        top_n_worst, llm_model_url, inference_params,
        user_id, app_id, save_results, pat,
        execution_steps=None, user_prompt="",
    ) -> dict:
        """Import and run the Snap-Moment Eval Analysis skill.

        If execution_steps are provided (from SKILL.md), they are
        logged for traceability but the eval analyser already follows
        a deterministic analysis pipeline internally.
        """
        logger.info("Executing: Snap-Moment Eval Analysis skill")

        if execution_steps:
            logger.info(
                "Execution plan loaded (%d steps) — eval analyser will "
                "follow its internal pipeline.",
                len(execution_steps),
            )

        snap_eval_dir = str(
            Path(__file__).parent.parent.parent.parent
            / "snap-eval-pipeline" / "snap-eval-ps" / "1"
        )

        import importlib
        saved_module = sys.modules.pop("pipeline_step", None)
        sys.path.insert(0, snap_eval_dir)
        try:
            se_module = importlib.import_module("pipeline_step")
            SnapEvalStep = se_module.SnapEvalStep
        finally:
            sys.path.remove(snap_eval_dir)
            if saved_module is not None:
                sys.modules["pipeline_step"] = saved_module

        step = SnapEvalStep()

        # Guard: snap eval requires at least one CSV input
        if not any([csv_path, csv_url, csv_text]):
            return {
                "status": "error",
                "message": (
                    "Snap-eval analysis was selected but no CSV data was "
                    "provided. Please supply one of: csv_path, csv_url, "
                    "or csv_text."
                ),
            }

        result_json = step.analyze(
            csv_path=csv_path,
            csv_url=csv_url,
            csv_text=csv_text,
            failure_threshold=failure_threshold,
            top_n_worst=top_n_worst,
            llm_model_url=llm_model_url,
            inference_params=inference_params,
            user_id=user_id,
            app_id=app_id,
            save_results=save_results,
        )

        try:
            return json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            return {"raw_output": str(result_json)}

    def _build_planning_prefix(
        self, execution_steps: list, user_prompt: str = "",
    ) -> str:
        """Build a planning prefix string from execution steps.

        This is injected into the LLM/VLM context so the model follows
        a deterministic execution plan defined in the skill's SKILL.md.
        """
        if not execution_steps:
            return ""

        lines = [
            "=== EXECUTION PLAN ===",
            "Follow these steps in order to complete the task:",
            "",
        ]
        for i, step in enumerate(execution_steps, 1):
            lines.append(f"  Step {i}. {step}")
        lines.append("")
        lines.append("=== END PLAN ===")
        return "\n".join(lines)

    def _run_passthrough(
        self, *, user_prompt, model_url, inference_params, pat,
    ) -> dict:
        """
        No skill matched — forward the raw user prompt to the model
        without any skill-specific system prompts or processing.

        This ensures the user always gets a response, while the router
        metadata clearly flags that no route was matched.
        """
        logger.info("Executing: Passthrough (no skill matched)")

        if not model_url:
            return {
                "status": "no_route_matched",
                "routed": False,
                "message": (
                    "⚠ No skill matched the prompt and no model_url was "
                    "provided to handle it as a general query. Please either "
                    "rephrase your request or supply a model_url."
                ),
                "user_prompt": user_prompt,
            }

        try:
            from clarifai.client.model import Model

            try:
                inf = json.loads(inference_params)
            except (json.JSONDecodeError, TypeError):
                inf = {"temperature": 0.3, "max_tokens": 4096}

            model = Model(url=model_url, pat=pat)
            response = model.predict_by_bytes(
                input_bytes=user_prompt.encode("utf-8"),
                input_type="text",
                inference_params={
                    k: float(v) if k == "temperature" else int(v)
                    for k, v in inf.items()
                },
            )

            raw_text = ""
            if response.outputs:
                output = response.outputs[0]
                if hasattr(output, "data") and hasattr(output.data, "text"):
                    raw_text = output.data.text.raw

            return {
                "status": "completed",
                "routed": False,
                "note": (
                    "⚠ No registered skill matched this prompt. The request "
                    "was forwarded directly to the model without any "
                    "skill-specific prompts or processing."
                ),
                "model_used": model_url,
                "response": raw_text or "(empty response from model)",
            }

        except Exception as exc:
            logger.error("Passthrough model call failed: %s", exc)
            return {
                "status": "error",
                "routed": False,
                "note": (
                    "⚠ No registered skill matched this prompt and the "
                    "direct model call also failed."
                ),
                "error": str(exc),
                "user_prompt": user_prompt,
            }


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    step = SmartRouterStep()
    parser = step.to_pipeline_parser()
    args = parser.parse_args()
    print(step.route(**vars(args)))
