#!/usr/bin/env python
"""
Python Code Review Pipeline Step — Entry Point

A reusable Clarifai pipeline step that performs static analysis on Python code
(pylint, flake8, bandit) and generates LLM-powered code reviews using any
Clarifai-hosted LLM/VLM model.

Plug into any pipeline via templateRef:
    templateRef:
        name: users/<USER>/apps/<APP>/pipeline_steps/python-code-review-ps
        template: users/<USER>/apps/<APP>/pipeline_steps/python-code-review-ps

Inputs (via argparse, mapped from pipeline_step_input_params):
    --repo_url             Git repository URL to review
    --branch               Branch to checkout (default: main)
    --file_patterns        Glob patterns for Python files (default: **/*.py)
    --code_text            Raw code to review (alternative to repo_url)
    --review_strictness    low | medium | high (default: medium)
    --llm_model_url        Clarifai model URL for LLM summarization
    --llm_inference_params JSON string with temperature, max_tokens, etc.
    --user_id              Clarifai user ID
    --app_id               Clarifai app ID
    --save_results         Whether to persist results to Clarifai
"""

import argparse
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("code-review-step")

# Local imports (co-located modules)
sys.path.insert(0, str(Path(__file__).parent))
from analyzers import StaticAnalyzer
from llm_reviewer import LLMReviewer


class CodeReviewStep:
    """
    Reusable Clarifai pipeline step: Python code review via static analysis + LLM.

    Follows the same introspection pattern used by Clarifai's training pipelines:
    - to_pipeline_parser() builds argparse from the review() signature.
    - The entry point parses CLI args and calls review(**vars(args)).

    Works with ANY Clarifai-hosted LLM — pass the model URL as a parameter.
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
        """Auto-generate argparse from the review() method signature.

        Mirrors the pattern in classifier-pipeline-resnet and
        detector-pipeline-yolof model.py classes.
        """
        parser = argparse.ArgumentParser(
            description="Python Code Review Pipeline Step"
        )
        sig = inspect.signature(cls.review)
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
    # Core review method — its signature defines the pipeline interface.
    # Every parameter here becomes a --flag in argparse AND a
    # pipeline_step_input_param in config.yaml.
    # ------------------------------------------------------------------

    def review(
        self,
        repo_url: str = "",
        branch: str = "main",
        file_patterns: str = "**/*.py",
        code_text: str = "",
        review_strictness: str = "medium",
        llm_model_url: str = "",
        llm_inference_params: str = '{"temperature": 0.3, "max_tokens": 4096}',
        user_id: str = "",
        app_id: str = "",
        save_results: bool = False,
    ) -> str:
        """
        Run a full Python code review: static analysis + LLM summary.

        Args:
            repo_url: Git repository URL to clone and review.
            branch: Git branch to checkout.
            file_patterns: Comma-separated glob patterns for files to scan.
            code_text: Raw Python code to review (alternative to repo_url).
            review_strictness: Filter severity — low (all), medium, high (errors only).
            llm_model_url: Any Clarifai model URL
                (e.g. https://clarifai.com/openai/chat-completion/models/gpt-4o).
            llm_inference_params: JSON string with temperature, max_tokens, etc.
            user_id: Clarifai user ID (for saving results).
            app_id: Clarifai app ID (for saving results).
            save_results: Persist review results to Clarifai as a text input.

        Returns:
            JSON string with complete review results.
        """
        pat = os.environ.get("CLARIFAI_PAT", "")

        logger.info("=" * 60)
        logger.info("PYTHON CODE REVIEW PIPELINE STEP")
        logger.info("=" * 60)

        # ── Phase 1: Acquire Code ────────────────────────────────────
        logger.info("[Phase 1/3] Acquiring code...")

        python_files: dict[str, str] = {}  # {relative_path: content}

        if code_text:
            logger.info("Mode: inline code text (%d chars)", len(code_text))
            python_files["inline_code.py"] = code_text
        elif repo_url:
            logger.info("Mode: repository clone — %s @ %s", repo_url, branch)
            python_files = self._clone_and_collect(repo_url, branch, file_patterns)
        else:
            raise ValueError(
                "Either --repo_url or --code_text must be provided."
            )

        logger.info("Collected %d Python file(s) for review.", len(python_files))

        if not python_files:
            logger.warning("No Python files found matching the given patterns.")
            return json.dumps({"status": "no_files", "findings": [], "review": ""})

        # ── Phase 2: Static Analysis ─────────────────────────────────
        logger.info("[Phase 2/3] Running static analysis (pylint, flake8, bandit)...")

        analyzer = StaticAnalyzer(strictness=review_strictness)
        findings = analyzer.analyze(python_files)

        logger.info("Static analysis complete — %d issue(s) found.", len(findings))

        # ── Phase 3: LLM Review ──────────────────────────────────────
        review_text = ""
        if llm_model_url:
            logger.info("[Phase 3/3] Generating LLM-powered review...")
            try:
                inference_params = json.loads(llm_inference_params)
            except json.JSONDecodeError:
                logger.warning("Invalid llm_inference_params JSON, using defaults.")
                inference_params = {"temperature": 0.3, "max_tokens": 4096}

            reviewer = LLMReviewer(
                model_url=llm_model_url,
                pat=pat,
                inference_params=inference_params,
            )
            review_text = reviewer.generate_review(python_files, findings)
            logger.info("LLM review generated successfully.")
        else:
            logger.info("[Phase 3/3] Skipped — no llm_model_url provided.")
            review_text = self._format_findings_as_text(findings)

        # ── Compile Results ───────────────────────────────────────────
        results = {
            "status": "completed",
            "files_reviewed": list(python_files.keys()),
            "total_findings": len(findings),
            "findings": findings,
            "review": review_text,
            "strictness": review_strictness,
        }

        # ── Optional: Save to Clarifai ────────────────────────────────
        if save_results and user_id and app_id and pat:
            logger.info("Saving results to Clarifai...")
            self._save_to_clarifai(results, user_id, app_id, pat)

        # ── Output ────────────────────────────────────────────────────
        results_json = json.dumps(results, indent=2)
        logger.info("\n" + "=" * 60)
        logger.info("REVIEW COMPLETE")
        logger.info("=" * 60)
        logger.info("\n%s", review_text)

        # Print structured JSON to stdout (capturable by Argo workflows)
        print(results_json)

        return results_json

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clone_and_collect(
        self, repo_url: str, branch: str, file_patterns: str
    ) -> dict[str, str]:
        """Clone a git repo and collect Python files matching the patterns."""
        work_dir = tempfile.mkdtemp(prefix="code-review-")
        repo_dir = os.path.join(work_dir, "repo")

        try:
            logger.info("Cloning %s (branch: %s)...", repo_url, branch)
            subprocess.run(
                [
                    "git", "clone",
                    "--depth", "1",
                    "--branch", branch,
                    repo_url, repo_dir,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Git clone failed: %s", e.stderr)
            raise RuntimeError(f"Failed to clone repository: {e.stderr}") from e
        except FileNotFoundError:
            raise RuntimeError(
                "git is not installed. Ensure git is in your Dockerfile."
            )

        python_files: dict[str, str] = {}
        patterns = [p.strip() for p in file_patterns.split(",")]
        repo_path = Path(repo_dir)

        for pattern in patterns:
            for filepath in repo_path.glob(pattern):
                if filepath.is_file() and filepath.suffix == ".py":
                    rel = str(filepath.relative_to(repo_path))
                    try:
                        python_files[rel] = filepath.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    except Exception as e:
                        logger.warning("Could not read %s: %s", rel, e)

        return python_files

    @staticmethod
    def _format_findings_as_text(findings: list) -> str:
        """Format static-analysis findings as readable text (no-LLM fallback)."""
        if not findings:
            return "✅ No issues found. Code looks clean!"

        lines = ["## Python Code Review Results\n"]
        by_severity: dict[str, list] = {}
        for f in findings:
            by_severity.setdefault(f.get("severity", "info"), []).append(f)

        for severity in ("error", "warning", "convention", "refactor", "info"):
            items = by_severity.get(severity, [])
            if not items:
                continue
            lines.append(f"\n### {severity.upper()} ({len(items)} issue{'s' if len(items) != 1 else ''})\n")
            for item in items:
                lines.append(
                    f"- **{item['file']}:{item['line']}** "
                    f"[{item['tool']}/{item['rule']}] {item['message']}"
                )

        lines.append(f"\n---\n**Total: {len(findings)} issue(s) found.**")
        return "\n".join(lines)

    @staticmethod
    def _save_to_clarifai(
        results: dict, user_id: str, app_id: str, pat: str
    ):
        """Persist review results to Clarifai as a text input."""
        try:
            from clarifai.client.user import User

            user = User(user_id=user_id, pat=pat)
            app = user.app(app_id=app_id)
            dataset = app.dataset(dataset_id="code-reviews")
            dataset.upload_from_bytes(
                input_id=f"review-{os.urandom(8).hex()}",
                text_bytes=json.dumps(results, indent=2).encode("utf-8"),
                labels=["code-review"],
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
    args = CodeReviewStep.to_pipeline_parser().parse_args()
    CodeReviewStep().review(**vars(args))


if __name__ == "__main__":
    main()
