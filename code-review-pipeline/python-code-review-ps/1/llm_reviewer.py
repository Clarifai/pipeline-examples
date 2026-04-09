"""
LLM Review Module

Sends static-analysis findings and source code to any Clarifai-hosted
LLM/VLM model and returns a structured, human-readable code review.

The model is specified at runtime via `llm_model_url`, so this module is
model-agnostic — it works with GPT-4o, Claude, Llama, Mistral, or any
text-generation model available on the Clarifai platform.
"""

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("code-review-step.llm_reviewer")


SYSTEM_PROMPT = """\
You are an expert Python code reviewer. You will receive:

1. Python source code files.
2. Static analysis findings from pylint, flake8, and bandit.

Produce a clear, actionable code review structured as follows:

## Summary
A 2–3 sentence overall assessment with a quality score (1–10).

## Critical Issues
Security vulnerabilities, bugs, and errors that must be fixed.

## Improvements
Recommended changes for performance, readability, and maintainability.

## Style & Conventions
Minor style or convention suggestions.

## Positive Observations
What the code does well.

Rules:
- Reference specific file names and line numbers.
- Explain WHY something is an issue and HOW to fix it.
- Prioritize: focus on the most impactful issues first.
- If there are security findings from bandit, always highlight them prominently.
- Be concise — skip trivial issues when there are many findings.
"""


class LLMReviewer:
    """Generate code reviews using any Clarifai-hosted LLM/VLM."""

    def __init__(
        self,
        model_url: str,
        pat: str = "",
        inference_params: Optional[dict] = None,
    ):
        """
        Args:
            model_url: Full Clarifai model URL, e.g.
                https://clarifai.com/openai/chat-completion/models/gpt-4o
            pat: Personal Access Token (falls back to CLARIFAI_PAT env var).
            inference_params: Dict with temperature, max_tokens, etc.
        """
        self.model_url = model_url
        self.pat = pat or os.environ.get("CLARIFAI_PAT", "")
        self.inference_params = inference_params or {
            "temperature": 0.3,
            "max_tokens": 4096,
        }

    def generate_review(
        self, files: Dict[str, str], findings: List[dict]
    ) -> str:
        """
        Generate an LLM-powered code review.

        Args:
            files: {filepath: source_code} mapping.
            findings: Normalized finding dicts from StaticAnalyzer.

        Returns:
            Human-readable review text.
        """
        prompt = self._build_prompt(files, findings)

        try:
            return self._call_llm(prompt)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            logger.info("Falling back to findings-only summary.")
            return self._fallback_summary(findings)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self, files: Dict[str, str], findings: List[dict]
    ) -> str:
        """Build the user prompt with code context and findings."""
        parts: List[str] = []

        # ── Source code section ───────────────────────────────────────
        parts.append("## Source Code\n")
        total_chars = 0
        max_chars = 50_000  # ~12.5 K tokens; leave room for response
        files_included = 0

        for filepath, content in sorted(files.items()):
            if total_chars > max_chars:
                remaining = len(files) - files_included
                parts.append(f"\n... ({remaining} more file(s) truncated)")
                break
            if len(content) > 10_000:
                content = content[:10_000] + "\n# ... (truncated at 10 000 chars)"
            parts.append(f"\n### {filepath}\n```python\n{content}\n```\n")
            total_chars += len(content)
            files_included += 1

        # ── Findings section ──────────────────────────────────────────
        parts.append("\n## Static Analysis Findings\n")

        if findings:
            by_tool: Dict[str, list] = {}
            for f in findings:
                by_tool.setdefault(f["tool"], []).append(f)
            for tool, items in by_tool.items():
                parts.append(f"\n### {tool} ({len(items)} issues)\n")
                for item in items[:50]:  # cap per tool
                    parts.append(
                        f"- {item['file']}:{item['line']} "
                        f"[{item['severity'].upper()}] "
                        f"{item['rule']}: {item['message']}"
                    )
                if len(items) > 50:
                    parts.append(f"  ... ({len(items) - 50} more)")
        else:
            parts.append("No issues found by static analysis tools.\n")

        parts.append("\n## Your Review\n")
        parts.append(
            "Please provide your structured code review "
            "following the format above.\n"
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Clarifai model call
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Call the Clarifai-hosted model and return the text response."""
        # Import here so the module can be loaded even without the SDK
        # (useful for unit testing the prompt-building logic).
        try:
            from clarifai.client import Model
        except ImportError:
            from clarifai.client.model import Model

        logger.info("Calling LLM: %s", self.model_url)

        model = Model(url=self.model_url, pat=self.pat)

        # Combine system prompt + user prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\n---\n\n{prompt}"

        # Build inference params
        inf_params = {}
        if "temperature" in self.inference_params:
            inf_params["temperature"] = float(self.inference_params["temperature"])
        if "max_tokens" in self.inference_params:
            inf_params["max_tokens"] = int(self.inference_params["max_tokens"])

        response = model.predict_by_bytes(
            input_bytes=full_prompt.encode("utf-8"),
            input_type="text",
            inference_params=inf_params if inf_params else None,
        )

        # Extract text from the response
        if response.outputs:
            output = response.outputs[0]
            if hasattr(output, "data") and hasattr(output.data, "text"):
                text = output.data.text.raw
                if text:
                    return text

        raise RuntimeError(
            "Empty response from LLM model. "
            "Check that llm_model_url points to a valid text-generation model."
        )

    # ------------------------------------------------------------------
    # Fallback (no LLM available)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_summary(findings: List[dict]) -> str:
        """Return a plain-text summary when the LLM is unavailable."""
        if not findings:
            return "✅ No issues found. Code passed all static analysis checks."

        lines = [
            "## Code Review (Static Analysis Only)\n",
            f"Found **{len(findings)} issue(s)** across the reviewed files.\n",
            "*LLM summarization was unavailable — showing raw findings.*\n",
        ]

        by_sev: Dict[str, list] = {}
        for f in findings:
            by_sev.setdefault(f["severity"], []).append(f)

        for sev in ("error", "warning", "convention", "refactor", "info"):
            items = by_sev.get(sev, [])
            if not items:
                continue
            lines.append(f"\n### {sev.upper()} ({len(items)})\n")
            for item in items[:20]:
                lines.append(
                    f"- **{item['file']}:{item['line']}** "
                    f"[{item['tool']}/{item['rule']}] {item['message']}"
                )
            if len(items) > 20:
                lines.append(f"  ... and {len(items) - 20} more")

        return "\n".join(lines)
