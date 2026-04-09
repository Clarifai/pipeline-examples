"""
Intent Classifier Module

Classifies user prompts into one of the registered skill intents using
a two-tier strategy:

    Tier 1 — Keyword & heuristic matching (zero-latency, no API calls).
    Tier 2 — LLM-based classification via any Clarifai-hosted model
             (used when Tier 1 confidence is low).

Skill descriptors are loaded dynamically from the SKILL.md files in
`.agents/skills/*/SKILL.md` via the `skill_loader` module.
Nothing is hardcoded — add a new SKILL.md and the router picks it up.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Re-export SkillDescriptor from skill_loader so existing imports work
from skill_loader import SkillDescriptor, load_skills

logger = logging.getLogger("smart-router.intent_classifier")


# ── Load Skills Dynamically ──────────────────────────────────────────
# All skill descriptors (name, when_to_use, keywords, patterns,
# execution_steps) are parsed from SKILL.md at import time.

SKILL_REGISTRY: Dict[str, SkillDescriptor] = load_skills(
    exclude=["clarifai-smart-router"],
)

if not SKILL_REGISTRY:
    logger.warning(
        "No skills loaded from SKILL.md files! "
        "Ensure .agents/skills/*/SKILL.md exist with proper frontmatter."
    )


# ── Classification Result ─────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """The output of intent classification."""

    intent: str                 # Skill name or "unknown"
    confidence: float           # 0.0 – 1.0
    method: str                 # "keyword", "pattern", "llm", "image_signal"
    reasoning: str              # Human-readable explanation
    all_scores: Dict[str, float] = field(default_factory=dict)


# ── Intent Classifier ─────────────────────────────────────────────────

class IntentClassifier:
    """
    Two-tier intent classifier for routing user prompts to the correct skill.

    Tier 1: Fast keyword + regex pattern matching (no API calls).
    Tier 2: LLM-based classification using any Clarifai model (optional).
    """

    # Confidence threshold — below this, escalate to LLM
    KEYWORD_CONFIDENCE_THRESHOLD = 0.35
    # Confidence gap — if top two skills are within this gap, use LLM
    AMBIGUITY_GAP = 0.15

    def __init__(
        self,
        llm_model_url: str = "",
        pat: str = "",
    ):
        self.llm_model_url = llm_model_url
        self.pat = pat or os.environ.get("CLARIFAI_PAT", "")
        self.skills = SKILL_REGISTRY

    def get_execution_steps(self, skill_name: str) -> List[str]:
        """Return the execution planning steps for a matched skill.

        These steps are loaded from the "## Execution Steps" section of
        the skill's SKILL.md file and should be injected into the LLM
        prompt to make task execution deterministic.
        """
        skill = self.skills.get(skill_name)
        if skill and skill.execution_steps:
            return skill.execution_steps
        return []

    def classify(
        self,
        user_prompt: str,
        has_image: bool = False,
        has_code: bool = False,
        has_csv: bool = False,
        force_llm: bool = False,
    ) -> ClassificationResult:
        """
        Classify the user's intent purely from the prompt text.

        The model can handle arbitrary image and code tasks on its own;
        skills are only invoked when the prompt clearly matches a
        registered skill's "When to Use" criteria.

        Args:
            user_prompt: The raw user prompt / task description.
            has_image: True if the user also provided an image input
                (used as a soft hint to boost chart_analysis score).
            has_code: True if the user also provided code / repo_url
                (used as a soft hint to boost code_review score).
            has_csv: True if the user also provided CSV data
                (used as a soft hint to boost snap_eval score).
            force_llm: Skip Tier 1, go straight to LLM classification.

        Returns:
            ClassificationResult with intent, confidence, and reasoning.
        """
        prompt_lower = user_prompt.lower().strip()

        if not prompt_lower:
            return ClassificationResult(
                intent="unknown", confidence=0.0,
                method="none", reasoning="Empty prompt.",
            )

        # NOTE: We intentionally do NOT auto-route based on input
        # modality (image / code).  The model handles arbitrary images
        # and code on its own; skills are only activated when the
        # prompt text clearly signals a registered skill.

        # ── Tier 1: Keyword + pattern matching ───────────────────────
        if not force_llm:
            tier1_result = self._tier1_classify(
                prompt_lower, has_image=has_image, has_code=has_code,
                has_csv=has_csv,
            )
            logger.info(
                "Tier 1 result: intent=%s, confidence=%.2f, method=%s",
                tier1_result.intent, tier1_result.confidence, tier1_result.method,
            )

            # Check if confident enough
            scores = sorted(tier1_result.all_scores.values(), reverse=True)
            if (
                tier1_result.confidence >= self.KEYWORD_CONFIDENCE_THRESHOLD
                and (len(scores) < 2 or (scores[0] - scores[1]) >= self.AMBIGUITY_GAP)
            ):
                return tier1_result

            logger.info(
                "Tier 1 confidence too low (%.2f) or ambiguous — escalating to Tier 2.",
                tier1_result.confidence,
            )

        # ── Tier 2: LLM classification ───────────────────────────────
        if self.llm_model_url:
            try:
                tier2_result = self._tier2_llm_classify(user_prompt)
                logger.info(
                    "Tier 2 result: intent=%s, confidence=%.2f",
                    tier2_result.intent, tier2_result.confidence,
                )
                return tier2_result
            except Exception as exc:
                logger.warning("Tier 2 LLM classification failed: %s", exc)
                # Fall back to Tier 1 result
                if not force_llm:
                    return tier1_result

        # If we got here, return Tier 1 result (even if low confidence)
        if not force_llm:
            return tier1_result

        return ClassificationResult(
            intent="unknown", confidence=0.0,
            method="none", reasoning="Classification failed — no LLM available.",
        )

    # ------------------------------------------------------------------
    # Tier 1: Keyword + Pattern Matching
    # ------------------------------------------------------------------

    def _tier1_classify(
        self,
        prompt_lower: str,
        has_image: bool = False,
        has_code: bool = False,
        has_csv: bool = False,
    ) -> ClassificationResult:
        """Fast keyword + regex classification.

        Input modality flags (has_image, has_code, has_csv) are applied
        as a small bonus (+0.10) to the relevant skill — enough to
        break ties but never enough to override a clear prompt match
        for a different skill.
        """
        scores: Dict[str, float] = {}
        match_details: Dict[str, List[str]] = {}

        for skill_name, skill in self.skills.items():
            score = 0.0
            details = []

            # Keyword matching (weighted by specificity)
            keyword_hits = 0
            for kw in skill.keywords:
                if kw.lower() in prompt_lower:
                    # Multi-word keywords score higher (more specific)
                    word_count = len(kw.split())
                    hit_score = 0.15 * word_count
                    score += hit_score
                    keyword_hits += 1
                    details.append(f"keyword:{kw}")

            # Pattern matching (higher weight — patterns are more precise)
            pattern_hits = 0
            for pat in skill.patterns:
                if re.search(pat, prompt_lower):
                    score += 0.25
                    pattern_hits += 1
                    details.append(f"pattern:{pat[:40]}")

            # File extension detection in the prompt
            for ext in skill.file_extensions:
                if ext in prompt_lower:
                    score += 0.10
                    details.append(f"extension:{ext}")

            # Soft modality hint — small bonus, never enough on its own
            if has_image and skill.has_image_input:
                score += 0.10
                details.append("hint:image_input")
            if has_code and not skill.has_image_input:
                score += 0.10
                details.append("hint:code_input")
            if has_csv and ".csv" in skill.file_extensions:
                score += 0.10
                details.append("hint:csv_input")

            # Normalize: cap at 1.0
            score = min(score, 1.0)
            scores[skill_name] = score
            match_details[skill_name] = details

        # Pick the winner
        if not scores or max(scores.values()) == 0:
            return ClassificationResult(
                intent="unknown", confidence=0.0,
                method="keyword", reasoning="No keyword or pattern matches found.",
                all_scores=scores,
            )

        best_skill = max(scores, key=scores.get)
        best_score = scores[best_skill]
        details_str = ", ".join(match_details[best_skill][:5])

        return ClassificationResult(
            intent=best_skill,
            confidence=best_score,
            method="keyword" if not any("pattern:" in d for d in match_details[best_skill]) else "pattern",
            reasoning=f"Matched on: {details_str}",
            all_scores=scores,
        )

    # ------------------------------------------------------------------
    # Tier 2: LLM-based Classification
    # ------------------------------------------------------------------

    def _tier2_llm_classify(self, user_prompt: str) -> ClassificationResult:
        """Use a Clarifai LLM to classify the intent."""
        from clarifai.client.model import Model

        # Build the classification prompt with skill descriptors
        skill_descriptions = []
        for skill in self.skills.values():
            when_to_use = "\n".join(f"  - {line}" for line in skill.when_to_use)
            skill_descriptions.append(
                f"### {skill.name} — {skill.display_name}\n"
                f"When to use:\n{when_to_use}"
            )

        system_prompt = (
            "You are an intent classification system. Your job is to determine "
            "which skill should handle a user's request.\n\n"
            "## Available Skills\n\n"
            + "\n\n".join(skill_descriptions)
            + "\n\n"
            "## Instructions\n"
            "Given the user's prompt, decide which skill is the best match.\n"
            "Respond ONLY with a JSON object — no other text:\n"
            '{"intent": "<skill_name>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}\n\n'
            "If no skill matches, use intent=\"unknown\".\n"
            "The skill_name MUST be one of: "
            + ", ".join(self.skills.keys())
            + ", or \"unknown\"."
        )

        full_prompt = f"{system_prompt}\n\n## User Prompt\n{user_prompt}"

        model = Model(url=self.llm_model_url, pat=self.pat)
        response = model.predict_by_bytes(
            input_bytes=full_prompt.encode("utf-8"),
            input_type="text",
            inference_params={"temperature": 0.0, "max_tokens": 256},
        )

        # Parse the LLM response
        raw = ""
        if response.outputs:
            output = response.outputs[0]
            if hasattr(output, "data") and hasattr(output.data, "text"):
                raw = output.data.text.raw

        if not raw:
            raise RuntimeError("Empty response from classification LLM.")

        # Extract JSON from response (handle markdown fences)
        json_match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if not json_match:
            raise RuntimeError(f"Could not parse LLM response as JSON: {raw[:200]}")

        parsed = json.loads(json_match.group())
        intent = parsed.get("intent", "unknown")
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = parsed.get("reasoning", "LLM classification")

        # Validate intent is a known skill
        if intent not in self.skills and intent != "unknown":
            logger.warning("LLM returned unknown skill '%s', defaulting to unknown.", intent)
            intent = "unknown"

        return ClassificationResult(
            intent=intent,
            confidence=confidence,
            method="llm",
            reasoning=reasoning,
            all_scores={intent: confidence},
        )
