"""
Skill Loader — Dynamic SKILL.md Parser

Discovers SKILL.md files in `.agents/skills/*/SKILL.md`, parses them to
extract all SkillDescriptor fields dynamically:

    • name, display_name      — from YAML frontmatter
    • when_to_use              — from "## When to Use" section
    • keywords & patterns      — auto-generated from when_to_use + description
    • execution_steps           — from "## Execution Steps" section
    • file_extensions, has_image_input, priority — inferred from content

This replaces all hardcoded skill descriptors in intent_classifier.py.
Every skill is defined in exactly one place: its SKILL.md file.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("smart-router.skill_loader")

# ── Paths ─────────────────────────────────────────────────────────────
# The workspace root: walk up from smart-router-ps/1/ to pipeline-examples/
_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parent.parent.parent  # pipeline-examples/
_SKILLS_DIR = _WORKSPACE_ROOT / ".agents" / "skills"


# ── Data Structures ───────────────────────────────────────────────────

@dataclass
class SkillDescriptor:
    """Defines one routable skill with intent signals loaded from SKILL.md."""

    name: str                          # Unique skill ID  (e.g. "code_review")
    display_name: str                  # Human-readable   (e.g. "Python Code Review")
    when_to_use: List[str]             # Lines from "## When to Use"
    keywords: List[str]                # Auto-generated keywords
    patterns: List[str]                # Auto-generated regex patterns
    execution_steps: List[str]         # Steps from "## Execution Steps"
    file_extensions: List[str] = field(default_factory=list)
    has_image_input: bool = False
    priority: int = 0


# ── SKILL.md Section Parser ──────────────────────────────────────────

def _parse_frontmatter(text: str) -> dict:
    """Extract YAML-ish frontmatter between --- markers.

    We do a lightweight key: value parse instead of pulling in PyYAML,
    since frontmatter in SKILL.md files is simple flat YAML.
    """
    fm = {}
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return fm

    block = match.group(1)
    current_key = None
    current_val_lines: List[str] = []

    for line in block.split("\n"):
        # Handle continuation of multi-line value (indented or >)
        if current_key and (line.startswith("  ") or line.strip() == ""):
            current_val_lines.append(line.strip())
            continue

        # Key: value line
        kv = re.match(r"^(\w[\w\-]*):\s*(.*)", line)
        if kv:
            # Save previous key
            if current_key:
                fm[current_key] = " ".join(
                    l for l in current_val_lines if l
                ).strip().strip(">").strip('"').strip("'")
            current_key = kv.group(1)
            val = kv.group(2).strip().strip(">").strip('"').strip("'")
            current_val_lines = [val] if val else []

    # Save last key
    if current_key:
        fm[current_key] = " ".join(
            l for l in current_val_lines if l
        ).strip().strip(">").strip('"').strip("'")

    return fm


def _extract_section(text: str, heading: str) -> str:
    """Extract the body of a markdown section by heading (## Heading).

    Returns everything between the target heading and the next heading
    of equal or higher level.
    """
    # Escape heading for regex, match ## Heading (flexible whitespace)
    pattern = re.compile(
        r"^##\s+" + re.escape(heading) + r"\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return ""

    start = match.end()
    # Find the next heading of same or higher level (## or #)
    next_heading = re.search(r"^#{1,2}\s+", text[start:], re.MULTILINE)
    if next_heading:
        end = start + next_heading.start()
    else:
        end = len(text)

    return text[start:end].strip()


def _extract_bullet_list(section_body: str) -> List[str]:
    """Extract bullet list items (- item) from a markdown section body."""
    items = []
    for line in section_body.split("\n"):
        line = line.strip()
        m = re.match(r"^[-*]\s+(.*)", line)
        if m:
            # Strip bold markers and leading/trailing whitespace
            item = m.group(1).strip()
            # Remove trailing markdown bold
            item = re.sub(r"\*\*([^*]+)\*\*", r"\1", item)
            items.append(item)
    return items


def _extract_numbered_list(section_body: str) -> List[str]:
    """Extract numbered list items (1. item) from a section body."""
    items = []
    for line in section_body.split("\n"):
        line = line.strip()
        m = re.match(r"^\d+\.\s+(.*)", line)
        if m:
            item = m.group(1).strip()
            item = re.sub(r"\*\*([^*]+)\*\*", r"\1", item)
            items.append(item)
    return items


# ── Keyword & Pattern Generation ─────────────────────────────────────

# Stopwords to filter out when generating keywords
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "up", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once", "and",
    "but", "or", "nor", "not", "so", "if", "as", "that", "this", "it",
    "its", "their", "they", "them", "we", "our", "you", "your", "he",
    "she", "him", "her", "my", "me", "i", "who", "what", "which",
    "when", "where", "how", "all", "each", "every", "any", "some",
    "no", "more", "most", "other", "such", "only", "own", "same",
    "than", "too", "very", "just", "also", "here", "there",
    "user", "asks", "wants", "needs", "provides",  # SKILL.md filler words
    "existing", "pipeline", "workflow", "set", "asks",
}

# Domain bigrams/trigrams that should be treated as single keywords
_PHRASE_PATTERNS = [
    r"code\s+review",
    r"static\s+analysis",
    r"code\s+quality",
    r"security\s+scan",
    r"chart\s+analysis",
    r"graph\s+analysis",
    r"trend\s+analysis",
    r"anomaly\s+detection",
    r"data\s+extraction",
    r"data\s+visualization",
    r"bar\s+chart",
    r"line\s+chart",
    r"pie\s+chart",
    r"scatter\s+plot",
    r"extract\s+data",
    r"code\s+reviewer",
    r"code\s+scan",
    r"data\s+points?",
    r"snap\s+detection",
    r"snap\s+moment",
    r"snap\s+eval",
    r"eval\s+results?",
    r"evaluation\s+results?",
    r"play\s+type",
    r"failure\s+rate",
    r"failure\s+pattern",
    r"model\s+accuracy",
    r"model\s+evaluation",
    r"prediction\s+error",
    r"camera\s+view",
    r"ground\s+truth",
    r"american\s+football",
]


def _generate_keywords(
    when_to_use: List[str],
    description: str,
    display_name: str,
) -> List[str]:
    """Auto-generate keywords from when_to_use lines + description.

    Strategy:
      1. Extract known domain phrases (bigrams/trigrams).
      2. Extract meaningful single words (non-stopword, len > 2).
      3. Add the display name tokens.
      4. Deduplicate while preserving order.
    """
    # Combine all text sources
    all_text = " ".join(when_to_use) + " " + description + " " + display_name
    all_text_lower = all_text.lower()

    keywords = []
    seen = set()

    # 1. Extract known phrase patterns
    for phrase_re in _PHRASE_PATTERNS:
        for m in re.finditer(phrase_re, all_text_lower):
            phrase = re.sub(r"\s+", " ", m.group())
            if phrase not in seen:
                keywords.append(phrase)
                seen.add(phrase)

    # 2. Extract meaningful single words
    words = re.findall(r"[a-z][a-z\-]+", all_text_lower)
    for w in words:
        if w not in _STOPWORDS and w not in seen and len(w) > 2:
            keywords.append(w)
            seen.add(w)

    return keywords


def _generate_patterns(
    when_to_use: List[str],
    keywords: List[str],
    has_image_input: bool,
) -> List[str]:
    """Auto-generate regex patterns from when_to_use lines + keywords.

    Strategy:
      1. Build verb ↔ noun cross-product patterns from when_to_use.
      2. Add tool-specific patterns (pylint, flake8, etc.) if found in keywords.
      3. Add image-related patterns if the skill processes images.
    """
    patterns = []

    # ── Verb-noun extraction from when_to_use ──────────────────────
    verbs = set()
    nouns = set()

    verb_re = re.compile(
        r"\b(review|analyze|analy[sz]e|check|scan|audit|lint|detect|extract|"
        r"compare|read|interpret|explain|describe|improve|fix|find|set up|"
        r"integrate|plug|add|identify|run|evaluate|assess|benchmark|"
        r"summari[sz]e|breakdown|diagnose)\b",
        re.IGNORECASE,
    )
    noun_re = re.compile(
        r"\b(code|python|script|repo|repository|chart|graph|plot|visualization|"
        r"image|figure|diagram|trend|anomal\w*|outlier|data|quality|bugs?|"
        r"errors?|issues?|vulnerabilit\w*|security|style|category|categories|"
        r"insight|pattern|csv|eval|evaluation|results?|predictions?|accuracy|"
        r"snap|football|play_type|view|league|failure|model|performance|"
        r"metrics?)\b",
        re.IGNORECASE,
    )

    for line in when_to_use:
        for m in verb_re.finditer(line):
            verbs.add(m.group().lower())
        for m in noun_re.finditer(line):
            nouns.add(m.group().lower())

    # Build verb.*noun and noun.*verb patterns
    for v in sorted(verbs):
        for n in sorted(nouns):
            patterns.append(rf"\b{re.escape(v)}\b.*\b{re.escape(n)}\b")
    # Also noun.*verb (reversed)
    for n in sorted(nouns)[:5]:  # limit reversed to avoid explosion
        for v in sorted(verbs)[:5]:
            rev = rf"\b{re.escape(n)}\b.*\b{re.escape(v)}\b"
            if rev not in patterns:
                patterns.append(rev)

    # ── Tool-specific patterns ──────────────────────────────────────
    tool_kw = [k for k in keywords if k in (
        "pylint", "flake8", "bandit", "lint", "linter",
    )]
    for t in tool_kw:
        patterns.append(rf"\b{re.escape(t)}\b")

    # ── Code structure patterns ─────────────────────────────────────
    if any(k in keywords for k in ("python", "code", "review", "script")):
        patterns.extend([
            r"```python",
            r"\bdef\s+\w+\s*\(",
            r"\bclass\s+\w+",
            r"\bimport\s+\w+",
        ])

    # ── Image / chart patterns ──────────────────────────────────────
    if has_image_input:
        patterns.extend([
            r"\bhttps?://\S+\.(png|jpg|jpeg|gif|webp|svg)\b",
            r"\b(png|jpg|jpeg)\b.*\b(chart|graph|plot)\b",
            r"\b(bar|line|pie|scatter|histogram|heat\s?map|area|box)\s*(chart|plot|graph)\b",
        ])

    # ── CSV / eval patterns ──────────────────────────────────────────
    if any(k in keywords for k in ("csv", "eval", "evaluation", "snap", "football")):
        patterns.extend([
            r"\b\.csv\b",
            r"\beval(uation)?\s+(result|csv|report|metric)\b",
            r"\bsnap\b.*\b(detect|moment|time|eval)\b",
            r"\b(play_type|play type)\b",
            r"\bpred(iction)?.*error\b",
            r"\bground\s*truth\b",
            r"\bfailure\s+(rate|pattern|analysis)\b",
            r"\bper\s+(play.type|view|league)\b",
        ])

    # Deduplicate
    seen = set()
    unique = []
    for p in patterns:
        if p not in seen:
            unique.append(p)
            seen.add(p)

    return unique


# ── Skill Name Mapping ────────────────────────────────────────────────
# The SKILL.md frontmatter `name` uses kebab-case (clarifai-code-review)
# but the router uses snake_case IDs (code_review).  This maps the
# canonical MD name → router ID.

_SKILL_NAME_MAP = {
    "clarifai-code-review": "code_review",
    "clarifai-chart-analysis": "chart_analysis",
    "clarifai-snap-eval": "snap_eval",
}

_DISPLAY_NAME_MAP = {
    "code_review": "Python Code Review",
    "chart_analysis": "Chart & Graph Analysis",
    "snap_eval": "Snap-Moment Eval Analysis",
}

_IMAGE_SKILLS = {"chart_analysis"}

_FILE_EXTENSION_MAP = {
    "code_review": [".py", ".pyw"],
    "chart_analysis": [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"],
    "snap_eval": [".csv"],
}


# ── Main Loader ───────────────────────────────────────────────────────

def load_skills(
    skills_dir: Optional[Path] = None,
    exclude: Optional[List[str]] = None,
) -> Dict[str, SkillDescriptor]:
    """Discover and parse all SKILL.md files into SkillDescriptors.

    Args:
        skills_dir: Override for .agents/skills/ directory path.
        exclude: List of frontmatter `name` values to skip
            (e.g. ["clarifai-smart-router"] to avoid loading self).

    Returns:
        Dict mapping router skill IDs → SkillDescriptor.
    """
    skills_dir = skills_dir or _SKILLS_DIR
    exclude = set(exclude or ["clarifai-smart-router"])

    registry: Dict[str, SkillDescriptor] = {}

    if not skills_dir.exists():
        logger.warning("Skills directory not found: %s", skills_dir)
        return registry

    # Discover SKILL.md files
    md_files = sorted(skills_dir.glob("*/SKILL.md"))
    logger.info("Discovered %d SKILL.md files in %s", len(md_files), skills_dir)

    for md_path in md_files:
        try:
            skill = _parse_skill_md(md_path, exclude)
            if skill:
                registry[skill.name] = skill
                logger.info(
                    "Loaded skill '%s': %d keywords, %d patterns, %d execution steps",
                    skill.name,
                    len(skill.keywords),
                    len(skill.patterns),
                    len(skill.execution_steps),
                )
        except Exception as exc:
            logger.error("Failed to parse %s: %s", md_path, exc)

    return registry


def _parse_skill_md(
    md_path: Path,
    exclude: set,
) -> Optional[SkillDescriptor]:
    """Parse a single SKILL.md into a SkillDescriptor."""
    text = md_path.read_text(encoding="utf-8")

    # ── Frontmatter ───────────────────────────────────────────────
    fm = _parse_frontmatter(text)
    md_name = fm.get("name", "")

    if not md_name:
        logger.warning("No 'name' in frontmatter of %s — skipping.", md_path)
        return None

    if md_name in exclude:
        logger.debug("Skipping excluded skill: %s", md_name)
        return None

    # Map to router ID
    router_id = _SKILL_NAME_MAP.get(md_name)
    if not router_id:
        # Fallback: convert kebab to snake
        router_id = md_name.replace("clarifai-", "").replace("-", "_")
        logger.info(
            "No explicit name mapping for '%s', using '%s'.", md_name, router_id,
        )

    description = fm.get("description", "")
    display_name = _DISPLAY_NAME_MAP.get(router_id, md_name.replace("-", " ").title())

    # ── When to Use ───────────────────────────────────────────────
    wtu_body = _extract_section(text, "When to Use")
    when_to_use = _extract_bullet_list(wtu_body)
    if not when_to_use:
        logger.warning("No 'When to Use' items found in %s.", md_path)

    # ── Execution Steps ───────────────────────────────────────────
    es_body = _extract_section(text, "Execution Steps")
    # Try numbered list first, then bullet list
    execution_steps = _extract_numbered_list(es_body)
    if not execution_steps:
        execution_steps = _extract_bullet_list(es_body)

    # ── Infer properties ──────────────────────────────────────────
    has_image_input = router_id in _IMAGE_SKILLS
    file_extensions = _FILE_EXTENSION_MAP.get(router_id, [])

    # ── Generate keywords & patterns ──────────────────────────────
    keywords = _generate_keywords(when_to_use, description, display_name)
    patterns = _generate_patterns(when_to_use, keywords, has_image_input)

    return SkillDescriptor(
        name=router_id,
        display_name=display_name,
        when_to_use=when_to_use,
        keywords=keywords,
        patterns=patterns,
        execution_steps=execution_steps,
        file_extensions=file_extensions,
        has_image_input=has_image_input,
        priority=0,
    )


# ── Convenience: load once at module level for import ─────────────────

def get_default_registry() -> Dict[str, SkillDescriptor]:
    """Load skills from the default .agents/skills/ directory."""
    return load_skills()


# Allow testing from CLI
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    registry = get_default_registry()
    print(f"\nLoaded {len(registry)} skills:\n")
    for sid, s in registry.items():
        print(f"  [{sid}] {s.display_name}")
        print(f"    When to Use ({len(s.when_to_use)} items):")
        for item in s.when_to_use:
            print(f"      - {item}")
        print(f"    Keywords ({len(s.keywords)}): {s.keywords[:10]}...")
        print(f"    Patterns ({len(s.patterns)}): {len(s.patterns)} generated")
        print(f"    Execution Steps ({len(s.execution_steps)}):")
        for step in s.execution_steps:
            print(f"      → {step}")
        print()
