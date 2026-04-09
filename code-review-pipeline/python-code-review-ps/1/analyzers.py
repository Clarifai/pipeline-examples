"""
Static Analysis Module

Runs pylint, flake8, and bandit on Python source files and returns a
normalized list of findings. Each tool is optional — if not installed in the
container, it is skipped gracefully so the step never hard-fails.

Findings schema:
    {
        "file":     str,   # relative file path
        "line":     int,   # 1-based line number
        "column":   int,   # 1-based column number
        "severity": str,   # error | warning | convention | refactor | info
        "rule":     str,   # tool-specific rule ID (e.g. E0001, B301)
        "message":  str,   # human-readable description
        "tool":     str,   # pylint | flake8 | bandit
    }
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("code-review-step.analyzers")

# Which severities to keep at each strictness level
STRICTNESS_FILTER = {
    "high": {"error", "fatal"},
    "medium": {"error", "fatal", "warning"},
    "low": {"error", "fatal", "warning", "convention", "refactor", "info"},
}


class StaticAnalyzer:
    """Runs multiple linters and normalizes their output."""

    def __init__(self, strictness: str = "medium"):
        self.strictness = strictness.lower()
        self.allowed = STRICTNESS_FILTER.get(
            self.strictness, STRICTNESS_FILTER["medium"]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, files: Dict[str, str]) -> List[dict]:
        """
        Analyze Python files with all available tools.

        Args:
            files: {relative_path: source_code} mapping.

        Returns:
            Deduplicated, strictness-filtered list of finding dicts.
        """
        # Write files to a temp dir so subprocess tools can read them
        work_dir = tempfile.mkdtemp(prefix="analysis-")
        file_paths: List[str] = []

        for rel_path, content in files.items():
            full = Path(work_dir) / rel_path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
            file_paths.append(str(full))

        # Run each tool
        findings: List[dict] = []
        findings.extend(self._run_pylint(work_dir, file_paths))
        findings.extend(self._run_flake8(work_dir, file_paths))
        findings.extend(self._run_bandit(work_dir, file_paths))

        # Filter by strictness
        filtered = [f for f in findings if f["severity"] in self.allowed]

        # Strip the temp-dir prefix from file paths
        for f in filtered:
            for prefix in (work_dir + "/", work_dir):
                if f["file"].startswith(prefix):
                    f["file"] = f["file"][len(prefix):]

        # Deduplicate by (file, line, message)
        seen: set = set()
        deduped: List[dict] = []
        for f in filtered:
            key = (f["file"], f["line"], f["message"])
            if key not in seen:
                seen.add(key)
                deduped.append(f)

        return deduped

    # ------------------------------------------------------------------
    # pylint
    # ------------------------------------------------------------------

    def _run_pylint(self, work_dir: str, paths: List[str]) -> List[dict]:
        findings: List[dict] = []
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "pylint",
                    "--output-format=json",
                    "--disable=C0114,C0115,C0116",  # skip missing docstrings
                ]
                + paths,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=work_dir,
            )
            if result.stdout.strip():
                for item in json.loads(result.stdout):
                    findings.append(
                        {
                            "file": item.get("path", ""),
                            "line": item.get("line", 0),
                            "column": item.get("column", 0),
                            "severity": self._map_pylint(item.get("type", "")),
                            "rule": item.get("message-id", ""),
                            "message": item.get("message", ""),
                            "tool": "pylint",
                        }
                    )
        except FileNotFoundError:
            logger.info("pylint not installed — skipping.")
        except subprocess.TimeoutExpired:
            logger.warning("pylint timed out.")
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("pylint error: %s", exc)
        return findings

    # ------------------------------------------------------------------
    # flake8
    # ------------------------------------------------------------------

    def _run_flake8(self, work_dir: str, paths: List[str]) -> List[dict]:
        findings: List[dict] = []
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "flake8",
                    "--max-line-length=120",
                ]
                + paths,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=work_dir,
            )
            # Default flake8 output: file:line:col: CODE message
            for line in (result.stdout or "").strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(":", 3)
                if len(parts) < 4:
                    continue
                code_msg = parts[3].strip()
                code = code_msg.split()[0] if code_msg else ""
                message = " ".join(code_msg.split()[1:]) if " " in code_msg else code_msg
                findings.append(
                    {
                        "file": parts[0].strip(),
                        "line": int(parts[1]) if parts[1].strip().isdigit() else 0,
                        "column": int(parts[2]) if parts[2].strip().isdigit() else 0,
                        "severity": self._map_flake8(code),
                        "rule": code,
                        "message": message,
                        "tool": "flake8",
                    }
                )
        except FileNotFoundError:
            logger.info("flake8 not installed — skipping.")
        except subprocess.TimeoutExpired:
            logger.warning("flake8 timed out.")
        except Exception as exc:
            logger.warning("flake8 error: %s", exc)
        return findings

    # ------------------------------------------------------------------
    # bandit  (security-focused)
    # ------------------------------------------------------------------

    def _run_bandit(self, work_dir: str, paths: List[str]) -> List[dict]:
        findings: List[dict] = []
        try:
            result = subprocess.run(
                [sys.executable, "-m", "bandit", "-f", "json", "-r"] + paths,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=work_dir,
            )
            output = result.stdout.strip()
            if output:
                raw = json.loads(output)
                for item in raw.get("results", []):
                    sev = item.get("issue_severity", "MEDIUM").lower()
                    # Map bandit severities to our schema
                    if sev == "high":
                        sev = "error"
                    elif sev == "medium":
                        sev = "warning"
                    elif sev == "low":
                        sev = "info"
                    findings.append(
                        {
                            "file": item.get("filename", ""),
                            "line": item.get("line_number", 0),
                            "column": 0,
                            "severity": sev,
                            "rule": item.get("test_id", ""),
                            "message": (
                                f"[{item.get('issue_confidence', '')}] "
                                f"{item.get('issue_text', '')}"
                            ),
                            "tool": "bandit",
                        }
                    )
        except FileNotFoundError:
            logger.info("bandit not installed — skipping.")
        except subprocess.TimeoutExpired:
            logger.warning("bandit timed out.")
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("bandit error: %s", exc)
        return findings

    # ------------------------------------------------------------------
    # Severity mappers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_pylint(pylint_type: str) -> str:
        return {
            "fatal": "error",
            "error": "error",
            "warning": "warning",
            "convention": "convention",
            "refactor": "refactor",
            "info": "info",
        }.get(pylint_type.lower(), "info")

    @staticmethod
    def _map_flake8(code: str) -> str:
        if not code:
            return "info"
        return {
            "E": "error",       # PEP 8 errors
            "W": "warning",     # PEP 8 warnings
            "F": "error",       # PyFlakes errors
            "C": "convention",  # McCabe complexity
            "N": "convention",  # pep8-naming
            "B": "warning",     # flake8-bugbear
        }.get(code[0].upper(), "info")
