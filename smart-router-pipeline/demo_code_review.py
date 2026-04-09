#!/usr/bin/env python3
"""
Demo: Smart Router → Code Review

Routes a single code-review request through the smart router.
GPT-4o reviews the code for bugs, security issues, and style problems.
"""
import json
import os
import sys
import textwrap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "smart-router-ps", "1"))

from pipeline_step import SmartRouterStep

MODEL_URL = "https://clarifai.com/openai/chat-completion/models/gpt-4o"

BUGGY_CODE = textwrap.dedent("""\
    import pickle
    import os

    def load_data(user_input):
        data = pickle.loads(user_input)  # Security: deserializing untrusted data
        return data

    def process(items):
        result = []
        for i in range(len(items)):  # Style: should use enumerate
            result.append(items[i] * 2)
        return result

    x = 10
    y = 0
    print(x / y)  # Bug: division by zero
""")


def main():
    print("=" * 72)
    print("  SMART ROUTER — CODE REVIEW DEMO")
    print("=" * 72)

    pat = os.environ.get("CLARIFAI_PAT", "")
    if not pat:
        print("\n  ⚠  CLARIFAI_PAT not set — LLM review will be skipped.\n")

    step = SmartRouterStep()
    result_json = step.route(
        user_prompt="Review this Python code for bugs, security issues, and style problems",
        model_url=MODEL_URL,
        code_text=BUGGY_CODE,
        review_strictness="medium",
        user_id="demo",
        app_id="demo",
    )

    data = json.loads(result_json)
    router = data.get("router", {})
    skill = data.get("skill_output", {})

    print(f"\n  Intent     : {router.get('intent')}")
    print(f"  Confidence : {router.get('confidence')}")
    print(f"  Method     : {router.get('method')}")
    print(f"  Reasoning  : {router.get('reasoning')}")
    print(f"  Scores     : {router.get('all_scores')}")

    print(f"\n  Status     : {skill.get('status')}")
    print(f"  Findings   : {skill.get('total_findings')}")
    print(f"  Files      : {skill.get('files_reviewed')}")

    review = skill.get("review", "")
    if review:
        print(f"\n  ── LLM REVIEW ──\n{textwrap.indent(review, '    ')}")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
