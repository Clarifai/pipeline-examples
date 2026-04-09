#!/usr/bin/env python3
"""Run the code review step with a sample buggy file."""
import os
import sys

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DEMO_DIR, "python-code-review-ps", "1"))

from pipeline_step import CodeReviewStep

# Read the sample file
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(DEMO_DIR, "sample_buggy_code.py"), "r") as f:
    code = f.read()

step = CodeReviewStep()
step.review(
    code_text=code,
    review_strictness="low",
    user_id="demo_user",
    app_id="demo_app",
)
