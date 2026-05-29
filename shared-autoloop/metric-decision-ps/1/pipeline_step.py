#!/usr/bin/env python
import sys
from pathlib import Path

# Make the `model/` sibling directory importable. The path includes the literal
# "1" (Clarifai runner versioning convention) — `__import__` accepts dotted
# strings starting with digits, but normal `import model.1.model` would not.
sys.path.insert(0, str(Path(__file__).parent / "models"))
model_module = __import__("model.1.model", fromlist=[''])
MODEL_CLASS = model_module.MODEL_CLASS


def main():
    args = MODEL_CLASS.to_pipeline_parser().parse_args()
    MODEL_CLASS().decide(**vars(args))


if __name__ == "__main__":
    main()
