#!/usr/bin/env python
"""Test model.py load_model + predict/generate end-to-end."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

import importlib
_mod = importlib.import_module("model.1.model")
UnslothLoRAVLLM = _mod.UnslothLoRAVLLM


def main():
    model = UnslothLoRAVLLM()

    print("Loading model...")
    model.load_model()
    print("Model loaded.\n")

    # Test predict
    print("--- predict ---")
    result = model.predict(prompt="Hello, how are you?", max_tokens=64)
    assert result and len(result) > 0, "predict returned empty response"
    print(result)

    # Test generate (streaming)
    print("\n--- generate ---")
    tokens = []
    for tok in model.generate(prompt="Write a haiku about coding.", max_tokens=64):
        tokens.append(tok)
        print(tok, end="", flush=True)
    print()
    assert any(t for t in tokens), "generate returned no tokens"

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
