#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "models"))
model_module = __import__("model.1.model", fromlist=[''])
model_class = [obj for name in dir(model_module) if isinstance(obj := getattr(model_module, name), type) and hasattr(obj, 'train')][0]

def main():
    args = model_class.to_pipeline_parser().parse_args()
    model_class().train(**vars(args))


if __name__ == "__main__":
    main()
