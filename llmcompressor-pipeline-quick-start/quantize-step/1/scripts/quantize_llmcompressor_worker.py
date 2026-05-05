#!/usr/bin/env python
"""
quantize_llmcompressor_worker.py — Python worker invoked by quantize_llmcompressor.sh.

You normally don't run this directly — the bash driver handles GPU selection,
HF snapshot materialization, etc. before invoking this. Run it standalone
only if you need to bypass the wrapper.

The worker:
  1. Loads the (already-local) HF model + tokenizer/processor.
  2. Builds a `QuantizationModifier` recipe with sensible defaults based on
     the model's architecture (MoE / VLM / dense).
  3. Loads & preprocesses a calibration dataset (chat-template aware).
  4. Calls llmcompressor.oneshot(...).
  5. Saves the compressed-tensors checkpoint to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# We import torch / transformers / llmcompressor lazily so that --help and
# argument validation work even if the env isn't fully set up yet.


def _add_default_ignore_patterns(ignore: list[str], cfg: dict) -> list[str]:
    """Augment the user-provided ignore list with sensible defaults for the
    detected architecture. Patterns use llmcompressor / compressed-tensors
    conventions: prefix with 're:' for regex, otherwise exact-match."""
    out = list(ignore)

    def add(p):
        if p not in out:
            out.append(p)

    # Universally needed
    add("lm_head")

    # MoE indicators
    is_moe = any(
        isinstance(cfg.get(k) or (cfg.get("text_config") or {}).get(k), int)
        and (cfg.get(k) or (cfg.get("text_config") or {}).get(k)) > 1
        for k in (
            "num_local_experts",
            "n_routed_experts",
            "num_experts",
            "moe_num_experts",
        )
    )
    if is_moe:
        # Most MoE archs name the router `mlp.gate` (DeepSeek/Kimi/Qwen3-MoE)
        # or `mlp.router` (GLM4 newer). Cover both. Anchor with $ so we don't
        # accidentally hit `gate_proj`.
        add("re:.*mlp.gate$")
        add("re:.*mlp.router$")
        add("re:.*router$")
        # Some archs (DeepSeek-V3) have a mixer/conv attribute that should
        # stay in BF16; the `re:.*linear_attn.*` pattern covers Qwen3.5/3-Next
        # hybrid linear-attention layers.
        add("re:.*linear_attn.*")
        # Shared-expert gates (Qwen3.5 etc.) are scalar gating coefficients.
        add("re:.*shared_expert_gate$")

    # VLM indicators — keep vision tower in BF16
    if any(k in cfg for k in ("vision_config", "vision_tower_config")) or any(
        k in cfg for k in ("image_token_id", "video_token_id")
    ):
        add("re:.*vision_tower.*")
        add("re:visual.*")
        add("re:.*vision_model.*")
        add("re:.*embed_vision.*")

    # Embeddings should always stay in BF16 — they aren't `Linear` so most
    # recipes won't touch them, but exact-name pattern catches any custom
    # embedding linears (e.g. multimodal `embed_*`).
    add("re:.*embed_tokens$")

    return out


def _detect_arch(cfg: dict) -> dict:
    """Cheap structural classification used for logging and recipe defaults."""
    info = {
        "model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures") or [],
        "is_vlm": any(
            k in cfg for k in ("vision_config", "vision_tower_config", "image_token_id")
        ),
        "is_moe": False,
        "n_experts": None,
        "n_layers": cfg.get("num_hidden_layers")
        or (cfg.get("text_config") or {}).get("num_hidden_layers"),
    }
    text_cfg = cfg.get("text_config") or {}
    for src in (cfg, text_cfg):
        for k in ("num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts"):
            v = src.get(k)
            if isinstance(v, int) and v > 1:
                info["is_moe"] = True
                info["n_experts"] = v
                return info
    return info


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src", required=True, help="Local checkpoint dir (or HF repo id).")
    p.add_argument("--out", required=True, help="Output dir for the NVFP4 checkpoint.")
    p.add_argument(
        "--scheme",
        default="NVFP4",
        help="Quantization scheme. Common values: NVFP4 (W4A4 fp4),"
        " NVFP4A16 (W4A16 fp4 weight-only), FP8_DYNAMIC, FP8 (per-tensor),"
        " W4A16 (int4 weight-only), MXFP4. See compressed-tensors for the full list.",
    )
    p.add_argument(
        "--ignore",
        default="",
        help="Extra ignore patterns, comma-separated. Use 're:<regex>' for regex,"
        " bare strings for exact matches. Merged with auto-detected defaults"
        " (MoE router, vision tower, embeddings).",
    )
    p.add_argument("--calib-dataset", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument(
        "--calib-split",
        default="train_sft",
        help='Dataset split (e.g. "train_sft", "train", "train[:5%%]").',
    )
    p.add_argument("--calib-size", type=int, default=512)
    p.add_argument("--calib-seq", type=int, default=2048)
    p.add_argument(
        "--moe-all-experts",
        choices=["true", "false"],
        default="true",
        help="Force every routed expert through calibration. Default: true.",
    )
    p.add_argument(
        "--pipeline",
        choices=["sequential", "basic", "independent", "datafree"],
        default=None,
        help="Calibration pipeline. Default: let llm-compressor infer "
        "(picks 'sequential' for QuantizationModifier — slow, ~10 min/layer "
        "on 30B-class MoE models with all-experts calibration). "
        "Use 'basic' for a single forward pass over all calib samples "
        "(~5-30 min total, ~0.5%% accuracy hit on MMLU vs sequential). "
        "Use 'datafree' for weight-only schemes (NVFP4A16, W4A16) — "
        "skips activation calibration entirely.",
    )
    p.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code (default: enabled for unrecognized arches).",
    )
    p.add_argument(
        "--save-uncompressed",
        action="store_true",
        help="Pass save_compressed=False to save_pretrained (raw fp4 tensors,"
        " bigger on disk, useful for debugging).",
    )
    p.add_argument(
        "--auto-defaults",
        choices=["true", "false"],
        default="true",
        help="Add MoE / VLM / lm_head ignore patterns automatically based on"
        " config.json. Default: true.",
    )
    p.add_argument(
        "--dtype",
        default="auto",
        help="Model load dtype. 'auto' uses the model's config.json torch_dtype.",
    )
    args = p.parse_args()

    src = args.src
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Inspect config to drive default ignore-list and detect VLM
    # ------------------------------------------------------------------
    config_path = Path(src) / "config.json"
    if not config_path.exists():
        print(f"ERROR: {config_path} missing — --src must be a local directory "
              "(snapshot it first via the bash driver).", file=sys.stderr)
        return 2
    cfg = json.loads(config_path.read_text())
    arch = _detect_arch(cfg)
    print(f"[inspect] model_type={arch['model_type']}  "
          f"architectures={arch['architectures']}  "
          f"is_moe={arch['is_moe']}  n_experts={arch['n_experts']}  "
          f"is_vlm={arch['is_vlm']}  n_layers={arch['n_layers']}")

    # ------------------------------------------------------------------
    # Build ignore list
    # ------------------------------------------------------------------
    user_ignore = [p.strip() for p in args.ignore.split(",") if p.strip()]
    if args.auto_defaults == "true":
        ignore = _add_default_ignore_patterns(user_ignore, cfg)
    else:
        ignore = user_ignore or ["lm_head"]
    print(f"[recipe] scheme={args.scheme}  ignore={ignore}")

    # ------------------------------------------------------------------
    # Load model + tokenizer/processor (now the heavy imports happen)
    # ------------------------------------------------------------------
    import torch  # noqa: F401 (used implicitly by HF / llmcompressor)
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    print(f"[load] model from {src}  dtype={args.dtype}")
    trust = not args.no_trust_remote_code
    model_kwargs = {"dtype": args.dtype, "trust_remote_code": trust}

    # VLMs typically need AutoProcessor; pure-text LLMs use AutoTokenizer.
    # For multimodal architectures the right `from_pretrained` class is the
    # explicit `Foo4ForConditionalGeneration` — but `AutoModelForCausalLM`
    # works for the language-only variants and most VLMs that ship a
    # text-CausalLM config. If the user hits a model where this fails, they
    # can pre-load the model and pass it via the Python API.
    model = AutoModelForCausalLM.from_pretrained(src, **model_kwargs)
    try:
        processor = AutoProcessor.from_pretrained(src, trust_remote_code=trust)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=trust)
    except Exception:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=trust)

    # ------------------------------------------------------------------
    # Calibration dataset — chat-template aware
    # ------------------------------------------------------------------
    print(f"[calib] dataset={args.calib_dataset}  split={args.calib_split}  "
          f"n={args.calib_size}  seq={args.calib_seq}")
    from datasets import load_dataset

    split_spec = f"{args.calib_split}[:{args.calib_size}]"
    ds = load_dataset(args.calib_dataset, split=split_spec)
    ds = ds.shuffle(seed=42)

    has_messages = "messages" in ds.column_names
    has_text = "text" in ds.column_names

    def preprocess(example):
        if has_messages:
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"], tokenize=False
                )
            }
        if has_text:
            return {"text": example["text"]}
        # Fallback: stringify whatever the first column has.
        first_col = ds.column_names[0]
        return {"text": str(example[first_col])}

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=args.calib_seq,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # ------------------------------------------------------------------
    # Quantize
    # ------------------------------------------------------------------
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    recipe = QuantizationModifier(
        targets="Linear",
        scheme=args.scheme,
        ignore=ignore,
    )
    moe_all = args.moe_all_experts == "true"
    print(f"[oneshot] moe_calibrate_all_experts={moe_all}  pipeline={args.pipeline or '(inferred)'}")

    oneshot_kwargs = dict(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.calib_seq,
        num_calibration_samples=args.calib_size,
        moe_calibrate_all_experts=moe_all,
        tokenizer=tokenizer,
    )
    if args.pipeline is not None:
        oneshot_kwargs["pipeline"] = args.pipeline
    oneshot(**oneshot_kwargs)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print(f"[save] writing to {out}  save_compressed={not args.save_uncompressed}")
    model.save_pretrained(str(out), save_compressed=not args.save_uncompressed)
    if processor is not None:
        processor.save_pretrained(str(out))
    else:
        tokenizer.save_pretrained(str(out))

    print(f"[done] checkpoint at: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
