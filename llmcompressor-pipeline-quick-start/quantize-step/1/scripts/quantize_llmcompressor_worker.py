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

    # Universally needed. The `lm_head` exact pattern catches the top-level
    # module; `re:.*lm_head$` also catches nested forms like
    # `language_model.lm_head` (common in VLMs / audio multimodals).
    add("lm_head")
    add("re:.*lm_head$")

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
        # Shared-expert gates (Qwen3.5 etc.) are scalar gating coefficients.
        add("re:.*shared_expert_gate$")

    # Built-in MTP / multi-token-prediction heads used for in-model
    # speculative decoding (Qwen3-Next, Qwen3.5, Qwen3.6, DeepSeek-V3 MTP).
    # Quantizing these tiny modules to W4A4 risks tanking the speculative
    # accept-rate while giving negligible compression savings. Detect via
    # `mtp_num_hidden_layers` (or nested under text_config for VLMs).
    text_cfg = cfg.get("text_config") or {}
    if any(
        isinstance(src.get("mtp_num_hidden_layers"), int)
        and src.get("mtp_num_hidden_layers") > 0
        for src in (cfg, text_cfg)
    ):
        add("re:^mtp\\..*")
        add("re:.*\\.mtp\\..*")

    # Hybrid attention paths — small but precision-sensitive sub-modules that
    # don't benefit from W4A4 and tend to drift quality when quantized. Apply
    # to BOTH MoE and dense models; Qwen3.5 / Qwen3.6 / Qwen3-Next include
    # `linear_attn` paths even in their dense variants.
    add("re:.*linear_attn.*")
    add("re:.*mamba.*")
    add("re:.*ssm.*")
    # Mamba/SSM 1-D conv (Falcon-Mamba, Mamba-2, Jamba, NemotronH). The
    # convs are nn.Conv1d (not nn.Linear) so `targets="Linear"` already
    # excludes them, but be defensive — some recipes broaden targets and
    # we don't want a Conv1d weight ever quantized.
    # Note: do NOT exclude the whole `*.mixer.*` subtree — Mamba's
    # in_proj/out_proj/x_proj/dt_proj are nn.Linear and CAN be NVFP4'd.
    add("re:.*\\.conv1d\\..*")
    add("re:.*\\.conv1d$")

    # VLM indicators — keep vision tower in BF16. Patterns intentionally use
    # a leading `.*` because module paths are typically `model.<sub>.…` — a
    # bare `re:visual.*` does NOT match `model.visual.blocks.0.…` (it would
    # only match names *starting* with `visual`).
    is_vlm = any(k in cfg for k in ("vision_config", "vision_tower_config")) or any(
        k in cfg for k in ("image_token_id", "video_token_id")
    )
    # Audio multimodal indicators (Qwen2-Audio, Whisper-style stacks, Phi-4
    # multimodal etc.). Detection mirrors the VLM check on audio-side keys.
    is_audio_mm = any(k in cfg for k in ("audio_config", "audio_tower_config")) or any(
        k in cfg for k in ("audio_token_id", "audio_token_index")
    )
    if is_vlm or is_audio_mm:
        # Vision side
        add("re:.*vision_tower.*")
        add("re:.*\\.visual\\..*")        # `model.visual.…` (Qwen3.5/3.6 VLM)
        add("re:.*\\.visual$")            # the visual submodule attribute itself
        add("re:.*vision_model.*")
        add("re:.*vision_encoder.*")
        add("re:.*visual_encoder.*")
        add("re:.*visual_tower.*")
        add("re:.*embed_vision.*")
        add("re:.*image_processor.*")
        add("re:.*image_proj.*")
        add("re:.*image_embedding.*")
        add("re:.*patch_embed.*")
        add("re:.*mm_projector.*")        # LLaVA-style multimodal projector
        add("re:.*multi_modal_projector.*")
        # Audio side
        add("re:.*audio_tower.*")          # Qwen2-Audio path: `audio_tower.…`
        add("re:.*audio_encoder.*")
        add("re:.*audio_model.*")
        add("re:.*audio_proj.*")
        add("re:.*audio_embed.*")

    # Embeddings should always stay in BF16 — they aren't `Linear` so most
    # recipes won't touch them, but exact-name pattern catches any custom
    # embedding linears (e.g. multimodal `embed_*`).
    add("re:.*embed_tokens$")

    return out


def _restore_mtp_weights(out_dir: Path, src_dir: str) -> None:
    """Some loader classes (Qwen3_5ForConditionalGeneration, Qwen3NextForCausalLM,
    DeepSeek-V3 multimodal variants) set `_keys_to_ignore_on_load_unexpected`
    to drop top-level `mtp.*` weights during from_pretrained — MTP is loaded
    later by the inference engine, not the HF model class. The downside is
    that `save_pretrained` then writes a checkpoint with no MTP, and vLLM /
    SGLang spec decoding (`--speculative-config qwen3_next_mtp`) breaks.

    This copies MTP tensors from the source snapshot into the output as a
    new BF16 shard and rebuilds `model.safetensors.index.json`. No-op when
    the source has no MTP or when MTP already survived the round-trip.
    """
    import json
    import shutil
    from collections import defaultdict
    from safetensors import safe_open
    from safetensors.torch import save_file

    src_p = Path(src_dir)

    src_index = src_p / "model.safetensors.index.json"
    if src_index.exists():
        src_weight_map = json.loads(src_index.read_text())["weight_map"]
    elif (src_p / "model.safetensors").exists():
        with safe_open(src_p / "model.safetensors", framework="pt") as f:
            src_weight_map = {k: "model.safetensors" for k in f.keys()}
    else:
        return

    mtp_keys = sorted(k for k in src_weight_map if k.startswith("mtp.") or ".mtp." in k)
    if not mtp_keys:
        return

    out_index = out_dir / "model.safetensors.index.json"
    out_single = out_dir / "model.safetensors"
    if out_index.exists():
        existing_weight_map = dict(json.loads(out_index.read_text())["weight_map"])
        existing_files = sorted(set(existing_weight_map.values()))
    elif out_single.exists():
        with safe_open(out_single, framework="pt") as f:
            existing_weight_map = {k: "model.safetensors" for k in f.keys()}
        existing_files = ["model.safetensors"]
    else:
        shards = sorted(out_dir.glob("model-*-of-*.safetensors"))
        if not shards:
            return
        existing_weight_map = {}
        for sh in shards:
            with safe_open(sh, framework="pt") as f:
                for k in f.keys():
                    existing_weight_map[k] = sh.name
        existing_files = [sh.name for sh in shards]

    if any(k.startswith("mtp.") or ".mtp." in k for k in existing_weight_map):
        print("[mtp-restore] MTP weights already present in output — no patch needed.")
        return

    print(f"[mtp-restore] Source has {len(mtp_keys)} MTP tensors that the loader class "
          f"dropped on save; restoring as a BF16 shard alongside quantized weights.")

    shard_to_keys = defaultdict(list)
    for k in mtp_keys:
        shard_to_keys[src_weight_map[k]].append(k)
    mtp_tensors = {}
    mtp_bytes = 0
    for sh, keys in shard_to_keys.items():
        with safe_open(src_p / sh, framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k)
                mtp_tensors[k] = t
                mtp_bytes += t.element_size() * t.numel()

    dtype_bytes = {
        "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
        "F8_E4M3": 1, "F8_E5M2": 1,
        "U8": 1, "I8": 1, "I32": 4, "I64": 8, "BOOL": 1,
    }
    existing_bytes = 0
    for fname in existing_files:
        with safe_open(out_dir / fname, framework="pt") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                shape = sl.get_shape()
                sz = dtype_bytes.get(sl.get_dtype().upper(), 4)
                n = 1
                for d in shape:
                    n *= d
                existing_bytes += n * sz

    new_total = len(existing_files) + 1
    rename_map = {}
    for i, old in enumerate(existing_files, start=1):
        rename_map[old] = f"model-{i:05d}-of-{new_total:05d}.safetensors"
    for old, new in rename_map.items():
        if old != new:
            shutil.move(str(out_dir / old), str(out_dir / new))

    new_weight_map = {k: rename_map[v] for k, v in existing_weight_map.items()}
    mtp_name = f"model-{new_total:05d}-of-{new_total:05d}.safetensors"
    save_file(mtp_tensors, str(out_dir / mtp_name), metadata={"format": "pt"})
    for k in mtp_tensors:
        new_weight_map[k] = mtp_name

    total_size = existing_bytes + mtp_bytes
    out_index.write_text(json.dumps(
        {"metadata": {"total_size": total_size}, "weight_map": new_weight_map},
        indent=2,
    ))
    print(f"[mtp-restore] Wrote {new_total}-shard layout, "
          f"{len(existing_weight_map)} quantized + {len(mtp_keys)} MTP keys, "
          f"total_size={total_size / 1024 ** 3:.2f} GiB")

    # Patch quantization_config.ignore in config.json. The recipe-time `re:^mtp\..*`
    # regex never matched anything because the Qwen3_5/Qwen3_6/Next loader class
    # discarded MTP on load — so the saved `ignore` list (the *expanded* form
    # llmcompressor walks from the live model) has zero MTP entries. Without
    # this, vLLM's compressed-tensors loader wraps every MTP Linear with NVFP4
    # weight_packed / weight_scale parameters and fails to load the BF16 MTP
    # weights we just restored — silently leaving MTP at random init, killing
    # spec-decoding accept rate.
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        qc = cfg.get("quantization_config")
        if qc and isinstance(qc.get("ignore"), list):
            mtp_linear_modules = set()
            for k in mtp_tensors:
                if not k.endswith(".weight"):
                    continue
                if mtp_tensors[k].dim() == 2:
                    mtp_linear_modules.add(k[: -len(".weight")])
            added = 0
            for m in sorted(mtp_linear_modules):
                if m not in qc["ignore"]:
                    qc["ignore"].append(m)
                    added += 1
            if added:
                cfg_path.write_text(json.dumps(cfg, indent=2))
                print(f"[mtp-restore] Patched quantization_config.ignore "
                      f"with {added} MTP Linear module names "
                      f"(prevents vLLM from wrapping them as NVFP4).")


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
    import transformers
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    print(f"[load] model from {src}  dtype={args.dtype}")
    trust = not args.no_trust_remote_code
    model_kwargs = {"dtype": args.dtype, "trust_remote_code": trust}

    # Loader strategy:
    #   * For VLMs: load via the explicit class declared in config.json
    #     `architectures` field (e.g. Qwen3_5ForConditionalGeneration). This
    #     preserves the full multimodal structure — vision tower, vision_config,
    #     embed_vision, etc. — in both the live model and the saved
    #     state_dict/config.json. Required so vLLM's multimodal dispatchers
    #     can load the resulting checkpoint (vLLM's qwen3_5 / gemma4 / llama4
    #     model registries route through the multimodal pipeline and expect a
    #     full Foo*Config with vision_config; AutoModelForCausalLM-saved
    #     checkpoints fail there with "Invalid type of HuggingFace config").
    #   * For non-VLM models: AutoModelForCausalLM is the right loader.
    #
    # Vision-tower Linears stay BF16 in either case via the auto-defaults
    # ignore list (`re:.*vision_tower.*`, `re:visual.*`, etc.). What changes
    # is whether the vision components survive in the saved checkpoint at all.
    loader_class = None
    if arch.get("is_vlm") and arch.get("architectures"):
        arch_name = arch["architectures"][0]
        loader_class = getattr(transformers, arch_name, None)
        if loader_class is not None:
            print(f"[load] VLM detected; using {arch_name}.from_pretrained "
                  "(preserves vision_config in saved checkpoint)")
        else:
            print(f"[load] VLM detected but {arch_name!r} not in transformers "
                  "namespace; falling back to AutoModelForCausalLM "
                  "(saved checkpoint will be text-only)")

    if loader_class is not None:
        model = loader_class.from_pretrained(src, **model_kwargs)
    else:
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

    _restore_mtp_weights(out, src)

    print(f"[done] checkpoint at: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
