# Parameter Reference: Autonomous ML Training Loop Pipeline

This document describes every parameter in the generic autoloop pipeline, organized by category. Each entry includes the parameter's role, which model types use it, defaults, and practical guidance.

---

## 1. Pipeline Control

### 1.1 Task Routing

#### `task_type`
| | |
|---|---|
| **Type** | string |
| **Values** | `"detection"`, `"classification"`, `"llm_finetune"` |
| **Required** | Yes |
| **Priority** | P0 — v1 |

The single discriminator that makes the pipeline generic. Everything downstream branches on this value:
- **Decision step**: selects the primary metric and direction (AP vs accuracy vs eval_loss)
- **HP adjustment step**: generates task-appropriate search spaces and adjustment rules
- **Eval step**: selects the correct evaluation method (COCOeval vs top-k accuracy vs loss)
- **Export step**: uses the correct packaging format (.pth + MMDet config vs LoRA adapter tar.gz)

Set this once at pipeline launch. It is immutable across retrain iterations.

---

### 1.2 Identity

#### `user_id`
| | |
|---|---|
| **Type** | string |
| **Model types** | All |
| **Required** | Yes |
| **Priority** | P0 — v1 |

Clarifai user ID. Used for dataset access, artifact upload/download, and model registration. Must match the PAT token's owner.

#### `app_id`
| | |
|---|---|
| **Type** | string |
| **Model types** | All |
| **Required** | Yes |
| **Priority** | P0 — v1 |

Clarifai application ID. All artifacts, datasets, and model versions are scoped to this app.

#### `model_id`
| | |
|---|---|
| **Type** | string |
| **Default** | `test_model` |
| **Model types** | All |
| **Priority** | P0 — v1 |

Identifier for the model being trained and exported. Used as the model name on the Clarifai platform and as part of the artifact ID for checkpoint storage.

---

### 1.3 Data

#### `dataset_id`
| | |
|---|---|
| **Type** | string |
| **Model types** | Detection, Classification |
| **Required** | Yes (vision tasks) |
| **Priority** | P0 — v1 |

Clarifai dataset ID containing the training images and annotations. The pipeline downloads this dataset via the Clarifai SDK's protobuf export mechanism.

Not used for LLM fine-tuning (which pulls from HuggingFace datasets).

#### `dataset_version_id`
| | |
|---|---|
| **Type** | string |
| **Default** | `""` (latest version) |
| **Model types** | Detection, Classification |
| **Priority** | P1 — v1 |

Pins to a specific dataset version for reproducibility. When empty, the latest version is used.

#### `concepts`
| | |
|---|---|
| **Type** | string (JSON array) |
| **Default** | `'["bird","cat"]'` |
| **Model types** | Detection, Classification |
| **Priority** | P0 — v1 |

JSON list of class names to train on. For detection, these are the object categories. For classification, these are the label names. Only inputs annotated with these concepts are included in training.

Example: `'["car","truck","bus"]'`

#### `dataset_name`
| | |
|---|---|
| **Type** | string |
| **Default** | `mlabonne/FineTome-100k` |
| **Model types** | LLM Fine-tuning |
| **Priority** | P0 — v1 |

HuggingFace dataset identifier. The pipeline calls `datasets.load_dataset(dataset_name)` to fetch training data. Must be in ShareGPT/conversational format or convertible via `standardize_sharegpt()`.

---

## 2. Shared Training Parameters

These parameters apply across all model types, though defaults differ.

### 2.1 Reproducibility

#### `seed`
| | |
|---|---|
| **Type** | int |
| **Default** | `-1` |
| **Constraints** | -1 to 999999 |
| **Model types** | All |
| **Priority** | P1 — v1 |

Random seed for reproducible training. Set to `-1` to disable seeding (non-deterministic). When using random search HP tuning, the effective seed for each iteration is derived as `seed + current_iteration` to ensure different but reproducible HP samples per retry.

### 2.2 Duration

#### `num_epochs`
| | |
|---|---|
| **Type** | int |
| **Constraints** | 1–1000 |
| **Model types** | All |
| **Priority** | P0 — v1 |

Training epochs per iteration. Task-specific defaults:
- **Detection**: 100 (YOLOF converges slowly due to single-scale feature map)
- **Classification**: 200 (ResNet with cosine annealing needs more epochs)
- **LLM Fine-tuning**: 1 (LoRA adapters typically converge in 1 epoch on large datasets)

When the `epoch_scale_factor` parameter is used (v2), epochs are multiplied by this factor on each retry.

### 2.3 Batch

#### `batch_size`
| | |
|---|---|
| **Type** | int |
| **Constraints** | 1–256 |
| **Model types** | All |
| **Priority** | P0 — v1 |

Per-device training batch size. Task-specific defaults:
- **Detection**: 16 (limited by GPU memory due to large feature maps at 512px)
- **Classification**: 64 (smaller 224px images allow larger batches)
- **LLM Fine-tuning**: 4 (combined with `gradient_accumulation_steps=4` for effective batch of 16)

For vision tasks, batch size directly affects the effective learning rate:
```
effective_lr = batch_size × num_gpus × per_item_lrate
```

### 2.4 Initialization

#### `pretrained_weights`
| | |
|---|---|
| **Type** | string |
| **Values** | `"coco"`, `"ImageNet-1k"`, `"None"`, or a file path |
| **Model types** | Detection, Classification |
| **Priority** | P0 — v1 |

Source of pretrained weights for transfer learning:
- **Detection**: `"coco"` downloads YOLOF ResNet-50 trained on COCO
- **Classification**: `"ImageNet-1k"` downloads ResNet-50 trained on ImageNet
- **`"None"`**: Train from scratch (rarely useful)
- **File path**: Used internally for warm-start retraining (previous iteration's checkpoint)

#### `base_model_name`
| | |
|---|---|
| **Type** | string |
| **Default** | `unsloth/Qwen3-0.6B` |
| **Model types** | LLM Fine-tuning |
| **Priority** | P0 — v1 |

HuggingFace model identifier for the base LLM. The pipeline loads this via Unsloth's `FastLanguageModel.from_pretrained()`. Supports any Unsloth-optimized model (Llama, Mistral, Qwen, Gemma, etc.).

### 2.5 Regularization

#### `weight_decay`
| | |
|---|---|
| **Type** | float |
| **Constraints** | 0.0–1.0 |
| **Model types** | All |
| **Priority** | P1 — v1 |

L2 regularization strength applied to all non-bias, non-normalization parameters. Task-specific defaults:
- **Detection**: 0.0001 (hardcoded in MMDet config)
- **Classification**: 0.01 (standard for ResNet fine-tuning)
- **LLM Fine-tuning**: 0.01 (standard for LoRA, helps prevent overfitting)

Higher values penalize large weights more aggressively — useful when the model overfits on small datasets.

### 2.6 LR Schedule

#### `warmup_ratio`
| | |
|---|---|
| **Type** | float |
| **Default** | varies |
| **Constraints** | 0.0–1.0 |
| **Model types** | Classification, LLM Fine-tuning |
| **Priority** | P1 — v1 |

Fraction of total training steps used for learning rate warmup (linear ramp from near-zero to peak LR). Prevents early training instability.
- **Classification**: 0.0001 (very short warmup; 5 iterations)
- **LLM Fine-tuning**: 0.06 (6% of total steps — standard for LLM SFT)

#### `warmup_iters`
| | |
|---|---|
| **Type** | int |
| **Default** | 5 |
| **Constraints** | 0–1000 |
| **Model types** | Classification |
| **Priority** | P2 — v1 |

Absolute warmup iteration count (alternative to `warmup_ratio`). Used by MMPretrain's `LinearLR` warmup. If both `warmup_ratio` and `warmup_iters` are set, the framework uses whichever corresponds to fewer steps.

---

## 3. Vision-Specific Training

### 3.1 Learning Rate

#### `per_item_lrate`
| | |
|---|---|
| **Type** | float |
| **Constraints** | 0.0–1.0 |
| **Model types** | Detection, Classification |
| **Priority** | P0 — v1 |

Per-item contribution to the effective learning rate. The actual optimizer LR is computed as:

```
effective_lr = batch_size × max(1, num_gpus) × per_item_lrate
```

Task-specific defaults:
- **Detection**: 0.001875 → effective LR of 0.03 at batch=16
- **Classification**: 1.953125e-5 → effective LR of 0.00125 at batch=64

This abstraction decouples the learning rate from batch size changes — you can change `batch_size` without retuning the LR.

#### `per_item_min_lrate`
| | |
|---|---|
| **Type** | float |
| **Default** | 1.5625e-8 |
| **Constraints** | 0.0–1.0 |
| **Model types** | Classification |
| **Priority** | P2 — v1 |

Minimum per-item learning rate at the end of cosine annealing. The actual minimum LR is `batch_size × num_gpus × per_item_min_lrate`. Ensures the optimizer doesn't fully stall at the tail of the schedule.

### 3.2 Backbone

#### `frozen_stages`
| | |
|---|---|
| **Type** | int |
| **Default** | 1 |
| **Constraints** | 0–4 |
| **Model types** | Detection |
| **Priority** | P0 — v1 |

Number of ResNet backbone stages to freeze during training. The ResNet-50 backbone has 4 stages:
- **0**: Fully trainable (all backbone weights update)
- **1**: Freeze stage 1 only (default — preserves low-level features)
- **4**: Freeze entire backbone (only train neck + head)

**Autoloop behavior**: When `unfreeze_on_retry=true`, the decision step decrements this by 1 on each retry (floored at 0), progressively allowing more backbone adaptation.

### 3.3 Input

#### `image_size`
| | |
|---|---|
| **Type** | string (JSON) / int |
| **Model types** | Detection, Classification |
| **Priority** | P0 — v1 |

Input image dimensions:
- **Detection**: JSON array `[512]` (min side length) or `[H, W]` for explicit dimensions. When `keep_aspect_ratio=true`, the max side is computed as `max_aspect_ratio × min_side`.
- **Classification**: Single integer `224` for square crop after resize.

Larger sizes improve accuracy (especially for small objects in detection) but increase GPU memory usage quadratically.

#### `max_aspect_ratio`
| | |
|---|---|
| **Type** | float |
| **Default** | 1.5 |
| **Constraints** | 1.0–5.0 |
| **Model types** | Detection |
| **Priority** | P1 — v1 |

Maximum aspect ratio multiplier when `keep_aspect_ratio=true`. The max image side is `max_aspect_ratio × min_side`. For example, with `image_size=[512]` and `max_aspect_ratio=1.5`, images are resized so the short side is 512px and the long side is at most 768px.

#### `keep_aspect_ratio`
| | |
|---|---|
| **Type** | bool |
| **Default** | true |
| **Model types** | Detection |
| **Priority** | P1 — v1 |

Whether to preserve the original image aspect ratio during resize. When `false`, images are force-resized to the exact `[H, W]` specified in `image_size`, which may distort objects.

### 3.4 Augmentation

#### `flip_probability`
| | |
|---|---|
| **Type** | float |
| **Default** | 0.5 |
| **Constraints** | 0.0–1.0 |
| **Model types** | Classification |
| **Priority** | P2 — v1 |

Probability of applying random flip augmentation during training. Set to 0.0 to disable flipping (useful for orientation-sensitive tasks like document classification).

#### `flip_direction`
| | |
|---|---|
| **Type** | string |
| **Default** | `horizontal` |
| **Values** | `"horizontal"`, `"vertical"` |
| **Model types** | Classification |
| **Priority** | P2 — v1 |

Direction of random flip augmentation. Vertical flips are useful for aerial/satellite imagery; horizontal flips are standard for natural images.

### 3.5 Labels

#### `concepts_mutually_exclusive`
| | |
|---|---|
| **Type** | bool |
| **Default** | false |
| **Model types** | Classification |
| **Priority** | P1 — v1 |

Controls single-label vs multi-label classification:
- **`true`**: Each image has exactly one class (softmax + CrossEntropyLoss)
- **`false`**: Each image can have multiple classes (sigmoid + BinaryCrossEntropyLoss)

### 3.6 Data

#### `min_samples_per_epoch`
| | |
|---|---|
| **Type** | int |
| **Default** | 300 |
| **Constraints** | 1–10000 |
| **Model types** | Detection |
| **Priority** | P1 — v1 |

Minimum number of data samples per training epoch. If the dataset is smaller than this, samples are repeated to meet the minimum. Prevents epochs from being too short on small datasets.

### 3.7 Benchmark

#### `inference_max_batch_size`
| | |
|---|---|
| **Type** | int |
| **Default** | 2 |
| **Constraints** | 1–32 |
| **Model types** | Detection, Classification |
| **Priority** | P1 — v1 |

Batch size used during GPU memory benchmarking after training. The benchmark step runs dummy inference to measure peak GPU memory, then writes the required `accelerator_memory` into the model's config.yaml for deployment.

---

## 4. LLM-Specific Training

### 4.1 Learning Rate

#### `learning_rate`
| | |
|---|---|
| **Type** | float |
| **Default** | 2e-4 |
| **Constraints** | 1e-6 to 1e-2 |
| **Model types** | LLM Fine-tuning |
| **Priority** | P0 — v1 |

Peak learning rate for the optimizer. Unlike vision's `per_item_lrate`, this is the direct value passed to `TrainingArguments`. Uses 8-bit AdamW optimizer by default.

#### `lr_scheduler_type`
| | |
|---|---|
| **Type** | string |
| **Default** | `cosine` |
| **Values** | `"cosine"`, `"linear"`, `"constant"` |
| **Model types** | LLM Fine-tuning |
| **Priority** | P1 — v1 |

Learning rate decay schedule after warmup:
- **`cosine`**: Standard for LLM SFT — smooth decay to near-zero
- **`linear`**: Linear decay to zero — simpler but equally effective for short training
- **`constant`**: No decay — LR stays at peak after warmup (useful for very short fine-tuning)

### 4.2 LoRA

#### `lora_r`
| | |
|---|---|
| **Type** | int |
| **Default** | 16 |
| **Values** | 4, 8, 16, 32, 64, 128 |
| **Model types** | LLM Fine-tuning |
| **Priority** | P0 — v1 |

LoRA adapter rank — the most impactful hyperparameter for LoRA fine-tuning. Controls the dimensionality of the low-rank decomposition:
- **Lower rank (4–8)**: Fewer trainable parameters, faster training, less expressive
- **Higher rank (32–128)**: More capacity, better for complex tasks, more VRAM

**Autoloop behavior**: If the model underfits (high eval_loss), the HP adjustment step may increase `lora_r` (e.g., 16 → 32 → 64) before reducing learning rate.

#### `lora_alpha`
| | |
|---|---|
| **Type** | int |
| **Default** | 16 |
| **Constraints** | 1–128 |
| **Model types** | LLM Fine-tuning |
| **Priority** | P1 — v1 |

LoRA scaling factor. The effective adapter weight is scaled by `lora_alpha / lora_r`. Convention is to set `lora_alpha = lora_r` (scaling factor = 1.0). When increasing `lora_r` during HP tuning, `lora_alpha` should be kept equal to `lora_r`.

#### `lora_dropout`
| | |
|---|---|
| **Type** | float |
| **Default** | 0.0 |
| **Constraints** | 0.0–0.5 |
| **Model types** | LLM Fine-tuning |
| **Priority** | P1 — v1 |

Dropout rate applied to LoRA adapter layers. Most LoRA fine-tuning uses 0.0 (no dropout), but adding small dropout (0.05) can help when the model overfits (train_loss << eval_loss). The HP adjustment step may increase this when overfitting is detected.

### 4.3 Batch

#### `gradient_accumulation_steps`
| | |
|---|---|
| **Type** | int |
| **Default** | 4 |
| **Constraints** | 1–64 |
| **Model types** | LLM Fine-tuning |
| **Priority** | P1 — v1 |

Number of forward/backward passes before an optimizer step. Effective batch size is `batch_size × gradient_accumulation_steps`. This allows simulating large batches on limited GPU memory.

With defaults: `4 × 4 = 16` effective batch size.

### 4.4 Input

#### `max_seq_length`
| | |
|---|---|
| **Type** | int |
| **Default** | 2048 |
| **Constraints** | 128–8192 |
| **Model types** | LLM Fine-tuning |
| **Priority** | P0 — v1 |

Maximum tokenized sequence length. Inputs longer than this are truncated. GPU memory scales linearly with sequence length (approximately). For chat/instruction tuning, 2048 is standard; for long-context tasks, increase to 4096+.

### 4.5 Quantization

#### `load_in_4bit`
| | |
|---|---|
| **Type** | bool |
| **Default** | true |
| **Model types** | LLM Fine-tuning |
| **Priority** | P1 — v1 |

Load base model weights in 4-bit quantization (QLoRA). Reduces base model VRAM by ~75% with minimal quality loss. For example, a 7B model drops from ~14GB to ~3.5GB. Only the LoRA adapters train in full precision.

Set to `false` for maximum quality (if VRAM allows).

### 4.6 Duration & Monitoring

#### `max_steps`
| | |
|---|---|
| **Type** | int |
| **Default** | -1 |
| **Model types** | LLM Fine-tuning |
| **Priority** | P2 — v1 |

Override `num_epochs` with a fixed step count. When set to -1, training runs for the full `num_epochs`. Useful for debugging (`max_steps=100` for a quick sanity check) or when the dataset size is unknown.

#### `logging_steps`
| | |
|---|---|
| **Type** | int |
| **Default** | 10 |
| **Constraints** | 1–1000 |
| **Model types** | LLM Fine-tuning |
| **Priority** | P2 — v1 |

Log training metrics (loss, learning rate, epoch) every N optimizer steps. Lower values give finer-grained loss curves but slightly increase overhead.

#### `save_steps`
| | |
|---|---|
| **Type** | int |
| **Default** | 100 |
| **Constraints** | 1–10000 |
| **Model types** | LLM Fine-tuning |
| **Priority** | P2 — v1 |

Save a checkpoint every N optimizer steps. Checkpoints include the LoRA adapter weights and optimizer state. More frequent saves allow resuming from closer to failure but consume disk space.

---

## 5. Autoloop Control Parameters

### 5.1 Decision

#### `metric_threshold`
| | |
|---|---|
| **Type** | float |
| **Model types** | All |
| **Priority** | P0 — v1 |

The quality gate — the trained model must meet or exceed this metric value to be accepted and exported. Task-specific guidance:
- **Detection**: 0.50 AP is a reasonable production baseline for custom object detectors
- **Classification**: 0.85 top-1 accuracy for most practical use cases
- **LLM Fine-tuning**: 1.5 eval_loss (lower is better — set based on baseline model's loss)

The decision step compares `primary_metric` against this threshold. When `metric_direction="maximize"`, the model passes if metric ≥ threshold. When `"minimize"`, it passes if metric ≤ threshold.

#### `primary_metric`
| | |
|---|---|
| **Type** | string |
| **Default** | `"auto"` |
| **Model types** | All |
| **Priority** | P0 — v1 |

Which metric drives the retrain decision. When set to `"auto"`:
- **Detection**: `AP` (mAP@IoU=0.50:0.95 — the standard COCO primary metric)
- **Classification**: `accuracy/top1` (top-1 classification accuracy)
- **LLM Fine-tuning**: `eval_loss` (validation cross-entropy loss)

Can be overridden to any metric the eval step produces (e.g., `"AP50"`, `"accuracy/top5"`, `"perplexity"`).

#### `metric_direction`
| | |
|---|---|
| **Type** | string |
| **Default** | `"auto"` |
| **Values** | `"maximize"`, `"minimize"`, `"auto"` |
| **Model types** | All |
| **Priority** | P1 — v1 |

Whether higher is better or lower is better. Auto-resolution:
- `AP`, `AP50`, `accuracy/*`, `AR*` → `"maximize"`
- `eval_loss`, `perplexity`, `*_loss` → `"minimize"`

### 5.2 Loop

#### `max_retrain_iterations`
| | |
|---|---|
| **Type** | int |
| **Default** | 3 |
| **Constraints** | 1–10 |
| **Model types** | All |
| **Priority** | P0 — v1 |

Maximum retrain attempts before the pipeline declares failure and exits with an error. Total training runs = `max_retrain_iterations + 1` (initial train + N retries).

Cost implication: each iteration costs ~2.5 GPU-hours (train + eval). Setting to 3 caps total cost at ~10 GPU-hours (~$10 on g5.xlarge).

#### `warm_start`
| | |
|---|---|
| **Type** | bool |
| **Default** | true |
| **Model types** | All |
| **Priority** | P0 — v1 |

Whether retrain iterations resume from the previous iteration's checkpoint:
- **`true`** (warm-start): Loads previous checkpoint, trains with adjusted HPs. Faster convergence, avoids wasting prior compute.
- **`false`** (cold-start): Always starts from `pretrained_weights`/`base_model_name`. Each iteration is independent — useful when prior training diverged badly.

### 5.3 HP Tuning

#### `tuning_strategy`
| | |
|---|---|
| **Type** | string |
| **Default** | `"schedule"` |
| **Values** | `"schedule"`, `"grid"`, `"random"` |
| **Model types** | All |
| **Priority** | P0 — v1 |

Hyperparameter adjustment strategy:
- **`schedule`**: Predefined decay — halve LR, unfreeze backbone (detection), increase epochs. Deterministic, zero overhead. Best default.
- **`grid`**: Enumerate all combos from `search_space`, try one per iteration. Systematic but may waste iterations on bad configs.
- **`random`**: Sample from distributions defined in `search_space`. Better coverage than grid for high-dimensional spaces.

See the [HP Tuning Step Design Doc](hp-tuning-design.md) for detailed strategy descriptions.

#### `search_space`
| | |
|---|---|
| **Type** | string (JSON) |
| **Default** | `"auto"` |
| **Model types** | All |
| **Priority** | P1 — v1 |

JSON defining the tunable hyperparameters and their values/ranges. When `"auto"`, generated based on `task_type`:

**Detection auto**:
```json
{
  "per_item_lrate": {"type": "schedule", "factor": 0.5},
  "frozen_stages": {"type": "schedule", "delta": -1, "min": 0}
}
```

**Classification auto**:
```json
{
  "per_item_lrate": {"type": "schedule", "factor": 0.5},
  "weight_decay": {"type": "grid", "values": [0.01, 0.001]},
  "num_epochs": {"type": "schedule", "factor": 1.5, "max": 500}
}
```

**LLM auto**:
```json
{
  "learning_rate": {"type": "schedule", "factor": 0.5},
  "lora_r": {"type": "grid", "values": [16, 32, 64]},
  "num_epochs": {"type": "grid", "values": [1, 2, 3]},
  "lora_dropout": {"type": "grid", "values": [0.0, 0.05]}
}
```

#### `lr_decay_factor`
| | |
|---|---|
| **Type** | float |
| **Default** | 0.5 |
| **Constraints** | 0.1–1.0 |
| **Model types** | All |
| **Priority** | P0 — v1 |

Multiplier applied to learning rate on each retry iteration (used by `schedule` and `random` strategies). With default 0.5, the LR halves each time:
```
Iteration 1: 0.001875 → Iteration 2: 0.0009375 → Iteration 3: 0.00046875
```

A factor of 0.3 is more aggressive (faster convergence to low LR); 0.7 is gentler (more iterations at moderate LR).

#### `unfreeze_on_retry`
| | |
|---|---|
| **Type** | bool |
| **Default** | true |
| **Model types** | Detection |
| **Priority** | P1 — v1 |

When true, the decision step decrements `frozen_stages` by 1 on each retry, allowing progressively more backbone adaptation. Floors at 0 (fully unfrozen). Only applies to detection tasks where backbone freezing is meaningful.

#### `hyperparams_override`
| | |
|---|---|
| **Type** | string (JSON) |
| **Default** | `""` |
| **Model types** | All |
| **Priority** | P1 — v1 |

JSON blob of hyperparameter overrides passed from the HP adjustment step to the train step. Values in this blob override the corresponding named workflow parameters. This is the mechanism by which adjusted HPs flow back into the training step.

Example: `'{"per_item_lrate": 0.0009375, "frozen_stages": 0}'`

The train step parses this and applies overrides before training:
```python
if hyperparams_override:
    overrides = json.loads(hyperparams_override)
    for key, value in overrides.items():
        setattr(self, key, value)
```

### 5.4 Stopping

#### `early_stop_min_delta`
| | |
|---|---|
| **Type** | float |
| **Default** | 0.01 |
| **Constraints** | 0.0–1.0 |
| **Model types** | All |
| **Priority** | P1 — v2 |

Minimum metric improvement between consecutive iterations. If improvement is below this threshold, the pipeline stops early with a failure to avoid wasting compute on diminishing returns.

Task-specific guidance:
- **Detection**: 0.01 AP improvement (~1% mAP) is a meaningful signal
- **Classification**: 0.01 accuracy improvement (~1%)
- **LLM**: 0.05 eval_loss improvement

#### `overfitting_detection`
| | |
|---|---|
| **Type** | bool |
| **Default** | true (LLM), false (vision) |
| **Model types** | LLM Fine-tuning |
| **Priority** | P1 — v2 |

When enabled, the decision step checks for overfitting by comparing train_loss and eval_loss. If `train_loss < eval_loss × 0.5`, this triggers regularization adjustments:
- Increase `lora_dropout` (0.0 → 0.05)
- Reduce `num_epochs`
- Reduce `lora_r`

Not enabled by default for vision tasks because MMDetection/MMPretrain already run validation every epoch within the training step.

---

## 6. Evaluation Parameters

#### `score_threshold`
| | |
|---|---|
| **Type** | float |
| **Default** | 0.05 |
| **Constraints** | 0.0–1.0 |
| **Model types** | Detection |
| **Priority** | P1 — v1 |

Minimum confidence score for a detection to be included in evaluation. Lower values (0.01–0.05) include more detections (higher recall, lower precision). Higher values (0.3+) are stricter.

The standard COCO evaluation uses 0.05 and lets the AP computation handle precision/recall trade-offs internally.

#### `iou_threshold`
| | |
|---|---|
| **Type** | float |
| **Default** | 0.6 |
| **Constraints** | 0.0–1.0 |
| **Model types** | Detection |
| **Priority** | P1 — v1 |

IoU threshold for Non-Maximum Suppression (NMS) during evaluation. Controls how much overlap is tolerated between detections:
- **0.5**: Standard — suppresses heavily overlapping boxes
- **0.6**: Slightly lenient (default)
- **0.3**: Aggressive suppression — better for crowded scenes

#### `eval_batch_size`
| | |
|---|---|
| **Type** | int |
| **Default** | 4 |
| **Constraints** | 1–64 |
| **Model types** | All |
| **Priority** | P1 — v1 |

Batch size for evaluation inference. Smaller than training batch_size because evaluation may need more memory for full-resolution images (detection) or generation (LLM).

#### `eval_config`
| | |
|---|---|
| **Type** | string (JSON) |
| **Default** | `"auto"` |
| **Model types** | All |
| **Priority** | P2 — v2 |

JSON blob for task-specific evaluation settings that don't fit as top-level parameters:

```json
// Detection
{"score_threshold": 0.05, "iou_threshold": 0.6, "max_per_img": 100}

// Classification
{"topk": [1, 5]}

// LLM
{"max_new_tokens": 256, "temperature": 0.0, "benchmark_prompts": 10}
```

---

## 7. Internal (Step-to-Step) Parameters

These parameters are not user-facing — they flow between pipeline steps via Argo output parameters.

#### `checkpoint_source`
How the eval step obtains the trained checkpoint. `"artifact"` for standalone evaluation (downloads from Clarifai store); `"path"` when chained after a train step in the autoloop.

#### `checkpoint_path`
Filesystem path to the trained checkpoint (`.pth` for vision, LoRA adapter directory for LLM). Written by the train step, consumed by eval and export steps.

#### `config_path`
Path to the MMDetection/MMPretrain config file (vision tasks only). Written by the train step, consumed by the eval step to configure the model architecture.

#### `eval_results_json`
Path to the evaluation results JSON file. Written by the eval step, consumed by the decision step. Contains the full metrics dict + metadata.

#### `current_iteration`
Loop iteration counter (1-indexed). Starts at 1, incremented by the decision step on each retrain. Used by the decision step to check against `max_retrain_iterations`.

#### `hp_history`
JSON array of all previous iterations' hyperparameter configs and their resulting metrics. Accumulated by the decision step and passed forward. Used by grid/random search to avoid repeats and by LLM-guided tuning (v3) for context.

#### `hyperparams_json`
JSON blob output by the HP adjustment step containing the adjusted hyperparameters for the next iteration. Consumed by the train step via `hyperparams_override`.

---

## 8. Future Parameters (v2–v3)

#### `epoch_scale_factor` (v2)
Multiply `num_epochs` by this factor on each retry. Default 1.0 (no scaling). Set to 1.5 to give lower LR more convergence time.

#### `max_parallel_runs` (v2)
Maximum concurrent training runs for parallel grid/random search. Requires Argo `withParam` fan-out and sufficient GPU budget.

#### `notification_webhook` (v3)
Slack or email webhook URL. The `report-failure` step sends a notification when max iterations are exhausted.

#### `multi_metric_condition` (v3)
JSON defining compound quality gates: `{"AP": ">=0.50", "AP50": ">=0.70"}`. All conditions must be met for the model to pass.
