# llm-profiler

Practical profiling utilities for PyTorch models (including LLMs), with an engineering-first focus.

This project provides lightweight, **model-agnostic** helpers to compare models and configurations
(FP32 vs quantized, before/after optimizations, etc.) using metrics that actually matter in practice.

---

## Features

- Parameter counts (total vs trainable vs frozen)
- Model size estimation:
  - params + buffers (theoretical)
  - serialized `state_dict` size (realistic, works with quantization)
- Per-layer statistics (top-k layers by parameter count)
- Weight distribution helpers (near-zero / sparsity)
- Inference latency (average + configurable percentile)
- Optional **generation latency** using `model.generate()`
- Pretty console output (optional, via `rich`)
- No hard dependency on Hugging Face (Transformers optional)

---

## Installation

For now, just copy `llm_profiler.py` into your project.

Optional dependencies:

- `rich` → pretty console output
- `pandas` → DataFrame layer stats
- `matplotlib` → histogram plots
- `transformers` → examples with Hugging Face models

```bash
pip install rich pandas matplotlib transformers
```

---

## Quickstart (Hugging Face)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_profiler import profile_model, print_profile_report

model_id = "facebook/opt-125m"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()

enc = tokenizer("Hello, profiling!", return_tensors="pt")
inputs = {k: v.to(device) for k, v in enc.items()}

report = profile_model(
    model,
    sample_inputs=inputs,
    device=device,
    percentile=95,
    num_runs=30,
    warmup_runs=10,
    measure_generate=True,
    max_new_tokens=32,
)

print_profile_report(report, top_k_layers=15)
```

---

## Notes on Model Size

- `model_size_mb` measures parameters + buffers and is best for FP32 / FP16.
- `state_dict_size_mb` measures the serialized weights and is more accurate for
  quantized or packed models (INT8, dynamic quantization, etc.).

---

## Examples

- `examples/profile_opt125m_int8.py`  
  Compare FP32 vs INT8 dynamic quantization on CPU.

- `examples/profile_forward_vs_generate.py`  
  Compare forward-pass latency vs generation latency.

---

## License

MIT License
