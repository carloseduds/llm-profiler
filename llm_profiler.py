"""llm_profiler.py

Model-agnostic utilities to profile PyTorch models (including LLMs).

This module focuses on practical metrics you typically need when comparing
"before vs after" model changes (FP32 vs quantized, pruning, freezing layers,
etc.):

- Parameter counts (total vs trainable vs frozen)
- Model size estimates:
  - In-memory estimate from parameters + buffers (good for FP32/FP16)
  - Serialized `state_dict` size (more realistic for quantized/packed models)
- Per-layer statistics (Linear/Conv/Norm/Embedding/etc.)
- Weight distribution helpers (histogram + sparsity/near-zero percentages)
- Inference latency (avg + configurable percentile)
- Optional generation latency using `model.generate()` when available
- Pretty terminal reporting (optional, via `rich`)

Notes
-----
- This module does *not* depend on Hugging Face Transformers.
  However, Transformers tokenizers often return dict-like objects (e.g.,
  `BatchEncoding`). To support those, inputs are treated as `Mapping`.

Author: Carlos Eduardo Correa
License: MIT
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore


TensorLikeInputs = Union[
    torch.Tensor,
    Sequence[torch.Tensor],
    Mapping[str, torch.Tensor],
]

__all__ = [
    "TensorLikeInputs",
    "count_parameters",
    "estimate_model_size_mb",
    "estimate_state_dict_size_mb",
    "human_readable_number",
    "collect_layer_stats",
    "plot_param_distribution",
    "measure_inference_time",
    "measure_generation_time",
    "profile_model",
    "make_dummy_input_ids",
    "cuda_memory_snapshot",
    "format_profile_report",
    "print_profile_report",
]


# ---------------------------------------------------------------------
# Parameter and size helpers
# ---------------------------------------------------------------------


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts.

    Parameters
    ----------
    model:
        A PyTorch module.

    Returns
    -------
    Dict[str, int]
        Dictionary with:
        - total_params
        - trainable_params
        - frozen_params
    """
    total = 0
    trainable = 0
    for param in model.parameters():
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
    frozen = total - trainable
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen,
    }


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    """Return the number of bytes occupied by a tensor."""
    return int(tensor.numel() * tensor.element_size())


def estimate_model_size_mb(model: nn.Module, include_buffers: bool = True) -> float:
    """Estimate model size in memory (MB) from parameters (+ optional buffers).

    This estimate is accurate for standard FP32/FP16/BF16 tensors, but it may not
    reflect the true footprint of quantized/packed models.

    Parameters
    ----------
    model:
        A PyTorch module.
    include_buffers:
        Whether to include buffers (e.g., running_mean/var for BatchNorm).

    Returns
    -------
    float
        Size in MB.
    """
    total_bytes = 0
    for param in model.parameters():
        total_bytes += _tensor_nbytes(param)

    if include_buffers:
        for buf in model.buffers():
            total_bytes += _tensor_nbytes(buf)

    return total_bytes / (1024**2)


def estimate_state_dict_size_mb(
    model: nn.Module,
    *,
    use_safe_serialization: bool = False,
) -> float:
    """Estimate the *serialized* size (MB) of the model's state_dict.

    Why this is useful:
    - For quantized or packed models, `model.parameters()` may not reflect the
      actual stored weights representation.
    - Measuring the serialized `state_dict` size often provides a more realistic
      "artifact size" metric.

    Parameters
    ----------
    model:
        A PyTorch module.
    use_safe_serialization:
        If True, attempts to use `safetensors` (if installed). Note: `safetensors`
        only supports tensors, so non-tensor entries are dropped in that branch.

    Returns
    -------
    float
        Size in MB of the serialized weights.
    """
    state = model.state_dict()

    def _to_cpu_state_value(val: Any) -> Any:
        if isinstance(val, torch.Tensor):
            return val.detach().cpu()
        return val

    cpu_state = {k: _to_cpu_state_value(v) for k, v in state.items()}

    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)

    try:
        if use_safe_serialization:
            try:
                from safetensors.torch import save_file  # type: ignore

                st_path = path.replace(".pt", ".safetensors")
                tensor_only = {
                    k: v for k, v in cpu_state.items() if isinstance(v, torch.Tensor)
                }
                save_file(tensor_only, st_path)
                real_path = st_path
            except Exception:
                torch.save(cpu_state, path)
                real_path = path
        else:
            torch.save(cpu_state, path)
            real_path = path

        size_bytes = os.path.getsize(real_path)
        return size_bytes / (1024**2)
    finally:
        for p in (path, path.replace(".pt", ".safetensors")):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def human_readable_number(n: int) -> str:
    """Format a large integer into K/M/B units."""
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1_000:.2f}K"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.2f}M"
    return f"{n / 1_000_000_000:.2f}B"


# ---------------------------------------------------------------------
# Per-layer statistics
# ---------------------------------------------------------------------

_DEFAULT_LAYER_TYPES: Tuple[type, ...] = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Embedding,
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.MultiheadAttention,
)


def _safe_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a CPU float32 numpy array (detached)."""
    return tensor.detach().float().cpu().numpy()


def _calc_sparsity_and_near_zero(
    param_arrays: List[np.ndarray],
    near_zero_threshold: float = 1e-6,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute sparsity (% exact zeros) and near-zero percentage.

    Parameters
    ----------
    param_arrays:
        List of numpy arrays (flattened later).
    near_zero_threshold:
        Absolute threshold for "near-zero" values.

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        (sparsity_pct, near_zero_pct). Returns (None, None) when empty.
    """
    if not param_arrays:
        return None, None

    flat = np.concatenate([p.reshape(-1) for p in param_arrays])
    if flat.size == 0:
        return None, None

    sparsity = float((flat == 0).mean() * 100.0)
    near_zero = float((np.abs(flat) < near_zero_threshold).mean() * 100.0)
    return sparsity, near_zero


def collect_layer_stats(
    model: nn.Module,
    layer_types: Optional[Tuple[type, ...]] = None,
    include_buffers: bool = False,
    compute_sparsity: bool = True,
    near_zero_threshold: float = 1e-6,
    store_param_arrays: bool = False,
) -> Union["pd.DataFrame", List[Dict[str, Any]]]:
    """Collect per-module statistics for a subset of layer types.

    By default, only common "parameter-heavy" layers are included
    (Linear/Conv/Embedding/Norm/etc.).

    Parameters
    ----------
    model:
        A PyTorch module.
    layer_types:
        Tuple of layer classes to include. If None, uses a default set.
    include_buffers:
        Whether to include buffers in per-layer size estimates.
    compute_sparsity:
        Whether to compute sparsity and near-zero percentages.
    near_zero_threshold:
        Threshold for near-zero percentage.
    store_param_arrays:
        If True, stores concatenated parameter arrays in the output (can be huge).

    Returns
    -------
    Union[pandas.DataFrame, List[Dict[str, Any]]]
        DataFrame when pandas is available, otherwise a list of dicts.
    """
    if layer_types is None:
        layer_types = _DEFAULT_LAYER_TYPES

    layer_stats: List[Dict[str, Any]] = []

    total_params = int(sum(p.numel() for p in model.parameters()))
    total_params = max(total_params, 1)  # avoid division by zero

    for name, module in model.named_modules():
        if name == "":
            # Skip the root module entry.
            continue

        if not isinstance(module, layer_types):
            continue

        params = list(module.parameters(recurse=False))
        if not params:
            continue

        param_count = int(sum(p.numel() for p in params))
        trainable_count = int(sum(p.numel() for p in params if p.requires_grad))

        dtype = str(params[0].dtype).replace("torch.", "")
        size_bytes = int(sum(_tensor_nbytes(p) for p in params))

        if include_buffers:
            buffers = list(module.buffers(recurse=False))
            size_bytes += int(sum(_tensor_nbytes(b) for b in buffers))

        size_mb = size_bytes / (1024**2)

        stat: Dict[str, Any] = {
            "name": name,
            "type": module.__class__.__name__,
            "parameters": param_count,
            "trainable_parameters": trainable_count,
            "dtype": dtype,
            "size_mb": size_mb,
            "parameters_pct": (param_count / total_params) * 100.0,
        }

        if compute_sparsity:
            arrays = [_safe_to_numpy(p) for p in params]
            sparsity_pct, near_zero_pct = _calc_sparsity_and_near_zero(
                arrays, near_zero_threshold
            )
            stat["sparsity_pct"] = sparsity_pct
            stat["near_zero_pct"] = near_zero_pct

        if store_param_arrays:
            stat["params_array"] = np.concatenate(
                [_safe_to_numpy(p).reshape(-1) for p in params]
            )

        layer_stats.append(stat)

    layer_stats.sort(key=lambda x: x["parameters"], reverse=True)

    if pd is not None:
        return pd.DataFrame(layer_stats)
    return layer_stats


# ---------------------------------------------------------------------
# Weight distribution (optional plotting)
# ---------------------------------------------------------------------


def plot_param_distribution(
    params: Union[np.ndarray, torch.Tensor],
    title: str,
    bins: int = 50,
    near_zero_threshold: float = 1e-6,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """Plot a histogram of weights and return summary statistics.

    Parameters
    ----------
    params:
        1D array or tensor of parameters.
    title:
        Plot title.
    bins:
        Number of histogram bins.
    near_zero_threshold:
        Threshold for near-zero percentage.
    show:
        Whether to display the plot (requires matplotlib).
    save_path:
        If provided, saves the plot to this path.

    Returns
    -------
    Dict[str, float]
        Summary stats: mean/std/median/min/max/sparsity_pct/near_zero_pct.
    """
    if isinstance(params, torch.Tensor):
        params_np = _safe_to_numpy(params).reshape(-1)
    else:
        params_np = np.asarray(params).reshape(-1)

    if params_np.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "sparsity_pct": float("nan"),
            "near_zero_pct": float("nan"),
        }

    mean = float(np.mean(params_np))
    std = float(np.std(params_np))
    median = float(np.median(params_np))
    vmin = float(np.min(params_np))
    vmax = float(np.max(params_np))
    sparsity_pct = float((params_np == 0).mean() * 100.0)
    near_zero_pct = float((np.abs(params_np) < near_zero_threshold).mean() * 100.0)

    if plt is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(params_np, bins=bins, density=False)
        plt.title(title)
        plt.xlabel("Weight value")
        plt.ylabel("Count")

        stats_text = (
            f"Mean: {mean:.6f}\n"
            f"Std: {std:.6f}\n"
            f"Median: {median:.6f}\n"
            f"Sparsity: {sparsity_pct:.2f}%\n"
            f"Near-zero (<{near_zero_threshold}): {near_zero_pct:.2f}%"
        )
        plt.gca().text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", alpha=0.2),
        )

        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "min": vmin,
        "max": vmax,
        "sparsity_pct": sparsity_pct,
        "near_zero_pct": near_zero_pct,
    }


# ---------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------


def _move_inputs_to_device(inputs: TensorLikeInputs, device: torch.device) -> TensorLikeInputs:
    """Move tensor inputs to a device, supporting tensor/sequence/mapping."""
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    if isinstance(inputs, (list, tuple)):
        return type(inputs)([t.to(device) for t in inputs])
    if isinstance(inputs, Mapping):
        return {k: v.to(device) for k, v in inputs.items()}
    raise TypeError(f"Unsupported inputs type: {type(inputs)}")


def _infer_device(model: nn.Module, explicit_device: Optional[torch.device]) -> torch.device:
    """Infer a device from the first model parameter (fallback to CPU)."""
    if explicit_device is not None:
        return explicit_device
    try:
        param = next(model.parameters())
        return param.device
    except StopIteration:
        return torch.device("cpu")


def _call_model_forward(model: nn.Module, inputs: TensorLikeInputs) -> Any:
    """Call model forward for supported input containers."""
    if isinstance(inputs, (list, tuple)):
        return model(*inputs)
    if isinstance(inputs, Mapping):
        return model(**inputs)
    return model(inputs)


def measure_inference_time(
    model: nn.Module,
    inputs: TensorLikeInputs,
    num_runs: int = 50,
    warmup_runs: int = 10,
    percentile: int = 95,
    device: Optional[Union[str, torch.device]] = None,
    use_amp_fp16: bool = False,
) -> Dict[str, float]:
    """Measure forward-pass inference latency (ms).

    Includes warmup runs and CUDA synchronization (when applicable).

    Parameters
    ----------
    model:
        A PyTorch module.
    inputs:
        One of:
        - a single tensor
        - a list/tuple of tensors
        - a Mapping[str, Tensor] (e.g., dict-like tokenizer outputs)
    num_runs:
        Number of timed runs.
    warmup_runs:
        Number of warmup runs (not timed).
    percentile:
        Percentile to report, e.g., 95 -> p95_ms.
    device:
        "cpu", "cuda", torch.device, or None to infer.
    use_amp_fp16:
        Use autocast fp16 on CUDA (does not change model weights).

    Returns
    -------
    Dict[str, float]
        avg_ms, pXX_ms, min_ms, max_ms, runs
    """
    model.eval()

    dev = torch.device(device) if device is not None else _infer_device(model, None)

    model = model.to(dev)
    inputs = _move_inputs_to_device(inputs, dev)

    with torch.no_grad():
        for _ in range(max(0, warmup_runs)):
            if use_amp_fp16 and dev.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = _call_model_forward(model, inputs)
            else:
                _ = _call_model_forward(model, inputs)

            if dev.type == "cuda":
                torch.cuda.synchronize()

    latencies_ms: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()

            if use_amp_fp16 and dev.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = _call_model_forward(model, inputs)
            else:
                _ = _call_model_forward(model, inputs)

            if dev.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    lat_np = np.array(latencies_ms, dtype=np.float64)
    p_val = float(np.percentile(lat_np, percentile))

    return {
        "avg_ms": float(lat_np.mean()),
        f"p{percentile}_ms": p_val,
        "min_ms": float(lat_np.min()),
        "max_ms": float(lat_np.max()),
        "runs": float(num_runs),
    }


def measure_generation_time(
    model: nn.Module,
    inputs: TensorLikeInputs,
    *,
    max_new_tokens: int = 50,
    num_runs: int = 20,
    warmup_runs: int = 5,
    percentile: int = 95,
    device: Optional[Union[str, torch.device]] = None,
    use_amp_fp16: bool = False,
) -> Dict[str, float]:
    """Measure generation latency (ms) via `model.generate()` (if available).

    Parameters
    ----------
    model:
        A PyTorch module with `.generate()` (common in Transformers models).
    inputs:
        Same as `measure_inference_time`. For LLMs, a Mapping with `input_ids`
        (and optionally `attention_mask`) is recommended.
    max_new_tokens:
        Number of tokens to generate.
    num_runs:
        Number of timed runs.
    warmup_runs:
        Number of warmup runs (not timed).
    percentile:
        Percentile to report, e.g., 95 -> p95_ms.
    device:
        "cpu", "cuda", torch.device, or None to infer.
    use_amp_fp16:
        Use autocast fp16 on CUDA (does not change model weights).

    Returns
    -------
    Dict[str, float]
        avg_ms, pXX_ms, min_ms, max_ms, runs, max_new_tokens
    """
    if not hasattr(model, "generate"):
        raise AttributeError("This model does not implement `.generate()`.")

    model.eval()
    dev = torch.device(device) if device is not None else _infer_device(model, None)

    model = model.to(dev)
    inputs = _move_inputs_to_device(inputs, dev)

    def _call_generate() -> Any:
        if isinstance(inputs, (list, tuple)):
            return model.generate(*inputs, max_new_tokens=max_new_tokens)
        if isinstance(inputs, Mapping):
            return model.generate(**inputs, max_new_tokens=max_new_tokens)
        return model.generate(inputs, max_new_tokens=max_new_tokens)

    with torch.no_grad():
        for _ in range(max(0, warmup_runs)):
            if use_amp_fp16 and dev.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = _call_generate()
            else:
                _ = _call_generate()

            if dev.type == "cuda":
                torch.cuda.synchronize()

    latencies_ms: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()

            if use_amp_fp16 and dev.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = _call_generate()
            else:
                _ = _call_generate()

            if dev.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    lat_np = np.array(latencies_ms, dtype=np.float64)
    p_val = float(np.percentile(lat_np, percentile))

    return {
        "avg_ms": float(lat_np.mean()),
        f"p{percentile}_ms": p_val,
        "min_ms": float(lat_np.min()),
        "max_ms": float(lat_np.max()),
        "runs": float(num_runs),
        "max_new_tokens": float(max_new_tokens),
    }


# ---------------------------------------------------------------------
# One-call full report
# ---------------------------------------------------------------------


def profile_model(
    model: nn.Module,
    sample_inputs: TensorLikeInputs,
    *,
    layer_types: Optional[Tuple[type, ...]] = None,
    device: Optional[Union[str, torch.device]] = None,
    num_runs: int = 50,
    warmup_runs: int = 10,
    percentile: int = 95,
    compute_sparsity: bool = True,
    near_zero_threshold: float = 1e-6,
    measure_generate: bool = False,
    max_new_tokens: int = 50,
) -> Dict[str, Any]:
    """Run a practical profiling suite in a single call.

    Returns a dictionary containing:
    - param_summary
    - model_size_mb (params+buffers)
    - state_dict_size_mb (serialized weights)
    - layer_stats
    - latency_ms
    - generation_latency_ms (optional, if enabled and supported)

    Parameters
    ----------
    model:
        A PyTorch module.
    sample_inputs:
        Inputs for the model forward/generation.
    layer_types:
        Which layer classes to include in per-layer stats (None -> defaults).
    device:
        Device for latency measurement. If None, inferred from model.
    num_runs, warmup_runs, percentile:
        Latency measurement configuration.
    compute_sparsity, near_zero_threshold:
        Per-layer sparsity metrics configuration.
    measure_generate:
        If True, measures `model.generate()` latency when available.
    max_new_tokens:
        Passed to `generate()`.

    Returns
    -------
    Dict[str, Any]
        Profiling report.
    """
    param_summary = count_parameters(model)
    model_size_mb = estimate_model_size_mb(model, include_buffers=True)
    state_dict_size_mb = estimate_state_dict_size_mb(model)

    layer_stats = collect_layer_stats(
        model,
        layer_types=layer_types,
        include_buffers=False,
        compute_sparsity=compute_sparsity,
        near_zero_threshold=near_zero_threshold,
        store_param_arrays=False,
    )

    latency_ms = measure_inference_time(
        model,
        inputs=sample_inputs,
        device=device,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
        percentile=percentile,
        use_amp_fp16=False,
    )

    out: Dict[str, Any] = {
        "param_summary": param_summary,
        "model_size_mb": model_size_mb,
        "state_dict_size_mb": state_dict_size_mb,
        "layer_stats": layer_stats,
        "latency_ms": latency_ms,
    }

    if measure_generate:
        try:
            out["generation_latency_ms"] = measure_generation_time(
                model,
                inputs=sample_inputs,
                device=device,
                num_runs=max(5, num_runs // 2),
                warmup_runs=max(2, warmup_runs // 2),
                percentile=percentile,
                max_new_tokens=max_new_tokens,
                use_amp_fp16=False,
            )
        except Exception as exc:
            out["generation_latency_ms"] = {"error": str(exc)}

    return out


# ---------------------------------------------------------------------
# LLM-friendly helpers (no Transformers dependency)
# ---------------------------------------------------------------------


def make_dummy_input_ids(
    vocab_size: int,
    batch_size: int = 1,
    seq_len: int = 32,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Create dummy `input_ids` for language models.

    Parameters
    ----------
    vocab_size:
        Vocabulary size.
    batch_size:
        Batch dimension.
    seq_len:
        Sequence length.
    device:
        "cpu", "cuda", torch.device, or None (defaults to CPU).
    dtype:
        Typically torch.long.

    Returns
    -------
    torch.Tensor
        Tensor of shape [batch_size, seq_len].
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    return torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=dev,
        dtype=dtype,
    )


def cuda_memory_snapshot() -> Dict[str, float]:
    """Return a snapshot of CUDA memory usage (MB).

    Returns zeros if CUDA is not available.

    Returns
    -------
    Dict[str, float]
        allocated_mb, reserved_mb, max_allocated_mb
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}

    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    max_alloc = torch.cuda.max_memory_allocated() / (1024**2)

    return {
        "allocated_mb": float(alloc),
        "reserved_mb": float(reserved),
        "max_allocated_mb": float(max_alloc),
    }


# ---------------------------------------------------------------------
# Pretty reporting (optional Rich)
# ---------------------------------------------------------------------


def _try_import_rich():
    """Try importing Rich components. Returns (Console, Table, Panel) or (None,...)."""
    try:
        from rich.console import Console  # type: ignore
        from rich.panel import Panel  # type: ignore
        from rich.table import Table  # type: ignore

        return Console, Table, Panel
    except Exception:
        return None, None, None


def format_profile_report(
    report: Dict[str, Any],
    top_k_layers: int = 15,
    decimals: int = 3,
) -> str:
    """Format a profile report into a readable plain-text string.

    This format is great for logs and CI environments.

    Parameters
    ----------
    report:
        Output from `profile_model`.
    top_k_layers:
        How many layers to display in the "Top layers" section.
    decimals:
        Decimal precision for MB and latency values.

    Returns
    -------
    str
        Human-readable report text.
    """
    ps = report.get("param_summary", {})
    latency = report.get("latency_ms", {})
    size_mb = report.get("model_size_mb", None)
    state_size_mb = report.get("state_dict_size_mb", None)

    lines: List[str] = []
    lines.append("==== PyTorch Model Profile ====")
    lines.append("")
    lines.append(">> Parameters")
    lines.append(
        f"  - Total:      {ps.get('total_params', 'N/A')} "
        f"({human_readable_number(int(ps.get('total_params', 0) or 0))})"
    )
    lines.append(
        f"  - Trainable:  {ps.get('trainable_params', 'N/A')} "
        f"({human_readable_number(int(ps.get('trainable_params', 0) or 0))})"
    )
    lines.append(
        f"  - Frozen:     {ps.get('frozen_params', 'N/A')} "
        f"({human_readable_number(int(ps.get('frozen_params', 0) or 0))})"
    )

    if size_mb is not None:
        lines.append("")
        lines.append(f">> Size (params+buffers): {float(size_mb):.{decimals}f} MB")

    if state_size_mb is not None:
        lines.append(f">> Size (state_dict):     {float(state_size_mb):.{decimals}f} MB")

    if latency:
        lines.append("")
        lines.append(">> Inference latency (forward)")
        p_keys = [k for k in latency.keys() if k.startswith("p") and k.endswith("_ms")]
        p_key = p_keys[0] if p_keys else None

        for k in ["avg_ms", p_key, "min_ms", "max_ms"]:
            if k and k in latency:
                lines.append(f"  - {k}: {float(latency[k]):.{decimals}f} ms")

    layer_stats = report.get("layer_stats", None)
    if layer_stats is not None:
        lines.append("")
        lines.append(f">> Top {top_k_layers} layers by parameter count")

        if pd is not None and hasattr(layer_stats, "to_dict"):
            rows = layer_stats.head(top_k_layers).to_dict(orient="records")
        else:
            rows = list(layer_stats)[:top_k_layers]

        header = (
            f"{'Layer':40} | {'Type':14} | {'Params':>12} | "
            f"{'MB':>8} | {'NearZero%':>9}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for r in rows:
            name = str(r.get("name", ""))[:40].ljust(40)
            typ = str(r.get("type", ""))[:14].ljust(14)
            params = int(r.get("parameters", 0))
            mb = float(r.get("size_mb", 0.0))
            nz = r.get("near_zero_pct", None)
            nz_s = f"{float(nz):.2f}" if nz is not None else "N/A"
            lines.append(f"{name} | {typ} | {params:12d} | {mb:8.{decimals}f} | {nz_s:>9}")

    return "\n".join(lines)


def print_profile_report(
    report: Dict[str, Any],
    top_k_layers: int = 15,
    decimals: int = 3,
) -> None:
    """Print a profile report using Rich (if available), otherwise plain text."""
    Console, Table, Panel = _try_import_rich()

    if Console is None:
        print(format_profile_report(report, top_k_layers=top_k_layers, decimals=decimals))
        return

    console = Console()
    ps = report.get("param_summary", {})
    latency = report.get("latency_ms", {})
    size_mb = float(report.get("model_size_mb", 0.0))
    state_size_mb = report.get("state_dict_size_mb", None)

    summary_lines = [
        (
            f"[bold]Total params:[/bold] {ps.get('total_params', 'N/A')}  "
            f"({human_readable_number(int(ps.get('total_params', 0) or 0))})"
        ),
        (
            f"[bold]Trainable:[/bold]   {ps.get('trainable_params', 'N/A')}  "
            f"({human_readable_number(int(ps.get('trainable_params', 0) or 0))})"
        ),
        (
            f"[bold]Frozen:[/bold]      {ps.get('frozen_params', 'N/A')}  "
            f"({human_readable_number(int(ps.get('frozen_params', 0) or 0))})"
        ),
        f"[bold]Model size:[/bold]    {size_mb:.{decimals}f} MB",
    ]

    if state_size_mb is not None:
        summary_lines.append(
            f"[bold]State_dict size:[/bold] {float(state_size_mb):.{decimals}f} MB"
        )

    console.print(Panel("\n".join(summary_lines), title="Model Summary", expand=False))

    if latency:
        lat_lines: List[str] = []
        p_keys = [k for k in latency.keys() if k.startswith("p") and k.endswith("_ms")]
        p_key = p_keys[0] if p_keys else None

        for k in ["avg_ms", p_key, "min_ms", "max_ms"]:
            if k and k in latency:
                lat_lines.append(f"[bold]{k}[/bold]: {float(latency[k]):.{decimals}f} ms")

        if lat_lines:
            console.print(Panel("\n".join(lat_lines), title="Inference Latency", expand=False))

    gen = report.get("generation_latency_ms")
    if isinstance(gen, dict) and "error" not in gen and gen:
        gen_lines: List[str] = []
        p_keys = [k for k in gen.keys() if k.startswith("p") and k.endswith("_ms")]
        p_key = p_keys[0] if p_keys else None

        for k in ["avg_ms", p_key, "min_ms", "max_ms"]:
            if k and k in gen:
                gen_lines.append(f"[bold]{k}[/bold]: {float(gen[k]):.{decimals}f} ms")

        if gen_lines:
            console.print(
                Panel("\n".join(gen_lines), title="Generation Latency", expand=False)
            )

    layer_stats = report.get("layer_stats", None)
    if layer_stats is None:
        return

    if pd is not None and hasattr(layer_stats, "to_dict"):
        rows = layer_stats.head(top_k_layers).to_dict(orient="records")
    else:
        rows = list(layer_stats)[:top_k_layers]

    table = Table(title=f"Top {top_k_layers} Layers (by params)")
    table.add_column("Layer", justify="left", overflow="fold")
    table.add_column("Type", justify="left")
    table.add_column("Params", justify="right")
    table.add_column("MB", justify="right")
    table.add_column("NearZero%", justify="right")

    for r in rows:
        name = str(r.get("name", ""))
        typ = str(r.get("type", ""))
        params = str(int(r.get("parameters", 0)))
        mb = f"{float(r.get('size_mb', 0.0)):.{decimals}f}"
        nz = r.get("near_zero_pct", None)
        nz_s = f"{float(nz):.2f}" if nz is not None else "N/A"
        table.add_row(name, typ, params, mb, nz_s)

    console.print(table)


if __name__ == "__main__":
    # Minimal smoke test (does not require Transformers).
    m = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    x = torch.randn(1, 128)

    rep = profile_model(m, sample_inputs=x, device="cpu", num_runs=20, warmup_runs=5)
    print_profile_report(rep)
