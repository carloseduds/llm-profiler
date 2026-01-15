"""
Compare FP32 vs INT8 dynamic quantization on CPU.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_profiler import profile_model, print_profile_report


MODEL_ID = "facebook/opt-125m"
DEVICE = "cpu"  # dynamic quantization is CPU-only


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    model_fp32 = AutoModelForCausalLM.from_pretrained(MODEL_ID).eval()

    enc = tokenizer("Profiling INT8 quantization", return_tensors="pt")
    inputs = {k: v for k, v in enc.items()}

    print("\n===== FP32 =====\n")
    rep_fp32 = profile_model(
        model_fp32,
        sample_inputs=inputs,
        device=DEVICE,
        num_runs=30,
        warmup_runs=10,
    )
    print_profile_report(rep_fp32)

    print("\n===== INT8 (dynamic quantization) =====\n")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {nn.Linear},
        dtype=torch.qint8,
    )

    rep_int8 = profile_model(
        model_int8,
        sample_inputs=inputs,
        device=DEVICE,
        num_runs=30,
        warmup_runs=10,
    )
    print_profile_report(rep_int8)


if __name__ == "__main__":
    main()
