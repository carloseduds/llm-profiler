"""
Compare forward-pass latency vs generation latency.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_profiler import profile_model, print_profile_report


MODEL_ID = "facebook/opt-125m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()

    enc = tokenizer("Forward vs generate latency", return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in enc.items()}

    report = profile_model(
        model,
        sample_inputs=inputs,
        device=DEVICE,
        num_runs=20,
        warmup_runs=10,
        measure_generate=True,
        max_new_tokens=64,
    )

    print_profile_report(report)


if __name__ == "__main__":
    main()
