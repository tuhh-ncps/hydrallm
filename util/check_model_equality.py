import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import torch
from transformers import AutoTokenizer
from model_implementations import load_model
from common_helpers import parse_dtype, pick_default_device


def main():
    p = argparse.ArgumentParser(
        description="Compare logits between two causal LM checkpoints."
    )
    p.add_argument(
        "--model1", type=str, required=True, help="Path or HF id for model1."
    )
    p.add_argument(
        "--model2", type=str, required=True, help="Path or HF id for model2."
    )

    p.add_argument(
        "--implementation1",
        type=str,
        default="auto_model",
        help="Implementation name for model1 (default: auto_model).",
    )
    p.add_argument(
        "--implementation2",
        type=str,
        default="auto_model",
        help="Implementation name for model2 (default: auto_model).",
    )

    p.add_argument(
        "--attn-implementation1",
        type=str,
        default="eager",
        help="Attention implementation for model1 (default: eager).",
    )
    p.add_argument(
        "--attn-implementation2",
        type=str,
        default="eager",
        help="Attention implementation for model2 (default: eager).",
    )

    p.add_argument(
        "--dtype1",
        type=str,
        default="auto",
        help="dtype for model1: auto|float32|float16|bfloat16 (default: auto).",
    )
    p.add_argument(
        "--dtype2",
        type=str,
        default="auto",
        help="dtype for model2: auto|float32|float16|bfloat16 (default: auto).",
    )

    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device, e.g. cpu, cuda, cuda:0 (default: auto-detect).",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Prompt to run (default provided).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Max |Δ logit| threshold for success (default: 1e-3).",
    )

    args = p.parse_args()

    device = args.device or pick_default_device()
    dtype1 = parse_dtype(args.dtype1, device)
    dtype2 = parse_dtype(args.dtype2, device)

    tok1 = AutoTokenizer.from_pretrained(args.model1)
    if tok1.pad_token is None:
        tok1.pad_token = tok1.eos_token

    tok2 = AutoTokenizer.from_pretrained(args.model2)
    if tok2.pad_token is None:
        tok2.pad_token = tok2.eos_token

    inputs1 = tok1(args.prompt, return_tensors="pt").to(device)
    inputs2 = tok2(args.prompt, return_tensors="pt").to(device)

    print(f"Loading model1: {args.model1}")
    model1 = load_model(
        args.model1,
        implementation=args.implementation1,
        dtype=dtype1,
        device=device,
        attn_implementation=args.attn_implementation1,
    ).eval()

    print(f"Loading model2: {args.model2}")
    model2 = load_model(
        args.model2,
        implementation=args.implementation2,
        dtype=dtype2,
        device=device,
        attn_implementation=args.attn_implementation2,
    ).eval()

    with torch.no_grad():
        logits1 = model1(**inputs1).logits
        logits2 = model2(**inputs2).logits

    diff = (logits1 - logits2).abs().max().item()
    print(f"Sanity Check Max |Δ logit|: {diff:.6f}")

    if diff > args.threshold:
        print("CRITICAL: Models are very different.")
    else:
        print("SUCCESS: Models are virtually the same.")


if __name__ == "__main__":
    main()
