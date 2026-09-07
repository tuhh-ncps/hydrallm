# util/eval_ppl_and_loss.py
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# --- make imports work even when running from util/ ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_implementations import load_model  # noqa: E402
from common_helpers import parse_dtype, pick_default_device  # noqa: E402


def read_text(text: str | None, text_file: str | None) -> str:
    if text is not None:
        return text
    if text_file is not None:
        return Path(text_file).read_text(encoding="utf-8")
    return (
        "The capital of France is Paris.\n"
        "This is a short test passage for perplexity evaluation.\n"
    )


@torch.no_grad()
def eval_loss_and_ppl(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int,
    stride: int,
    forward_kwargs: dict,
) -> tuple[float, float, int]:
    enc = tokenizer(text, return_tensors="pt")
    input_ids_all = enc["input_ids"][0]
    if input_ids_all.numel() < 2:
        raise ValueError("Text too short after tokenization (need >= 2 tokens).")

    total_nll = 0.0
    total_tokens = 0

    seq_len = int(input_ids_all.numel())
    for start in range(0, seq_len - 1, stride):
        end = min(start + max_length, seq_len)
        input_ids = input_ids_all[start:end].unsqueeze(0).to(device)

        # Predict next-token over this window; mask out first token so it has a target
        labels = input_ids.clone()
        labels[:, 0] = -100

        attention_mask = torch.ones_like(input_ids, device=device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **forward_kwargs,
        )

        n_tokens = int((labels != -100).sum().item())
        total_nll += float(out.loss) * n_tokens
        total_tokens += n_tokens

        if end == seq_len:
            break

    mean_loss = total_nll / max(1, total_tokens)
    ppl = math.exp(mean_loss)
    return mean_loss, ppl, total_tokens


def main():
    p = argparse.ArgumentParser("Evaluate mean loss and perplexity on a test text.")
    p.add_argument("--model", type=str, required=True, help="Path or HF id.")
    p.add_argument(
        "--implementation",
        type=str,
        default="auto_model",
        help="e.g. auto_model | hydra_gemma_from_flex | matformer_gemma | flex_gemma | ...",
    )
    p.add_argument("--attn-implementation", type=str, default="eager")
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="auto|float32|float16|bfloat16 (default: auto)",
    )
    p.add_argument(
        "--device", type=str, default=None, help="cpu|cuda|cuda:0 (default: auto)"
    )

    # text
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--text-file", type=str, default=None)

    # eval params
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--stride", type=int, default=None, help="default: max-length")

    # Hydra heads
    p.add_argument("--current-num-heads", type=int, default=None)

    # MatFormer FFN width
    p.add_argument("--granularity", type=int, default=None)
    p.add_argument("--mlp-ratio", type=float, default=None)

    args = p.parse_args()

    device_str = args.device or pick_default_device()
    dtype = parse_dtype(args.dtype, device_str)

    if args.stride is None:
        args.stride = args.max_length

    # Build dynamic forward kwargs (supported by your wrappers)
    forward_kwargs: dict = {}
    if args.current_num_heads is not None:
        forward_kwargs["current_num_heads"] = args.current_num_heads
    if args.granularity is not None:
        forward_kwargs["granularity"] = args.granularity
    if args.mlp_ratio is not None:
        forward_kwargs["mlp_ratio"] = args.mlp_ratio

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Model
    model = load_model(
        args.model,
        implementation=args.implementation,
        dtype=dtype,
        device=device_str,  # keep it simple: put the whole model on one device
        attn_implementation=args.attn_implementation,
    ).eval()

    # Determine where to place inputs (robust even if model wrappers differ)
    try:
        input_device = next(model.parameters()).device
    except StopIteration:
        input_device = torch.device(device_str)

    text = read_text(args.text, args.text_file)

    mean_loss, ppl, n_tokens = eval_loss_and_ppl(
        model=model,
        tokenizer=tok,
        text=text,
        device=input_device,
        max_length=args.max_length,
        stride=args.stride,
        forward_kwargs=forward_kwargs,
    )

    print(f"model={args.model}")
    print(f"implementation={args.implementation}")
    if forward_kwargs:
        print(f"dynamic_forward_kwargs={forward_kwargs}")
    print(f"evaluated_tokens={n_tokens}")
    print(f"mean_loss={mean_loss:.6f}")
    print(f"perplexity={ppl:.6f}")


if __name__ == "__main__":
    main()
