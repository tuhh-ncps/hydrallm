from typing import Dict, Optional, List
import torch
from transformers import AutoProcessor
from PIL import Image
from transformers import pipeline
from random import randint
import re

from common_helpers import normalize_mlp_ratio, parse_dtype
from model_implementations import (
    load_tokenizer_with_fallback,
    load_model,
)


# def create_conversation(input):
#     return {
#         "messages": [
#             {"role": "user", "content": input},
#         ]
#     }


@torch.inference_mode()
def run_inference_from_cfg(cfg: Dict):
    inf = cfg["inference"]
    model_path = inf["model_path"]
    tokenizer_path = inf.get("tokenizer_path")
    device = inf["device"]

    dtype = parse_dtype(inf.get("dtype", "auto"), device)
    implementation = inf.get("implementation", "hydra_gemma_from_flex")
    attn_implementation = inf.get("attn_implementation", "auto")

    # --- k_heads controls -----------------------------------------------------
    use_k_heads = bool(inf.get("use_k_heads", False))
    k_heads: Optional[int] = inf.get("k_heads", None)
    # --------------------------------------------------------------------------

    prompt = inf.get("prompt", "Hello")
    image_paths: Optional[List[str]] = inf.get("image_paths", None)

    tok = load_tokenizer_with_fallback(model_path, tokenizer_path, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model = load_model(
        model_path,
        implementation=implementation,
        dtype=dtype,
        attn_implementation=attn_implementation,
        device=device,
    )
    model.eval()

    model_config = getattr(model.config, "text_config", model.config)
    mlp_ratio = normalize_mlp_ratio(
        inf.get("ffn_granularity_ratio"),
        getattr(model_config, "num_hidden_layers", None),
    )

    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    # optionally also:
    model.generation_config.bos_token_id = tok.bos_token_id

    # pipe = pipeline("text-generation", model=model, tokenizer=tok)
    # chat_prompt = ""

    # Prepare multimodal or text‑only inputs
    if image_paths:
        required = len(image_paths)
        actual = prompt.count("<start_of_image>")
        if actual != required:
            raise ValueError(
                f"Prompt must contain {required} '<start_of_image>' tokens, but found {actual}."
            )

        processor = AutoProcessor.from_pretrained(model_path)
        images = [Image.open(p).convert("RGB") for p in image_paths]
        processed = processor(images=images, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in processed.items() if hasattr(v, "to")}
    else:
        # pipe = pipeline("text-generation", model=model, tokenizer=tok)
        # chat_prompt = pipe.tokenizer.apply_chat_template(
        #     create_conversation(prompt), tokenize=False, add_generation_prompt=True
        # )
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generation parameters
    gen_kwargs = dict(
        max_new_tokens=int(inf.get("max_new_tokens", 64)),
        do_sample=bool(inf.get("do_sample", False)),
        no_repeat_ngram_size=4,
        repetition_penalty=1.5,
        temperature=float(inf.get("temperature", 0.7)),
        top_p=float(inf.get("top_p", 0.9)),
        # pad_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )
    top_k = inf.get("top_k")
    if top_k is not None:
        gen_kwargs["top_k"] = int(top_k)

    # --- Apply dynamic head scaling ------------------------------------------
    # If user requested use_k_heads and a value is provided,
    # propagate it into model.generate() as current_num_heads.
    print("eos_token:", tok.eos_token, tok.eos_token_id)
    print("pad_token:", tok.pad_token, tok.pad_token_id)

    print("model.config.eos_token_id:", getattr(model.config, "eos_token_id", None))
    print(
        "model.generation_config.eos_token_id:",
        getattr(model.generation_config, "eos_token_id", None),
    )
    if use_k_heads and k_heads is not None:
        gen_kwargs["current_num_heads"] = int(k_heads)
        print(f"[Info] Using dynamic attention heads: current_num_heads={k_heads}")
    if mlp_ratio is not None:
        gen_kwargs["mlp_ratio"] = mlp_ratio
        print(f"[Info] Using dynamic FFN ratio: mlp_ratio={mlp_ratio}")
    # --------------------------------------------------------------------------

    # Perform generation (ForCausalLM / ForConditionalGeneration both accept it)
    outputs = model.generate(**inputs, **gen_kwargs)
    # outputs2 = pipe(chat_prompt, max_new_tokens=128, disable_compile=True)
    seq = outputs[0].tolist()
    print("EOS present?", tok.eos_token_id in seq)
    text = tok.decode(outputs[0], skip_special_tokens=True)
    print(text)
    # print(f"{outputs2[0]['generated_text'][len(chat_prompt):].strip()}")
    return text
