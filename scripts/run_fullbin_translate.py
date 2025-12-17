#!/usr/bin/env python3
"""
Chạy thử model đã merge 16-bit:
    final_models/Qwen2.5-7B-Medical-Full-Bin

Dùng trực tiếp Transformers (không qua Unsloth), với format prompt:

    Translate the following English text to Vietnamese:
    Input: ...
    Output:

hoặc

    Translate the following Vietnamese text to English:
    Input: ...
    Output:
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(direction: str, text: str) -> str:
    if direction == "en-vi":
        instr = "Translate the following English text to Vietnamese:"
    else:
        instr = "Translate the following Vietnamese text to English:"
    return f"{instr}\nInput: {text}\nOutput:"


def main():
    parser = argparse.ArgumentParser(
        description="Chạy thử Qwen2.5-7B-Medical-Full-Bin để dịch en-vi / vi-en."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="final_models/Qwen2.5-7B-Medical-Full-Bin",
        help="Đường dẫn tới model đã merge 16-bit.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="en-vi",
        choices=["en-vi", "vi-en"],
        help="Chiều dịch.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Câu/đoạn văn nguồn cần dịch.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Số token tối đa model sinh ra.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📥 Loading full model from: {args.model_dir} (device={device})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    prompt = build_prompt(args.direction, args.text)
    print("\n====== PROMPT GỬI VÀO MODEL ======")
    print(prompt)
    print("==================================\n")

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Cắt phần sau "Output:"
    marker = "Output:"
    idx = decoded.rfind(marker)
    if idx != -1:
        translation = decoded[idx + len(marker) :].strip()
    else:
        translation = decoded.strip()

    print("====== BẢN DỊCH ======")
    print(translation)
    print("======================")


if __name__ == "__main__":
    main()


