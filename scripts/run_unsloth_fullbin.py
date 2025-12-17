#!/usr/bin/env python3
"""
Chạy Qwen2.5-7B-Medical-Full-Bin với Unsloth + FlashAttention 2 (FA2)
====================================================================

- Model gốc (đã merge 16-bit) nằm tại:
    final_models/Qwen2.5-7B-Medical-Full-Bin

- Script này:
    * Dùng Unsloth `FastLanguageModel.from_pretrained` để bật FA2 tự động.
    * Tùy chọn `load_in_4bit=True` để giảm VRAM và tăng tốc inference.
    * Dùng đúng format prompt như khi train:
          {instruction}
          Input: {text}
          Output:
"""

import argparse

import torch
from unsloth import FastLanguageModel


def build_prompt(direction: str, text: str) -> str:
    """Tạo prompt đúng format đã train."""
    if direction == "en-vi":
        instr = "Translate the following English text to Vietnamese:"
    else:
        instr = "Translate the following Vietnamese text to English:"
    return f"{instr}\nInput: {text}\nOutput:"


def main():
    parser = argparse.ArgumentParser(
        description="Chạy Qwen2.5-7B-Medical-Full-Bin với Unsloth + FA2 để dịch en-vi / vi-en."
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
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Nếu set, không dùng 4-bit (load full 16-bit, tốn VRAM hơn, chậm hơn).",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📥 Loading full model (via Unsloth) from: {args.model_dir} (device={device})")

    load_in_4bit = not args.no_4bit

    # Unsloth sẽ tự bật FlashAttention 2 nếu GPU hỗ trợ (RTX 5060 Ti có hỗ trợ)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=2048,
        dtype=None,          # để Unsloth tự chọn bf16/fp16 phù hợp
        load_in_4bit=load_in_4bit,
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
            # Tham số chống lặp / thoái hóa
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


