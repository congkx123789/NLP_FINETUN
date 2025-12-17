#!/usr/bin/env python3
"""
Test Qwen2.5-7B-Medical (đã merge) với Unsloth trên bộ public_test.*
và lưu câu trả lời của model ra file JSON.

Ví dụ chạy:
    # Dịch Anh -> Việt
    python scripts/test_unsloth.py --direction en-vi --max-samples 50

    # Dịch Việt -> Anh
    python scripts/test_unsloth.py --direction vi-en --max-samples 50
"""

import argparse
import json
from pathlib import Path

import torch
from unsloth import FastLanguageModel


def build_prompt(direction: str, src_sentence: str) -> str:
    """Tạo prompt giống style training (instruction + Input + Output)."""
    if direction == "en-vi":
        instruction = "Dịch đoạn văn sau từ tiếng Anh sang tiếng Việt trong lĩnh vực y tế."
    else:
        instruction = "Dịch đoạn văn sau từ tiếng Việt sang tiếng Anh trong lĩnh vực y tế."

    return f"{instruction}\nInput: {src_sentence}\nOutput:"


def main():
    parser = argparse.ArgumentParser(description="Test Qwen2.5-7B-Medical với Unsloth và lưu kết quả ra JSON.")
    parser.add_argument(
        "--direction",
        type=str,
        default="en-vi",
        choices=["en-vi", "vi-en"],
        help="Hướng dịch: en-vi hoặc vi-en.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Giới hạn số câu test (mặc định: dùng toàn bộ file).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Tên file JSON output (mặc định: tự sinh theo direction).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Số token tối đa model sinh ra cho mỗi câu.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Chỉ đường đến model đã merge
    model_dir = base_dir / "final_models/Qwen2.5-7B-Medical-Full-Bin"

    # File test nguồn/đích
    test_dir = base_dir / "Test"
    if args.direction == "en-vi":
        src_path = test_dir / "public_test.en.txt"
        tgt_path = test_dir / "public_test.vi.txt"
        default_output = test_dir / "public_test.en-vi.pred.jsonl"
    else:
        src_path = test_dir / "public_test.vi.txt"
        tgt_path = test_dir / "public_test.en.txt"
        default_output = test_dir / "public_test.vi-en.pred.jsonl"

    output_path = Path(args.output) if args.output else default_output

    print("📂 Model dir :", model_dir)
    print("📂 Source    :", src_path)
    print("📂 Target    :", tgt_path)
    print("📂 Output    :", output_path)

    # Đọc dữ liệu test
    with open(src_path, "r", encoding="utf-8") as f_src:
        src_lines = [line.strip() for line in f_src.readlines()]
    with open(tgt_path, "r", encoding="utf-8") as f_tgt:
        tgt_lines = [line.strip() for line in f_tgt.readlines()]

    assert len(src_lines) == len(tgt_lines), "Số dòng source và target không khớp!"

    if args.max_samples is not None:
        src_lines = src_lines[: args.max_samples]
        tgt_lines = tgt_lines[: args.max_samples]

    print(f"🔢 Số câu sẽ test: {len(src_lines)}")

    # Load model với Unsloth (4bit để tiết kiệm VRAM khi test)
    print("🧠 Đang load model merged với Unsloth (4bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)  # Bật chế độ inference tối ưu

    device = model.device

    # Chạy dịch từng câu và lưu JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        for idx, (src, tgt) in enumerate(zip(src_lines, tgt_lines), start=1):
            prompt = build_prompt(args.direction, src)

            inputs = tokenizer(
                [prompt],
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = outputs[0][inputs["input_ids"].shape[1] :]
            pred = tokenizer.decode(generated, skip_special_tokens=True).strip()

            record = {
                "id": idx,
                "direction": args.direction,
                "source": src,
                "target": tgt,
                "prediction": pred,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if idx % 50 == 0 or idx == 1 or idx == len(src_lines):
                print(f"✅ Đã xử lý {idx}/{len(src_lines)} câu")

    print(f"🎉 Hoàn tất! Kết quả được lưu tại: {output_path}")


if __name__ == "__main__":
    main()


