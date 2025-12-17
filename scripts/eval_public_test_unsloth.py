#!/usr/bin/env python3
"""
Đánh giá model Qwen2.5-7B-Medical bằng Unsloth 4-bit trên bộ public_test.json
(cả hai chiều en-vi và vi-en), KHÔNG dùng vLLM.

Nguồn:
    data/public_test.json  (đã chứa cả en-vi & vi-en)

Đích:
    data/public_test.pred.jsonl

Mỗi dòng output:
    {
      "id": <int>,
      "direction": "en-vi" | "vi-en",
      "source": "...",
      "target": "...",        # tham chiếu
      "prediction": "..."     # model dịch
    }

Chạy:
    cd /home/alida/Documents/Cursor/NLP_fine_tun
    python scripts/eval_public_test_unsloth.py --max-samples 50
"""

import argparse
import json
from pathlib import Path

import torch
from unsloth import FastLanguageModel


def build_prompt(direction: str, source: str) -> str:
    """Tạo prompt giống format training (instruction + Input/Output)."""
    if direction == "en-vi":
        instruction = "Dịch đoạn văn sau từ tiếng Anh sang tiếng Việt trong lĩnh vực y tế."
    else:
        instruction = "Dịch đoạn văn sau từ tiếng Việt sang tiếng Anh trong lĩnh vực y tế."

    return f"{instruction}\nInput: {source}\nOutput:"


def main():
    parser = argparse.ArgumentParser(description="Eval public_test.json bằng Unsloth 4-bit (không dùng vLLM).")
    parser.add_argument(
        "--input",
        type=str,
        default="data/public_test.json",
        help="Đường dẫn file JSON test (mặc định: data/public_test.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/public_test.pred.jsonl",
        help="Đường dẫn file JSONL output (mặc định: data/public_test.pred.jsonl).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Giới hạn số mẫu để test nhanh (mặc định: dùng toàn bộ).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size cho inference (mặc định: 8).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Số token tối đa model sinh ra.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / "final_models/Qwen2.5-7B-Medical-Full-Bin"

    input_path = base_dir / args.input
    output_path = base_dir / args.output

    print(f"📂 Model dir : {model_dir}")
    print(f"📂 Input     : {input_path}")
    print(f"💾 Output    : {output_path}")

    # Load dữ liệu test
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples is not None:
        data = data[: args.max_samples]

    print(f"🔢 Số mẫu sẽ test: {len(data)}")

    # Load model 4-bit với Unsloth
    print("🧠 Đang load model merged với Unsloth (4-bit, không dùng vLLM)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)
    device = model.device

    # Batching
    tokenizer.padding_side = "left"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        batch_size = args.batch_size
        for i in range(0, len(data), batch_size):
            batch_samples = data[i : i + batch_size]

            prompts = [build_prompt(s["direction"], s["source"]) for s in batch_samples]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Cắt phần mới sinh
            gen_seqs = outputs[:, inputs["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(gen_seqs, skip_special_tokens=True)

            for sample, pred in zip(batch_samples, decoded):
                record = {
                    "id": sample.get("id"),
                    "direction": sample.get("direction"),
                    "source": sample.get("source"),
                    "target": sample.get("target"),
                    "prediction": pred.strip(),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"✅ Đã xử lý {min(i + batch_size, len(data))}/{len(data)} mẫu", flush=True)

    print(f"🎉 Hoàn tất! Kết quả được lưu tại: {output_path}")


if __name__ == "__main__":
    main()




