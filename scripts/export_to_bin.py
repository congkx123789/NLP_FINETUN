#!/usr/bin/env python3
"""
Export Qwen2.5-7B + LoRA (checkpoint-10000) thành 1 model đã hàn chết bản vá,
lưu ra đĩa dạng .safetensors (hoặc .bin nếu cần).

Chạy:
    cd /home/alida/Documents/Cursor/NLP_fine_tun
    python scripts/export_to_bin.py
"""

from unsloth import FastLanguageModel

import torch  # noqa: F401 - giữ lại nếu sau này cần
import os
from pathlib import Path


def main():
    # --- CẤU HÌNH ---
    base_dir = Path(__file__).parent.parent

    # 1. Đường dẫn đến thư mục chứa bản vá (nơi có adapter_model.safetensors của checkpoint-10000)
    #    Ví dụ: saves/qwen2_5-7b/unsloth/mixed_maxsteps10000_fa2/checkpoint-10000
    adapter_path = base_dir / "saves/qwen2_5-7b/unsloth/mixed_maxsteps10000_fa2/checkpoint-10000"

    # 2. Thư mục đầu ra (Nơi sẽ chứa model.safetensors hoặc pytorch_model.bin và vocab/tokenizer)
    output_dir = base_dir / "final_models/Qwen2.5-7B-Medical-Full-Bin"

    print(f"⏳ Đang load Adapter từ: {adapter_path}...")

    # Load model gốc + adapter
    # Dùng 4-bit để tiết kiệm VRAM lúc merge
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    print("\n" + "=" * 50)
    print("💾 ĐANG HỢP NHẤT (MERGE) VÀ LƯU FILE...")
    print("   Quá trình này sẽ tạo ra file nặng khoảng 15GB.")
    print("=" * 50)

    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)

    # --- QUAN TRỌNG: LỆNH LƯU RA DẠNG BIN/SAFETENSORS ---
    # save_method="merged_16bit": Hợp nhất thành 1 khối float16
    model.save_pretrained_merged(
        str(output_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    # Lưu ý:
    # - Mặc định Unsloth sẽ lưu thành file đuôi .safetensors (hiện đại hơn .bin)
    # - Đa số các tool (vLLM, HuggingFace, Python) đều đọc được .safetensors y hệt .bin
    #
    # Nếu BẮT BUỘC phải cần file tên là pytorch_model.bin, bỏ comment block dưới:
    #
    # print("\n🔁 Đang xuất thêm bản .bin (safe_serialization=False)...")
    # model.merge_and_unload()
    # model.save_pretrained(str(output_dir), safe_serialization=False)
    # tokenizer.save_pretrained(str(output_dir))

    print(f"\n✅ XONG! Hãy kiểm tra thư mục: {output_dir}")
    print("Bạn sẽ thấy các file: config.json, model.safetensors (hoặc bin), vocab.json, tokenizer.json...")


if __name__ == "__main__":
    main()


