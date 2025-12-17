#!/usr/bin/env python3
"""
Merge LoRA adapter đã train vào base model Qwen2.5-7B để tạo:
1) Bản 16-bit merged (cho Python/vLLM/HuggingFace)
2) Bản GGUF (cho Ollama/LM Studio)
"""

from unsloth import FastLanguageModel
import torch  # noqa: F401 - giữ lại nếu sau này cần
import os

# --- CẤU HÌNH ---

# Đường dẫn đến bản vá bạn vừa train (Folder chứa adapter_model.safetensors)
adapter_path = "saves/qwen2_5-7b/unsloth/mixed_maxsteps10000_fa2"

# Tên folder đầu ra
output_16bit = "final_models/Qwen2.5-7B-Medical-16bit"  # Folder chứa model 16bit
output_gguf = "final_models/Qwen2.5-7B-Medical-GGUF"    # Folder chứa file GGUF


def main():
    print(f"⏳ Đang load model từ: {adapter_path}...")

    # 1. Load Model & Tokenizer (4-bit để tiết kiệm VRAM khi xử lý)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Tạo folder cha nếu chưa có
    os.makedirs("final_models", exist_ok=True)

    # ---------------------------------------------------------
    # BƯỚC 1: XUẤT BẢN 16-BIT (Merged Float16)
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("💾 BƯỚC 1: Đang hàn bản vá ra định dạng 16-bit (safetensors)...")
    print(f"   Lưu tại: {output_16bit}")
    print("=" * 50)

    model.save_pretrained_merged(
        output_16bit,
        tokenizer,
        save_method="merged_16bit",  # Hàn chết vào model gốc
    )

    print("✅ Đã xong bản 16-bit!")

    # ---------------------------------------------------------
    # BƯỚC 2: XUẤT BẢN GGUF (Cho Ollama/LM Studio)
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("💾 BƯỚC 2: Đang chuyển đổi sang GGUF (Quantization q4_k_m)...")
    print("   Quá trình này sẽ tốn nhiều RAM và CPU, hãy kiên nhẫn...")
    print(f"   Lưu tại: {output_gguf}")
    print("=" * 50)

    model.save_pretrained_gguf(
        output_gguf,
        tokenizer,
        quantization_method="q4_k_m",  # Chuẩn cân bằng nhất hiện nay
    )

    print("\n🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH!")
    print(f"1. Bản 16-bit: nằm trong '{output_16bit}'")
    print(f"2. Bản GGUF : nằm trong '{output_gguf}'")


if __name__ == "__main__":
    main()


