#!/usr/bin/env python3
"""
Test nhanh model đã merge bằng vLLM (khuyên dùng để đạt tốc độ cao).

Hướng dẫn:
    pip install vllm
    python scripts/test_fast_vllm.py
"""

import time
from vllm import LLM, SamplingParams

# Đường dẫn tuyệt đối đến model đã merge
MODEL_PATH = "/home/alida/Documents/Cursor/NLP_fine_tun/final_models/Qwen2.5-7B-Medical-Full-Bin"


def main():
    print(f"⏳ Đang load model vào vLLM từ: {MODEL_PATH}...")

    # Load model (vLLM tự quản lý VRAM rất hiệu quả)
    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",            # Chạy 16-bit cho nhẹ
        gpu_memory_utilization=0.85,  # Dùng 85% VRAM, chừa ít cho màn hình
        trust_remote_code=True,
    )

    # Cấu hình sinh chữ
    sampling_params = SamplingParams(temperature=0.3, max_tokens=200)

    # Danh sách câu test (có thể nhét nhiều câu tùy ý)
    prompts = [
        "Translate to Vietnamese: The patient presents with severe abdominal pain.",
        "Translate to Vietnamese: Dosage: Take 500mg twice daily after meals.",
        "Translate to Vietnamese: MRI scan reveals a mass in the left lung.",
        "Translate to English: Bệnh nhân có tiền sử dị ứng với các loại hải sản.",
        "Translate to English: Chỉ định: Phẫu thuật nội soi cắt ruột thừa.",
    ]

    print("\n" + "=" * 50)
    print("🚀 BẮT ĐẦU CHẠY...")
    start_time = time.time()

    # Chạy inference (batch)
    outputs = llm.generate(prompts, sampling_params)

    end_time = time.time()
    print(f"✅ Đã xong! Tổng thời gian: {end_time - start_time:.2f} giây")
    print("=" * 50 + "\n")

    # In kết quả
    for output in outputs:
        print(f"🔹 Input: {output.prompt}")
        print(f"🔸 Output: {output.outputs[0].text.strip()}")
        print("-" * 30)


if __name__ == "__main__":
    main()


