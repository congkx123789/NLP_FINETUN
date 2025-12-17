#!/usr/bin/env python3
"""
Test tốc độ vLLM với Qwen2.5-7B-Medical (full bin) trên RTX 5060 Ti.

Chạy:
    cd /home/alida/Documents/Cursor/NLP_fine_tun
    python test_vllm_final.py
"""

from vllm import LLM, SamplingParams
import time

# Đường dẫn model Full-Bin của bạn
MODEL_PATH = "/home/alida/Documents/Cursor/NLP_fine_tun/final_models/Qwen2.5-7B-Medical-Full-Bin"


def main():
    print("⏳ Đang khởi động vLLM...")

    # Cấu hình tối ưu cho RTX 5060 Ti 16GB (nhưng vẫn có nguy cơ OOM với fp16)
    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        gpu_memory_utilization=0.9,  # Dùng 90% VRAM
        max_model_len=2048,          # Giới hạn độ dài để tiết kiệm nhớ
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    prompts = [
        "Translate to Vietnamese: The patient has a severe headache.",
        "Translate to Vietnamese: Take 2 tablets twice daily.",
        "Translate to English: Bệnh nhân bị đau bụng dữ dội.",
    ]

    sampling_params = SamplingParams(temperature=0.3, max_tokens=128)

    print("🚀 BẮT ĐẦU CHẠY...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()

    print(f"✅ Xong! Tổng thời gian: {end - start:.2f}s")
    for o in outputs:
        print(f"Input: {o.prompt}")
        print(f"Output: {o.outputs[0].text.strip()}")
        print("-" * 20)


if __name__ == "__main__":
    main()


