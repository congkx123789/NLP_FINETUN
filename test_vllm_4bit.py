#!/usr/bin/env python3
"""
Test vLLM ở chế độ 4-bit (bitsandbytes) với Qwen2.5-7B-Medical.

Chạy:
    cd /home/alida/Documents/Cursor/NLP_fine_tun
    python test_vllm_4bit.py
"""

from vllm import LLM, SamplingParams
import time

# Đường dẫn model Full-Bin của bạn
MODEL_PATH = "/home/alida/Documents/Cursor/NLP_fine_tun/final_models/Qwen2.5-7B-Medical-Full-Bin"


def main():
    print("⏳ Đang khởi động vLLM chế độ 4-bit (bitsandbytes)...")

    llm = LLM(
        model=MODEL_PATH,
        # Nén 4-bit bằng bitsandbytes
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        dtype="float16",             # Tính toán 16-bit
        gpu_memory_utilization=0.9,  # Tận dụng 90% VRAM
        max_model_len=2048,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    prompts = [
        "Translate to Vietnamese: The patient has a severe headache.",
        "Translate to English: Bệnh nhân bị đau bụng dữ dội.",
    ]

    sampling_params = SamplingParams(temperature=0.3, max_tokens=128)

    print("🚀 BẮT ĐẦU CHẠY...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()

    print(f"✅ Xong! Tổng thời gian: {end - start:.2f}s")
    print("⚡ vLLM 4-bit thường sẽ nhanh hơn Unsloth thuần túy trên batch lớn.")
    for o in outputs:
        print(f">> {o.outputs[0].text.strip()}")


if __name__ == "__main__":
    main()


