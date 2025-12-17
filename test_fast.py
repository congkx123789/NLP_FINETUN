#!/usr/bin/env python3
"""
Test tốc độ model đã merge bằng vLLM sau khi đã cài đặt thành công.

Chạy:
    python test_fast.py
"""

from vllm import LLM, SamplingParams
import time

# Đường dẫn đến model của bạn (folder chứa file safetensors to đùng)
MODEL_PATH = "/home/alida/Documents/Cursor/NLP_fine_tun/final_models/Qwen2.5-7B-Medical-Full-Bin"


def main():
    print(f"⏳ Đang khởi động vLLM với model: {MODEL_PATH}...")

    # 1. Load Model
    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",              # Chạy 16-bit
        gpu_memory_utilization=0.7,   # Giảm bớt để tránh OOM
        trust_remote_code=True,
        tensor_parallel_size=1,       # Chạy 1 GPU
        max_model_len=4096,           # Giảm context để nhẹ hơn
        enforce_eager=True,           # Tắt torch.compile để bớt tốn RAM
    )

    # 2. Cấu hình sinh chữ
    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=256,
        stop=["<|endoftext|>", "###"],
    )

    # 3. Bộ câu hỏi test
    prompts = [
        "Translate to Vietnamese: The patient has a severe headache and nausea.",
        "Translate to Vietnamese: Take one tablet explicitly after meals.",
        "Translate to English: Bệnh nhân bị gãy xương đùi trái do tai nạn.",
        "Translate to English: Chỉ định phẫu thuật nội soi cắt ruột thừa.",
    ]

    print("\n" + "=" * 50)
    print("🚀 BẮT ĐẦU TEST TỐC ĐỘ...")
    start_time = time.time()

    # 4. Chạy batch inference
    outputs = llm.generate(prompts, sampling_params)

    end_time = time.time()
    total_time = end_time - start_time

    print("=" * 50)
    print(f"✅ XONG! Tổng thời gian: {total_time:.2f} giây")
    print(f"⚡ Tốc độ trung bình: {len(prompts) / total_time:.2f} câu/giây")
    print("=" * 50 + "\n")

    # 5. In kết quả
    for output in outputs:
        print(f"🔹 Input:  {output.prompt}")
        print(f"🔸 Output: {output.outputs[0].text.strip()}")
        print("-" * 30)


if __name__ == "__main__":
    main()


