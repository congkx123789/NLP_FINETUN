#!/usr/bin/env python3
"""
Script benchmark tốc độ Unsloth trên RTX 5060 Ti với:
- 4-bit quantization
- FastLanguageModel.for_inference
- Batch size = 8

Chạy:
    cd /home/alida/Documents/Cursor/NLP_fine_tun
    python scripts/max_speed_unsloth.py
"""

from unsloth import FastLanguageModel
import torch
import time

# --- CẤU HÌNH ---
MODEL_PATH = "/home/alida/Documents/Cursor/NLP_fine_tun/final_models/Qwen2.5-7B-Medical-Full-Bin"


def main():
    print(f"⏳ Đang load model với Flash Attention 2 từ: {MODEL_PATH}...")

    # 1. Load Model (Bắt buộc 4-bit để nhẹ và nhanh trên card 16GB)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # 2. KÍCH HOẠT TĂNG TỐC (QUAN TRỌNG NHẤT)
    FastLanguageModel.for_inference(model)

    # 3. Dữ liệu test (Giả lập 16 câu hỏi y tế)
    prompts = [
        "Translate to Vietnamese: The patient presents with severe abdominal pain.",
        "Translate to Vietnamese: Dosage: Take 500mg twice daily after meals.",
        "Translate to Vietnamese: MRI scan reveals a mass in the left lung.",
        "Translate to English: Bệnh nhân có tiền sử dị ứng với các loại hải sản.",
        "Translate to English: Chỉ định: Phẫu thuật nội soi cắt ruột thừa.",
        "Translate to Vietnamese: Acute kidney failure is a rapid loss of kidney function.",
        "Translate to Vietnamese: The doctor prescribed antibiotics for the infection.",
        "Translate to English: Bệnh nhân bị gãy xương đùi phải do tai nạn giao thông.",
    ] * 2  # Nhân đôi lên thành 16 câu để test khả năng chịu tải

    print(f"\n🚀 Đang xử lý {len(prompts)} câu với batch_size = 8...")
    start_time = time.time()

    # 4. CHẠY BATCH (Thay vì vòng lặp for từng câu)
    batch_size = 8
    results = []

    # Thêm padding side left cho decoder (Bắt buộc khi batching)
    tokenizer.padding_side = "left"

    device = model.device

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]

        # Tokenize 1 cục
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                use_cache=True,  # Bắt buộc True để nhanh
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode kết quả
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)

    end_time = time.time()
    total_time = end_time - start_time

    print("=" * 50)
    print(f"⚡ Tốc độ xử lý: {len(prompts) / total_time:.2f} câu/giây")
    print(f"⏱️ Tổng thời gian: {total_time:.2f} giây")
    print("=" * 50)

    # In thử vài kết quả
    for res in results[:2]:
        print(f">> {res}")


if __name__ == "__main__":
    main()




