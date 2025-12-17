#!/usr/bin/env python3
"""
Batch inference với Unsloth ở 4-bit để tránh OOM trên RTX 5060 Ti.

Chạy:
    cd /home/alida/Documents/Cursor/NLP_fine_tun
    python final_test_unsloth.py
"""

from unsloth import FastLanguageModel
import torch
import json

try:
    from tqdm import tqdm
except ImportError:  # Fallback nếu chưa cài tqdm
    def tqdm(x, **kwargs):
        return x


# --- CẤU HÌNH ---
# Đường dẫn model đã merge
MODEL_PATH = "/home/alida/Documents/Cursor/NLP_fine_tun/final_models/Qwen2.5-7B-Medical-Full-Bin"

# (Tuỳ chọn) File test JSON nếu sau này bạn muốn dùng
INPUT_FILE = "data/public_test.json"
OUTPUT_FILE = "predictions.jsonl"


print(f"⏳ Đang load model 4-bit từ: {MODEL_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,  # CHÌA KHÓA ĐỂ KHÔNG BỊ OOM
)
FastLanguageModel.for_inference(model)  # Tăng tốc 2x


def batch_translate(texts, batch_size: int = 8):
    """Dịch batch câu Anh->Việt với prompt đơn giản."""
    results = []

    prompts = [
        (f"Translate to Vietnamese: {text}" if "Translate" not in text else text)
        for text in texts
    ]

    for i in tqdm(range(0, len(prompts), batch_size), desc="Đang dịch..."):
        batch = prompts[i : i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cleaned = [d.split("Vietnamese:")[-1].strip() for d in decoded]
        results.extend(cleaned)

    return results


def main():
    # --- CHẠY THỬ VỚI DỮ LIỆU MẪU ---
    test_sentences = [
        "The patient was diagnosed with type 2 diabetes.",
        "Acute respiratory distress syndrome (ARDS) is a life-threatening condition.",
        "Dùng thuốc sau khi ăn 30 phút.",
        "Bệnh nhân có tiền sử cao huyết áp vô căn.",
    ]

    print("\n🚀 Bắt đầu test thử...")
    translations = batch_translate(test_sentences, batch_size=4)

    for src, tgt in zip(test_sentences, translations):
        print("-" * 40)
        print(f"Input:  {src}")
        print(f"Output: {tgt}")

    print("\n✅ Hoàn tất! Model đang chạy ở chế độ 4-bit, an toàn VRAM.")


if __name__ == "__main__":
    main()


