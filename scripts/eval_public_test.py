#!/usr/bin/env python3
"""
Đánh giá / chạy inference model Unsloth (Qwen2.5-7B) trên file public_test.json.
- Dùng FastLanguageModel (Unsloth) -> Flash Attention 2 tự bật nếu GPU hỗ trợ.
- Input:  data/public_test.json  (id, direction, source, target)
- Output: results/public_test_predictions.tsv
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

import torch
from unsloth import FastLanguageModel


def build_prompt(item: Dict) -> str:
    """
    Tạo prompt ĐÚNG VỚI FORMAT ĐÃ TRAIN:
        instruction
        Input: ...
        Output:
    (y hệt như trong train_unsloth.py)
    """
    direction = item.get("direction")
    source = item.get("source", "")

    if direction == "en-vi":
        instruction = "Translate the following English text to Vietnamese:"
    else:
        instruction = "Translate the following Vietnamese text to English:"

    # Lúc train: "Output: {output_text}" -> lúc test chỉ để "Output:" cho model tự điền
    prompt = f"{instruction}\nInput: {source}\nOutput:"
    return prompt


def extract_answer(full_text: str) -> str:
    """
    Tách phần dịch sau 'Output:' khỏi toàn bộ chuỗi generate.
    """
    marker = "Output:"
    idx = full_text.rfind(marker)
    if idx == -1:
        return full_text.strip()
    return full_text[idx + len(marker) :].strip()


def run_inference(
    model_dir: str,
    test_file: str,
    output_tsv: str,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    resume: bool = False,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()
    print(f"📥 Đang load model từ: {model_dir} (device={device})")

    # Dùng Unsloth model (4bit) để giữ Flash Attention 2 + tối ưu VRAM
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    model.eval()

    # Load test json
    test_path = Path(test_file)
    with test_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📊 Số câu test: {len(data)}")

    # Chuẩn bị output
    out_path = Path(output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Sẽ ghi kết quả vào: {out_path}")

    # Lưu thêm kết quả dạng JSON cho dễ đọc / phân tích
    json_results: List[Dict] = []
    json_path = out_path.with_suffix(".json")

    # Xử lý logic resume: nếu đã có TSV/JSON thì tiếp tục từ dòng kế tiếp
    start_index = 0
    tsv_mode = "w"
    write_header = True

    if resume and out_path.exists():
        lines: List[str]
        with out_path.open("r", encoding="utf-8") as f_prev:
            lines = f_prev.readlines()
        if len(lines) > 1:
            # Đã có sẵn prediction trước đó
            start_index = len(lines) - 1  # trừ header
            tsv_mode = "a"
            write_header = False
            print(f"🔁 Resume từ mẫu thứ {start_index} (dựa trên TSV hiện có).")
        else:
            start_index = 0
            tsv_mode = "w"
            write_header = True

        # Nếu đã có file JSON cũ thì load vào để nối thêm
        if json_path.exists():
            import json as _json

            with json_path.open("r", encoding="utf-8") as jf_prev:
                try:
                    json_results = _json.load(jf_prev)
                except Exception:
                    json_results = []
    else:
        start_index = 0
        tsv_mode = "w"
        write_header = True

    with out_path.open(tsv_mode, encoding="utf-8") as fw:
        # Header
        if write_header:
            fw.write("id\tdirection\tsource\ttarget_ref\tprediction\n")

        # Batch inference
        total = len(data)
        if start_index >= total:
            print(f"✅ Tất cả {total} mẫu đã có prediction, không cần resume.")
            return

        for start in range(start_index, total, batch_size):
            end = min(start + batch_size, total)
            batch_items: List[Dict] = data[start:end]

            prompts = [build_prompt(item) for item in batch_items]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # greedy cho MT ổn định
                    use_cache=True,
                    # 🚫 Chống lặp / thoái hóa văn bản
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2,
                    eos_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for item, prompt, full_out in zip(batch_items, prompts, decoded):
                # Trong full_out có cả prompt + answer -> tách phần answer
                # Phòng trường hợp tokenizer bỏ bớt prompt, vẫn fallback bằng marker 'Output:'
                if full_out.startswith(prompt):
                    answer = full_out[len(prompt) :].strip()
                else:
                    answer = extract_answer(full_out)

                pred_clean = answer.replace(chr(9), " ")
                src_clean = item.get("source", "").replace(chr(9), " ")
                tgt_clean = item.get("target", "").replace(chr(9), " ")

                # Ghi TSV (phục vụ nộp bài / tính BLEU)
                fw.write(
                    f"{item.get('id')}\t{item.get('direction')}\t"
                    f"{src_clean}\t"
                    f"{tgt_clean}\t"
                    f"{pred_clean}\n"
                )

                # Gom để xuất thêm file JSON dễ đọc
                json_results.append(
                    {
                        "id": item.get("id"),
                        "direction": item.get("direction"),
                        "source": src_clean,
                        "target_ref": tgt_clean,
                        "prediction": pred_clean,
                    }
                )

            print(f"✅ Đã xử lý {end}/{total} câu", flush=True)

    elapsed = time.time() - start_time
    mins = elapsed / 60.0
    print(f"🎉 Hoàn thành! Kết quả được lưu tại: {out_path}")
    print(f"⏱️ Tổng thời gian inference: {elapsed:.1f} giây (~{mins:.2f} phút) cho {total} câu.")

    # Ghi thêm file JSON đẹp cho bạn dễ xem bằng editor / Jupyter
    import json as _json

    with json_path.open("w", encoding="utf-8") as jf:
        _json.dump(json_results, jf, ensure_ascii=False, indent=2)
    print(f"📄 Đồng thời đã lưu bản JSON dễ đọc tại: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Chạy inference model Unsloth (Qwen2.5-7B) trên public_test.json"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="saves/qwen2_5-7b/unsloth/mixed_maxsteps10000_fa2",
        help="Thư mục model Unsloth đã fine-tune (adapter).",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/public_test.json",
        help="File JSON test (format: id, direction, source, target).",
    )
    parser.add_argument(
        "--output-tsv",
        type=str,
        default="results/public_test_predictions.tsv",
        help="File TSV output chứa prediction.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size khi inference.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Số token tối đa model sinh ra.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục từ TSV/JSON hiện có, không chạy lại từ đầu.",
    )

    args = parser.parse_args()
    run_inference(
        model_dir=args.model_dir,
        test_file=args.test_file,
        output_tsv=args.output_tsv,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()



