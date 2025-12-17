#!/usr/bin/env python3
"""
Script để chuyển đổi dữ liệu song ngữ từ file text sang JSON format cho training
"""

import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def read_parallel_files(vi_file, en_file):
    """Đọc các file song ngữ và trả về danh sách các cặp câu"""
    print(f"📖 Đang đọc file: {vi_file} và {en_file}")
    
    with open(vi_file, 'r', encoding='utf-8') as f_vi:
        vi_lines = [line.strip() for line in f_vi if line.strip()]
    
    with open(en_file, 'r', encoding='utf-8') as f_en:
        en_lines = [line.strip() for line in f_en if line.strip()]
    
    if len(vi_lines) != len(en_lines):
        print(f"⚠️  Cảnh báo: Số dòng không khớp! Vi: {len(vi_lines)}, En: {len(en_lines)}")
        min_len = min(len(vi_lines), len(en_lines))
        vi_lines = vi_lines[:min_len]
        en_lines = en_lines[:min_len]
        print(f"   Đã cắt xuống {min_len} cặp câu")
    
    return list(zip(vi_lines, en_lines))

def create_json_dataset(pairs, direction, output_file, test_size=0.1):
    """
    Tạo dataset JSON từ các cặp câu
    
    Args:
        pairs: List of (vi_text, en_text) tuples
        direction: "en-vi" hoặc "vi-en"
        output_file: File output JSON
        test_size: Tỷ lệ validation set
    """
    print(f"📝 Đang tạo dataset {direction}...")
    
    # Tạo train/val split
    train_pairs, val_pairs = train_test_split(
        pairs, 
        test_size=test_size, 
        random_state=42,
        shuffle=True
    )
    
    print(f"   Train: {len(train_pairs)} cặp câu")
    print(f"   Val: {len(val_pairs)} cặp câu")
    
    # Tạo train dataset
    train_data = []
    for vi_text, en_text in train_pairs:
        if direction == "en-vi":
            item = {
                "instruction": "Translate the following English text to Vietnamese:",
                "input": en_text,
                "output": vi_text
            }
        else:  # vi-en
            item = {
                "instruction": "Translate the following Vietnamese text to English:",
                "input": vi_text,
                "output": en_text
            }
        train_data.append(item)
    
    # Tạo val dataset
    val_data = []
    for vi_text, en_text in val_pairs:
        if direction == "en-vi":
            item = {
                "instruction": "Translate the following English text to Vietnamese:",
                "input": en_text,
                "output": vi_text
            }
        else:  # vi-en
            item = {
                "instruction": "Translate the following Vietnamese text to English:",
                "input": vi_text,
                "output": en_text
            }
        val_data.append(item)
    
    # Lưu train file
    train_file = output_file.replace('.json', '_train.json')
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã lưu train file: {train_file}")
    
    # Lưu val file
    val_file = output_file.replace('.json', '_val.json')
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã lưu val file: {val_file}")
    
    return train_file, val_file

def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu cho training")
    parser.add_argument("--vi-file", type=str, 
                       default="data/raw/train.vi.txt",
                       help="File tiếng Việt")
    parser.add_argument("--en-file", type=str,
                       default="data/raw/train.en.txt",
                       help="File tiếng Anh")
    parser.add_argument("--output-dir", type=str,
                       default="data",
                       help="Thư mục output")
    parser.add_argument("--test-size", type=float, default=0.1,
                       help="Tỷ lệ validation set (default: 0.1)")
    
    args = parser.parse_args()
    
    # Đảm bảo output directory tồn tại
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Bắt đầu xử lý dữ liệu...")
    print("=" * 60)
    
    # Đọc file song ngữ
    pairs = read_parallel_files(args.vi_file, args.en_file)
    print(f"✅ Đã đọc {len(pairs)} cặp câu")
    print()
    
    # Tạo dataset cho cả hai chiều
    print("📌 Tạo dataset Anh -> Việt")
    print("-" * 60)
    en_vi_train, en_vi_val = create_json_dataset(
        pairs,
        "en-vi",
        str(output_dir / "vlsp_medical_en_vi.json"),
        args.test_size
    )
    print()
    
    print("📌 Tạo dataset Việt -> Anh")
    print("-" * 60)
    vi_en_train, vi_en_val = create_json_dataset(
        pairs,
        "vi-en",
        str(output_dir / "vlsp_medical_vi_en.json"),
        args.test_size
    )
    print()
    
    print("=" * 60)
    print("✅ Hoàn thành xử lý dữ liệu!")
    print()
    print("📁 Các file đã tạo:")
    print(f"   - {en_vi_train}")
    print(f"   - {en_vi_val}")
    print(f"   - {vi_en_train}")
    print(f"   - {vi_en_val}")

if __name__ == "__main__":
    main()

