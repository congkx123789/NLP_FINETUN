#!/usr/bin/env python3
"""
Gộp dữ liệu VLSP en-vi & vi-en thành một bộ mixed (instruction/input/output)
để train 2 chiều trong một model duy nhất.
"""

import json
import random
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    base_dir = Path(__file__).parent.parent / "data"

    en_vi_train_path = base_dir / "vlsp_medical_en_vi_train.json"
    en_vi_val_path = base_dir / "vlsp_medical_en_vi_val.json"
    vi_en_train_path = base_dir / "vlsp_medical_vi_en_train.json"
    vi_en_val_path = base_dir / "vlsp_medical_vi_en_val.json"

    print("📥 Đang load dữ liệu VLSP en-vi & vi-en ...")
    en_vi_train = load_json(en_vi_train_path)
    en_vi_val = load_json(en_vi_val_path)
    vi_en_train = load_json(vi_en_train_path)
    vi_en_val = load_json(vi_en_val_path)

    print(f"  en-vi train: {len(en_vi_train)}")
    print(f"  en-vi val  : {len(en_vi_val)}")
    print(f"  vi-en train: {len(vi_en_train)}")
    print(f"  vi-en val  : {len(vi_en_val)}")

    # 1) Mix train
    mixed_train = []
    mixed_train.extend(en_vi_train)
    mixed_train.extend(vi_en_train)

    # 2) Mix val
    mixed_val = []
    mixed_val.extend(en_vi_val)
    mixed_val.extend(vi_en_val)

    # 3) Shuffle để tránh học lệch / catastrophic forgetting
    random.seed(42)
    random.shuffle(mixed_train)
    random.shuffle(mixed_val)

    # 4) Lưu ra file mới
    mixed_train_path = base_dir / "vlsp_medical_mixed_train.json"
    mixed_val_path = base_dir / "vlsp_medical_mixed_val.json"

    with open(mixed_train_path, "w", encoding="utf-8") as f:
        json.dump(mixed_train, f, ensure_ascii=False, indent=2)
    with open(mixed_val_path, "w", encoding="utf-8") as f:
        json.dump(mixed_val, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã tạo {mixed_train_path} với {len(mixed_train)} dòng.")
    print(f"✅ Đã tạo {mixed_val_path} với {len(mixed_val)} dòng.")


if __name__ == "__main__":
    main()


