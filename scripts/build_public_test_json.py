#!/usr/bin/env python3
"""
Tạo file JSON test chứa cả hai chiều EN->VI và VI->EN từ bộ public_test.*.txt.

Nguồn:
    Test/public_test.en.txt
    Test/public_test.vi.txt

Đích:
    data/public_test.json

Mỗi phần tử JSON có dạng:
    {
        "id": <int>,          # chỉ số cặp gốc (1..N)
        "direction": "en-vi" hoặc "vi-en",
        "source": "<câu nguồn>",
        "target": "<câu đích>"
    }

Chạy:
    cd /home/alida/Documents/Cursor/NLP_fine_tun
    python scripts/build_public_test_json.py
"""

import json
from pathlib import Path


def main():
    base_dir = Path(__file__).parent.parent
    test_dir = base_dir / "Test"
    data_dir = base_dir / "data"

    en_path = test_dir / "public_test.en.txt"
    vi_path = test_dir / "public_test.vi.txt"
    out_path = data_dir / "public_test.json"

    print(f"📂 EN file : {en_path}")
    print(f"📂 VI file : {vi_path}")
    print(f"💾 Output  : {out_path}")

    with open(en_path, "r", encoding="utf-8") as f_en:
        en_lines = [l.rstrip("\n") for l in f_en.readlines()]
    with open(vi_path, "r", encoding="utf-8") as f_vi:
        vi_lines = [l.rstrip("\n") for l in f_vi.readlines()]

    if len(en_lines) != len(vi_lines):
        raise ValueError(f"Số dòng không khớp: EN={len(en_lines)}, VI={len(vi_lines)}")

    records = []
    for idx, (en, vi) in enumerate(zip(en_lines, vi_lines), start=1):
        # Chiều Anh -> Việt
        records.append(
            {
                "id": idx,
                "direction": "en-vi",
                "source": en,
                "target": vi,
            }
        )
        # Chiều Việt -> Anh
        records.append(
            {
                "id": idx,
                "direction": "vi-en",
                "source": vi,
                "target": en,
            }
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f_out:
        json.dump(records, f_out, ensure_ascii=False, indent=2)

    print(f"✅ Xong! Đã ghi {len(records)} mẫu (cả en-vi & vi-en) vào: {out_path}")


if __name__ == "__main__":
    main()


