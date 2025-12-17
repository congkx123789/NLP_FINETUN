#!/usr/bin/env python3
"""
Gộp 2 file test song ngữ EN/VI thành 1 file duy nhất, mỗi dòng:
    <en>\t<vi>

Nguồn:
    Test/public_test.en.txt
    Test/public_test.vi.txt

Đích:
    Test/public_test.en-vi.tsv

Chạy:
    cd /home/alida/Documents/Cursor/NLP_fine_tun
    python scripts/mix_test_data.py
"""

from pathlib import Path


def main():
    base_dir = Path(__file__).parent.parent
    test_dir = base_dir / "Test"

    en_path = test_dir / "public_test.en.txt"
    vi_path = test_dir / "public_test.vi.txt"
    out_path = test_dir / "public_test.en-vi.tsv"

    print(f"📂 EN file : {en_path}")
    print(f"📂 VI file : {vi_path}")
    print(f"💾 Output  : {out_path}")

    with open(en_path, "r", encoding="utf-8") as f_en:
        en_lines = [l.rstrip("\n") for l in f_en.readlines()]
    with open(vi_path, "r", encoding="utf-8") as f_vi:
        vi_lines = [l.rstrip("\n") for l in f_vi.readlines()]

    if len(en_lines) != len(vi_lines):
        raise ValueError(f"Số dòng không khớp: EN={len(en_lines)}, VI={len(vi_lines)}")

    with open(out_path, "w", encoding="utf-8") as f_out:
        for en, vi in zip(en_lines, vi_lines):
            f_out.write(f"{en}\t{vi}\n")

    print(f"✅ Xong! Đã ghi {len(en_lines)} cặp câu vào: {out_path}")


if __name__ == "__main__":
    main()


