#!/usr/bin/env python3
"""
Tính BLEU cho bộ public_test dựa trên file:
  - Input:  results/public_test_predictions.tsv  (id, direction, source, target_ref, prediction)
  - Output: In ra BLEU tổng + BLEU cho en-vi và vi-en.
"""

import argparse
from pathlib import Path

import sacrebleu


def load_tsv(path: Path):
    refs = []
    hyps = []
    dirs = []

    with path.open("r", encoding="utf-8") as f:
        # Bỏ dòng header
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            _id, direction, _src, ref, hyp = parts
            dirs.append(direction)
            refs.append(ref)
            hyps.append(hyp)

    return dirs, refs, hyps


def compute_and_print_bleu(name: str, refs, hyps):
    if not hyps:
        print(f"{name}: Không có câu nào, bỏ qua.")
        return
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    print(f"{name}: BLEU = {bleu.score:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Tính BLEU trên file results/public_test_predictions.tsv"
    )
    parser.add_argument(
        "--pred-tsv",
        type=str,
        default="results/public_test_predictions.tsv",
        help="File TSV chứa prediction (id, direction, source, target_ref, prediction).",
    )
    args = parser.parse_args()

    tsv_path = Path(args.pred_tsv)
    if not tsv_path.exists():
        print(f"❌ Không tìm thấy file: {tsv_path}")
        return

    dirs, refs, hyps = load_tsv(tsv_path)
    print(f"📊 Tổng số câu có prediction: {len(hyps)}")

    # BLEU chung
    compute_and_print_bleu("Tất cả (en-vi + vi-en)", refs, hyps)

    # BLEU theo chiều
    en_vi_refs = [r for d, r in zip(dirs, refs) if d == "en-vi"]
    en_vi_hyps = [h for d, h in zip(dirs, hyps) if d == "en-vi"]
    vi_en_refs = [r for d, r in zip(dirs, refs) if d == "vi-en"]
    vi_en_hyps = [h for d, h in zip(dirs, hyps) if d == "vi-en"]

    compute_and_print_bleu("Chiều en-vi", en_vi_refs, en_vi_hyps)
    compute_and_print_bleu("Chiều vi-en", vi_en_refs, vi_en_hyps)


if __name__ == "__main__":
    main()


