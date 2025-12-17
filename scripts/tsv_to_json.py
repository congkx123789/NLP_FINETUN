#!/usr/bin/env python3
"""
Convert file TSV kết quả sang JSON dễ đọc.
"""

import json
import sys
from pathlib import Path

def tsv_to_json(tsv_path: str, json_path: str = None):
    """Convert TSV sang JSON."""
    tsv_file = Path(tsv_path)
    if not tsv_file.exists():
        print(f"❌ Không tìm thấy file: {tsv_path}")
        return
    
    if json_path is None:
        json_path = str(tsv_file.with_suffix(".json"))
    
    results = []
    
    with tsv_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            print("❌ File TSV trống!")
            return
        
        # Bỏ header
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            
            results.append({
                "id": parts[0],
                "direction": parts[1],
                "source": parts[2],
                "target_ref": parts[3],
                "prediction": parts[4],
            })
    
    json_file = Path(json_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    
    with json_file.open("w", encoding="utf-8") as jf:
        json.dump(results, jf, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã convert {len(results)} dòng từ TSV sang JSON:")
    print(f"   TSV: {tsv_path}")
    print(f"   JSON: {json_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tsv_to_json.py <tsv_file> [json_file]")
        sys.exit(1)
    
    tsv_path = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else None
    tsv_to_json(tsv_path, json_path)

