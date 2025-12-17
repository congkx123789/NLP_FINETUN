## Qwen2.5-7B Medical MT (EN–VI / VI–EN) – Unsloth + FlashAttention2

Dự án này fine-tune **Qwen2.5-7B** cho dịch máy y tế Anh–Việt / Việt–Anh trên dữ liệu VLSP (constrained), tối ưu cho **RTX 5060 Ti 16GB** với **Unsloth + QLoRA 4-bit + FlashAttention 2**.  
Bạn đã có:
- Bản **adapter 4-bit đã fine-tune**: `saves/qwen2_5-7b/unsloth/mixed_maxsteps10000_fa2`
- Bản **full 16-bit đã merge**: `final_models/Qwen2.5-7B-Medical-Full-Bin`
- Bản **GGUF** cho Ollama / llama.cpp: `Qwen2.5-7B.Q4_K_M.gguf`

---

## 1. Chuẩn bị môi trường

```bash
cd /home/alida/Documents/Cursor/NLP_fine_tun
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unsloth_nightly
```

Môi trường này đã cài:
- `torch` nightly + CUDA 12.8 (cu128)
- `unsloth` (colab-new) + FA2
- `trl`, `peft`, `bitsandbytes`, `sacrebleu`, v.v.

---

## 2. Cấu trúc dữ liệu

### 2.1. Dữ liệu VLSP đã chuẩn hóa

Trong `data/`:
- Một chiều:
  - `vlsp_medical_en_vi_train.json`, `vlsp_medical_en_vi_val.json`
  - `vlsp_medical_vi_en_train.json`, `vlsp_medical_vi_en_val.json`
- Dữ liệu **mix 2 chiều** (train 1 model dịch 2 chiều bằng instruction):
  - `vlsp_medical_mixed_train.json`
  - `vlsp_medical_mixed_val.json`

Format mỗi dòng (Alpaca-style đơn giản):

```json
{
  "instruction": "Translate the following English text to Vietnamese:",
  "input": "Text source...",
  "output": "Text target..."
}
```

### 2.2. Public test

- File gốc: `data/public_test.json`
  - Trường: `id`, `direction` ("en-vi" / "vi-en"), `source`, `target`
- File kết quả:
  - TSV: `results/public_test_predictions.tsv`
  - JSON: `results/public_test_predictions.json`

---

## 3. Train với Unsloth (adapter 4-bit, FA2)

Script chính: `scripts/train_unsloth.py`

### 3.1. Train mixed (EN–VI + VI–EN) 10.000 steps

```bash
cd /home/alida/Documents/Cursor/NLP_fine_tun
python scripts/train_unsloth.py \
  --direction mixed \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --max-seq-length 2048 \
  --max-steps 10000 \
  --output-dir saves/qwen2_5-7b/unsloth/mixed_maxsteps10000_fa2
```

Các thiết lập chính bên trong:
- **QLoRA 4-bit** (`load_in_4bit=True`)
- **bf16 = True**, **fp16 = False**
- **FlashAttention2 = True** (nếu GPU hỗ trợ – RTX 5060 Ti OK)
- **packing = True** trong `SFTTrainer` (gộp nhiều câu ngắn → tăng tốc)
- `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`

Log training được lưu trong:
- `training_mixed_maxsteps10000.log` (nếu dùng `tee`)

---

## 4. Merge adapter → full 16-bit + GGUF

Script: `scripts/merge_all.py`

Chức năng:
1. Load adapter Unsloth: `saves/qwen2_5-7b/unsloth/mixed_maxsteps10000_fa2`
2. `save_pretrained_merged(...)` → 16-bit safetensors:
   - `final_models/Qwen2.5-7B-Medical-Full-Bin`
3. `save_pretrained_gguf(...)` → GGUF (ví dụ `q4_k_m`):
   - `final_models/Qwen2.5-7B-Medical-GGUF`

Sau đó bạn đã copy/đổi tên 1 bản GGUF ra ngoài:
- `Qwen2.5-7B.Q4_K_M.gguf`

---

## 5. Chạy inference trên public_test (Unsloth + FA2)

Script: `scripts/eval_public_test.py`

### 5.1. Chạy full 6000 câu (adapter Unsloth)

```bash
python scripts/eval_public_test.py \
  --model-dir saves/qwen2_5-7b/unsloth/mixed_maxsteps10000_fa2 \
  --test-file data/public_test.json \
  --output-tsv results/public_test_predictions.tsv \
  --batch-size 8 \
  --max-new-tokens 128
```

Đặc điểm:
- Dùng `FastLanguageModel.from_pretrained` → **bật FA2 + 4-bit**
- Prompt đúng format train:

  ```text
  Translate the following English text to Vietnamese:
  Input: <source>
  Output:
  ```

- Tham số chống lặp:
  - `repetition_penalty=1.2`
  - `no_repeat_ngram_size=2`
  - `eos_token_id=tokenizer.eos_token_id`
- Ghi ra:
  - TSV: `results/public_test_predictions.tsv`
  - JSON: `results/public_test_predictions.json`
  - Log thời gian: `eval_public_test.log` (hoặc `eval_public_test_resume.log`)

### 5.2. Resume khi bị crash / timeout

`eval_public_test.py` hỗ trợ:

```bash
python scripts/eval_public_test.py \
  --model-dir saves/qwen2_5-7b/unsloth/mixed_maxsteps10000_fa2 \
  --test-file data/public_test.json \
  --output-tsv results/public_test_predictions.tsv \
  --batch-size 8 \
  --max-new-tokens 128 \
  --resume
```

- `--resume`: đọc TSV/JSON hiện có, tiếp tục từ mẫu kế tiếp (không chạy lại từ đầu).

---

## 6. Tính BLEU cho public_test

Script: `scripts/compute_bleu_public_test.py`

```bash
python scripts/compute_bleu_public_test.py \
  --pred-tsv results/public_test_predictions.tsv
```

In ra:
- Số câu có prediction
- **BLEU tổng** (en-vi + vi-en)
- **BLEU en-vi**
- **BLEU vi-en**

---

## 7. Đồng bộ TSV ↔ JSON kết quả

Để convert lại khi TSV thay đổi:

Script: `scripts/tsv_to_json.py`

```bash
python scripts/tsv_to_json.py results/public_test_predictions.tsv
```

Tạo/ghi đè:
- `results/public_test_predictions.json`

---

## 8. Chạy model full 16-bit trực tiếp (Transformers)

Script: `scripts/run_fullbin_translate.py`

### 8.1. EN → VI

```bash
python scripts/run_fullbin_translate.py \
  --model-dir final_models/Qwen2.5-7B-Medical-Full-Bin \
  --direction en-vi \
  --text "The patient has a severe headache and a history of hypertension."
```

### 8.2. VI → EN

```bash
python scripts/run_fullbin_translate.py \
  --model-dir final_models/Qwen2.5-7B-Medical-Full-Bin \
  --direction vi-en \
  --text "Bệnh nhân có tiền sử tăng huyết áp và hiện đang đau đầu dữ dội."
```

Script sẽ:
- Load model 16-bit bằng `AutoModelForCausalLM.from_pretrained`.
- Prompt: `instruction + "Input:" + "Output:"`.
- In ra phần sau `"Output:"` là bản dịch.

---

## 9. Chạy model full với Unsloth + FA2 (tối ưu GPU)

Nếu muốn tận dụng FA2 + 4-bit cả với bản đã merge:

Script: `scripts/run_unsloth_fullbin.py`

### 9.1. EN → VI (4-bit + FA2)

```bash
python scripts/run_unsloth_fullbin.py \
  --model-dir final_models/Qwen2.5-7B-Medical-Full-Bin \
  --direction en-vi \
  --text "The patient has a severe headache and a history of hypertension."
```

### 9.2. VI → EN

```bash
python scripts/run_unsloth_fullbin.py \
  --model-dir final_models/Qwen2.5-7B-Medical-Full-Bin \
  --direction vi-en \
  --text "Bệnh nhân có tiền sử tăng huyết áp và hiện đang đau đầu dữ dội."
```

Nếu muốn load full 16-bit (không 4-bit):

```bash
python scripts/run_unsloth_fullbin.py \
  --model-dir final_models/Qwen2.5-7B-Medical-Full-Bin \
  --direction en-vi \
  --text "The patient has a severe headache and a history of hypertension." \
  --no-4bit
```

---

## 10. Dùng GGUF với Ollama / llama.cpp (tóm tắt)

### 10.1. llama.cpp

- Model GGUF: `Qwen2.5-7B.Q4_K_M.gguf`

Ví dụ chạy CLI:

```bash
./llama.cpp/llama-cli \
  -m Qwen2.5-7B.Q4_K_M.gguf \
  -p "Translate the following English text to Vietnamese:\nInput: The patient has a severe headache.\nOutput:"
```

### 10.2. Ollama (gợi ý)

Tạo file `Modelfile` (ví dụ trong thư mục này):

```text
FROM ./Qwen2.5-7B.Q4_K_M.gguf
PARAMETER temperature 0.1
PARAMETER top_p 0.9
TEMPLATE """{{ .Prompt }}"""
```

Sau đó:

```bash
ollama create qwen2.5-7b-medical -f Modelfile
ollama run qwen2.5-7b-medical
```

Rồi gõ prompt dịch với đúng format `instruction + Input + Output` như trên.

---

## 11. Ghi chú quan trọng

- **Train & Test phải cùng format prompt**:
  - Bạn đã dùng:  
    `{instruction}\nInput: {input}\nOutput: {output}`
  - Khi inference, luôn gửi:  
    `{instruction}\nInput: {source}\nOutput:`
- **FA2 chỉ bật khi dùng Unsloth** (`FastLanguageModel.from_pretrained`).
- Với RTX 5060 Ti 16GB:
  - Ưu tiên `load_in_4bit=True`, `bf16=True`, `packing=True`.
  - `batch_size * gradient_accumulation_steps` ≈ 16 là hợp lý.


