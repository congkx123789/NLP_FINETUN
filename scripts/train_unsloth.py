#!/usr/bin/env python3
"""
Script training với Unsloth - Tối ưu cho RTX 5060 Ti 16GB GDDR7
"""

import argparse
import json
from pathlib import Path
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from datasets import load_dataset
import torch

def load_json_dataset(file_path):
    """Load dataset từ JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert sang format cho Unsloth
    texts = []
    for item in data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        
        # Format: instruction + input + output
        text = f"{instruction}\nInput: {input_text}\nOutput: {output_text}"
        texts.append({"text": text})
    
    return texts

class SimpleLogCallback(TrainerCallback):
    """In ra loss & learning rate theo thời gian thực (mỗi logging_steps)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        epoch = logs.get("epoch", state.epoch)
        loss = logs.get("loss", logs.get("train_loss", None))
        lr = logs.get("learning_rate", None)
        if loss is not None or lr is not None:
            msg = f"📉 Step {step}"
            if epoch is not None:
                msg += f" | epoch={epoch:.2f}"
            if loss is not None:
                msg += f" | loss={loss:.4f}"
            if lr is not None:
                msg += f" | lr={lr:.6f}"
            print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train với Unsloth - Tối ưu tốc độ")
    parser.add_argument("--train-file", type=str, default=None, help="File train JSON")
    parser.add_argument("--val-file", type=str, default=None, help="File validation JSON")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    # direction:
    #   - en-vi : chỉ train chiều Anh -> Việt
    #   - vi-en : chỉ train chiều Việt -> Anh
    #   - mixed: train cả 2 chiều trong 1 model (dữ liệu đã mix sẵn, dùng instruction làm "công tắc")
    parser.add_argument(
        "--direction",
        type=str,
        default="en-vi",
        choices=["en-vi", "vi-en", "mixed"],
        help="Hướng dịch. 'mixed' dùng file JSON đã gộp cả en-vi & vi-en.",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of epochs (nếu không dùng --max-steps)")
    parser.add_argument("--max-steps", type=int, default=None, help="Train theo số bước (ưu tiên hơn epochs nếu được set)")
    parser.add_argument("--use-torch-compile", action="store_true", help="Enable torch.compile (requires no 4-bit quantization)")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"], help="Quantization type")
    
    args = parser.parse_args()
    
    # Tự động tìm file nếu không được chỉ định
    base_dir = Path(__file__).parent.parent
    if args.train_file is None:
        if args.direction == "en-vi":
            args.train_file = str(base_dir / "data/vlsp_medical_en_vi_train.json")
            args.val_file = args.val_file or str(base_dir / "data/vlsp_medical_en_vi_val.json")
            args.output_dir = args.output_dir or str(base_dir / "saves/qwen2_5-7b/unsloth/en_vi")
        elif args.direction == "vi-en":
            args.train_file = str(base_dir / "data/vlsp_medical_vi_en_train.json")
            args.val_file = args.val_file or str(base_dir / "data/vlsp_medical_vi_en_val.json")
            args.output_dir = args.output_dir or str(base_dir / "saves/qwen2_5-7b/unsloth/vi_en")
        else:  # mixed
            # YÊU CẦU: đã tạo sẵn các file:
            #   data/vlsp_medical_mixed_train.json
            #   data/vlsp_medical_mixed_val.json
            args.train_file = str(base_dir / "data/vlsp_medical_mixed_train.json")
            args.val_file = args.val_file or str(base_dir / "data/vlsp_medical_mixed_val.json")
            args.output_dir = args.output_dir or str(base_dir / "saves/qwen2_5-7b/unsloth/mixed")
    
    # Tạo output directory nếu chưa có
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("🚀 Khởi động training với Unsloth (Tối ưu RTX 5060 Ti)")
    print("=" * 60)
    print(f"📂 Train file: {args.train_file}")
    print(f"📂 Val file: {args.val_file}")
    print(f"📂 Output dir: {args.output_dir}")
    print(f"📂 Direction: {args.direction}")
    print("")
    
    # 1. Load Model với quantization tùy chọn và Flash Attention 2
    print("📥 Đang load model với Flash Attention 2...")
    print("   ⚡ Flash Attention 2 tự động bật nếu GPU hỗ trợ (RTX 5060 Ti có hỗ trợ)")
    
    # Xử lý quantization dựa trên yêu cầu torch.compile
    # Lưu ý: torch.compile() KHÔNG tương thích với BẤT KỲ quantization nào khi dùng PEFT
    if args.use_torch_compile:
        print("   ⚠️  torch.compile được bật - BỎ QUA quantization (không tương thích với PEFT)")
        print("   ℹ️  Sẽ dùng bf16 + gradient checkpointing để tiết kiệm VRAM")
        load_in_4bit = False
        load_in_8bit = False
    else:
        load_in_4bit = (args.quantization == "4bit")
        load_in_8bit = (args.quantization == "8bit")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/home/alida/Documents/Cursor/NLP_fine_tun/models/Qwen2.5-7B",
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        # Flash Attention 2 tự động bật trong Unsloth nếu GPU hỗ trợ
    )
    
    # 2. Config LoRA với Unsloth optimizations
    print("⚙️  Đang cấu hình LoRA với Unsloth optimizations...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
        random_state=3407,
    )
    
    # 3. Torch Compile để tối ưu cho RTX 5060 Ti (Blackwell architecture)
    if args.use_torch_compile:
        try:
            print("⚡ Đang bật torch.compile để tối ưu cho RTX 5060 Ti (Blackwell)...")
            print("   ⚠️  Lưu ý: torch.compile KHÔNG tương thích với quantization khi dùng PEFT")
            print("   ✅ Đã bỏ quantization - model sẽ dùng bf16 + gradient checkpointing")
            print("   💡 Nếu thiếu VRAM, có thể cần giảm batch-size")
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            print("✅ torch.compile đã được bật! Tốc độ sẽ tăng thêm 10-20%")
        except Exception as e:
            print(f"⚠️  torch.compile không khả dụng: {e}")
            print("   Training vẫn sẽ chạy bình thường với Unsloth (đã nhanh 2-3x)")
    else:
        print("ℹ️  torch.compile chưa được bật")
        print("   Để bật, thêm flag --use-torch-compile (sẽ bỏ quantization)")
        print("   Unsloth đã tối ưu sẵn, tốc độ vẫn rất nhanh")
    
    
    # 4. Load dataset
    print("📊 Đang load dataset...")
    # Load JSON files directly
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    
    # Convert to format cho Unsloth
    def format_text(example):
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output_text = example.get('output', '')
        text = f"{instruction}\nInput: {input_text}\nOutput: {output_text}"
        return {"text": text}
    
    train_dataset = train_dataset.map(format_text, remove_columns=train_dataset.column_names)
    
    eval_dataset = None
    if args.val_file:
        eval_dataset = load_dataset("json", data_files=args.val_file, split="train")
        eval_dataset = eval_dataset.map(format_text, remove_columns=eval_dataset.column_names)
    
    # 5. Training Arguments tối ưu cho RTX 5060 Ti + dữ liệu lớn
    # Ưu tiên train theo steps nếu --max-steps được set, ngược lại dùng epochs như bình thường.
    training_args_kwargs = dict(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,  # Bắt buộc cho RTX 5060 Ti
        logging_steps=10,
        optim="adamw_8bit",  # Tiết kiệm VRAM
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=args.output_dir,
        dataloader_num_workers=4,  # Tối ưu CPU, tránh nghẽn I/O
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=2000 if eval_dataset else None,
    )
    if args.max_steps is not None:
        training_args_kwargs["max_steps"] = args.max_steps
    else:
        training_args_kwargs["num_train_epochs"] = args.epochs

    training_args = TrainingArguments(**training_args_kwargs)
    
    # 6. Trainer với packing (Tăng tốc 2-3x)
    print("🎯 Bắt đầu training với packing=True (Tăng tốc 2-3x)...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=True,  # ⚡ QUAN TRỌNG: Gộp dữ liệu để train siêu nhanh
        args=training_args,
        callbacks=[SimpleLogCallback()],
    )
    
    # 6. Train
    train_output = trainer.train()
    
    # 7. Thống kê thời gian train (dễ đọc)
    metrics = getattr(train_output, "metrics", None) or {}
    # Fallback: lấy từ trainer.state nếu cần
    if not metrics and trainer.state.log_history:
        for log in reversed(trainer.state.log_history):
            if "train_runtime" in log:
                metrics = log
                break
    train_runtime = float(metrics.get("train_runtime", 0.0))
    train_epochs = float(metrics.get("epoch", args.epochs))
    train_loss = float(metrics.get("train_loss", -1.0))
    total_seconds = int(train_runtime)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print(f"⏱️  Thời gian train: {train_runtime:.2f} giây (~{hours}h {minutes}m {seconds}s) | epochs={train_epochs} | train_loss={train_loss:.4f}")
    
    # 8. Save
    print("💾 Đang lưu model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ Hoàn thành! Model đã được lưu tại: {args.output_dir}")

if __name__ == "__main__":
    main()

