#!/bin/bash
# Script để train cả hai chiều với Unsloth (Tốc độ cao nhất)

echo "🚀 Training với Unsloth - Tối ưu RTX 5060 Ti"
echo "=============================================="
echo ""

# Kích hoạt venv
source venv/bin/activate

# Kiểm tra Unsloth
python3 -c "import unsloth; print('✅ Unsloth đã được cài đặt')" 2>/dev/null || {
    echo "❌ Unsloth chưa được cài đặt!"
    echo "Đang cài đặt Unsloth..."
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps "xformers<0.0.27" trl peft accelerate bitsandbytes
}

echo ""
echo "📌 Bước 1/2: Training Anh -> Việt với Unsloth"
echo "=============================================="
nohup python3 scripts/train_unsloth.py \
    --direction en-vi \
    --max-seq-length 512 \
    --batch-size 8 \
    --lora-rank 8 \
    --epochs 1.0 \
    > training_log_unsloth_en_vi.txt 2>&1 &

EN_VI_PID=$!
echo $EN_VI_PID > training_unsloth_en_vi.pid
echo "✅ Training Anh->Việt đã khởi động (PID: $EN_VI_PID)"
echo "📝 Log: training_log_unsloth_en_vi.txt"
echo ""

sleep 5

echo "📌 Bước 2/2: Training Việt -> Anh với Unsloth"
echo "=============================================="
nohup python3 scripts/train_unsloth.py \
    --direction vi-en \
    --max-seq-length 512 \
    --batch-size 8 \
    --lora-rank 8 \
    --epochs 1.0 \
    > training_log_unsloth_vi_en.txt 2>&1 &

VI_EN_PID=$!
echo $VI_EN_PID > training_unsloth_vi_en.pid
echo "✅ Training Việt->Anh đã khởi động (PID: $VI_EN_PID)"
echo "📝 Log: training_log_unsloth_vi_en.txt"
echo ""

echo "=============================================="
echo "✅ Cả hai training với Unsloth đã được khởi động!"
echo ""
echo "📊 PIDs:"
echo "   - Anh->Việt: $EN_VI_PID"
echo "   - Việt->Anh: $VI_EN_PID"
echo ""
echo "📋 Các lệnh hữu ích:"
echo "   - Xem log Anh->Việt: tail -f training_log_unsloth_en_vi.txt"
echo "   - Xem log Việt->Anh: tail -f training_log_unsloth_vi_en.txt"
echo "   - Kiểm tra GPU: watch -n 1 nvidia-smi"
echo "   - Dừng training: kill $EN_VI_PID $VI_EN_PID"
echo ""
echo "⚡ Unsloth với Flash Attention 2 và torch.compile"

