# ✅ Cài đặt hoàn tất!

## Môi trường đã được thiết lập thành công

- **Conda Environment**: `unsloth_env` (Python 3.10)
- **GPU**: NVIDIA GeForce RTX 5060 Ti ✅
- **CUDA Version**: 12.8 ✅
- **PyTorch Version**: 2.9.1+cu128 ✅
- **Unsloth**: Đã cài đặt và sẵn sàng ✅

## Cách sử dụng

### 1. Kích hoạt môi trường mỗi lần mở Terminal:

```bash
conda activate unsloth_env
```

Hoặc nếu conda chưa được khởi động tự động:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unsloth_env
```

### 2. Chạy training script:

```bash
cd /home/alida/Documents/Cursor/NLP_fine_tun
python scripts/train_unsloth.py --direction en-vi
```

Hoặc cho chiều Việt-Anh:

```bash
python scripts/train_unsloth.py --direction vi-en
```

## Lưu ý về Flash Attention 2

Flash Attention 2 chưa được cài đặt vì cần CUDA toolkit. Tuy nhiên, **Unsloth đã có tối ưu hóa riêng** và sẽ hoạt động tốt mà không cần Flash Attention 2.

Nếu bạn muốn cài Flash Attention 2 sau (để tăng tốc thêm), bạn cần:
1. Cài đặt CUDA toolkit từ NVIDIA
2. Thiết lập biến môi trường `CUDA_HOME`
3. Chạy: `pip install flash-attn --no-build-isolation`

## Kiểm tra cài đặt

Để kiểm tra lại mọi thứ đã sẵn sàng:

```bash
conda activate unsloth_env
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'CUDA: {torch.version.cuda}'); import unsloth; print('Unsloth OK!')"
```

## Các thư viện đã cài đặt

- ✅ Unsloth (với tối ưu hóa 2-3x tốc độ)
- ✅ PyTorch 2.9.1 với CUDA 12.8
- ✅ Transformers, Datasets, PEFT, TRL
- ✅ BitsAndBytes (cho quantization)
- ✅ XFormers
- ✅ Tất cả các thư viện hỗ trợ cần thiết

---

**Bạn đã sẵn sàng để bắt đầu training! 🚀**



