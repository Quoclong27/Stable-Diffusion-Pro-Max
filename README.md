<div align="center">

# 🎨 Image Processing Studio
### *Stable Diffusion Pro Max*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-GPU-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

> Bộ công cụ xử lý ảnh thông minh tích hợp 3 tác vụ AI: **Ghép ảnh đa phơi sáng**, **Chỉnh sửa ảnh bằng AI** và **Mở rộng ảnh tự động** — tất cả trong một giao diện web duy nhất.

</div>

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Tính năng](#-tính-năng)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt & Khởi chạy](#-cài-đặt--khởi-chạy)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cấu hình nâng cao](#-cấu-hình-nâng-cao)

---

## 🌟 Tổng quan

**Image Processing Studio** là một web application kết hợp nhiều mô hình AI tiên tiến vào một nền tảng thống nhất. Giao diện được xây dựng bằng **React + Chakra UI** ở frontend và **FastAPI + PyTorch** ở backend, hỗ trợ xử lý ảnh GPU thời gian thực với quản lý VRAM thông minh.

```
[React Frontend] ──► [FastAPI Backend] ──► [Model Manager] ──► [GPU / CUDA]
                                                  │
                          ┌───────────────────────┼──────────────────────┐
                          ▼                       ▼                      ▼
                   [Task 1: MEF]        [Task 2: Inpaint]      [Task 3: Outpaint]
                 UFRetinex-ToneNet      SDXL + LoRA           SDXL + ControlNet
```

---

## ✨ Tính năng

### 🖼️ Task 1 — Multi-Exposure Fusion (Ghép ảnh đa phơi sáng)

| Tính năng | Mô tả |
|-----------|-------|
| **Tải nhiều ảnh** | Hỗ trợ 2+ ảnh với các mức phơi sáng khác nhau |
| **SIFT Alignment** | Tự động căn chỉnh ảnh bằng thuật toán SIFT + RANSAC |
| **Phase 1 — MEF** | Mạng UFRetinex tổng hợp HDR từ nhiều ảnh LDR |
| **Phase 2 — ToneNet** | Tăng cường màu sắc và ánh sáng với mô hình ToneNet |
| **Xử lý 1 ảnh** | Nâng chất lượng ảnh đơn lẻ qua pipeline Phase 2 |

### ✏️ Task 2 — Inpainting (Chỉnh sửa ảnh bằng AI)

| Chế độ | Mô tả |
|--------|-------|
| **Add** | Thêm đối tượng mới vào vùng được vẽ mask |
| **Delete** | Xóa đối tượng và tự động điền nền phù hợp |
| **Replace** | Thay thế đối tượng bằng nội dung theo prompt |
| **Depth Estimation** | Ước lượng chiều sâu để tạo mask chính xác hơn |
| **Prompt Enhancer** | Nâng cao prompt bằng Qwen LLM tích hợp |

> 🔧 Được xây dựng trên **Stable Diffusion XL** với **LoRA fine-tuned** cho tiếng Việt và ảnh thực tế.

### 🌅 Task 3 — Outpainting (Mở rộng ảnh)

| Tính năng | Mô tả |
|-----------|-------|
| **Nhiều tỷ lệ khung hình** | 1:1, 4:3, 3:4, 16:9, 9:16 và tuỳ chỉnh |
| **Vị trí linh hoạt** | Đặt ảnh gốc tại Middle / Left / Right / Top / Bottom |
| **Overlap Control** | Kiểm soát vùng chồng lấp từng cạnh độc lập |
| **ControlNet Union** | Đảm bảo kết quả liền mạch với ảnh gốc |
| **Color Matching** | Tự động khớp màu sắc vùng mở rộng với ảnh gốc |
| **Output lên đến 4K** | Giới hạn đầu ra tối đa 4096px, hỗ trợ upscale 50%/33%/25% |

---

## 🏗️ Kiến trúc hệ thống

```
Final-Web-Demo/
├── 🐍 app.py                    # Gradio entry point (legacy)
├── 🔧 model_manager.py          # Quản lý GPU/CPU swap thông minh
├── 📦 requirements.txt
├── 🐳 Dockerfile                # CUDA 12.8 + Python 3.10 + Node 20
├── 🐳 docker-compose.yml
│
├── 🖼️  task1/                   # Multi-Exposure Fusion
│   ├── model.py                 # UFRetinex + ToneNet loader
│   ├── inference.py             # SIFT alignment + MEF pipeline
│   ├── nets/                    # Mạng Restormer + custom net
│   └── model/phase_1 & phase_2/ # Trọng số đã huấn luyện
│
├── ✏️  task2/                   # Inpainting & Editing
│   ├── model.py                 # SDXL pipeline + Depth model
│   ├── inference.py             # Mask processing + inpaint
│   ├── prompt_enhancer.py       # Qwen-based prompt enhancement
│   └── ckp/                     # LoRA weights (.safetensors)
│
├── 🌅 task3/                    # Outpainting
│   ├── inference_preserve_blend_v2.py  # Core pipeline
│   ├── controlnet_union.py      # Custom ControlNet Union
│   ├── pipeline_fill_sd_xl.py   # SDXL fill pipeline
│   └── lora_best/               # Fine-tuned LoRA adapter
│
└── ⚛️  react_app/
    ├── backend/main.py          # FastAPI server (port 7860)
    └── frontend/src/            # React + Chakra UI + Vite
        ├── components/
        │   ├── Task1Tab.jsx
        │   ├── Task2Tab.jsx
        │   └── Task3Tab.jsx
        └── App.jsx              # Tab routing + health polling
```

---

## 💻 Yêu cầu hệ thống

| Thành phần | Tối thiểu | Khuyến nghị |
|------------|-----------|-------------|
| **GPU** | NVIDIA 8GB VRAM | NVIDIA 16GB+ VRAM |
| **CUDA** | 11.8+ | 12.8 |
| **RAM** | 16 GB | 32 GB |
| **Ổ đĩa** | 20 GB | 50 GB (cho HF cache) |
| **Python** | 3.10 | 3.10 |
| **Docker** | 24.0+ | Latest |

> ⚠️ **Bắt buộc có GPU NVIDIA** với CUDA. Hệ thống sử dụng `torch.cuda` và quản lý VRAM thông qua `ModelManager`.

---

## 🚀 Cài đặt & Khởi chạy

### Phương pháp 1 — Docker (Khuyến nghị)

```bash
# 1. Clone repository
git clone <repo-url>
cd Final-Web-Demo

# 2. Đảm bảo NVIDIA Container Toolkit đã được cài
nvidia-smi   # Kiểm tra GPU

# 3. Khởi chạy với Docker Compose
docker compose up --build

# 4. Truy cập ứng dụng
open http://localhost:7860
```

**Tuỳ chỉnh qua biến môi trường (`.env`):**
```env
PORT=7860
MODEL_PHASE_1=task1/model/phase_1/final.pth
MODEL_PHASE_2=task1/model/phase_2/final_with_tonenet.pth
T2_DEVICE=cuda:0
HF_CACHE=/home/diffusion/.cache/huggingface
```

---

### Phương pháp 2 — Chạy trực tiếp (Local)

```bash
# 1. Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 2. Cài PyTorch với CUDA
pip install torch>=2.0.1 --index-url https://download.pytorch.org/whl/cu128

# 3. Cài các dependencies còn lại
pip install -r requirements.txt

# 4. Build React frontend
cd react_app/frontend
npm install
npm run build
cd ../..

# 5. Khởi chạy server
python -m uvicorn react_app.backend.main:app --host 0.0.0.0 --port 7860
```

> 💡 Để chạy giao diện Gradio (legacy): `python app.py`

---

## 📖 Hướng dẫn sử dụng

### Task 1 — Ghép ảnh đa phơi sáng

1. **Tải ảnh**: Chọn 2+ ảnh cùng cảnh với các mức sáng khác nhau
2. **Căn chỉnh**: Bật/tắt tùy chọn *Căn chỉnh ảnh* (SIFT alignment)
3. **Tăng cường**: Bật *Tăng cường màu sắc - ánh sáng* để áp dụng ToneNet
4. Nhấn **🚀 Ghép & Xử lý**

> 💡 Chỉ cần 1 ảnh nếu chỉ muốn tăng cường ánh sáng qua Phase 2.

### Task 2 — Chỉnh sửa ảnh

1. **Tải ảnh** vào canvas
2. **Vẽ mask** trên vùng muốn chỉnh sửa
3. **Chọn chế độ**: Add / Delete / Replace
4. **Nhập prompt** mô tả nội dung mong muốn (tuỳ chọn: dùng *Enhance Prompt*)
5. Điều chỉnh **Steps** (10–50) và **Strength** (0.1–1.0)
6. Nhấn **▶ Xử lý**

### Task 3 — Mở rộng ảnh

1. **Tải ảnh** gốc
2. **Chọn tỷ lệ đầu ra**: 1:1, 4:3, 16:9, …
3. **Chọn vị trí** ảnh gốc trên canvas (Middle, Left, Right, …)
4. **Nhập prompt** mô tả nội dung vùng mở rộng
5. Xem **Preview** (vùng đỏ = nơi AI sẽ vẽ thêm)
6. Nhấn **🚀 Generate**

---

## ⚙️ Cấu hình nâng cao

### Model Manager — Quản lý VRAM

Hệ thống sử dụng `ModelManager` để tự động chuyển model giữa GPU ↔ CPU khi đổi tab, tránh OOM:

```python
# Preload tất cả models lên CPU RAM khi khởi động
preload_task1_cpu()   # UFRetinex + ToneNet
preload_task2_cpu()   # SDXL pipeline + Depth model
preload_task3_cpu()   # SDXL fill + ControlNet Union

# Khi user nhấn nút xử lý → task đó được đưa lên GPU
manager.activate("task1")  # Push task khác về CPU, đưa task1 lên GPU
```

### SDXL Output Resolutions (Task 3)

| Tỷ lệ | Độ phân giải |
|-------|-------------|
| 1:1   | 1024 × 1024 |
| 4:3   | 1360 × 1024 |
| 3:4   | 1024 × 1360 |
| 16:9  | 1344 × 768  |
| 9:16  | 768 × 1344  |
| Custom | Tuỳ chỉnh (tối đa 4096px) |

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Công nghệ |
|-------|-----------|
| **Frontend** | React 18, Chakra UI, Vite |
| **Backend API** | FastAPI, Uvicorn |
| **AI Framework** | PyTorch 2.0+, Diffusers, Transformers |
| **Computer Vision** | OpenCV, Pillow, NumPy |
| **Models** | UFRetinex, SDXL, ControlNet Union, LoRA |
| **Deployment** | Docker, NVIDIA Container Toolkit, CUDA 12.8 |

</div>

---

<div align="center">

Made with ❤️ by **VietDynamic Team**

</div>
