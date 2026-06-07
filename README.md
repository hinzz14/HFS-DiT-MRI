# HFS-DiT-FM MRI Reconstruction Studio

Hệ thống dashboard so sánh trực quan hiệu năng phục dựng ảnh MRI khớp gối giữa các mô hình:
1. **Zero-Filled** (Ảnh nhiễu ban đầu do thiếu tần số)
2. **U-Net Baseline** (Mô hình hồi quy truyền thống)
3. **Standard DiT-FM (No HFS)** (Khuếch tán Flow Matching cơ bản)
4. **HFS-DiT-FM (Ours)** (Giải pháp đề xuất tích hợp HFS và nhất quán K-space)

---

## 📂 Cấu trúc thư mục dự án trên máy mới

Sau khi clone repo từ GitHub về máy mới, bạn cần đặt các thư mục dữ liệu và checkpoints đã tải sẵn vào đúng cấu trúc như sau:

```text
MRI_Project/
├── HFS_DiT_FM/           # Mã nguồn mô hình
├── fastMRI/              # Thư viện fastMRI local
├── static/               # Giao diện web app (HTML/CSS/JS)
├── web_app.py            # Script chạy FastAPI backend
├── requirements.txt      # File thư viện cần cài đặt
├── data/
│   └── singlecoil_val/   # Đặt các tệp .h5 của tập Validation tại đây
└── experiments/          # Đặt các thư mục checkpoints của mô hình tại đây
    ├── unet_baseline/
    ├── unet_baseline_8x/
    ├── hfs_dit_fm/
    ├── hfs_dit_fm_8x/
    ├── no_hfs_dit_fm_4x/
    └── no_hfs_dit_fm_8x/
```

---

## 🛠️ Hướng dẫn cài đặt & Chạy trên máy mới

### Bước 1: Tạo môi trường Conda mới
Khởi tạo môi trường ảo Python 3.9 bằng Conda:
```bash
conda create -n dit_mri python=3.9 -y
conda activate dit_mri
```

### Bước 2: Cài đặt các thư viện cần thiết
Sử dụng file `requirements.txt` để tự động cài đặt toàn bộ dependencies (bao gồm PyTorch, FastAPI, TorchMetrics, LPIPS, OpenCV, v.v.):
```bash
pip install -r requirements.txt
```

### Bước 3: Khởi chạy ứng dụng Web
Chạy lệnh uvicorn để bật server:
```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000
```
Truy cập vào địa chỉ `http://localhost:8000` trên trình duyệt để sử dụng.

---
