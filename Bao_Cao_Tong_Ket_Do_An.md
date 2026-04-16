# BÁO CÁO ĐỒ ÁN KỸ THUẬT: KHÔI PHỤC ẢNH MRI BẰNG MÔ HÌNH HFS-DiT FLOW MATCHING

## 1. Đặt Quy Vấn Đề (Introduction)
Chụp Cộng hưởng từ (MRI) là phương pháp chẩn đoán hình ảnh hàng đầu hiện nay, nhưng có nhược điểm lớn là thời gian thu nhận dữ liệu (K-space) rất lâu. Giải pháp phổ biến là **Undersampling (Lấy mẫu thưa)** giúp tăng tốc độ chụp (Gia tốc 4x, 8x), tuy nhiên hệ quả là hình ảnh đầu ra bị artifacts và vỡ nét.
Để giải quyết bài toán Khôi phục ảnh (Reconstruction) từ dữ liệu Undersampled, nghiên cứu này đề xuất một kiến trúc trí tuệ nhân tạo thế hệ mới kết hợp giữa **Diffusion Transformer (DiT)**, lý thuyết **Flow Matching (Optimal Transport)** và cơ chế dồn trọng tâm vào Không gian tần số dao động cao **High-Frequency Space (HFS)**. Mục tiêu là đè bẹp các giới hạn truyền thống của mạng Convolution (U-Net) vốn đang gặp nút thắt ở các mức gia tốc cực hạn.

---

## 2. Phương Pháp Chuyên Sâu (Methodology)

### 2.1. Giới hạn của U-Net Baseline
U-Net Baseline, vốn phụ thuộc vào Convolutional Neural Network (CNN), chịu điểm yếu cố hữu là **Tầm nhìn hạn hẹp (Local Receptive Field)**. Ở độ gia tốc thấp như 4x, U-Net có thể "nối" các chi tiết liền kề khá tốt. Tuy nhiên ở gia tốc 8x (gần như bị đứt gãy hoàn toàn cấu trúc K-space), việc thiếu hụt Receptive Field toàn cục khiến U-Net trở nên bất lực và sinh ra ảnh chụp sụn khớp bị nhòe mờ. 

### 2.2. Đột phá từ Transformer và Flow Matching
Lõi của mạng đề xuất sử dụng **Diffusion Transformer (DiT)** đi kèm bộ tính toán **Flow Matching**.
- **DiT:** Cơ chế Global Attention giúp mạng nhìn xuyên suốt được toàn cảnh bức ảnh, bất chấp mọi mức độ gián đoạn dữ liệu. Trọng số của việc "mất liên kết" ở 8x được bù đắp hoàn toàn bởi Attention.
- **Flow Matching:** Thay vì đi đường vòng khuếch tán nhiễu SDE/DDPM mất hàng ngàn bước, Flow Matching thiết lập một đường bay thẳng (Optimal Transport) nối trực tiếp pha nhiễu ban đầu ($X_0$) với pha ảnh Ground Truth ($X_1$). Đường chim bay này được mô hình hóa bằng Vector Field: $v_t = X_1 - X_0$. Điểm ưu việt là khả năng giải phương trình ODE siêu nhanh với Euler/Heun Solver.

### 2.3. Trái Tim Của Kiến Trúc: HFS (High-Frequency Space)
Đây là đóng góp "ăn tiền" nhất của đồ án. Trong tín hiệu K-space:
- Lõi trung tâm (**Low-Frequency - LF**): Chứa hình hài đại thể của khớp gối (tỉ lệ nhỏ nhưng năng lượng cao nhất). Lúc nào cũng được giữ lại dù ở mức gia tốc nào.
- Vùng ngoài (**High-Frequency - HF**): Chứa độ phân giải, rìa, chi tiết (bị cắt bỏ đến 80-90% khi undersampling).

**Cách Standard DiT (No-HFS) gục ngã:**
Các nghiên cứu trước đây bắt DiT nhảy từ Cục nhiễu tạp (Gaussian Noise) sang vẽ lại cả bức ảnh Y tế. Cách này bất khả thi, vì mạng tốn quá nhiều sức lực để mò mẫm vẽ lại cái "khung khớp gối" (LF) khiến PSNR rớt thê thảm.

**Cách HFS-DiT chiến thắng:**
Chỉ đạo thuật toán bám vào sự thật là: "Khung xương khớp LF đã có sẵn!".
1. Chúng ta lấy LF sẵn có làm khung lõi.
2. Chỉ rắc nhiễu Gaussian $N_{HF}$ vào vùng ngoài (HF) bị thiếu hụt.
3. Dùng toàn bộ tổ hợp này làm điểm xuất phát $X_0$ và cấp thẳng nó cho DiT làm la bàn (Conditioning). 
4. Nhiệm vụ của mô hình giờ đây từ "Sinh lại mặt cắt sụn/khớp" biến thành một nhiệm vụ siêu nhẹ: **"Gọt bỏ cái nhiễu $N_{HF}$ ở viền để trả lại chi tiết nét"**. 

---

## 3. Bản Đồ Tư Duy Kiến Trúc (Architecture Diagram)

Sơ đồ quá trình kiến tạo Flow ban đầu (Forward Process) và Mạng DiT.

```mermaid
graph TD
    subgraph K-Space Physics (Sơ chế Dữ liệu Đầu Vào)
    A[K-space Đầy đủ (Full)] --> B(Mask Undersampling 4x/8x)
    B --> C[Phần Lõi Tần Số Thấp: LF]
    B --> D[Phần Rìa Bị Cắt Bỏ: Khu Vực HF Trống]
    C --> E[X_Zf: Ảnh Mờ (Zero-Filled)]
    
    F[Nhiễu Mù Gaussian thuần túy] --> G(Mask Func Inverted)
    G --> H[Nhiễu Tần Số Cao N_HF]
    
    C -->|Ghép dải tần| H
    H -->|Nghịch đảo FFT2| J[X_0: Điểm Khởi Đầu HFS Flow]
    end

    A --> |Nghịch đảo FFT2| K[X_1: Ảnh Đích Ground Truth]
    
    subgraph Flow Matching Process (Toán Học Vận Chuyển)
    J -.-> |Vector Field v_t| K
    end

    subgraph Mạng Diffusion Transformer - DiT
    E_n[X_t tại bước t] --> P(Patchify & Conv)
    J --> Q{Condition Channel}
    Q --> P
    Time[Thời gian t] --> P
    P --> R[Các khối DiT Block: Self-Attention, AdaLN Zero]
    R --> S(Unpatchify)
    S --> T[Output: Gradient v_pred = X_1 - X_0]
    end
```

---

## 4. Ma Trận Nghiên Cứu Lược Bỏ (Ablation Study Matrix)

Để chứng minh luận điểm, hệ thống được huấn luyện song song và độc lập thành 4 kịch bản tạo thành ma trận đối chứng:
- **Biến số 1 (Cơ chế Mạng):** U-Net Baseline vs DiT Có HFS vs DiT Không HFS.
- **Biến số 2 (Độ nén):** R=4x (Khoảng 8% lõi) vs R=8x (Khoảng 4% lõi).

Mọi mô hình được kiểm chứng khách quan trên Tập Test gồm 3903 lát cắt MRI Singlecoil, giải ODE Flow với 50 epochs.

---

## 5. Bảng Kết Quả Định Lượng (Quantitative Results)

Sự sống còn của HFS và giới hạn của U-Net được trình bày trực quan.

| Mô Hình | Cơ Chế HFS | PSNR (Gia Tốc 4x) | SSIM (Gia Tốc 4x) | PSNR (Gia Tốc 8x) | SSIM (Gia Tốc 8x) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **U-Net** (Baseline)| N/A | 19.85 | 0.6394 | 16.66 | 0.5412 |
| **DiT Flow** (Ablation)| ❌ Không_HFS | 14.27 | 0.4117 | 12.67 | 0.3363 |
| **DiT Flow** (Proposed)| ✅ Có_HFS | **21.16** | **0.6627** | **18.50** | **0.5763** |

> [!NOTE] 
> Biên độ Sai số: Kết quả hiển thị lấy trung bình trên gần 4,000 mặt cắt ngang khớp gối ngẫu nhiên độc lập.

---

## 6. Phân Tích Sự Đột Phá (Discussions & Findings)

Từ bảng số liệu, luận án rút ra 3 phát hiện khoa học mang tính quyết định:

### 6.1. HFS là xương sống sinh tồn của Diffusion
Sự hoán vị rớt thảm hại của "DiT Không HFS" (từ 21.16 dB rớt mốc 14.27 dB) là tiếng chuông cảnh tỉnh cho các thiết kế ngây thơ: Kiến trúc DiT nguyên bản sinh ra cho Generate (Sora/Dall-E) sẽ **hoàn toàn vô giá trị (PSNR < 15 dB)** nếu đem giải bài toán Y Khoa bằng cách đoán nghịch đảo từ Noise Mù. Việc bơm thẳng kênh dẫn đường bằng Cấu trúc Tần số thấp là tối cần thiết.

### 6.2. U-Net đạt cực hạn ở 8x, Nhường ngôi cho Attention
Sức mạnh của HFS-DiT thể hiện rõ trong việc nới rộng vòng cách biệt (Delta gap):
- Tại **4x**: Nó tốt hơn U-Net **1.31 dB**.
- Tại **8x**: Khoảng cách này bị đục thủng xa hơn tới **1.84 dB**.
U-Net bị giảm tới -3.19 dB khả năng suy luận khi thông tin K-space chạm ngưỡng nén cực đại (chỉ còn 4% center). Trong lúc đó, ma trận Self-attention của DiT vẫn giữ được dây liên kết cấu trúc xương khớp xa gần vững vàng.

### 6.3. Kiến trúc Đào Tạo Liền Mạch (End-to-End Elegance)
Khung Framework hiện tại được đóng gói triệt để qua PyTorch Lightning:
- Mọi logic nạp Flow `v_pred = x_1 - x_0` được khép kín trong `training_step`.
- Cơ chế Inference dùng Euler Solvers có kèm theo Data Match (`mask * K-space_org`) giúp bảo toàn tính chung thủy tuyệt đối cho ảnh gốc (Data Consistency).
Lĩnh vực Flow Matching hiện đại đã chứng tỏ được ưu thế Vượt Bậc so với DDPM truyền thống trong mảng Khôi phục Inverse Problems.
