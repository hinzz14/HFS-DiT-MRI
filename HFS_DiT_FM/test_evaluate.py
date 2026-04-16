import os
import argparse
import torch
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import fastmri
from fastmri.data.mri_data import SliceDataset
from fastmri.data.subsample import RandomMaskFunc
from data.transforms import HFSDataTransform
from pl_modules.flow_module import FlowMatchingDiTModule
from utils import r2c, c2r, fft2c, ifft2c

def main(args):
    print(f"🔍 Đang Load Model từ Checkpoint đắt giá: {args.ckpt_path}")
    model = FlowMatchingDiTModule.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.cuda()
    
    # Chuẩn bị luồng Data Validation với Mask 8% lõi (Gia tốc tia x4)
    mask_func = RandomMaskFunc(center_fractions=[0.08], accelerations=[4])
    transform = HFSDataTransform(mask_func)
    
    val_dir = os.path.join(args.data_path, "eval_data")
    dataset = SliceDataset(root=val_dir, transform=transform, challenge="singlecoil")
    
    # Rút ảnh ra test ngẫu nhiên hoặc chỉ định
    print(f"🖼 Đang lấy Slice ảnh thứ {args.slice_index}...")
    x1_image, mask = dataset[args.slice_index]
    x1_image = x1_image.unsqueeze(0).cuda()
    mask = mask.unsqueeze(0).cuda()
    
    B = 1
    
    # CHUẨN HÓA DỮ LIỆU ĐỂ GIỐNG QUÁ TRÌNH TRAINING
    std = x1_image.std(dim=(1, 2, 3), keepdim=True)
    x1_image_norm = x1_image / (std + 1e-11)
    
    # 1. BIẾN ĐỔI CHỒNG K-SPACE (HFS PREPARATION)
    x1_complex = r2c(x1_image_norm)
    k1 = fft2c(x1_complex)
    k1_low = k1 * mask          # <- ĐÂY LÀ PHẦN LÕI ẢNH MÀ MÌNH ĐÃ BIẾT KHI CHỤP!
    
    noise_img = torch.randn_like(x1_image_norm)
    k0_noise = fft2c(r2c(noise_img))
    k0_high = k0_noise * (1 - mask) 
    
    k0 = k1_low + k0_high       # Lắp ghép điểm x_0 chuẩn HFS
    x_t = c2r(ifft2c(k0)).type(torch.float32)
    x0_image = x_t.clone() # Giữ lại x0 để làm reference model condition
    
    print(f"⏳ Bắt đầu giải phương trình ODE (Inference step) với {args.num_steps} nhịp...")
    dt = 1.0 / args.num_steps
    
    # ODE SOLVER (Bộ Giải Euler Cơ Bản Thuần Túy)
    with torch.no_grad():
        for i in range(args.num_steps):
            t_val = i * dt
            t_tensor = torch.full((B,), t_val, device=model.device)
            
            # GỌI MẠNG DIT ĐỂ DỰ ĐOÁN HƯỚNG CHUYỂN ĐỘNG VECTỞ (VECTOR FIELD FLOW)
            # Truyền x_t và x0_image làm điều kiện
            v_pred = model(x_t, x0_image, t_tensor)
            
            # TĨNH TIẾN 1 BƯỚC TỚI BỨC ẢNH CẦN TÌM
            x_t = x_t + v_pred * dt
            
            # -- TIÊM THUỐC ĐẢM BẢO CHẤT LƯỢNG DATA (DATA CONSISTENCY) --
            # Dù máy Flow linh tinh, ta luôn Lôi cổ nó về lại K-space 
            # để ép cứng cái Tần số trung tâm (k1_low) là con số vật lý bất biến!!!
            k_t = fft2c(r2c(x_t))
            k_t_high = k_t * (1 - mask)
            
            k_t_corrected = k1_low + k_t_high # Cứu rỗi chất lượng Ảnh!
            x_t = c2r(ifft2c(k_t_corrected)).type(torch.float32)
            
    # HÀM HIỆN ẢNH (Biến số Phức 2-mặt phẳng thành Số Độ Bão Hoà Magnitude Grayscale 1D)
    def magnitude(x_complex):
        return torch.sqrt(x_complex[:, 0:1, ...]**2 + x_complex[:, 1:2, ...]**2)
        
    # Scale lại theo giá trị ban đầu để khớp với ground truth thật
    x_t_unnorm = x_t * std
    pred_mag = magnitude(x_t_unnorm)
    targ_mag = magnitude(x1_image)
    
    # QUY ĐỔI GIỚI HẠN SCALING (Sắp xếp độ chói từ 0.0 -> 1.0 cực sáng)
    max_val = targ_mag.max() + 1e-7
    pred_mag = pred_mag / max_val
    targ_mag = targ_mag / max_val
    
    # CẮT ĐUÔI TÀN DƯ (Clamp)
    # Lọc bay tất cả những điểm mảnh vỡ nhiễu vọt lố qua ngưỡng 1.0 để PSNR không nổ
    pred_mag = torch.clamp(pred_mag, 0.0, 1.0)
    targ_mag = torch.clamp(targ_mag, 0.0, 1.0)
    
    # GOẠI THƯ VIỆN TEST ĐIỂM SỐ NHƯ BẢN U-NET GỐC CỦA CHỦ NHÂN
    psnr_calc = PeakSignalNoiseRatio(data_range=1.0).cuda()
    ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    
    psnr = psnr_calc(pred_mag.cuda(), targ_mag.cuda())
    ssim = ssim_calc(pred_mag.cuda(), targ_mag.cuda())
    
    print("\n" + "="*60)
    print("💎 KẾT QUẢ ĐÁNH GIÁ RECONSTRUCTION (HFS + DiT + Flow Matching):")
    print(f"   ➤ PSNR : {psnr.item():.2f} dB")
    print(f"   ➤ SSIM : {ssim.item():.4f}")
    print("="*60 + "\n")
    
    # ------------------ KHU XUẤT ẢNH CHO CHỦ NHÂN MỞ XEM NGAY -----------------
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    
    input_mag = magnitude(c2r(ifft2c(k1_low)) * std).squeeze().cpu().numpy()
    plt.imshow(input_mag, cmap='gray')
    plt.title("Input (Nhờn Nát - ACS K-space Freq)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mag.squeeze().cpu().numpy(), cmap='gray')
    plt.title(f"HFS-DiT-FM Reconstructed (PSNR: {psnr.item():.2f})")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(targ_mag.squeeze().cpu().numpy(), cmap='gray')
    plt.title("Ground Truth (Target Chụp Kín)")
    plt.axis('off')
    
    plt.tight_layout()
    save_path = "/home/loipd/MRI_Project/hfs_dit_fm_result.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"🎉 Khui File Ảnh Tươi Sống Trong Vòng Tròn Màu Mới Này: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='Đường dẫn tuyệt đối đến file Trọng_lượng .ckpt mà Bạn Train Rất Lâu')
    parser.add_argument('--data_path', type=str, default='/home/loipd/MRI_Project')
    parser.add_argument('--slice_index', type=int, default=20, help='Vị trí của Slice ảnh mang ra mổ xẻ đánh giá')
    parser.add_argument('--num_steps', type=int, default=50, help='Số mốc thời gian Euler nhảy vi phân (Càng to càng nét rực rỡ nhưng chậm hơn) - SDE cũ dùng hẳn 1000 step')
    args = parser.parse_args()
    main(args)
