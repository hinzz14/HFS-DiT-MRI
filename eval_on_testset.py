import os
import sys
import argparse
import torch
import pathlib
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# Add HFS_DiT_FM to sys path so we can import its modules
sys.path.append("/home/loipd/MRI_Project/HFS_DiT_FM")

# Import fastMRI
import fastmri
from fastmri.data.mri_data import SliceDataset
from fastmri.data.subsample import RandomMaskFunc
from fastmri.pl_modules import UnetModule

# Import HFS DiT FM 
from data.transforms import HFSDataTransform
from pl_modules.flow_module import FlowMatchingDiTModule
from utils import r2c, c2r, fft2c, ifft2c

def magnitude(x_complex):
    """Tính magnitude từ ảnh complex chuẩn của HFS (B, 2, H, W)"""
    return torch.sqrt(x_complex[:, 0:1, ...]**2 + x_complex[:, 1:2, ...]**2)

def main(args):
    print("="*60)
    print("🤖 BẮT ĐẦU ĐÁNH GIÁ 2 MÔ HÌNH TRÊN TOÀN BỘ TẬP TEST")
    print("="*60)
    
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"📦 Đang Load Model U-Net baseline từ:\n   {args.unet_ckpt}")
    unet_model = UnetModule.load_from_checkpoint(args.unet_ckpt)
    unet_model.eval()
    unet_model.to(device)
    
    print(f"📦 Đang Load Model HFS DiT FM từ:\n   {args.hfs_ckpt}")
    hfs_model = FlowMatchingDiTModule.load_from_checkpoint(args.hfs_ckpt)
    
    # Ép dùng mạng chính (Net) thay vì mạng mô phỏng (EMA) do EMA update quá chậm ở 20 epoch đầu
    if args.disable_ema:
        hfs_model.ema_net = None

    hfs_model.eval()
    hfs_model.to(device)
    
    # Sử dụng HFSDataTransform làm chuẩn vì nó crop về 320x320 và chuẩn hóa
    center_frac = 0.08 if args.acceleration == 4 else 0.04
    mask_func = RandomMaskFunc(center_fractions=[center_frac], accelerations=[args.acceleration])
    transform = HFSDataTransform(mask_func)
    
    print(f"📂 Đang load tập dữ liệu Test từ:\n   {args.data_path}")
    dataset = SliceDataset(
        root=args.data_path,
        transform=transform,
        challenge="singlecoil"
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        drop_last=False
    )
    
    psnr_calc = PeakSignalNoiseRatio(data_range=1.0, dim=(1, 2, 3)).to(device)
    ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    total_psnr_unet = 0.0
    total_ssim_unet = 0.0
    total_psnr_hfs = 0.0
    total_ssim_hfs = 0.0
    num_samples = 0
    
    print(f"Số lượng slice cần đánh giá: {len(dataset)}")
    print(f"Sử dụng Batch Size: {args.batch_size}")
    
    for batch_idx, (x1_image, mask) in enumerate(tqdm(dataloader, desc="Đang đánh giá")):
        x1_image = x1_image.to(device)
        mask = mask.to(device)
        
        B = x1_image.shape[0]
        
        # 1. TIỀN XỬ LÝ CHUNG VÀ LẤY ẢNH ĐẦU VÀO, TRẢ LỜI ĐÁP ÁN
        # Tạo K-space nhiễu thiếu điểm chóp
        x1_complex = r2c(x1_image)
        k1 = fft2c(x1_complex)
        k1_low = k1 * mask 
        
        # Tính Ground Truth Magnitude
        targ_mag = magnitude(x1_image)
        targ_mag_clamp = torch.clamp(targ_mag, 0.0, 1.0)
        
        # Tính Input Zero-Filled Magnitude (Nhiễu)
        undersampled_complex = c2r(ifft2c(k1_low)).type(torch.float32)
        input_mag = magnitude(undersampled_complex)
        
        # 2. CHẠY U-NET BASELINE
        # Chuẩn bị input cho U-Net: (B, H, W)
        unet_input = input_mag.squeeze(1) # (B, H, W)
        mean = unet_input.mean(dim=(-2, -1), keepdim=True)
        std = unet_input.std(dim=(-2, -1), keepdim=True)
        
        unet_input_norm = (unet_input - mean) / (std + 1e-11)
        unet_input_norm = unet_input_norm.clamp(-6, 6)
        
        with torch.no_grad():
            unet_output_norm = unet_model(unet_input_norm)
            
        # Un-normalize U-Net output
        unet_output = unet_output_norm * std + mean
        pred_mag_unet = unet_output.unsqueeze(1) # (B, 1, H, W)
        pred_mag_unet_clamp = torch.clamp(pred_mag_unet, 0.0, 1.0)
        
        # 3. CHẠY HFS DiT FLOW MATCHING
        # Cần scale cho HFS model
        std_hfs = x1_image.std(dim=(1, 2, 3), keepdim=True)
        x1_image_norm = x1_image / (std_hfs + 1e-11)
        
        x1_complex_norm = r2c(x1_image_norm)
        k1_norm = fft2c(x1_complex_norm)
        k1_low_norm = k1_norm * mask 
        
        # Kênh Condition lúc nào cũng là Low-Frequency Zero-Filled dành cho bản Không HFS
        x_zf_cond = c2r(ifft2c(k1_low_norm)).type(torch.float32)
        
        # Thiết lập điểm X_0 tùy theo việc dùng HFS hay Standard Flow Matching
        if not args.no_hfs:
            noise_scale = 0.1
            noise_img = torch.randn_like(x1_image_norm) * noise_scale
            k0_noise = fft2c(r2c(noise_img))
            k0_high = k0_noise * (1 - mask) 
            k0_norm = k1_low_norm + k0_high       
            x_t_norm = c2r(ifft2c(k0_norm)).type(torch.float32)
            cond = c2r(ifft2c(k0_norm)).type(torch.float32) # Sửa lỗi chí mạng: HFS bắt buộc lấy x0 làm Cond
        else:
            x_t_norm = torch.randn_like(x1_image_norm)
            cond = x_zf_cond
            
        x0_image_norm = x_t_norm.clone()
        
        dt = 1.0 / args.num_steps
        with torch.no_grad():
            for i in range(args.num_steps):
                t_val = i * dt
                t_tensor = torch.full((B,), t_val, device=device)
                
                # Bước 1: Tính Vector V hiện tại (Euler)
                v_pred = hfs_model(x_t_norm, cond, t_tensor)
                x_t_proj = x_t_norm + v_pred * dt
                
                # Chèn Data Consistency cho nháp Euler
                k_t = fft2c(r2c(x_t_proj))
                k_t_high = k_t * (1 - mask)
                x_t_proj = c2r(ifft2c(k1_low_norm + k_t_high)).type(torch.float32)
                
                if args.solver == 'heun' and i < args.num_steps - 1:
                    # Bước 2: Hiệu chuẩn Heun (Đoán xa thêm 1 bước tính vector V ở tương lai)
                    t_next = t_tensor + dt
                    v_pred_next = hfs_model(x_t_proj, cond, t_next)
                    
                    # Lấy trung bình cộng 2 vector
                    v_heun = (v_pred + v_pred_next) / 2.0
                    x_t_norm = x_t_norm + v_heun * dt
                    
                    # Data Consistency sau chót
                    k_t_final = fft2c(r2c(x_t_norm))
                    k_t_high_final = k_t_final * (1 - mask)
                    x_t_norm = c2r(ifft2c(k1_low_norm + k_t_high_final)).type(torch.float32)
                else:
                    x_t_norm = x_t_proj
                
        x_t = x_t_norm * std_hfs
        pred_mag_hfs = magnitude(x_t)
        pred_mag_hfs_clamp = torch.clamp(pred_mag_hfs, 0.0, 1.0)
        
        # 4. TÍNH ĐIỂM
        # torchmetrics trả về giá trị trung bình của batch
        # Tuy nhiên ta tính tổng điểm rồi chia cho số lượng samples 
        psnr_unet_val = psnr_calc(pred_mag_unet_clamp, targ_mag_clamp) * B
        ssim_unet_val = ssim_calc(pred_mag_unet_clamp, targ_mag_clamp) * B
        
        psnr_hfs_val = psnr_calc(pred_mag_hfs_clamp, targ_mag_clamp) * B
        ssim_hfs_val = ssim_calc(pred_mag_hfs_clamp, targ_mag_clamp) * B
        
        total_psnr_unet += psnr_unet_val.item()
        total_ssim_unet += ssim_unet_val.item()
        total_psnr_hfs += psnr_hfs_val.item()
        total_ssim_hfs += ssim_hfs_val.item()
        
        num_samples += B
        
    avg_psnr_unet = total_psnr_unet / num_samples
    avg_ssim_unet = total_ssim_unet / num_samples
    avg_psnr_hfs = total_psnr_hfs / num_samples
    avg_ssim_hfs = total_ssim_hfs / num_samples
    
    print("\n" + "="*50)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ TRÊN TOÀN BỘ TẬP DỮ LIỆU TEST")
    print("="*50)
    print(f"1️⃣ U-Net Baseline     : Average PSNR = {avg_psnr_unet:.2f} dB | Average SSIM = {avg_ssim_unet:.4f}")
    print(f"2️⃣ HFS DiT FM (Flow)  : Average PSNR = {avg_psnr_hfs:.2f} dB | Average SSIM = {avg_ssim_hfs:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dùng đường dẫn chuẩn cho baseline
    parser.add_argument('--unet_ckpt', type=str, default='/home/loipd/MRI_Project/experiments/unet_baseline/checkpoints/epoch=45-step=99912.ckpt')
    # Lưu ý: checkpoint này lấy tại thư mục Projec do typo trước đó, nếu sau này train lại, hãy update đường dẫn này!
    parser.add_argument('--hfs_ckpt', type=str, default='/home/loipd/MRI_Project/experiments/hfs_dit_fm/checkpoints/hfs-dit-fm-epoch=96-val_loss=0.0001.ckpt')
    # Chỉ định tập test data mới 
    parser.add_argument('--data_path', type=str, default='/home/loipd/MRI_Project/data/singlecoil_test')
    # Thêm tham số batch_size
    parser.add_argument('--batch_size', type=int, default=32, help='Kích thước batch. Mặc định là 4 để không tràn RAM GPU, bạn có thể tăng lên 8 nếu GPU còn RAM.')
    parser.add_argument('--num_steps', type=int, default=50, help='Số bước lấy mẫu ODE của Flow Matching')
    parser.add_argument('--solver', type=str, default='euler', choices=['euler', 'heun'], help='Loại ODE solver để sinh ảnh')
    parser.add_argument('--disable_ema', action='store_true', help='Vô hiệu hóa mạng EMA để test thẳng mạng chính')
    parser.add_argument('--no_hfs', action='store_true', help='Tắt HFS để test Baseline Standard Flow Matching')
    parser.add_argument('--acceleration', type=int, default=4, choices=[4, 8], help='Mức độ gia tốc cắt K-space R=4 hoặc R=8')
    args = parser.parse_args()
    main(args)
