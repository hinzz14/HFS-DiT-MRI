import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import pathlib

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

def run_inference(model, model_type, no_hfs, x1_image, input_mag, mask, num_steps):
    B = 1
    if model_type == 'unet':
        unet_input = input_mag.squeeze(1) # (B, H, W)
        mean = unet_input.mean(dim=(-2, -1), keepdim=True)
        std = unet_input.std(dim=(-2, -1), keepdim=True)
        
        unet_input_norm = (unet_input - mean) / (std + 1e-11)
        unet_input_norm = unet_input_norm.clamp(-6, 6)
        
        with torch.no_grad():
            unet_output_norm = model(unet_input_norm)
            
        unet_output = unet_output_norm * std + mean
        pred_mag_unet = unet_output.unsqueeze(1) # (B, 1, H, W)
        return torch.clamp(pred_mag_unet, 0.0, 1.0)
        
    elif model_type == 'dit':
        std_hfs = x1_image.std(dim=(1, 2, 3), keepdim=True)
        x1_image_norm = x1_image / (std_hfs + 1e-11)
        
        x1_complex_norm = r2c(x1_image_norm)
        k1_norm = fft2c(x1_complex_norm)
        k1_low_norm = k1_norm * mask 
        
        x_zf_cond = c2r(ifft2c(k1_low_norm)).type(torch.float32)
        
        if not no_hfs:
            noise_scale = 0.1
            noise_img = torch.randn_like(x1_image_norm) * noise_scale
            k0_noise = fft2c(r2c(noise_img))
            k0_high = k0_noise * (1 - mask) 
            k0_norm = k1_low_norm + k0_high       
            x_t_norm = c2r(ifft2c(k0_norm)).type(torch.float32)
            cond = c2r(ifft2c(k0_norm)).type(torch.float32)
        else:
            x_t_norm = torch.randn_like(x1_image_norm)
            cond = x_zf_cond
            
        dt = 1.0 / num_steps
        with torch.no_grad():
            for i in range(num_steps):
                t_val = i * dt
                t_tensor = torch.full((B,), t_val, device=model.device)
                
                v_pred = model(x_t_norm, cond, t_tensor)
                x_t_norm = x_t_norm + v_pred * dt
                
                # Data Consistency
                k_t = fft2c(r2c(x_t_norm))
                k_t_high = k_t * (1 - mask)
                k_t_corrected = k1_low_norm + k_t_high 
                x_t_norm = c2r(ifft2c(k_t_corrected)).type(torch.float32)
                
        x_t = x_t_norm * std_hfs
        pred_mag_hfs = magnitude(x_t)
        return torch.clamp(pred_mag_hfs, 0.0, 1.0)

def load_model(ckpt_path, model_type):
    if model_type == 'unet':
        model = UnetModule.load_from_checkpoint(ckpt_path)
    else:
        model = FlowMatchingDiTModule.load_from_checkpoint(ckpt_path)
    model.eval()
    model.cuda()
    return model

def main(args):
    print("="*60)
    print("🤖 BẮT ĐẦU QUÁ TRÌNH TRỌNG TÀI ĐÁNH GIÁ 2 MÔ HÌNH BẤT KỲ")
    print("="*60)
    
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    
    print(f"📦 Đang Load Model 1 ({args.model1_name}) từ:\n   {args.model1_ckpt}")
    model1 = load_model(args.model1_ckpt, args.model1_type)
    
    print(f"📦 Đang Load Model 2 ({args.model2_name}) từ:\n   {args.model2_ckpt}")
    model2 = load_model(args.model2_ckpt, args.model2_type)
    
    # Cấu hình mặt nạ gia tốc
    center_frac = 0.08 if args.acceleration == 4 else 0.04
    mask_func = RandomMaskFunc(center_fractions=[center_frac], accelerations=[args.acceleration])
    transform = HFSDataTransform(mask_func)
    
    val_dir = os.path.join(args.data_path, "eval_data")
    dataset = SliceDataset(
        root=val_dir,
        transform=transform,
        challenge="singlecoil"
    )
    
    slice_idx = args.slice_idx
    print(f"🖼 Đang lấy Slice ảnh thứ {slice_idx} từ Dataset...")
    x1_image, mask = dataset[slice_idx]
    
    # Đưa lên GPU
    x1_image = x1_image.unsqueeze(0).cuda()
    mask = mask.unsqueeze(0).cuda()
    
    # ---------------------------------------------------------
    # 1. TIỀN XỬ LÝ CHUNG
    # ---------------------------------------------------------
    x1_complex = r2c(x1_image)
    k1 = fft2c(x1_complex)
    k1_low = k1 * mask 
    
    targ_mag = magnitude(x1_image)
    targ_mag_clamp = torch.clamp(targ_mag, 0.0, 1.0)
    
    undersampled_complex = c2r(ifft2c(k1_low)).type(torch.float32)
    input_mag = magnitude(undersampled_complex)
    input_mag_clamp = torch.clamp(input_mag, 0.0, 1.0)
    
    # ---------------------------------------------------------
    # 2. CHẠY INFERENCE CHO MODEL 1
    # ---------------------------------------------------------
    print(f"🧠 {args.model1_name} đang vào việc...")
    pred1_mag = run_inference(model1, args.model1_type, args.model1_no_hfs, x1_image, input_mag_clamp, mask, args.num_steps)
    
    # ---------------------------------------------------------
    # 3. CHẠY INFERENCE CHO MODEL 2
    # ---------------------------------------------------------
    print(f"⏳ {args.model2_name} đang vào việc...")
    pred2_mag = run_inference(model2, args.model2_type, args.model2_no_hfs, x1_image, input_mag_clamp, mask, args.num_steps)

    # ---------------------------------------------------------
    # 4. CHẤM ĐIỂM (EVALUATION METRICS)
    # ---------------------------------------------------------
    print("📊 Đang chấm điểm bài thi...")
    psnr_calc = PeakSignalNoiseRatio(data_range=1.0).cuda()
    ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    
    psnr1 = psnr_calc(pred1_mag, targ_mag_clamp)
    ssim1 = ssim_calc(pred1_mag, targ_mag_clamp)
    
    psnr2 = psnr_calc(pred2_mag, targ_mag_clamp)
    ssim2 = ssim_calc(pred2_mag, targ_mag_clamp)
    
    # ---------------------------------------------------------
    # 5. XUẤT HÌNH ẢNH RA LÒ
    # ---------------------------------------------------------
    print("🎨 Đang vẽ biểu đồ so sánh...")
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(input_mag_clamp.squeeze().cpu().numpy(), cmap='bone')
    plt.title(f"Input ({args.acceleration}x Acceleration)")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(pred1_mag.squeeze().cpu().numpy(), cmap='bone')
    plt.title(f"{args.model1_name}\nPSNR: {psnr1.item():.2f} | SSIM: {ssim1.item():.4f}")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(pred2_mag.squeeze().cpu().numpy(), cmap='bone')
    plt.title(f"{args.model2_name}\nPSNR: {psnr2.item():.2f} | SSIM: {ssim2.item():.4f}")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(targ_mag_clamp.squeeze().cpu().numpy(), cmap='bone')
    plt.title("Ground Truth (Target)")
    plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(args.data_path, "model_comparison_result.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    print(f"🎉 Xong! Bản so sánh vĩ đại đã được xuất tại:\n   --> {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script so sánh Đa Năng cho mọi Model (Unet, DiT HFS, DiT No-HFS)")
    
    # Cấu hình Model 1
    parser.add_argument('--model1_ckpt', type=str, required=True, help='Đường dẫn Checkpoint Model 1')
    parser.add_argument('--model1_type', type=str, choices=['unet', 'dit'], required=True, help='Kiểu Model 1')
    parser.add_argument('--model1_name', type=str, default='Model 1', help='Tên hiển thị Model 1')
    parser.add_argument('--model1_no_hfs', action='store_true', help='Truyền cờ này nếu Model 1 là DiT Không có HFS')
    
    # Cấu hình Model 2
    parser.add_argument('--model2_ckpt', type=str, required=True, help='Đường dẫn Checkpoint Model 2')
    parser.add_argument('--model2_type', type=str, choices=['unet', 'dit'], required=True, help='Kiểu Model 2')
    parser.add_argument('--model2_name', type=str, default='Model 2', help='Tên hiển thị Model 2')
    parser.add_argument('--model2_no_hfs', action='store_true', help='Truyền cờ này nếu Model 2 là DiT Không có HFS')
    
    # Môi trường chung
    parser.add_argument('--data_path', type=str, default='/home/loipd/MRI_Project')
    parser.add_argument('--acceleration', type=int, choices=[4, 8], default=4, help='Gia tốc muốn Test: 4 hay 8')
    parser.add_argument('--slice_idx', type=int, default=20, help='Vị trí lát cắt MRI (20 là vị trí giữa, thấy mô khớp rõ nhất)')
    parser.add_argument('--num_steps', type=int, default=50, help='Số bước Flow Matching')
    
    args = parser.parse_args()
    main(args)
