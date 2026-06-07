import os
import sys
import io
import time
import base64
import threading
import torch
import numpy as np
import pathlib
from PIL import Image
import matplotlib.cm as cm

# FastAPI Imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add HFS_DiT_FM to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "HFS_DiT_FM"))

import fastmri
from fastmri.data.mri_data import SliceDataset
from fastmri.data.subsample import RandomMaskFunc
from fastmri.pl_modules import UnetModule

from data.transforms import HFSDataTransform
from pl_modules.flow_module import FlowMatchingDiTModule
from utils import r2c, c2r, fft2c, ifft2c

# Initialize FastAPI
app = FastAPI(title="HFS-DiT-FM Reconstruction Studio")

# Mount Static Files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(BASE_DIR, "static/index.html"))

# Lazy load LPIPS model
lpips_fn = None
def get_lpips_fn(device):
    global lpips_fn
    if lpips_fn is None:
        import lpips
        print("📦 Loading LPIPS AlexNet model...")
        lpips_fn = lpips.LPIPS(net='alex').eval().to(device)
    return lpips_fn

def calculate_laplacian_var(img_tensor):
    """Compute Laplacian Variance for a single (1, 1, H, W) magnitude tensor."""
    img_np = img_tensor.squeeze().cpu().numpy()
    img_min = img_np.min()
    img_max = img_np.max()
    img_np = ((img_np - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)
    import cv2
    return float(cv2.Laplacian(img_np, cv2.CV_64F).var())

def to_lpips_input(mag_img, mx):
    """Convert (1, 1, H, W) magnitude to (1, 3, H, W) in [-1, 1] for LPIPS."""
    normed = torch.clamp(mag_img / mx, 0.0, 1.0)
    normed_3ch = normed.repeat(1, 3, 1, 1)  # LPIPS expects 3-channel
    return normed_3ch * 2.0 - 1.0  # scale to [-1, 1]

# Model Caching Setup
models_cache = {}
cache_lock = threading.Lock()

def get_models(acceleration: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with cache_lock:
        if acceleration not in models_cache:
            print(f"📦 Loading checkpoints for {acceleration}x acceleration...")
            torch.serialization.add_safe_globals([pathlib.PosixPath])
            
            if acceleration == 4:
                unet_ckpt = os.path.join(BASE_DIR, "experiments/unet_baseline/checkpoints/epoch=45-step=99912.ckpt")
                dit_ckpt = os.path.join(BASE_DIR, "experiments/hfs_dit_fm/checkpoints/hfs-dit-fm-epoch=96-val_loss=0.0001.ckpt")
                nohfs_ckpt = os.path.join(BASE_DIR, "experiments/no_hfs_dit_fm_4x/checkpoints/hfs-dit-fm-epoch=91-val_loss=0.4236.ckpt")
            else:
                unet_ckpt = os.path.join(BASE_DIR, "experiments/unet_baseline_8x/checkpoints/epoch=40-step=89052.ckpt")
                dit_ckpt = os.path.join(BASE_DIR, "experiments/hfs_dit_fm_8x/checkpoints/hfs-dit-fm-epoch=98-val_loss=0.0001.ckpt")
                nohfs_ckpt = os.path.join(BASE_DIR, "experiments/no_hfs_dit_fm_8x/checkpoints/hfs-dit-fm-epoch=70-val_loss=0.4734.ckpt")
                if not os.path.exists(nohfs_ckpt):
                    nohfs_ckpt = os.path.join(BASE_DIR, "experiments/no_hfs_dit_fm_8x/checkpoints/last.ckpt")
                
            unet_model = UnetModule.load_from_checkpoint(unet_ckpt).eval().to(device)
            dit_model = FlowMatchingDiTModule.load_from_checkpoint(dit_ckpt).eval().to(device)
            nohfs_model = FlowMatchingDiTModule.load_from_checkpoint(nohfs_ckpt).eval().to(device)
            
            models_cache[acceleration] = (unet_model, dit_model, nohfs_model)
            print(f"✅ Models for {acceleration}x loaded successfully.")
        return models_cache[acceleration]

class ReconstructionRequest(BaseModel):
    slice_idx: int
    acceleration: int
    center_fraction: float
    ode_steps: int

def magnitude(x):
    return torch.sqrt(x[:, 0:1, ...]**2 + x[:, 1:2, ...]**2)

def to_base64_pil(arr, crop_bbox=None):
    """
    Stretches contrast by clipping the top/bottom percentiles and 
    converts to a base64 PNG string.
    This creates an extremely sharp clinical-quality visual presentation.
    """
    if crop_bbox is not None:
        y1, y2, x1, x2 = crop_bbox
        arr = arr[y1:y2, x1:x2]
    
    # Contrast Stretching (Percentile clipping)
    vmin = np.percentile(arr, 1)
    vmax = np.percentile(arr, 99.5)
    arr_clipped = np.clip(arr, vmin, vmax)
    arr_norm = (arr_clipped - vmin) / (vmax - vmin + 1e-8)
    
    # Convert to uint8
    arr_uint8 = (arr_norm * 255).astype(np.uint8)
    
    # Encode to PNG
    img = Image.fromarray(arr_uint8, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_b64}"

def error_map_to_base64_pil(err_arr, crop_bbox=None, vmax=0.15):
    """
    Applies JET colormap directly to the error map and encodes to base64.
    """
    if crop_bbox is not None:
        y1, y2, x1, x2 = crop_bbox
        err_arr = err_arr[y1:y2, x1:x2]
        
    err_norm = np.clip(err_arr / vmax, 0.0, 1.0)
    err_rgba = cm.jet(err_norm)
    err_rgb = (err_rgba[..., :3] * 255).astype(np.uint8)
    
    img = Image.fromarray(err_rgb, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_b64}"

@app.post("/api/reconstruct")
def reconstruct(req: ReconstructionRequest):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load models (lazy/cached)
        unet_model, dit_model, nohfs_model = get_models(req.acceleration)
        
        # 2. Setup Transform and Dataset
        mask_func = RandomMaskFunc(center_fractions=[req.center_fraction], accelerations=[req.acceleration])
        transform = HFSDataTransform(mask_func)
        
        val_dir = os.path.join(BASE_DIR, "data/singlecoil_val")
        dataset = SliceDataset(root=val_dir, transform=transform, challenge="singlecoil")
        
        # Check slice index validity
        if req.slice_idx < 0 or req.slice_idx >= len(dataset):
            raise HTTPException(status_code=400, detail=f"Slice index must be between 0 and {len(dataset)-1}")
            
        x1_image, mask = dataset[req.slice_idx]
        x1_image = x1_image.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        
        # 3. Setup Ground Truth & Zero-Filled
        targ_mag = magnitude(x1_image)
        
        x1_complex = r2c(x1_image)
        k1 = fft2c(x1_complex)
        k1_low = k1 * mask 
        undersampled_complex = c2r(ifft2c(k1_low)).type(torch.float32)
        input_mag = magnitude(undersampled_complex)
        
        # Normalize and compute Zero-filled metrics
        mx = targ_mag.max() + 1e-7
        t_norm = torch.clamp(targ_mag / mx, 0.0, 1.0)
        zf_norm = torch.clamp(input_mag / mx, 0.0, 1.0)
        
        zf_nmse = (torch.sum((t_norm - zf_norm)**2) / (torch.sum(t_norm**2) + 1e-11)).item() * 100
        # Compute PSNR & SSIM for Zero-filled
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
        psnr_calc = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        
        zf_psnr = psnr_calc(zf_norm, t_norm).item()
        zf_ssim = ssim_calc(zf_norm, t_norm).item()
        
        # 4. Run U-Net Inference
        t_start = time.time()
        unet_input = input_mag.squeeze(1)
        mean = unet_input.mean(dim=(-2, -1), keepdim=True)
        std = unet_input.std(dim=(-2, -1), keepdim=True)
        unet_input_norm = (unet_input - mean) / (std + 1e-11)
        unet_input_norm = unet_input_norm.clamp(-6, 6)
        with torch.no_grad():
            unet_output_norm = unet_model(unet_input_norm)
        unet_output = unet_output_norm * std + mean
        pred_mag_unet = unet_output.unsqueeze(1)
        unet_time = time.time() - t_start
        
        unet_norm = torch.clamp(pred_mag_unet / mx, 0.0, 1.0)
        unet_nmse = (torch.sum((t_norm - unet_norm)**2) / (torch.sum(t_norm**2) + 1e-11)).item() * 100
        unet_psnr = psnr_calc(unet_norm, t_norm).item()
        unet_ssim = ssim_calc(unet_norm, t_norm).item()
        
        # 5. Run HFS DiT FM Inference
        t_start = time.time()
        std_hfs = x1_image.std(dim=(1, 2, 3), keepdim=True)
        x1_image_norm = x1_image / (std_hfs + 1e-11)
        x1_complex_norm = r2c(x1_image_norm)
        k1_norm = fft2c(x1_complex_norm)
        k1_low_norm = k1_norm * mask 
        
        # HFS conditioning (low freq + noise high freq)
        noise_scale = 0.1
        noise_img = torch.randn_like(x1_image_norm) * noise_scale
        k0_noise = fft2c(r2c(noise_img))
        k0_high = k0_noise * (1 - mask) 
        k0_norm = k1_low_norm + k0_high       
        x_t_norm = c2r(ifft2c(k0_norm)).type(torch.float32)
        cond = x_t_norm.clone()
        
        dt = 1.0 / req.ode_steps
        B = 1
        with torch.no_grad():
            for i in range(req.ode_steps):
                t_val = i * dt
                t_tensor = torch.full((B,), t_val, device=device)
                v_pred = dit_model(x_t_norm, cond, t_tensor)
                x_t_proj = x_t_norm + v_pred * dt
                # Consistency
                k_t = fft2c(r2c(x_t_proj))
                k_t_high = k_t * (1 - mask)
                x_t_norm = c2r(ifft2c(k1_low_norm + k_t_high)).type(torch.float32)
                
        pred_mag_hfs = magnitude(x_t_norm * std_hfs)
        dit_time = time.time() - t_start
        
        dit_norm = torch.clamp(pred_mag_hfs / mx, 0.0, 1.0)
        dit_nmse = (torch.sum((t_norm - dit_norm)**2) / (torch.sum(t_norm**2) + 1e-11)).item() * 100
        dit_psnr = psnr_calc(dit_norm, t_norm).item()
        dit_ssim = ssim_calc(dit_norm, t_norm).item()
        
        # 6. Run Standard DiT FM Inference (No HFS)
        t_start_nohfs = time.time()
        x_t_norm_nohfs = torch.randn_like(x1_image_norm)
        cond_nohfs = c2r(ifft2c(k1_low_norm)).type(torch.float32)
        
        with torch.no_grad():
            for i in range(req.ode_steps):
                t_val = i * dt
                t_tensor = torch.full((B,), t_val, device=device)
                v_pred = nohfs_model(x_t_norm_nohfs, cond_nohfs, t_tensor)
                x_t_proj = x_t_norm_nohfs + v_pred * dt
                # Consistency
                k_t = fft2c(r2c(x_t_proj))
                k_t_high = k_t * (1 - mask)
                x_t_norm_nohfs = c2r(ifft2c(k1_low_norm + k_t_high)).type(torch.float32)
                
        pred_mag_nohfs = magnitude(x_t_norm_nohfs * std_hfs)
        nohfs_time = time.time() - t_start_nohfs
        
        nohfs_norm = torch.clamp(pred_mag_nohfs / mx, 0.0, 1.0)
        nohfs_nmse = (torch.sum((t_norm - nohfs_norm)**2) / (torch.sum(t_norm**2) + 1e-11)).item() * 100
        nohfs_psnr = psnr_calc(nohfs_norm, t_norm).item()
        nohfs_ssim = ssim_calc(nohfs_norm, t_norm).item()
        
        # 7. Compute LPIPS & Laplacian Variance
        lpips_model = get_lpips_fn(device)
        
        gt_lap_var = calculate_laplacian_var(targ_mag)
        zf_lap_var = calculate_laplacian_var(input_mag)
        unet_lap_var = calculate_laplacian_var(pred_mag_unet)
        dit_lap_var = calculate_laplacian_var(pred_mag_hfs)
        nohfs_lap_var = calculate_laplacian_var(pred_mag_nohfs)
        
        # Prepare inputs for LPIPS
        t_lp = to_lpips_input(targ_mag, mx)
        zf_lp = to_lpips_input(input_mag, mx)
        u_lp = to_lpips_input(pred_mag_unet, mx)
        h_lp = to_lpips_input(pred_mag_hfs, mx)
        nohfs_lp = to_lpips_input(pred_mag_nohfs, mx)
        
        with torch.no_grad():
            zf_lpips = lpips_model(zf_lp, t_lp).item()
            unet_lpips = lpips_model(u_lp, t_lp).item()
            dit_lpips = lpips_model(h_lp, t_lp).item()
            nohfs_lpips = lpips_model(nohfs_lp, t_lp).item()
            
        # 8. Convert numpy maps
        gt_np = t_norm.squeeze().cpu().numpy()
        zf_np = zf_norm.squeeze().cpu().numpy()
        unet_np = unet_norm.squeeze().cpu().numpy()
        dit_np = dit_norm.squeeze().cpu().numpy()
        nohfs_np = nohfs_norm.squeeze().cpu().numpy()
        
        unet_err_np = np.abs(gt_np - unet_np)
        dit_err_np = np.abs(gt_np - dit_np)
        nohfs_err_np = np.abs(gt_np - nohfs_np)
        
        # Crop region coordinates
        crop_bbox = (120, 220, 110, 210)
        
        return {
            "success": True,
            "images": {
                "gt": to_base64_pil(gt_np),
                "zf": to_base64_pil(zf_np),
                "unet": to_base64_pil(unet_np),
                "nohfs": to_base64_pil(nohfs_np),
                "dit": to_base64_pil(dit_np),
                "gt_crop": to_base64_pil(gt_np, crop_bbox),
                "zf_crop": to_base64_pil(zf_np, crop_bbox),
                "unet_crop": to_base64_pil(unet_np, crop_bbox),
                "nohfs_crop": to_base64_pil(nohfs_np, crop_bbox),
                "dit_crop": to_base64_pil(dit_np, crop_bbox),
                "unet_err": error_map_to_base64_pil(unet_err_np),
                "nohfs_err": error_map_to_base64_pil(nohfs_err_np),
                "dit_err": error_map_to_base64_pil(dit_err_np)
            },
            "metrics": {
                "gt": {"lap_var": gt_lap_var},
                "zf": {"nmse": zf_nmse, "psnr": zf_psnr, "ssim": zf_ssim, "lpips": zf_lpips, "lap_var": zf_lap_var},
                "unet": {"nmse": unet_nmse, "psnr": unet_psnr, "ssim": unet_ssim, "lpips": unet_lpips, "lap_var": unet_lap_var, "time": unet_time},
                "nohfs": {"nmse": nohfs_nmse, "psnr": nohfs_psnr, "ssim": nohfs_ssim, "lpips": nohfs_lpips, "lap_var": nohfs_lap_var, "time": nohfs_time},
                "dit": {"nmse": dit_nmse, "psnr": dit_psnr, "ssim": dit_ssim, "lpips": dit_lpips, "lap_var": dit_lap_var, "time": dit_time}
            }
        }
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
