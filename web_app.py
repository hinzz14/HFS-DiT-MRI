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
loaded_checkpoints_mtime = {}
cache_lock = threading.Lock()

def get_models(acceleration: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with cache_lock:
        if acceleration == 4:
            unet_ckpt = os.path.join(BASE_DIR, "experiments/unet_baseline/checkpoints/epoch=45-step=99912.ckpt")
            nohfs_ckpt = os.path.join(BASE_DIR, "experiments/no_hfs_dit_fm_4x/checkpoints/hfs-dit-fm-epoch=91-val_loss=0.4236.ckpt")
            hfs_ckpt = os.path.join(BASE_DIR, "experiments/hfs_dit_fm_4x/checkpoints/last.ckpt")
        else:
            unet_ckpt = os.path.join(BASE_DIR, "experiments/unet_baseline_8x/checkpoints/epoch=40-step=89052.ckpt")
            nohfs_ckpt = os.path.join(BASE_DIR, "experiments/no_hfs_dit_fm_8x/checkpoints/hfs-dit-fm-epoch=70-val_loss=0.4734.ckpt")
            if not os.path.exists(nohfs_ckpt):
                nohfs_ckpt = os.path.join(BASE_DIR, "experiments/no_hfs_dit_fm_8x/checkpoints/last.ckpt")
            hfs_ckpt = os.path.join(BASE_DIR, "experiments/hfs_dit_fm_8x/checkpoints/last.ckpt")
            
        unet_mtime = os.path.getmtime(unet_ckpt) if os.path.exists(unet_ckpt) else 0
        nohfs_mtime = os.path.getmtime(nohfs_ckpt) if os.path.exists(nohfs_ckpt) else 0
        hfs_mtime = os.path.getmtime(hfs_ckpt) if os.path.exists(hfs_ckpt) else 0
        
        cached = models_cache.get(acceleration)
        
        need_load_unet = cached is None or loaded_checkpoints_mtime.get((acceleration, 'unet')) != unet_mtime
        need_load_nohfs = cached is None or loaded_checkpoints_mtime.get((acceleration, 'nohfs')) != nohfs_mtime
        need_load_hfs = cached is None or loaded_checkpoints_mtime.get((acceleration, 'hfs')) != hfs_mtime
        
        if cached is None:
            unet_model, nohfs_model, hfs_model = None, None, None
        else:
            unet_model, nohfs_model, hfs_model = cached
            
        torch.serialization.add_safe_globals([pathlib.PosixPath])
        
        if need_load_unet:
            print(f"📦 Loading Unet checkpoint: {unet_ckpt}")
            unet_model = UnetModule.load_from_checkpoint(unet_ckpt).eval().to(device)
            loaded_checkpoints_mtime[(acceleration, 'unet')] = unet_mtime
            
        if need_load_nohfs:
            print(f"📦 Loading No-HFS checkpoint: {nohfs_ckpt}")
            nohfs_model = FlowMatchingDiTModule.load_from_checkpoint(nohfs_ckpt).eval().to(device)
            loaded_checkpoints_mtime[(acceleration, 'nohfs')] = nohfs_mtime
            
        if need_load_hfs:
            # First try loading the exact hfs_ckpt
            loaded_successfully = False
            if os.path.exists(hfs_ckpt):
                try:
                    print(f"📦 Loading HFS checkpoint: {hfs_ckpt}")
                    hfs_model = FlowMatchingDiTModule.load_from_checkpoint(hfs_ckpt).eval().to(device)
                    loaded_checkpoints_mtime[(acceleration, 'hfs')] = hfs_mtime
                    loaded_successfully = True
                    print(f"✅ HFS Model loaded from {hfs_ckpt}")
                except Exception as e:
                    print(f"⚠️ Error loading HFS checkpoint {hfs_ckpt}: {e}")
            
            if not loaded_successfully:
                import glob
                hfs_pattern = os.path.join(os.path.dirname(hfs_ckpt), "*.ckpt")
                hfs_files = glob.glob(hfs_pattern)
                # Filter out last.ckpt to avoid double loading the same error if it failed
                hfs_files = [f for f in hfs_files if not f.endswith("last.ckpt")]
                if hfs_files:
                    # Sort by modification time to get the latest completed epoch checkpoint
                    hfs_files = sorted(hfs_files, key=os.path.getmtime, reverse=True)
                    latest_hfs_file = hfs_files[0]
                    file_mtime = os.path.getmtime(latest_hfs_file)
                    
                    if loaded_checkpoints_mtime.get((acceleration, 'hfs_file')) != latest_hfs_file or loaded_checkpoints_mtime.get((acceleration, 'hfs')) != file_mtime:
                        try:
                            print(f"📦 Loading HFS checkpoint: {latest_hfs_file}")
                            hfs_model = FlowMatchingDiTModule.load_from_checkpoint(latest_hfs_file).eval().to(device)
                            loaded_checkpoints_mtime[(acceleration, 'hfs')] = file_mtime
                            loaded_checkpoints_mtime[(acceleration, 'hfs_file')] = latest_hfs_file
                            loaded_successfully = True
                            print(f"✅ HFS Model loaded from {latest_hfs_file}")
                        except Exception as e:
                            print(f"⚠️ Error loading HFS checkpoint {latest_hfs_file}: {e}")
                            
            if hfs_model is None:
                print(f"⚠️ HFS checkpoint not found or failed to load. Mirroring No-HFS.")
                
        models_cache[acceleration] = (unet_model, nohfs_model, hfs_model)
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
        
        # Load models from cache
        unet_model, nohfs_model, hfs_model = get_models(req.acceleration)
        
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
        
        # 5. Run Standard DiT-FM (No HFS) Inference
        t_start_nohfs = time.time()
        std_hfs = x1_image.std(dim=(1, 2, 3), keepdim=True)
        x1_image_norm = x1_image / (std_hfs + 1e-11)
        x1_complex_norm = r2c(x1_image_norm)
        k1_norm = fft2c(x1_complex_norm)
        k1_low_norm = k1_norm * mask 
        
        x_t_norm_nohfs = torch.randn_like(x1_image_norm)
        cond_nohfs = c2r(ifft2c(k1_low_norm)).type(torch.float32)
        
        dt = 1.0 / req.ode_steps
        B = 1
        with torch.no_grad():
            for i in range(req.ode_steps):
                t_val = i * dt
                t_tensor = torch.full((B,), t_val, device=device)
                v_pred = nohfs_model(x_t_norm_nohfs, cond_nohfs, t_tensor)
                x_t_norm_nohfs = x_t_norm_nohfs + v_pred * dt
                
        k_final_nohfs = fft2c(r2c(x_t_norm_nohfs))
        k_final_nohfs_high = k_final_nohfs * (1 - mask)
        x_t_norm_nohfs = c2r(ifft2c(k1_low_norm + k_final_nohfs_high)).type(torch.float32)
        pred_mag_nohfs = magnitude(x_t_norm_nohfs * std_hfs)
        nohfs_time = time.time() - t_start_nohfs
        
        nohfs_norm = torch.clamp(pred_mag_nohfs / mx, 0.0, 1.0)
        nohfs_nmse = (torch.sum((t_norm - nohfs_norm)**2) / (torch.sum(t_norm**2) + 1e-11)).item() * 100
        nohfs_psnr = psnr_calc(nohfs_norm, t_norm).item()
        nohfs_ssim = ssim_calc(nohfs_norm, t_norm).item()
        
        # 6. Run HFS-DiT-FM (Ours) Inference
        if hfs_model is not None:
            t_start_hfs = time.time()
            noise_scale = 1.0
            noise_img = torch.randn_like(x1_image_norm) * noise_scale
            k0_noise = fft2c(r2c(noise_img))
            k0_high = k0_noise * (1 - mask) 
            k0_norm = k1_low_norm + k0_high       
            x_t_norm_hfs = c2r(ifft2c(k0_norm)).type(torch.float32)
            cond_hfs = c2r(ifft2c(k1_low_norm)).type(torch.float32)
            
            with torch.no_grad():
                for i in range(req.ode_steps):
                    t_val = i * dt
                    t_tensor = torch.full((B,), t_val, device=device)
                    v_pred = hfs_model(x_t_norm_hfs, cond_hfs, t_tensor)
                    x_t_norm_hfs = x_t_norm_hfs + v_pred * dt
                    
                    # Data Consistency (DC) tại mỗi bước nhảy để tránh trôi lệch phân phối K-space
                    k_t = fft2c(r2c(x_t_norm_hfs))
                    k_t_high = k_t * (1 - mask)
                    x_t_norm_hfs = c2r(ifft2c(k1_low_norm + k_t_high)).type(torch.float32)
                    
            pred_mag_hfs = magnitude(x_t_norm_hfs * std_hfs)
            hfs_time = time.time() - t_start_hfs
            
            hfs_norm = torch.clamp(pred_mag_hfs / mx, 0.0, 1.0)
            hfs_nmse = (torch.sum((t_norm - hfs_norm)**2) / (torch.sum(t_norm**2) + 1e-11)).item() * 100
            hfs_psnr = psnr_calc(hfs_norm, t_norm).item()
            hfs_ssim = ssim_calc(hfs_norm, t_norm).item()
        else:
            hfs_norm = nohfs_norm.clone()
            pred_mag_hfs = pred_mag_nohfs.clone()
            hfs_nmse = nohfs_nmse
            hfs_psnr = nohfs_psnr
            hfs_ssim = nohfs_ssim
            hfs_time = 0.0
            
        # 7. Compute LPIPS & Laplacian Variance
        lpips_model = get_lpips_fn(device)
        
        gt_lap_var = calculate_laplacian_var(targ_mag)
        zf_lap_var = calculate_laplacian_var(input_mag)
        unet_lap_var = calculate_laplacian_var(pred_mag_unet)
        nohfs_lap_var = calculate_laplacian_var(pred_mag_nohfs)
        hfs_lap_var = calculate_laplacian_var(pred_mag_hfs)
        
        # Prepare inputs for LPIPS
        t_lp = to_lpips_input(targ_mag, mx)
        zf_lp = to_lpips_input(input_mag, mx)
        u_lp = to_lpips_input(pred_mag_unet, mx)
        nh_lp = to_lpips_input(pred_mag_nohfs, mx)
        h_lp = to_lpips_input(pred_mag_hfs, mx)
        
        with torch.no_grad():
            zf_lpips = lpips_model(zf_lp, t_lp).item()
            unet_lpips = lpips_model(u_lp, t_lp).item()
            nohfs_lpips = lpips_model(nh_lp, t_lp).item()
            hfs_lpips = lpips_model(h_lp, t_lp).item()
            
        # 8. Convert numpy maps
        gt_np = t_norm.squeeze().cpu().numpy()
        zf_np = zf_norm.squeeze().cpu().numpy()
        unet_np = unet_norm.squeeze().cpu().numpy()
        nohfs_np = nohfs_norm.squeeze().cpu().numpy()
        hfs_np = hfs_norm.squeeze().cpu().numpy()
        
        unet_err_np = np.abs(gt_np - unet_np)
        nohfs_err_np = np.abs(gt_np - nohfs_np)
        hfs_err_np = np.abs(gt_np - hfs_np)
        
        # Crop region coordinates
        crop_bbox = (120, 220, 110, 210)
        
        return {
            "success": True,
            "images": {
                "gt": to_base64_pil(gt_np),
                "zf": to_base64_pil(zf_np),
                "unet": to_base64_pil(unet_np),
                "nohfs": to_base64_pil(nohfs_np),
                "dit": to_base64_pil(hfs_np),
                "unet_err": error_map_to_base64_pil(unet_err_np),
                "nohfs_err": error_map_to_base64_pil(nohfs_err_np),
                "dit_err": error_map_to_base64_pil(hfs_err_np)
            },
            "metrics": {
                "gt": {"lap_var": gt_lap_var},
                "zf": {"nmse": zf_nmse, "psnr": zf_psnr, "ssim": zf_ssim, "lpips": zf_lpips, "lap_var": zf_lap_var},
                "unet": {"nmse": unet_nmse, "psnr": unet_psnr, "ssim": unet_ssim, "lpips": unet_lpips, "lap_var": unet_lap_var, "time": unet_time},
                "nohfs": {"nmse": nohfs_nmse, "psnr": nohfs_psnr, "ssim": nohfs_ssim, "lpips": nohfs_lpips, "lap_var": nohfs_lap_var, "time": nohfs_time},
                "dit": {"nmse": hfs_nmse, "psnr": hfs_psnr, "ssim": hfs_ssim, "lpips": hfs_lpips, "lap_var": hfs_lap_var, "time": hfs_time}
            }
        }
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
