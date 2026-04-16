import torch
import fastmri
from fastmri.data import transforms as T
import numpy as np

class HFSDataTransform:
    """
    Tiền xử lý dữ liệu cho Flow Matching MRI 2-Kênh K-space
    """
    def __init__(self, mask_func):
        self.mask_func = mask_func

    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        # 1. Xử lý K-space vật lý
        kspace_torch = T.to_tensor(kspace) # (H, W, 2)
        
        # Sắp xếp lại luồng ma trận
        # FastMRI to_tensor sinh ra (H, W, 2). Mình cần (2, H, W)
        kspace_torch = kspace_torch.permute(2, 0, 1) # (2, H, W)
        
        # 2. Xén (Crop) thành kích thước chẵn (320x320)
        # Tại vì DiT dùng patch size 8x8 -> 320 chia hết cho 8
        crop_size = (320, 320)
        
        from utils import r2c, c2r, fft2c, ifft2c
        kspace_complex = r2c(kspace_torch.unsqueeze(0)).squeeze(0) # (H, W)
        image_complex = fastmri.ifft2c(kspace_torch.permute(1, 2, 0)) # (H, W, 2)
        # Tự viết hàm xén tâm (Crop ảnh C=2)
        h, w, c = image_complex.shape
        ch, cw = crop_size
        h_start = (h - ch) // 2
        w_start = (w - cw) // 2
        image_complex = image_complex[h_start:h_start+ch, w_start:w_start+cw, :] # (320, 320, 2)
        
        # [QUAN TRỌNG] Chuẩn hóa dải năng lượng (Normalization) để CÂN BẰNG vs NOISE
        # Dữ liệu vật lý FastMRI siêu nhỏ (cỡ 1e-5). Nhiễu của mình dùng torch.randn() lại bự 1.0!
        # Không có lệnh này, mô hình Mù Màu vì Noise đè bẹp Tín hiệu gấp Triệu Lần!
        std_val = torch.std(image_complex) + 1e-11
        image_complex = image_complex / std_val
        
        # Quay về K-space cropped để có K-space chuẩn
        kspace_cropped = fastmri.fft2c(image_complex) # (320, 320, 2)
        
        # 3. Tạo Mask (Che K-space mô phỏng chụp thiểu)
        seed = None
        # Mask trả về dạng (1, 1, 2) hoặc tương tự trong fastMRI
        masked_kspace, mask_tensor, _ = T.apply_mask(kspace_cropped, self.mask_func, seed)
        
        # 4. Định hình dữ liệu xuất ra
        # x1_image: Ảnh 2 kênh chuẩn chỉnh (C, H, W)
        x1_image = image_complex.permute(2, 0, 1) # (2, 320, 320)
        
        # Mask 2D chuẩn: (320, 320) -> Phóng to mảng 1D mask lên K-space
        # mask_tensor có dạng (1, W, 1). Đúc nó thành (320, 320)
        mask_2d = mask_tensor.squeeze(-1).squeeze(0).repeat(320, 1) # (320, 320)
        
        return x1_image, mask_2d
