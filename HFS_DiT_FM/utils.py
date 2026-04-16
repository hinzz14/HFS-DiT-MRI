import torch
import torch.fft

def r2c(tensor):
    """
    Chuyển tensor 2-channel (B, 2, H, W) thành số phức Complex (B, H, W)
    """
    assert tensor.shape[1] == 2
    return torch.complex(tensor[:, 0], tensor[:, 1])

def c2r(tensor):
    """
    Chuyển số phức Complex (B, H, W) về tensor 2-channel (B, 2, H, W)
    """
    return torch.stack([tensor.real, tensor.imag], dim=1)

def fft2c(img_complex):
    """
    Thực hiện FFT 2D căn giữa cho Tensor phức (B, H, W)
    """
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img_complex, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))

def ifft2c(kspace_complex):
    """
    Thực hiện IFFT 2D căn giữa để thu hồi Ảnh Phức (B, H, W) từ K-space
    """
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace_complex, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))
