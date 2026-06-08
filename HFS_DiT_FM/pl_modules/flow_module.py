import torch
import pytorch_lightning as pl
import copy
from models.dit import DiT
from utils import r2c, c2r, fft2c, ifft2c

class FlowMatchingDiTModule(pl.LightningModule):
    """
    Lightning Module đóng gói thuật toán Flow Matching kết hợp HFS.
    Huấn luyện mạng DiT học hướng Vector trên miền Image Domain.
    """
    def __init__(self, hidden_size=512, depth=12, num_heads=8, lr=2e-4, use_hfs=True):
        super().__init__()
        self.save_hyperparameters()
        self.use_hfs = use_hfs
        self.net = DiT(in_channels=4, out_channels=2, hidden_size=hidden_size, depth=depth, num_heads=num_heads)
        
        # Thêm EMA Model
        self.ema_net = copy.deepcopy(self.net)
        for param in self.ema_net.parameters():
            param.requires_grad = False
        self.ema_decay = 0.9999
        
        self.lr = lr
        
    def forward(self, x, x0, t):
        # Nối x và x0 theo chiều kênh (dim=1) để làm condition rõ ràng
        net_input = torch.cat([x, x0], dim=1)
        # Sử dụng EMA khi Inference (không phải training)
        if hasattr(self, 'ema_net') and self.ema_net is not None and not self.training:
            return self.ema_net(net_input, t)
        return self.net(net_input, t)
        
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Cập nhật tham số EMA mỗi khi train xong 1 batch
        with torch.no_grad():
            for param_q, param_k in zip(self.net.parameters(), self.ema_net.parameters()):
                param_k.data.mul_(self.ema_decay).add_(param_q.data, alpha=1 - self.ema_decay)
                
    def on_load_checkpoint(self, checkpoint):
        # Xử lý tương thích: nếu load checkpoint cũ chưa có EMA, tự động nhân bản từ net sang
        state_dict = checkpoint["state_dict"]
        has_ema = any(k.startswith("ema_net.") for k in state_dict.keys())
        if not has_ema:
            for k in list(state_dict.keys()):
                if k.startswith("net."):
                    ema_k = k.replace("net.", "ema_net.", 1)
                    state_dict[ema_k] = state_dict[k].clone()
        
    def training_step(self, batch, batch_idx):
        # Dataloader sẽ trả về x1_image (Ground Truth 2 kênh) và mask phân hoạch K-space
        x1_image, mask = batch
        B = x1_image.shape[0]
        
        # 1. BIẾN ĐỔI CHỒNG K-SPACE VÀ CHUẨN HÓA DỮ LIỆU
        # Scale input sao cho variance về quanh 1.0 để model dễ học
        std = x1_image.std(dim=(1, 2, 3), keepdim=True)
        x1_image = x1_image / (std + 1e-11)
        
        x1_complex = r2c(x1_image)
        k1 = fft2c(x1_complex)
        
        # Tách phổ theo HFS (High-Frequency Space)
        k1_low = k1 * mask
        
        # 2. KHỞI TẠO KHỞI ĐIỂM ($x_0$) VÀ ĐIỀU KIỆN (COND)
        # Condition lúc nào cũng là ảnh bị mờ Zero-Filled (để nối vào làm bản lề cho mạng Nơ-ron suy luận)
        x_zf_cond = c2r(ifft2c(k1_low)).type(torch.float32)
        
        if self.use_hfs:
            # HFS: Ảnh X0 xuất phát từ Zero-filled kết hợp với Nhiễu tần số cao (scale 1.0)
            noise_scale = 1.0
            noise_img = torch.randn_like(x1_image) * noise_scale # (B, 2, H, W)
            k0_noise = fft2c(r2c(noise_img))
            k0_high = k0_noise * (1 - mask)
            
            k0 = k1_low + k0_high
            x0_image = c2r(ifft2c(k0)).type(torch.float32)
        else:
            # STANDARD FLOW MATCHING (Không có HFS): Bắt đầu rễ đắng từ Pure Gaussian Noise
            x0_image = torch.randn_like(x1_image)
            
        # 3. RÚT THĂM THỜI GIAN FLOW MATCHING (OT-FLOW)
        t = torch.rand(B, device=self.device)
        t_view = t.view(B, 1, 1, 1)
        
        x_t = (1 - t_view) * x0_image + t_view * x1_image
        
        # Vector đích
        v_target = x1_image - x0_image
        
        # 4. MẠNG NƠ RON DỰ ĐOÁN
        cond_input = x_zf_cond
            
        v_pred = self(x_t, cond_input, t)
        
        # 5. TỐI ƯU LOSS (Ưu tiên phạt sai số ở tần số cao)
        # Vì phần tần số thấp là 0 trong v_target, nếu ta chỉ dùng MSE, model dư sức đoán
        # nhưng để hỗ trợ model học nét ảnh (viền), ta có thể giữ nguyên MSE hoặc tính trên mask
        loss = torch.nn.functional.mse_loss(v_pred, v_target)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x1_image, mask = batch
        B = x1_image.shape[0]
        
        std = x1_image.std(dim=(1, 2, 3), keepdim=True)
        x1_image = x1_image / (std + 1e-11)
        
        x1_complex = r2c(x1_image)
        k1 = fft2c(x1_complex)
        k1_low = k1 * mask
        
        x_zf_cond = c2r(ifft2c(k1_low)).type(torch.float32)
        
        if self.use_hfs:
            noise_scale = 1.0
            noise_img = torch.randn_like(x1_image) * noise_scale
            k0_noise = fft2c(r2c(noise_img))
            k0_high = k0_noise * (1 - mask)
            
            k0 = k1_low + k0_high
            x0_image = c2r(ifft2c(k0)).type(torch.float32)
        else:
            x0_image = torch.randn_like(x1_image)
        
        t = torch.rand(B, device=self.device)
        t_view = t.view(B, 1, 1, 1)
        x_t = (1 - t_view) * x0_image + t_view * x1_image
        v_target = x1_image - x0_image
        
        cond_input = x_zf_cond
            
        v_pred = self(x_t, cond_input, t)
        loss = torch.nn.functional.mse_loss(v_pred, v_target)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss
        
    def configure_optimizers(self):
        # Mạng Transformer (DiT) bắt buộc phải dùng AdamW có weight decay để Regularization, 
        # Adam bình thường rất dễ làm tự hoảng Gradient.
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Thêm Linear Warmup + Cosine Annealing (Tránh Gradient bùng nổ ở những epoch đầu)
        max_epochs = self.trainer.max_epochs if self.trainer.max_epochs is not None else 100
        warmup_epochs = 5
        
        def lr_lambda(current_step):
            if current_step < warmup_epochs:
                return float(current_step) / float(max(1, warmup_epochs))
            else:
                import math
                progress = float(current_step - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
                
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
