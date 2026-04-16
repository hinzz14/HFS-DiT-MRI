import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from fastmri.data.mri_data import SliceDataset
from fastmri.data.subsample import RandomMaskFunc
from data.transforms import HFSDataTransform
from pl_modules.flow_module import FlowMatchingDiTModule

def main(args):
    # Khóa ngẫu nhiên chặt chẽ để đảm bảo tái lập thử nghiệm
    pl.seed_everything(42)
    
    # 1. Ống xả Dữ liệu (Dataset & Dataloader)
    # Cấu hình gia tốc (Acceleration) dựa vào Argument
    center_frac = 0.08 if args.acceleration == 4 else 0.04
    mask_func = RandomMaskFunc(center_fractions=[center_frac], accelerations=[args.acceleration])
    transform = HFSDataTransform(mask_func)
    
    train_dir = os.path.join(args.data_path, "singlecoil_train")
    val_dir = os.path.join(args.data_path, "singlecoil_val")
    
    train_dataset = SliceDataset(root=train_dir, transform=transform, challenge="singlecoil")
    val_dataset = SliceDataset(root=val_dir, transform=transform, challenge="singlecoil")
    
    # Bật num_workers lên 4 để nạp dữ liệu siêu mượt (Pytorch Lightning xử lý Leak RAM rất giỏi)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 2. Khối Trái tim Mạng Nơ Ron
    use_hfs_flag = not args.no_hfs
    model = FlowMatchingDiTModule(lr=args.lr, use_hfs=use_hfs_flag)
    
    # 3. Kỹ thuật Lưu Checkpoint Tự Động & Early Stopping
    # Tạo tên thư mục lưu kết quả tự động theo tham số để khỏi gõ nhầm đè kết quả lên nhau
    if args.default_root_dir == '/home/loipd/MRI_Project/experiments/hfs_dit_fm':
        exp_name = "no_hfs" if args.no_hfs else "hfs"
        accel_name = f"{args.acceleration}x"
        root_dir = f"/home/loipd/MRI_Project/experiments/{exp_name}_dit_fm_{accel_name}"
    else:
        root_dir = args.default_root_dir
        
    os.makedirs(root_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(root_dir, "checkpoints"),
        filename="hfs-dit-fm-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=40,    # Nới lỏng lên 40 epoch cho DiT (thời gian khởi động lâu)
        verbose=True,
        mode="min"
    )
    
    # 4. Trình Điều Khối (Trainer) - KÍCH HOẠT TÍNH NĂNG TĂNG TỐC TENSOR CORE 16-BIT
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=16, # <--- Cấu hình Vàng của User Feedback (Sửa lỗi cho tương thích Lightning ver cũ)
        callbacks=[checkpoint_callback, early_stop_callback],
        default_root_dir=root_dir,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run
    )
    
    # 5. Bấm nút Khởi Động
    print("🚀 ĐANG KHỞI ĐỘNG ĐỘNG CƠ HFS-DiT FLOW MATCHING SOTA TRÊN TENSE CORE...")
    
    last_ckpt_path = os.path.join(root_dir, "checkpoints", "last.ckpt")
    if os.path.exists(last_ckpt_path):
        print(f"🔄 ĐÃ PHÁT HIỆN CHECKPOINT TẠI {last_ckpt_path}! TIẾP TỤC HUẤN LUYỆN (RESUME)...")
        trainer.fit(model, train_loader, val_loader, ckpt_path=last_ckpt_path)
    else:
        trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/loipd/MRI_Project/data', help='Đường dẫn gốc chứa singlecoil_train và singlecoil_val')
    parser.add_argument('--batch_size', type=int, default=16, help='Kích cỡ nạp dữ liệu')
    parser.add_argument('--num_workers', type=int, default=2, help='Số luồng load data')
    parser.add_argument('--max_epochs', type=int, default=100, help='Tổng số Epoch tối đa')
    parser.add_argument('--default_root_dir', type=str, default='/home/loipd/MRI_Project/experiments/hfs_dit_fm', help='Nơi lưu Checkpoint và Logs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--fast_dev_run', action='store_true', help='Lệnh chạy mô phỏng 1 batch cực nhanh để kiểm tra hình thái mã nguồn trước khi train')
    parser.add_argument('--no_hfs', action='store_true', help='Nếu bật, vô hiệu hóa HFS và train theo Standard Flow Matching (Ablation).')
    parser.add_argument('--acceleration', type=int, default=4, choices=[4, 8], help='Gia tốc K-space (4x hoặc 8x)')
    args = parser.parse_args()
    
    main(args)
