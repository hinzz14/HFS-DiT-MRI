#!/bin/bash
# Kịch bản Evaluation tự động cho toàn bộ Experimental Matrix (Treo máy qua đêm)
# Đảm bảo bạn đã activate môi trường: conda activate dit_mri

echo "=========================================================="
echo "BẮT ĐẦU ĐÁNH GIÁ MA TRẬN ABLATION (TỔNG CỘNG 4 KỊCH BẢN)"
echo "Tự động phân luồng ghi log... Bạn có thể đi ngủ!"
echo "=========================================================="

# 1. Khai báo các đường dẫn Checkpoint mạnh nhất (Best Checkpoints)
UNET_4X="experiments/unet_baseline/checkpoints/epoch=45-step=99912.ckpt"
UNET_8X="experiments/unet_baseline_8x/checkpoints/epoch=40-step=89052.ckpt"

HFS_4X="experiments/hfs_dit_fm/checkpoints/hfs-dit-fm-epoch=96-val_loss=0.0001.ckpt"
NOHFS_4X="experiments/no_hfs_dit_fm_4x/checkpoints/hfs-dit-fm-epoch=91-val_loss=0.4236.ckpt"

HFS_8X="experiments/hfs_dit_fm_8x/checkpoints/hfs-dit-fm-epoch=98-val_loss=0.0001.ckpt"
NOHFS_8X="experiments/no_hfs_dit_fm_8x/checkpoints/hfs-dit-fm-epoch=84-val_loss=0.4735.ckpt"

# 2. Xóa tệp kết quả cũ nếu có
rm -f final_ablation_results.txt
touch final_ablation_results.txt

# --- KỊCH BẢN 1: HFS-DiT 4x vs U-Net 4x ---
echo "Đang đánh giá kịch bản 1: HFS 4x (Khoảng 55 phút)..."
python eval_on_testset.py --unet_ckpt $UNET_4X --hfs_ckpt $HFS_4X --acceleration 4 > log_hfs_4x.txt
echo "--- KẾT QUẢ KỊCH BẢN 1 (GIA TỐC 4X - CÓ HFS) ---" >> final_ablation_results.txt
tail -n 6 log_hfs_4x.txt >> final_ablation_results.txt
echo -e "\n" >> final_ablation_results.txt

# --- KỊCH BẢN 2: Non-HFS DiT 4x vs U-Net 4x ---
echo "Đang đánh giá kịch bản 2: Không HFS 4x (Khoảng 55 phút)..."
python eval_on_testset.py --unet_ckpt $UNET_4X --hfs_ckpt $NOHFS_4X --no_hfs --acceleration 4 > log_nohfs_4x.txt
echo "--- KẾT QUẢ KỊCH BẢN 2 (GIA TỐC 4X - KHÔNG HFS) ---" >> final_ablation_results.txt
tail -n 6 log_nohfs_4x.txt >> final_ablation_results.txt
echo -e "\n" >> final_ablation_results.txt

# --- KỊCH BẢN 3: HFS-DiT 8x vs U-Net 8x ---
echo "Đang đánh giá kịch bản 3: HFS 8x (Khoảng 55 phút)..."
python eval_on_testset.py --unet_ckpt $UNET_8X --hfs_ckpt $HFS_8X --acceleration 8 > log_hfs_8x.txt
echo "--- KẾT QUẢ KỊCH BẢN 3 (GIA TỐC 8X - CÓ HFS) ---" >> final_ablation_results.txt
tail -n 6 log_hfs_8x.txt >> final_ablation_results.txt
echo -e "\n" >> final_ablation_results.txt

# --- KỊCH BẢN 4: Non-HFS DiT 8x vs U-Net 8x ---
echo "Đang đánh giá kịch bản 4: Không HFS 8x (Khoảng 55 phút)..."
python eval_on_testset.py --unet_ckpt $UNET_8X --hfs_ckpt $NOHFS_8X --no_hfs --acceleration 8 > log_nohfs_8x.txt
echo "--- KẾT QUẢ KỊCH BẢN 4 (GIA TỐC 8X - KHÔNG HFS) ---" >> final_ablation_results.txt
tail -n 6 log_nohfs_8x.txt >> final_ablation_results.txt

echo "=========================================================="
echo "CHÚC MỪNG HOÀN THÀNH! XIN HÃY MỞ FILE final_ablation_results.txt ĐỂ XEM ĐIỂM SỐ GỘP"
echo "=========================================================="
