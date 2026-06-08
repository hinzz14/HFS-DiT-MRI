// Cập nhật giá trị hiển thị của sliders khi kéo
document.getElementById('slice-idx').addEventListener('input', (e) => {
    document.getElementById('slice-val').textContent = e.target.value;
});

document.getElementById('center-fraction').addEventListener('input', (e) => {
    document.getElementById('cf-val').textContent = e.target.value;
});

document.getElementById('ode-steps').addEventListener('input', (e) => {
    document.getElementById('steps-val').textContent = e.target.value;
});

// Tự động điều chỉnh CF khi thay đổi hệ số gia tốc
document.querySelectorAll('input[name="acceleration"]').forEach((radio) => {
    radio.addEventListener('change', (e) => {
        const acc = parseInt(e.target.value);
        const cfSlider = document.getElementById('center-fraction');
        const cfVal = document.getElementById('cf-val');
        if (acc === 4) {
            cfSlider.value = 0.08;
            cfVal.textContent = "0.08";
        } else if (acc === 8) {
            cfSlider.value = 0.04;
            cfVal.textContent = "0.04";
        }
    });
});

// Hàm hỗ trợ chọn lát cắt nhanh (Golden Slices)
function setSlice(idx) {
    document.getElementById('slice-idx').value = idx;
    document.getElementById('slice-val').textContent = idx;
    
    // Đặt tham số mẫu đẹp
    if (idx === 727 || idx === 1438 || idx === 3030) {
        document.getElementById('center-fraction').value = 0.04;
        document.getElementById('cf-val').textContent = "0.04";
    }
    
    // Tự động chạy phục dựng
    triggerRecon();
}

// Xử lý gửi biểu mẫu
document.getElementById('recon-form').addEventListener('submit', (e) => {
    e.preventDefault();
    triggerRecon();
});

async function triggerRecon() {
    const btnSubmit = document.getElementById('btn-submit');
    const loader = btnSubmit.querySelector('.loader-spinner');
    const btnText = btnSubmit.querySelector('.btn-text');
    const statusText = document.getElementById('status-text');

    // Thu thập tham số
    const sliceIdx = parseInt(document.getElementById('slice-idx').value);
    const acceleration = parseInt(document.querySelector('input[name="acceleration"]:checked').value);
    const centerFraction = parseFloat(document.getElementById('center-fraction').value);
    const odeSteps = parseInt(document.getElementById('ode-steps').value);

    // Vô hiệu hóa nút và hiện loader
    btnSubmit.disabled = true;
    loader.classList.remove('hidden');
    btnText.textContent = "ĐANG XỬ LÝ TRÊN GPU...";
    statusText.textContent = `[Processing] Đang gửi yêu cầu... Slice #${sliceIdx}, Gia tốc ${acceleration}x, Center Fraction ${centerFraction}, Steps ${odeSteps}.`;

    try {
        const response = await fetch('/api/reconstruct', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                slice_idx: sliceIdx,
                acceleration: acceleration,
                center_fraction: centerFraction,
                ode_steps: odeSteps
            })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Server error");
        }

        const res = await response.json();
        
        if (res.success) {
            // 1. Cập nhật các ảnh chính (Base64)
            document.getElementById('img-gt').src = res.images.gt;
            document.getElementById('img-zf').src = res.images.zf;
            document.getElementById('img-unet').src = res.images.unet;
            document.getElementById('img-dit').src = res.images.dit;

            // 2. Cập nhật Error Maps
            document.getElementById('err-unet').src = res.images.unet_err;
            document.getElementById('err-dit').src = res.images.dit_err;

            // 3. Cập nhật bảng chỉ số (Metrics)
            // Ground Truth
            document.getElementById('m-lapvar-gt').textContent = `${res.metrics.gt.lap_var.toFixed(2)}`;

            // Zero-filled
            document.getElementById('m-nmse-zf').textContent = `${res.metrics.zf.nmse.toFixed(3)}%`;
            document.getElementById('m-psnr-zf').textContent = `${res.metrics.zf.psnr.toFixed(2)} dB`;
            document.getElementById('m-ssim-zf').textContent = `${(res.metrics.zf.ssim * 100).toFixed(2)}%`;
            document.getElementById('m-lpips-zf').textContent = `${res.metrics.zf.lpips.toFixed(4)}`;
            document.getElementById('m-lapvar-zf').textContent = `${res.metrics.zf.lap_var.toFixed(2)}`;

            // U-Net
            document.getElementById('m-nmse-unet').textContent = `${res.metrics.unet.nmse.toFixed(3)}%`;
            document.getElementById('m-psnr-unet').textContent = `${res.metrics.unet.psnr.toFixed(2)} dB`;
            document.getElementById('m-ssim-unet').textContent = `${(res.metrics.unet.ssim * 100).toFixed(2)}%`;
            document.getElementById('m-lpips-unet').textContent = `${res.metrics.unet.lpips.toFixed(4)}`;
            document.getElementById('m-lapvar-unet').textContent = `${res.metrics.unet.lap_var.toFixed(2)}`;
            document.getElementById('m-time-unet').textContent = `${res.metrics.unet.time.toFixed(3)}s`;


            // HFS-DiT-FM
            document.getElementById('m-nmse-dit').textContent = `${res.metrics.dit.nmse.toFixed(3)}%`;
            document.getElementById('m-psnr-dit').textContent = `${res.metrics.dit.psnr.toFixed(2)} dB`;
            document.getElementById('m-ssim-dit').textContent = `${(res.metrics.dit.ssim * 100).toFixed(2)}%`;
            document.getElementById('m-lpips-dit').textContent = `${res.metrics.dit.lpips.toFixed(4)}`;
            document.getElementById('m-lapvar-dit').textContent = `${res.metrics.dit.lap_var.toFixed(2)}`;
            document.getElementById('m-time-dit').textContent = `${res.metrics.dit.time.toFixed(3)}s`;

            // Ghi nhận trạng thái hoàn thành
            statusText.textContent = `[Success] Hoàn thành phục dựng! HFS-DiT-FM (PSNR: ${res.metrics.dit.psnr.toFixed(2)} dB) chạy trong ${res.metrics.dit.time.toFixed(2)} giây.`;
        } else {
            throw new Error("Reconstruction returned success=false");
        }

    } catch (error) {
        console.error(error);
        statusText.textContent = `[Error] Thất bại: ${error.message}`;
    } finally {
        // Kích hoạt lại nút
        btnSubmit.disabled = false;
        loader.classList.add('hidden');
        btnText.textContent = "BẮT ĐẦU PHỤC DỰNG";
    }
}

// Chạy tự động lần đầu tiên khi tải trang
window.addEventListener('DOMContentLoaded', () => {
    triggerRecon();
});
