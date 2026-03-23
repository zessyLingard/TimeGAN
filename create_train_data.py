import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def generate_csv_dataset(n_samples=50000):
    print("[-] Đang sinh dữ liệu WebSocket (Wide-Band) & Nhiễu tự nhiên...")
    
    # 1. Khởi tạo nhãn (Bit 0 và Bit 1)
    np.random.seed(42)
    labels = np.random.randint(0, 2, n_samples)
    
    # 2. Rải đều Base IPD
    base_ipd = np.zeros(n_samples)
    mask_0 = (labels == 0)
    mask_1 = (labels == 1)
    
    base_ipd[mask_0] = np.random.uniform(low=0.40, high=0.65, size=np.sum(mask_0))
    base_ipd[mask_1] = np.random.uniform(low=0.85, high=1.10, size=np.sum(mask_1))
    
    # 3. Thêm nhiễu mạng tự nhiên
    normal_jitter = np.random.normal(loc=0.0, scale=0.03, size=n_samples)
    lag_spikes = np.random.exponential(scale=0.02, size=n_samples)
    
    ipds = base_ipd + normal_jitter + lag_spikes
    ipds = np.clip(ipds, 0.2, 1.5)
    
    # ==========================================
    # LƯU FILE CSV
    # ==========================================
    dir_raw = "helper/data/raw"
    dir_labels = "helper/data/raw_labels"
    os.makedirs(dir_raw, exist_ok=True)
    os.makedirs(dir_labels, exist_ok=True)
    
    path_raw = os.path.join(dir_raw, "heartbeat_ipd.csv")
    path_labels = os.path.join(dir_labels, "heartbeat_labels.csv")
    
    pd.DataFrame(ipds, columns=["IPD"]).to_csv(path_raw, index=False)
    pd.DataFrame(labels, columns=["Bit_Label"]).to_csv(path_labels, index=False)
    
    # ==========================================
    # VẼ BIỂU ĐỒ TRỰC QUAN HÓA (VISUALIZATION)
    # ==========================================
    print("[-] Đang vẽ biểu đồ phân phối...")
    plt.figure(figsize=(12, 6))
    
    # Tách dữ liệu ra 2 mảng để vẽ 2 màu khác nhau
    ipd_bit0 = ipds[mask_0]
    ipd_bit1 = ipds[mask_1]
    
    # Vẽ Histogram
    # Bins=100 chia đồ thị thành 100 cột nhỏ để nhìn rõ chi tiết
    # Alpha=0.7 làm màu hơi trong suốt để thấy rõ ranh giới
    plt.hist(ipd_bit0, bins=100, color='#2980b9', alpha=0.7, label='Bit 0 (Base: 0.40s - 0.65s)')
    plt.hist(ipd_bit1, bins=100, color='#e74c3c', alpha=0.7, label='Bit 1 (Base: 0.85s - 1.10s)')
    
    # Vẽ vạch Threshold 0.75s
    plt.axvline(x=0.75, color='#27ae60', linestyle='dashed', linewidth=2.5, label='Ngưỡng Giải Mã (Threshold = 0.75s)')
    
    # Trang trí biểu đồ
    plt.title('Phân phối Kênh ẩn Wide-Band (Đã bao gồm Network Jitter & Lag)', fontsize=14, fontweight='bold')
    plt.xlabel('Thời gian trễ - IPD (giây)', fontsize=12)
    plt.ylabel('Tần suất (Số lượng gói tin)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Lưu file ảnh
    vis_path = os.path.join("helper/data", "distribution_preview.png")
    plt.tight_layout()
    plt.savefig(vis_path, dpi=300) # dpi=300 cho ảnh nét căng để dán vào báo cáo
    
    print(f"[+] Hoàn tất! Đã lưu biểu đồ cực nét tại: {vis_path}")
    print(f"    -> Dữ liệu IPD: {path_raw}")
    print(f"    -> Nhãn Bit:    {path_labels}")

if __name__ == "__main__":
    generate_csv_dataset()