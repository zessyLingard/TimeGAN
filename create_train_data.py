import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def create_dataset(num_samples=50000):
    print(f"[-] Đang tạo {num_samples} mẫu dữ liệu train (Mixture of Gaussians)...")
    
    # Decision boundary
    THRESHOLD = 0.125
    
    data = []
    labels = []
    # Tạo dữ liệu với phân phối TỰ NHIÊN - có overlap
    # Nhưng LABEL dựa trên threshold, không phải dựa trên ý định ban đầu
    for _ in range(num_samples):
        # Chọn ngẫu nhiên muốn tạo Bit 0 hay Bit 1
        intended_bit = np.random.randint(0, 2)
        
        if intended_bit == 0:
            # Bit 0: Trung tâm vùng 0.08s - 0.125s
            sub_choice = np.random.randint(0, 3)
            if sub_choice == 0:
                val = np.random.normal(loc=0.09, scale=0.008)
            elif sub_choice == 1:
                val = np.random.normal(loc=0.102, scale=0.008)
            else:
                val = np.random.normal(loc=0.115, scale=0.008)
        else:
            # Bit 1: Trung tâm vùng 0.125s - 0.20s
            sub_choice = np.random.randint(0, 3)
            if sub_choice == 0:
                val = np.random.normal(loc=0.135, scale=0.01)
            elif sub_choice == 1:
                val = np.random.normal(loc=0.160, scale=0.01)
            else:
                val = np.random.normal(loc=0.185, scale=0.01)
        
        # Đảm bảo không âm
        val = max(0.001, val)
        
        # QUAN TRỌNG: Label dựa trên THRESHOLD, không phải intended_bit
        # Điều này giúp model học đúng decision boundary
        actual_label = 0 if val < THRESHOLD else 1
        
        data.append(val)
        labels.append(actual_label)
    
    data = np.array(data)
    labels = np.array(labels)
    
    # 3. Lưu file
    output_dir_data = r"helper/data/raw"
    output_dir_labels = r"helper/data/raw_labels"
    os.makedirs(output_dir_data, exist_ok=True)
    os.makedirs(output_dir_labels, exist_ok=True)
    
    # Lưu Data
    pd.DataFrame(data, columns=["IPD"]).to_csv(
        os.path.join(output_dir_data, "custom_traffic_data.csv"), index=False
    )
    
    # Lưu Labels
    pd.DataFrame(labels, columns=["Label"]).to_csv(
        os.path.join(output_dir_labels, "custom_traffic_labels.csv"), index=False
    )
    
    # Thống kê
    bit0_count = np.sum(labels == 0)
    bit1_count = np.sum(labels == 1)
    
    print("[+] Đã tạo xong dữ liệu Mixed Gaussian!")
    print(f"    - Threshold: {THRESHOLD}s")
    print(f"    - Bit 0 (IPD < {THRESHOLD}s): {bit0_count} samples ({100*bit0_count/num_samples:.1f}%)")
    print(f"    - Bit 1 (IPD >= {THRESHOLD}s): {bit1_count} samples ({100*bit1_count/num_samples:.1f}%)")
    
    # Vẽ Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(data[labels==0], bins=50, alpha=0.6, label='Bit 0 (IPD < 0.125s)', color='blue')
    plt.hist(data[labels==1], bins=50, alpha=0.6, label='Bit 1 (IPD >= 0.125s)', color='red')
    plt.axvline(x=THRESHOLD, color='green', linestyle='--', linewidth=2, label=f'Threshold = {THRESHOLD}s')
    plt.title("Phân phối dữ liệu Train (Label = f(IPD), không phải f(intended))")
    plt.xlabel("IPD (s)")
    plt.legend()
    plt.savefig("data_distribution")
    plt.show()

if __name__ == "__main__":
    create_dataset()