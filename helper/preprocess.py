import pandas as pd
import numpy as np
import os

def generate_traffic_dataset():
    # --- 1. CẤU HÌNH ĐƯỜNG DẪN (Đã sửa để khớp với cấu trúc thư mục thực tế) ---
    # Lấy đường dẫn của file script hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))  # c:\Users\hung\Downloads\MC-TimeGAN-main\helper
    # Thư mục gốc của dự án
    project_root = os.path.dirname(current_dir)  # c:\Users\hung\Downloads\MC-TimeGAN-main
    
    # Đường dẫn File Gốc (Input) - Đã sửa: legit.csv nằm trong helper/data/raw/
    INPUT_FILE = os.path.join(current_dir, "data", "raw", "legit.csv")
    
    # Đường dẫn File Đầu ra (Output) - Đã sửa: Output nằm trong thư mục helper/data/
    output_data_dir = os.path.join(current_dir, "data", "raw")
    output_label_dir = os.path.join(current_dir, "data", "raw_labels")
    
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    OUT_DATA = os.path.join(output_data_dir, "my_traffic_data.csv")
    OUT_LABELS = os.path.join(output_label_dir, "my_traffic_labels.csv")

    print(f"[-] Đang đọc file từ: {INPUT_FILE}")

    # --- 2. XỬ LÝ DỮ LIỆU ---
    try:
        # Đọc file CSV
        df = pd.read_csv(INPUT_FILE)
        
        # Trích xuất cột "Time" (thay vì cột đầu tiên)
        if 'Time' not in df.columns:
            raise ValueError("Cột 'Time' không tồn tại trong file CSV.")
        
        ipd = pd.to_numeric(df['Time'], errors='coerce').dropna().values
        
        # Lọc bỏ giá trị âm hoặc 0 nếu có (để tránh lỗi)
        ipd = ipd[ipd > 0]
        
        print(f"[-] Đã đọc {len(ipd)} giá trị từ cột 'Time'.")
        print(f"    Min: {np.min(ipd):.6f}s | Max: {np.max(ipd):.6f}s")

        # --- 3. GÁN NHÃN (LABELING) ---
        # Dùng Median (Trung vị) để chia đôi dữ liệu: 50% Nhanh (0), 50% Chậm (1)
        threshold = np.median(ipd)
        print(f"[-] Ngưỡng Median tính được: {threshold:.6f}s")
        
        # Tạo mảng nhãn: 0 nếu < threshold, ngược lại là 1
        labels = (ipd > threshold).astype(int)
        
        # Thống kê nhãn
        unique, counts = np.unique(labels, return_counts=True)
        print(f"    Phân bố nhãn: {dict(zip(unique, counts))}")

        # --- 4. LƯU FILE ---
        # TimeGAN thường nhận file CSV không có header hoặc có header tùy chỉnh
        # Để an toàn và dễ dùng nhất, ta lưu 1 cột duy nhất, có header để dễ debug
        
        # Lưu file Data (IPD từ cột Time)
        pd.DataFrame(ipd, columns=['IPD']).to_csv(OUT_DATA, index=False)
        print(f"[+] Đã lưu dữ liệu IPD vào: {OUT_DATA}")
        
        # Lưu file Label
        pd.DataFrame(labels, columns=['Label']).to_csv(OUT_LABELS, index=False)
        print(f"[+] Đã lưu dữ liệu Nhãn vào: {OUT_LABELS}")
        
        print("\n=> HOÀN TẤT! Bạn có thể dùng 2 file này để train TimeGAN.")
        
    except Exception as e:
        print(f"[!] LỖI: {e}")
        print("Hãy kiểm tra lại đường dẫn file legit.csv hoặc tên cột trong file.")

if __name__ == "__main__":
    generate_traffic_dataset()