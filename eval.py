import argparse
import os
import sys
import numpy as np
import pandas as pd
import zlib
import math
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# --- CÁC THUẬT TOÁN DETECTION (Dựa trên bài báo 2023) ---

def calc_epsilon_similarity(data, epsilon=0.01):
    if len(data) < 2: return 0
    
    # 1. Sắp xếp IPD
    sorted_data = np.sort(data)
    
    # 2. Tính sự khác biệt tương đối giữa các gói liền kề
    # lambda_i = |t_{i+1} - t_i| / t_i
    # Để tránh chia cho 0, ta cộng một lượng rất nhỏ (eps)
    diffs = np.diff(sorted_data)
    relative_diffs = diffs / (sorted_data[:-1] + 1e-9)
    
    # 3. Đếm số lượng cặp có sự khác biệt nhỏ hơn epsilon
    count = np.sum(relative_diffs < epsilon)
    
    # 4. Score = Tỷ lệ phần trăm
    score = count / len(relative_diffs)
    return score

def encode_compressibility_string(val):
    """Mã hóa IPD thành chuỗi ký tự theo quy tắc bài báo (Section II.B.2)"""
    # Ví dụ: 0.00346 -> B35 (B là số lượng số 0 sau dấu phẩy)
    if val <= 0: return "Z00"
    
    # Đếm số lượng số 0 sau dấu phẩy (Leading zeros)
    # log10(0.00346) = -2.46 -> ceil = -2 -> abs = 2. 
    # Bài báo quy định: chuyển số lượng này thành chữ cái (A=0, B=1...)
    # Nhưng để đơn giản và nhanh, ta dùng format string
    s = "{:.10f}".format(val).split('.')[1]
    leading_zeros = 0
    for char in s:
        if char == '0': leading_zeros += 1
        else: break
    
    # Chuyển số 0 thành chữ cái (A=0, B=1, C=2...)
    prefix = chr(ord('A') + min(leading_zeros, 25))
    
    # Lấy 2 chữ số có nghĩa tiếp theo
    # val * 10^(leading_zeros) -> ví dụ 0.00346 * 1000 = 3.46
    significant = val * (10**(leading_zeros + 1))
    digits = "{:02d}".format(int(significant))[:2]
    
    return prefix + digits

def calc_compressibility_score(data):
    """Tính Compressibility Score"""
    # 1. Chuyển đổi chuỗi IPD thành chuỗi ký tự đặc biệt
    # (Mô phỏng cách nén của Cabuk et al.)
    string_rep = "".join([encode_compressibility_string(x) for x in data])
    
    # 2. Nén chuỗi (Dùng zlib/gzip)
    compressed = zlib.compress(string_rep.encode('utf-8'))
    
    # 3. Tính tỷ lệ nén
    # Score = Độ dài nén / Độ dài gốc
    # Traffic có quy luật (Covert) sẽ nén tốt hơn (Score thấp hơn)
    # Traffic ngẫu nhiên (Real) sẽ khó nén (Score cao hơn)
    score = len(compressed) / len(string_rep)
    return score

# --- HÀM CHÍNH ---

def evaluate(real_path, fake_path, window_size=100):
    print(f"[-] Đang đọc dữ liệu...")
    df_real = pd.read_csv(real_path)
    df_fake = pd.read_csv(fake_path)
    
    real_data = df_real.iloc[:, 0].values
    fake_data = df_fake.iloc[:, 0].values
    
    print(f"    Real samples: {len(real_data)}")
    print(f"    Fake samples: {len(fake_data)}")
    
    # Cắt dữ liệu thành các cửa sổ (Windows) để chấm điểm từng đoạn
    # Bài báo dùng window=2000, nhưng data bạn ít nên dùng 100 hoặc 200 thôi
    def get_windows(data, size):
        return [data[i:i+size] for i in range(0, len(data), size) if len(data[i:i+size])==size]

    real_windows = get_windows(real_data, window_size)
    fake_windows = get_windows(fake_data, window_size)
    
    print(f"[-] Đang chạy Detection trên các cửa sổ (Size={window_size})...")
    print(f"    Số mẫu Real: {len(real_windows)} | Số mẫu Fake: {len(fake_windows)}")
    
    # --- 1. Epsilon Similarity ---
    scores_real_eps = [calc_epsilon_similarity(w, epsilon=0.01) for w in real_windows]
    scores_fake_eps = [calc_epsilon_similarity(w, epsilon=0.01) for w in fake_windows]
    
    # --- 2. Compressibility ---
    scores_real_comp = [calc_compressibility_score(w) for w in real_windows]
    scores_fake_comp = [calc_compressibility_score(w) for w in fake_windows]
    
    # --- 3. Tính AUC ---
    # Gán nhãn: Real = 0, Fake = 1
    y_true = [0]*len(real_windows) + [1]*len(fake_windows)
    
    # Epsilon Score: Fake thường CAO hơn Real (High similarity)
    y_scores_eps = scores_real_eps + scores_fake_eps
    auc_eps = roc_auc_score(y_true, y_scores_eps)
    
    # Compressibility Score: Fake thường THẤP hơn Real (Dễ nén hơn)
    # Để tính AUC thuận chiều (Fake điểm cao), ta lấy 1 - Score hoặc -Score
    # Ở đây ta dùng -Score vì Fake nén tốt -> Score nhỏ -> -Score lớn
    y_scores_comp = [-s for s in scores_real_comp] + [-s for s in scores_fake_comp]
    auc_comp = roc_auc_score(y_true, y_scores_comp)
    
    # --- 4. KS Test (So sánh tổng thể) ---
    ks_stat, ks_pvalue = ks_2samp(real_data, fake_data)

    print("\n" + "="*50)
    print("   KẾT QUẢ ĐÁNH GIÁ KHẢ NĂNG BỊ PHÁT HIỆN")
    print("="*50)
    
    print(f"\n1. Epsilon-Similarity Detection (AUC): {auc_eps:.4f}")
    if auc_eps < 0.6:
        print("   ✅ RẤT TỐT! Hệ thống Detection không phân biệt được.")
    else:
        print("   ⚠️ CẢNH BÁO! Có nguy cơ bị phát hiện bởi thuật toán này.")
        
    print(f"\n2. Compressibility Detection (AUC):    {auc_comp:.4f}")
    if auc_comp < 0.6:
        print("   ✅ RẤT TỐT! Độ nén tương đương traffic thật.")
    else:
        print("   ⚠️ CẢNH BÁO! Traffic giả quá 'đều', dễ bị nén.")

    print(f"\n3. Kolmogorov-Smirnov Test:")
    print(f"   - Statistic (Khoảng cách): {ks_stat:.4f} (Càng nhỏ càng tốt)")
    print(f"   - P-value: {ks_pvalue:.4e}")
    if ks_pvalue > 0.05:
        print("   ✅ TUYỆT VỜI! Không thể bác bỏ giả thuyết 2 phân phối là một.")
    else:
        print("   ⚠️ Khác biệt thống kê có ý nghĩa.")

    # Vẽ biểu đồ phân phối điểm số
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(scores_real_eps, alpha=0.5, label='Real', bins=20)
    plt.hist(scores_fake_eps, alpha=0.5, label='Fake (TimeGAN)', bins=20)
    plt.title(f"Epsilon-Similarity Scores (AUC={auc_eps:.2f})")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(scores_real_comp, alpha=0.5, label='Real', bins=20)
    plt.hist(scores_fake_comp, alpha=0.5, label='Fake (TimeGAN)', bins=20)
    plt.title(f"Compressibility Scores (AUC={auc_comp:.2f})")
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", default=r"covert_ipd.csv")
    parser.add_argument("--real", default=r"helper/data/raw/my_traffic_data.csv")
    parser.add_argument("--window", type=int, default=100, help="Độ dài cửa sổ để chấm điểm (nên >= 100)")
    
    args = parser.parse_args()
    evaluate(args.real, args.fake, args.window)