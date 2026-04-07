"""
CTC-GAN Decoder (Covert Timing Channel)
Pipeline: IPD (CSV) → GAN decode (Shared Seed) → Bits → BCH decode (C binary) → File

AES decryption is handled by bch_decode (compile with -DUSE_AES if needed).
Place password in pass.txt to enable AES.
"""
import os
import sys
import argparse
import subprocess
import tempfile
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# ============================================================
# KIẾN TRÚC MODEL TỪ ĐỒ ÁN (BẮT BUỘC GIỐNG ENCODER)
# ============================================================
def get_rnn_class(module_name):
    return nn.GRU if module_name == "gru" else nn.LSTM

class NoiseGenerator(nn.Module):
    def __init__(self, module_name, input_dim, hidden_dim, num_layers, noise_scale=0.15):
        super().__init__()
        rnn_class = get_rnn_class(module_name)
        self.rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        self.noise_scale = noise_scale

    def forward(self, z, c_scaled):
        x = torch.cat([z, c_scaled], dim=-1)
        seq, _ = self.rnn(x)
        delta_t = self.noise_head(seq) * self.noise_scale
        final_ipd_scaled = c_scaled + delta_t
        return final_ipd_scaled, delta_t

# ============================================================
# PRE-SHARED PARAMETERS (PHẢI GIỐNG ENCODER)
# ============================================================
MODEL_PATH = "helper/models/covert_channel_generator.pth"
BCH_DECODE_BIN = "./bch_decode"

# Hằng số vật lý
PHYS_MIN = 0.0
PHYS_MAX = 2.5
BIT0_TIME = 0.5
BIT1_TIME = 1.0

BIT0_SCALED = (BIT0_TIME - PHYS_MIN) / (PHYS_MAX - PHYS_MIN)  # 0.20
BIT1_SCALED = (BIT1_TIME - PHYS_MIN) / (PHYS_MAX - PHYS_MIN)  # 0.40

# Hyperparams của GAN
SEQ_LEN = 24
HIDDEN_DIM = 24
NUM_LAYERS = 3
MODULE_NAME = "gru"
NOISE_SCALE = 0.4
SEED = 2025

# BCH Params (phải khớp với encoder)
BCH_M = 8
BCH_N = 255
BCH_T = 8
BCH_LOW_DELAY = 0.5    # ms - đại diện bit=1
BCH_HIGH_DELAY = 1  # ms - đại diện bit=0
BCH_THRESHOLD = 0.75   # ms - ngưỡng phân biệt 0/1

def inverse_scale(data_scaled):
    return data_scaled * (PHYS_MAX - PHYS_MIN) + PHYS_MIN

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def gan_decode(model, received_ipds, num_bits, device):
    """
    Giải mã bằng Shared Seed Hypothesis Testing (Tuần tự).
    Do RNN có 'trí nhớ', ta phải giải mã từng bit và dùng 
    kết quả đó làm bối cảnh (context) để đoán bit tiếp theo.
    """
    print(f"\n[+] Giải mã bằng Shared Seed Hypothesis Testing...")
    set_seed(SEED)
    
    num_windows = (num_bits + SEQ_LEN - 1) // SEQ_LEN
    decoded_bits = []
    
    ipd_idx = 0
    for w in range(num_windows):
        # 1. Khởi tạo hạt giống MỘT LẦN cho cả cửa sổ (giống hệt encoder)
        z_tensor = torch.rand(1, SEQ_LEN, 1, device=device)
        
        # Mảng lưu vết các bit đã giải mã trong cửa sổ hiện tại (Khởi tạo toàn 0)
        current_window_c = np.zeros((1, SEQ_LEN, 1), dtype=np.float32)
        
        for i in range(SEQ_LEN):
            if len(decoded_bits) >= num_bits:
                break
                
            if ipd_idx >= len(received_ipds):
                print(f"⚠️ Hết IPD nhận được, đệm bit 0 cho phần còn lại.")
                decoded_bits.append(0)
                continue
                
            received = received_ipds[ipd_idx]
            
            # --- THỬ GIẢ THUYẾT BIT = 0 ---
            current_window_c[0, i, 0] = BIT0_SCALED
            c_tensor_0 = torch.tensor(current_window_c, device=device)
            with torch.no_grad():
                _, expected_delta_0 = model(z_tensor, c_tensor_0)
            expected_ipd_0 = inverse_scale(BIT0_SCALED + expected_delta_0[0, i, 0].item())
            
            # --- THỬ GIẢ THUYẾT BIT = 1 ---
            current_window_c[0, i, 0] = BIT1_SCALED
            c_tensor_1 = torch.tensor(current_window_c, device=device)
            with torch.no_grad():
                _, expected_delta_1 = model(z_tensor, c_tensor_1)
            expected_ipd_1 = inverse_scale(BIT1_SCALED + expected_delta_1[0, i, 0].item())
            
            # --- ĐO KHOẢNG CÁCH & CHỐT BIT ---
            dist_0 = abs(received - expected_ipd_0)
            dist_1 = abs(received - expected_ipd_1)
            
            chosen_bit = 0 if dist_0 < dist_1 else 1
            decoded_bits.append(chosen_bit)
            
            # CẬP NHẬT TRẠNG THÁI: Ghi đè bit vừa đoán đúng để làm nền tảng đoán nhịp i+1
            current_window_c[0, i, 0] = BIT0_SCALED if chosen_bit == 0 else BIT1_SCALED
            
            ipd_idx += 1
            
    return decoded_bits


def bch_decode_bits(decoded_bits, original_size=None):
    """
    Chuyển decoded bits thành IAT values, ghi vào temp file,
    rồi gọi ./bch_decode binary để BCH decode + AES decrypt.
    Trả về bytes đã giải mã.
    """
    # Chuyển bits → IAT values (bit=1 → low_delay, bit=0 → high_delay)
    iat_values = []
    for bit in decoded_bits:
        iat_values.append(BCH_LOW_DELAY if bit == 1 else BCH_HIGH_DELAY)
    
    # Ghi IAT values vào temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
        f.write(','.join(f'{v:.1f}' for v in iat_values))
    
    try:
        # Gọi ./bch_decode
        cmd = [
            BCH_DECODE_BIN,
            str(BCH_M), str(BCH_N), str(BCH_T),
            temp_path, str(BCH_THRESHOLD)
        ]
        if original_size is not None:
            cmd.append(str(original_size))
        
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode != 0:
            stderr_text = result.stderr.decode('utf-8', errors='replace')
            print(f"bch_decode stderr:\n{stderr_text}", file=sys.stderr)
            raise RuntimeError(f"bch_decode failed with code {result.returncode}")
        
        # Print BCH decode info from stderr
        if result.stderr:
            for line in result.stderr.decode('utf-8', errors='replace').strip().split('\n'):
                print(f"  [bch] {line}")
        
        # stdout contains raw decoded bytes
        return result.stdout
    finally:
        os.unlink(temp_path)


def main():
    parser = argparse.ArgumentParser(description="CTC-GAN Decoder")
    parser.add_argument("--input", required=True, help="Input CSV file (IPDs from server)")
    parser.add_argument("--output", default="decoded_output", help="Output file path")
    parser.add_argument("--original-size", type=int, default=None,
                        help="Original file size in bytes (for exact truncation)")
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print("CTC-GAN DECODER (SHARED SEED - TANH)")
    print(f"{'='*50}")
    
    device = torch.device("cpu")
    
    # 1. Load model
    print(f"[1] Đang nạp AI Model...")
    model = NoiseGenerator(MODULE_NAME, input_dim=2, hidden_dim=HIDDEN_DIM, 
                           num_layers=NUM_LAYERS, noise_scale=NOISE_SCALE).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"    ✓ Model loaded from {MODEL_PATH}")
    except:
        full_dict = torch.load('covert_channel_full_model.pth', map_location=device)
        model.load_state_dict(full_dict['generator'])
        print(f"    ✓ Model loaded from covert_channel_full_model.pth")
    model.eval()
    
    # 2. Đọc IPD CSV
    print(f"[2] Đọc IPD từ {args.input}...")
    
    # header=None để tránh việc Pandas tự ý nuốt dòng đầu tiên làm tên cột
    df = pd.read_csv(args.input, header=None)
    
    # Ép kiểu cột đầu tiên về số thực, tự động bỏ qua chữ (như 'IPDs' hoặc 'IPD') nếu có
    received_ipds = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().tolist()
    
    print(f"    ✓ Đã đọc {len(received_ipds)} IPD values")
    
    # 3. GAN decode — tổng số bits = số IPDs (mỗi IPD mang 1 bit)
    num_bits = len(received_ipds)
    print(f"[3] GAN decode {num_bits} bits...")
    decoded_bits = gan_decode(model, received_ipds, num_bits, device)
    print(f"    ✓ GAN đã giải mã {len(decoded_bits)} bits")
    
    # 4. BCH decode (gọi C binary, AES tùy thuộc pass.txt)
    print(f"[4] BCH decode via {BCH_DECODE_BIN}...")
    decoded_bytes = bch_decode_bits(decoded_bits, args.original_size)
    print(f"    ✓ BCH decoded: {len(decoded_bytes)} bytes")
    
    # 5. Ghi file output
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        f.write(decoded_bytes)
    
    print(f"\n{'='*50}")
    print(f"🎉 HOÀN THÀNH! File đã lưu tại: {args.output}")
    print(f"   Kích thước: {len(decoded_bytes)} bytes")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
