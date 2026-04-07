"""
CTC-GAN Encoder (Covert Timing Channel)
Pipeline: file → BCH encode (C binary, AES optional via pass.txt) → GAN (Shared Seed) → IPD (CSV)

AES encryption is handled by bch_encode (compile with -DUSE_AES if needed).
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
import helper

# ============================================================
# KIẾN TRÚC MODEL TỪ ĐỒ ÁN
# ============================================================
def get_rnn_class(module_name):
    return nn.GRU if module_name == "gru" else nn.LSTM

class NoiseGenerator(nn.Module):
    """Bộ Sinh Nhiễu Cộng Gộp cho Covert Timing Channel.
       Phiên bản Tanh Additive (Không giới hạn cứng)
    """
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
# PRE-SHARED PARAMETERS (PHẢI GIỐNG FILE HUẤN LUYỆN)
# ============================================================
MODEL_PATH = "helper/models/covert_channel_generator.pth"
BCH_ENCODE_BIN = "./bch_encode"

# Hằng số vật lý (từ đồ án)
PHYS_MIN = 0.0
PHYS_MAX = 2.5
BIT0_TIME = 0.5
BIT1_TIME = 1.0

BIT0_SCALED = (BIT0_TIME - PHYS_MIN) / (PHYS_MAX - PHYS_MIN) # 0.20
BIT1_SCALED = (BIT1_TIME - PHYS_MIN) / (PHYS_MAX - PHYS_MIN) # 0.40

# Hyperparams của GAN
SEQ_LEN = 24
HIDDEN_DIM = 24
NUM_LAYERS = 3
MODULE_NAME = "gru"
NOISE_SCALE = 0.4  # Phải khớp với lúc training
SEED = 2025         # Shared seed cực kỳ quan trọng

# BCH Params (dùng cho bch_encode binary, PHẢI KHỚP VỚI DECODER)
BCH_LOW_DELAY = 0.5    # ms - đại diện bit=1
BCH_HIGH_DELAY = 1     # ms - đại diện bit=0
BCH_THRESHOLD = 0.75   # ms - ngưỡng phân biệt 0/1

def inverse_scale(data_scaled):
    return data_scaled * (PHYS_MAX - PHYS_MIN) + PHYS_MIN

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def bch_encode_file(input_file):
    """Gọi ./bch_encode binary và parse output IAT thành bits."""
    result = subprocess.run(
        [BCH_ENCODE_BIN, input_file, str(BCH_LOW_DELAY), str(BCH_HIGH_DELAY)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"bch_encode stderr:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"bch_encode failed with code {result.returncode}")
    
    # Parse stderr for info
    if result.stderr:
        for line in result.stderr.strip().split('\n'):
            print(f"  [bch] {line}")
    
    # Parse stdout: CSV of IAT values → convert to bits
    iat_str = result.stdout.strip()
    if not iat_str:
        raise RuntimeError("bch_encode produced no output")
    
    iat_values = [float(x) for x in iat_str.split(',')]
    
    # IAT → bits: <= threshold → 1, > threshold → 0
    encoded_bits = []
    for iat in iat_values:
        encoded_bits.append(1 if iat <= BCH_THRESHOLD else 0)
    
    return encoded_bits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="File to encode")
    parser.add_argument("--output", default="covert_ipd.csv", help="Output CSV")
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print("CTC-GAN ENCODER (SHARED SEED - TANH)")
    print(f"{'='*50}")
    
    device = torch.device("cpu")
    
    # 1. Khởi tạo và load Model
    model = NoiseGenerator(MODULE_NAME, input_dim=2, hidden_dim=HIDDEN_DIM, 
                           num_layers=NUM_LAYERS, noise_scale=NOISE_SCALE).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"[1] GAN model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Lỗi load model: Có thể bạn lưu full dict. Đang thử cách 2...")
        full_dict = torch.load('covert_channel_full_model.pth', map_location=device)
        model.load_state_dict(full_dict['generator'])
        print(f"[1] GAN model loaded from covert_channel_full_model.pth")
        
    model.eval()

    # 2. Đọc file size (để decoder biết cắt đúng)
    file_size = os.path.getsize(args.file)
    print(f"[2] Input file: {args.file} ({file_size} bytes)")
    
    # 3. BCH encode (gọi C binary, AES tùy thuộc pass.txt)
    print(f"[3] BCH encoding via {BCH_ENCODE_BIN}...")
    encoded_bits = bch_encode_file(args.file)
    print(f"    → {len(encoded_bits)} encoded bits")
    
    # 4. GAN encode (Nhúng thông điệp vào Covert Channel)
    print("[4] Nhúng gói tin qua GAN...")
    
    # Reset seed để đảm bảo vector Z đồng bộ giữa người gửi và nhận
    set_seed(SEED)
    
    # Cắt thành các cửa sổ có độ dài SEQ_LEN (24)
    num_windows = (len(encoded_bits) + SEQ_LEN - 1) // SEQ_LEN
    ipds = []
    
    for w in range(num_windows):
        start_idx = w * SEQ_LEN
        end_idx = min(start_idx + SEQ_LEN, len(encoded_bits))
        window_bits = encoded_bits[start_idx:end_idx]
        
        # Nếu cửa sổ cuối bị thiếu, đệm thêm bit 0 (sẽ bị bỏ qua khi ghi)
        pad_len = SEQ_LEN - len(window_bits)
        if pad_len > 0:
            window_bits.extend([0] * pad_len)
            
        # Chuyển mảng bit [0, 1] thành tensor condition (C_scaled) [0.2, 0.4]
        window_c_scaled = np.where(np.array(window_bits) == 0, BIT0_SCALED, BIT1_SCALED)
        
        # Chuyển sang PyTorch Tensor, batch=1
        c_tensor = torch.tensor(window_c_scaled, dtype=torch.float32, device=device).view(1, SEQ_LEN, 1)
        
        # Sinh nhiễu Z cùng seed
        z_tensor = torch.rand(1, SEQ_LEN, 1, device=device)
        
        # Đưa vào mô hình để cộng nhiễu tàng hình
        with torch.no_grad():
            final_ipd_scaled, _ = model(z_tensor, c_tensor)
            
        # Giải nén về số giây vật lý
        final_ipd_phys = inverse_scale(final_ipd_scaled.cpu().numpy().flatten())
        
        # Chỉ lưu đúng số lượng bit thực tế của cửa sổ này
        actual_len = SEQ_LEN - pad_len
        ipds.extend(final_ipd_phys[:actual_len].tolist())
        
    print(f"[4] GAN encoded: {len(ipds)} IPDs")
    
      # 5. Save — format: comma-separated milliseconds, 1 dòng (client.c đọc dạng này)
    ipds_ms = [ipd * 1000.0 for ipd in ipds]  # giây → milliseconds
    with open(args.output, 'w') as f:
        f.write(','.join(f'{v:.3f}' for v in ipds_ms))
    print(f"[5] Saved: {args.output} (ms, comma-separated)")
    
    print(f"\n{'='*50}")
    print(f"Hoàn thành! {file_size} bytes → {len(ipds)} IPDs")
    print(f"Original file size: {file_size} (pass to decoder)")
    print(f"Tổng thời gian gửi dự kiến: {sum(ipds):.1f} giây")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()