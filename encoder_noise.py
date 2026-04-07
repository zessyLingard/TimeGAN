"""
CTC-GAN Encoder (Covert Timing Channel)
Pipeline: file → AES-256 → BCH → GAN (Shared Seed) → IPD (CSV)

Reads AES key from aes_key.txt
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import bch_wrapper

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

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
MODEL_PATH = "models/covert_channel_generator.pth" 
KEY_FILE = "aes_key.txt"

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
NOISE_SCALE = 0.4 # Phải khớp với lúc training
SEED = 2025 # Shared seed cực kỳ quan trọng

# BCH Params
BCH_N = 255
BCH_K = 191
BCH_T = 8

def inverse_scale(data_scaled):
    return data_scaled * (PHYS_MAX - PHYS_MIN) + PHYS_MIN

def load_key():
    with open(KEY_FILE, 'r') as f:
        key_hex = f.read().strip()
    return bytes.fromhex(key_hex)

def bytes_to_bits(data):
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    return bits

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="File to encode")
    parser.add_argument("--output", default="covert_ipd.csv", help="Output CSV")
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print("CTC-GAN ENCODER (SHARED SEED - TANH)")
    print(f"{'='*50}")
    
    device = torch.device("cpu")
    
    # 1. Load AES key
    key = load_key()
    print(f"[1] AES key loaded from {KEY_FILE}")
    
    # 2. Khởi tạo và load Model
    model = NoiseGenerator(MODULE_NAME, input_dim=2, hidden_dim=HIDDEN_DIM, 
                           num_layers=NUM_LAYERS, noise_scale=NOISE_SCALE).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"[2] GAN model (generator) loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Lỗi load model: Có thể bạn lưu full dict. Đang thử cách 2...")
        full_dict = torch.load('covert_channel_full_model.pth', map_location=device)
        model.load_state_dict(full_dict['generator'])
        print(f"[2] GAN model loaded from covert_channel_full_model.pth")
        
    model.eval()

    # 3. Khởi tạo BCH
    k = bch_wrapper._BCH_K
    print(f"[3] BCH({BCH_N},{k}) ready, t={BCH_T}")
    
    # 4. Đọc file
    with open(args.file, 'rb') as f:
        plaintext = f.read()
    print(f"[4] Read {len(plaintext)} bytes: {args.file}")
    
    # 5. AES encrypt
    nonce = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = nonce + encryptor.update(plaintext) + encryptor.finalize()
    print(f"[5] AES encrypted: {len(ciphertext)} bytes")
    
    data_bits = bytes_to_bits(ciphertext)
    
    # 6. BCH encode
    num_blocks = (len(data_bits) + k - 1) // k
    padded_bits = data_bits + [0] * (num_blocks * k - len(data_bits))
    encoded_bits = []
    for b in range(num_blocks):
        block = padded_bits[b*k:(b+1)*k]
        codeword = bch_wrapper.encode(block)
        encoded_bits.extend(codeword)
    print(f"[6] BCH encoded: {len(encoded_bits)} bits ({num_blocks} blocks)")
    
    # 7. GAN encode (Nhúng thông điệp vào Covert Channel)
    print("[7] Nhúng gói tin qua GAN...")
    
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
        
    print(f"[7] GAN encoded: {len(ipds)} IPDs")
    
    # 8. Save
    pd.DataFrame({'IPDs': ipds}).to_csv(args.output, index=False)
    print(f"[8] Saved: {args.output}")
    
    print(f"\n{'='*50}")
    print(f"Hoàn thành! {len(plaintext)} bytes → {len(ipds)} IPDs")
    print(f"Tổng thời gian gửi dự kiến: {sum(ipds):.1f} giây")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()