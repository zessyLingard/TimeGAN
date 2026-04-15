"""
CTC Decoder (Final: Shared Seed + Dynamic Shift)
"""
import os
import argparse
import subprocess
import tempfile
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# KIẾN TRÚC PURE GAN
class NoiseGenerator(nn.Module):
    def __init__(self, module_name, z_dim, hidden_dim, num_layers, noise_scale=0.10):
        super().__init__()
        rnn_class = nn.GRU if module_name == "gru" else nn.LSTM
        self.rnn = rnn_class(input_size=z_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.base_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.noise_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())
        self.smearing_head = nn.Linear(hidden_dim, 1)
        self.noise_scale = noise_scale

    def forward(self, z):
        seq, _ = self.rnn(z)
        base_prob = self.base_head(seq)
        delta_prob = self.noise_head(seq) * self.noise_scale
        smearing_noise = self.smearing_head(seq)
        return torch.clamp(base_prob + delta_prob, 0.0, 1.0), delta_prob, smearing_noise

MODEL_PATH = "covert_channel_full_model.pth"
REAL_DATA_PATH = "real_ipds_doH.csv"
BCH_DECODE_BIN = "./bch_decode"

SHARED_SEED = 2026 
PAYLOAD_THRESHOLD = 0.10

BCH_M, BCH_N, BCH_T = 8, 255, 8
BCH_LOW_DELAY, BCH_HIGH_DELAY, BCH_THRESHOLD = 0.5, 1, 0.75

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def bch_decode_bits(decoded_bits, original_size=None):
    iat_values = [BCH_LOW_DELAY if b == 1 else BCH_HIGH_DELAY for b in decoded_bits]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
        f.write(','.join(f'{v:.1f}' for v in iat_values))
    try:
        cmd = [BCH_DECODE_BIN, str(BCH_M), str(BCH_N), str(BCH_T), temp_path, str(BCH_THRESHOLD)]
        if original_size: cmd.append(str(original_size))
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0: raise RuntimeError("BCH Decode failed")
        return result.stdout
    finally:
        os.unlink(temp_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="decoded_secret.txt")
    parser.add_argument("--original-size", type=int, default=None)
    args = parser.parse_args()
    device = torch.device("cpu")

    print("[1] Load Pure GAN Model & CSV...")
    model = NoiseGenerator("gru", 1, 24, 3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['generator'])
    model.eval()

    real_ipds = pd.to_numeric(pd.read_csv(REAL_DATA_PATH, header=None).iloc[:, 0], errors='coerce').dropna().values
    sorted_real = np.sort(real_ipds)
    emp_cdf = np.arange(1, len(sorted_real) + 1) / len(sorted_real)
    _, unique_idx = np.unique(emp_cdf, return_index=True)
    prob_to_time = interp1d(emp_cdf[unique_idx], sorted_real[unique_idx], bounds_error=False, fill_value=(sorted_real[0], sorted_real[-1]))

    print(f"[2] Khởi tạo AI Blueprint (Shared Seed = {SHARED_SEED})...")
    set_seed(SHARED_SEED)
    N_CHUNKS = 5000 
    z_val = torch.rand(N_CHUNKS, 24, 1, device=device)
    with torch.no_grad(): gan_probs_t, _, _ = model(z_val)
    ai_blueprint_times = prob_to_time(gan_probs_t.numpy()).flatten()

    print("[3] Phân tích IPD nhận được từ Cổng 3334...")
    df = pd.read_csv(args.input, header=None)
    received_ipds = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().tolist()
    
    # server.c ghi ra file bằng giây (VD: 0.123456), nên KHÔNG CẦN CHIA 1000
    if received_ipds[0] > 10.0: received_ipds = [x / 1000.0 for x in received_ipds]

    print("[4] Giải mã bằng Dynamic Index Sync...")
    decoded_bits = []
    sync_rng = np.random.RandomState(SHARED_SEED) # PRNG đồng bộ y hệt Alice
    
    # Do TCP không mất gói, Index (i) của Bob khớp 100% với Index của Alice
    for i in range(len(received_ipds)):
        ai_time = ai_blueprint_times[i]
        
        if ai_time >= PAYLOAD_THRESHOLD:
            r_time = received_ipds[i]
            
            # Sinh khoảng dịch ngẫu nhiên Y HỆT Alice
            dynamic_shift = sync_rng.uniform(0.05, 0.15)
            
            target_bit0 = max(ai_time - dynamic_shift, 0.01)
            target_bit1 = ai_time + dynamic_shift
            
            if abs(r_time - target_bit0) < abs(r_time - target_bit1):
                decoded_bits.append(0)
            else:
                decoded_bits.append(1)

    print(f"    ✓ Nhặt được {len(decoded_bits)} Bits Data.")
    print("[5] Gọi BCH Decode...")
    decoded_bytes = bch_decode_bits(decoded_bits, args.original_size)
    with open(args.output, 'wb') as f: f.write(decoded_bytes)
    print(f"🎉 HOÀN THÀNH! Đã khôi phục file vào: {args.output}")

if __name__ == "__main__":
    main()