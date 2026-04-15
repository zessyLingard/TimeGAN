"""
CTC-GAN Encoder (Final: Shared Seed + Dynamic Shift)
"""
import os
import argparse
import subprocess
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ============================================================
# KIẾN TRÚC PURE GAN
# ============================================================
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

# ============================================================
# CẤU HÌNH HỆ THỐNG
# ============================================================
MODEL_PATH = "covert_channel_full_model.pth"
REAL_DATA_PATH = "real_ipds_doH.csv"
BCH_ENCODE_BIN = "./bch_encode"

SHARED_SEED = 2026 
PAYLOAD_THRESHOLD = 0.10

# BCH params (Khớp với file C cũ của bạn)
BCH_LOW_DELAY, BCH_HIGH_DELAY, BCH_THRESHOLD = 0.5, 1, 0.75

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def bch_encode_file(input_file):
    result = subprocess.run([BCH_ENCODE_BIN, input_file, str(BCH_LOW_DELAY), str(BCH_HIGH_DELAY)], capture_output=True, text=True)
    if result.returncode != 0: raise RuntimeError("BCH Encode failed")
    iat_values = [float(x) for x in result.stdout.strip().split(',')]
    return [1 if iat <= BCH_THRESHOLD else 0 for iat in iat_values]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--output", default="covert_ipd.csv")
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

    print("[2] Đọc Data thật & Mã hóa BCH...")
    encoded_bits = bch_encode_file(args.file)
    
    print(f"[3] Khởi tạo AI Blueprint (Shared Seed = {SHARED_SEED})...")
    set_seed(SHARED_SEED)
    N_CHUNKS = max((len(encoded_bits) * 20) // 24, 500)
    z_val = torch.rand(N_CHUNKS, 24, 1, device=device)
    with torch.no_grad():
        gan_probs_t, _, _ = model(z_val)
    gan_times_np = prob_to_time(gan_probs_t.numpy()).flatten()

    ipds, bit_idx = [], 0
    sync_rng = np.random.RandomState(SHARED_SEED) # PRNG đồng bộ động

    for ai_time in gan_times_np:
        if bit_idx >= len(encoded_bits): break
        
        if ai_time < PAYLOAD_THRESHOLD:
            ipds.append(round(float(ai_time), 6)) # Xả rác tự nhiên
        else:
            current_bit = encoded_bits[bit_idx]
            
            # 🔥 ĐỘT PHÁ: Dynamic Shift ngẫu nhiên cho từng gói
            dynamic_shift = sync_rng.uniform(0.05, 0.15)
            
            if current_bit == 0:
                target_time = max(ai_time - dynamic_shift, 0.01)
            else:
                target_time = ai_time + dynamic_shift
                
            smear = np.random.uniform(-0.02, 0.02)
            ipds.append(round(float(target_time + smear), 6))
            bit_idx += 1

    ipds_ms = [ipd * 1000.0 for ipd in ipds]
    with open(args.output, 'w') as f:
        f.write(','.join(f'{v:.3f}' for v in ipds_ms))
    print(f"🎉 Hoàn thành! Ghi {len(ipds)} gói tin ra {args.output}")

if __name__ == "__main__":
    main()