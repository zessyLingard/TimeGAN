import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import sys
import subprocess
import os

# ==========================================
# 1. CONSTANTS & HYPERPARAMETERS
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Protocol
FRAME_SIZE    = 10
BITS_PER_FILE = 510
TOTAL_PACKETS = 5100

# Neural Network (Matches kaggle_notebook.py exactly)
CONTEXT_W = 32
NOISE_DIM = 16
G_HIDDEN  = 128
G_LAYERS  = 3

COND_COVER = 0
COND_BIT0  = 1
COND_BIT1  = 2

# Thresholds
GAP_THRESHOLD    = 0.08
BIT0_LO, BIT0_HI = 0.1, 0.2
BIT1_LO, BIT1_HI = 0.45, 0.7
DECODE_THRESHOLD = 0.325

# ==========================================
# 2. DATA LOADING & SCALING
# ==========================================
print("[*] Loading VPN dataset for Cover extraction and scaling factors...")
try:
    df = pd.read_csv("data/vpn_legit.csv")
except FileNotFoundError:
    df = pd.read_csv("vpn_legit.csv")

raw_series = df["IPD" if "IPD" in df.columns else "IPDs"]
raw_series = pd.to_numeric(raw_series, errors='coerce').dropna()
raw_ipds = raw_series.values.astype(np.float64)[500_000:600_000]
LOG_OFFSET = 1e-3
log_ipds   = np.log(raw_ipds + LOG_OFFSET)
LOG_MIN    = float(log_ipds.min())
LOG_MAX    = float(log_ipds.max())

def to_norm(val):
    return (np.log(val + LOG_OFFSET) - LOG_MIN) / (LOG_MAX - LOG_MIN)

NORM_COVER_HI = to_norm(GAP_THRESHOLD - 0.001)
NORM_BIT0_LO  = to_norm(BIT0_LO)
NORM_BIT0_HI  = to_norm(BIT0_HI)
NORM_BIT1_LO  = to_norm(BIT1_LO)
NORM_BIT1_HI  = to_norm(BIT1_HI)

# ==========================================
# 3. GENERATOR ARCHITECTURE
# ==========================================
class ConditionalBN(nn.Module):
    def __init__(self, num_features, num_conditions=3):
        super().__init__()
        self.bn    = nn.BatchNorm1d(num_features, affine=False)
        self.gamma = nn.Embedding(num_conditions, num_features)
        self.beta  = nn.Embedding(num_conditions, num_features)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x, cond_idx):
        x = self.bn(x)
        return self.gamma(cond_idx) * x + self.beta(cond_idx)

class Generator(nn.Module):
    def __init__(self, hidden=G_HIDDEN, layers=G_LAYERS, noise_dim=NOISE_DIM):
        super().__init__()
        self.noise_dim   = noise_dim
        self.cond_embed  = nn.Embedding(3, 16)

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                            num_layers=layers, batch_first=True, dropout=0.2)
        self.ln   = nn.LayerNorm(hidden)
        self.cbn  = ConditionalBN(hidden, num_conditions=3)

        backbone_in = hidden + 16
        self.backbone = nn.Sequential(
            nn.Linear(backbone_in, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        self.head_mean   = nn.Linear(128, 1)
        self.head_logvar = nn.Linear(128, 1)
        nn.init.constant_(self.head_logvar.bias, -4.0)

    def forward(self, context, cond_idx, noise):
        _, (h_n, _) = self.lstm(context)
        h    = self.ln(h_n[-1])
        h    = self.cbn(h, cond_idx)
        cond = self.cond_embed(cond_idx)
        x    = torch.cat([h, cond], dim=-1)

        feat   = self.backbone(x)
        mean   = self.head_mean(feat)
        logvar = self.head_logvar(feat)
        logvar = torch.clamp(logvar, -8, 0)

        std    = torch.exp(0.5 * logvar)
        sample = mean + std * noise[:, :1]
        return torch.clamp(sample, 1e-6, 1.0 - 1e-6), logvar

# ==========================================
# 4. GENERATION FUNCTION
# ==========================================
@torch.no_grad()
def generate_covert_traffic(G, payload_bits):
    G.eval()
    total_covers = TOTAL_PACKETS - BITS_PER_FILE
    counts = np.ones(BITS_PER_FILE, dtype=int)
    remaining = total_covers - BITS_PER_FILE
    
    weights = np.random.exponential(scale=1.0, size=BITS_PER_FILE)
    weights /= weights.sum()
    added = np.round(weights * remaining).astype(int)
    
    diff = remaining - added.sum()
    if diff > 0:
        np.add.at(added, np.random.choice(BITS_PER_FILE, size=diff, replace=True), 1)
    elif diff < 0:
        for _ in range(-diff):
            idx = np.random.choice(np.where(added > 0)[0])
            added[idx] -= 1
            
    cover_counts = counts + added
    conditions = []
    for bit, c_count in zip(payload_bits, cover_counts):
        conditions.extend([COND_COVER] * c_count)
        conditions.append(COND_BIT0 if bit == 0 else COND_BIT1)

    conds_t = torch.tensor(conditions, dtype=torch.long, device=device).unsqueeze(0)

    real_covers = raw_ipds[raw_ipds < GAP_THRESHOLD]
    real_covers_norm = (np.log(real_covers + LOG_OFFSET) - LOG_MIN) / (LOG_MAX - LOG_MIN)
    cover_idx = np.random.randint(0, max(1, len(real_covers_norm) - TOTAL_PACKETS))

    init_vals   = real_covers[cover_idx:cover_idx+CONTEXT_W]
    init_norm   = (np.log(init_vals + LOG_OFFSET) - LOG_MIN) / (LOG_MAX - LOG_MIN)
    context     = torch.tensor(init_norm, dtype=torch.float32, device=device).reshape(1, CONTEXT_W, 1)

    generated = []
    for t in range(TOTAL_PACKETS):
        cond_idx = conds_t[:, t]
        noise    = torch.randn(1, NOISE_DIM, device=device)

        ipd_norm, _ = G(context, cond_idx, noise)

        c = cond_idx.item()
        if c == COND_COVER:
            ipd_norm_val = float(real_covers_norm[cover_idx % len(real_covers_norm)])
            cover_idx += 1
            ipd_norm = torch.tensor([[ipd_norm_val]], device=device)
        elif c == COND_BIT0:
            ipd_norm = torch.clamp(ipd_norm, min=NORM_BIT0_LO + 1e-4, max=NORM_BIT0_HI - 1e-4)
        elif c == COND_BIT1:
            ipd_norm = torch.clamp(ipd_norm, min=NORM_BIT1_LO + 1e-4, max=NORM_BIT1_HI - 1e-4)

        val = max(0.0, float(np.exp(ipd_norm.item() * (LOG_MAX - LOG_MIN) + LOG_MIN) - LOG_OFFSET))
        generated.append(val)
        context = torch.cat([context[:, 1:, :], ipd_norm.unsqueeze(1)], dim=1)

    return np.array(generated)

# ==========================================
# 5. MAIN ENCODER SCRIPT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Full Covert Channel Encoder (AES + BCH + GAN)")
    parser.add_argument("input_file", help="The secret text file to encode")
    parser.add_argument("output_file", help="The comma-separated timing file for client.c")
    parser.add_argument("--weights", default="stealth_ctc_generator_v2.pth", help="Path to trained GAN weights")
    args = parser.parse_args()

    if not os.path.exists("./bch_encode"):
        print("[*] Compiling bch_encode.c...")
        subprocess.run(["gcc", "-o", "bch_encode", "bch_encode.c", "-lcrypto"], check=True)

    print(f"[*] Encrypting and Encoding: {args.input_file}")
    result = subprocess.run(
        ["./bch_encode", args.input_file, "0.1", "1.0"],
        capture_output=True, text=True, check=True
    )
    
    raw_str = result.stdout.strip()
    payload_bits = []
    if raw_str:
        for val in raw_str.split(","):
            if val:
                payload_bits.append(1 if float(val) > 0.5 else 0)
                
    print(f"[*] Extracted {len(payload_bits)} secure payload bits.")

    if len(payload_bits) < BITS_PER_FILE:
        payload_bits.extend([0] * (BITS_PER_FILE - len(payload_bits)))
    elif len(payload_bits) > BITS_PER_FILE:
        payload_bits = payload_bits[:BITS_PER_FILE]

    print(f"[*] Loading GAN weights from {args.weights}...")
    G = Generator().to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
        G.load_state_dict(checkpoint["generator_state_dict"])
    else:
        G.load_state_dict(checkpoint)
    G.eval()

    print("[*] Synthesizing stealthy IPD delays via Reparameterized GAN...")
    traffic_ipds_seconds = generate_covert_traffic(G, np.array(payload_bits))

    float_ipds = [f"{ipd:.9f}" for ipd in traffic_ipds_seconds]
    output_str = "\n".join(float_ipds)
    
    with open(args.output_file, 'w') as f:
        f.write(output_str + "\n")
        
    print(f"✅ Success! Saved {len(traffic_ipds_seconds)} IPDs to {args.output_file}")
    print(f"   -> Run Client: python3 simple_client.py <IP> {args.output_file}")

if __name__ == "__main__":
    main()
