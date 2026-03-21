"""
Covert Timing Channel Encoder - TRUE Noise Removal Method
Theo ý tưởng của Giáo Viên: output = f(z) + g(c)

Decoder có thể remove noise: signal = output - f(z)
"""
import os
import argparse
import torch
import numpy as np
import pandas as pd
import json

import model_classes
from model_classes import (
    MCTimeGAN, 
    ConditioningNetwork, 
    Embedder, 
    Recovery, 
    Generator, 
    Supervisor, 
    Discriminator
)

# Constants
DEFAULT_MODEL = "helper/models/mctimegan_model.pth"
DEFAULT_HORIZON = 24
DEFAULT_SEED = 2025

MIN_VAL = 0.0615832379381183
MAX_VAL = 0.2239434066266675

def set_shared_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def file_to_bits(file_path):
    """Convert file to bits"""
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    
    bits = []
    for byte in file_bytes:
        bits.extend([int(b) for b in format(byte, '08b')])
    
    print(f"[+] File: {os.path.basename(file_path)}")
    print(f"    Size: {len(file_bytes)} bytes = {len(bits)} bits")
    
    return bits, len(file_bytes)

def generate_ipd_with_noise_removal(model, bits, horizon, seed):
    """
    TRUE Noise Removal Method
    
    Nguyên lý:
      output(z, c=0) = f(z) + β      ← baseline (noise)
      output(z, c=1) = f(z) + α + β  ← baseline + signal
      
      Delta = α = CONSTANT (đã verify!)
    
    Encoder:
      - bit=0: Gửi baseline (chỉ có noise)
      - bit=1: Gửi baseline + alpha (noise + signal)
    
    Decoder:
      - Generate CÙNG baseline
      - Remove baseline: clean = received - baseline
      - Threshold trên clean signal
    """
    num_windows = (len(bits) + horizon - 1) // horizon
    
    print(f"\n[+] Generating IPD with TRUE Noise Removal Method...")
    print(f"    Windows: {num_windows}")
    
    set_shared_seed(seed)
    
    all_ipd = []
    all_baselines = []
    all_alphas = []
    
    for window_idx in range(num_windows):
        # Save RNG state
        torch_state = torch.get_rng_state()
        np_state = np.random.get_state()
        
        # Generate baseline (c=0) = f(z) + β
        cond_0 = np.zeros((1, horizon, 1), dtype=np.float32)
        with torch.no_grad():
            out_0 = model.transform(cond_0.shape, cond=cond_0)
        if out_0.ndim == 2:
            out_0 = out_0[:, :, np.newaxis]
        baseline = (out_0 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
        
        # Restore RNG and generate with c=1 = f(z) + α + β
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        
        cond_1 = np.ones((1, horizon, 1), dtype=np.float32)
        with torch.no_grad():
            out_1 = model.transform(cond_1.shape, cond=cond_1)
        if out_1.ndim == 2:
            out_1 = out_1[:, :, np.newaxis]
        out_1_scaled = (out_1 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
        
        # Calculate alpha (signal difference)
        delta = out_1_scaled - baseline
        alpha = delta.mean()  # Use mean as fixed alpha for this window
        
        all_baselines.append(baseline)
        all_alphas.append(alpha)
        
        # Encode bits in this window
        start_idx = window_idx * horizon
        end_idx = min(start_idx + horizon, len(bits))
        window_bits = bits[start_idx:end_idx]
        
        for i, bit in enumerate(window_bits):
            if bit == 0:
                ipd = baseline[i]  # Send baseline only (noise)
            else:
                ipd = baseline[i] + alpha  # Send baseline + signal
            
            all_ipd.append(ipd)
    
    # Statistics
    avg_baseline = np.mean([b.mean() for b in all_baselines])
    avg_alpha = np.mean(all_alphas)
    
    print(f"    ✓ Baseline (noise) mean: {avg_baseline:.4f}s")
    print(f"    ✓ Alpha (signal strength): {avg_alpha:.4f}s")
    print(f"    ✓ Separation: {avg_alpha:.4f}s ({avg_alpha/avg_baseline*100:.1f}%)")
    
    return np.array(all_ipd), avg_alpha

def main():
    parser = argparse.ArgumentParser(description="Encode with TRUE Noise Removal Method")
    parser.add_argument("--file", required=True, help="Secret file to encode")
    parser.add_argument("--output", default="covert_ipd.csv", help="Output IPD CSV file")
    parser.add_argument("--metadata", default="covert_metadata.json", help="Metadata JSON file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shared seed")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Time horizon")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("COVERT TIMING CHANNEL ENCODER")
    print("Method: TRUE Noise Removal (output = noise + signal)")
    print(f"{'='*60}")
    
    # Setup
    device = torch.device("cpu")
    model_classes.device = device
    
    # Load model
    print(f"\n[1] Loading model...")
    if not os.path.exists(args.model):
        print(f"    ❌ Model not found: {args.model}")
        return
    
    model = torch.load(args.model, map_location=device, weights_only=False)
    model.eval()
    print(f"    ✓ Model loaded: {args.model}")
    
    # Read file
    print(f"\n[2] Reading file...")
    bits, file_size = file_to_bits(args.file)
    
    # Encode
    print(f"\n[3] Encoding (seed={args.seed})...")
    ipd_sequence, alpha = generate_ipd_with_noise_removal(model, bits, args.horizon, args.seed)
    
    print(f"    ✓ Encoded {len(ipd_sequence)} IPDs")
    print(f"    IPD range: [{ipd_sequence.min():.4f}, {ipd_sequence.max():.4f}]")
    print(f"    IPD mean: {ipd_sequence.mean():.4f}")
    
    # Save
    print(f"\n[4] Saving...")
    pd.DataFrame({'IPD': ipd_sequence}).to_csv(args.output, index=False)
    
    threshold = alpha / 2  # Optimal threshold = midpoint
    
    metadata = {
        "filename": os.path.basename(args.file),
        "num_bits": len(bits),
        "num_packets": len(ipd_sequence),
        "seed": args.seed,
        "horizon": args.horizon,
        "alpha": float(alpha),
        "threshold": float(threshold),
        "method": "noise_removal",
        "min_val": MIN_VAL,
        "max_val": MAX_VAL
    }
    
    with open(args.metadata, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"    ✓ {args.output}")
    print(f"    ✓ {args.metadata}")
    
    print(f"\n{'='*60}")
    print("✅ ENCODING COMPLETE")
    print(f"{'='*60}")
    print(f"\nKey Parameters:")
    print(f"  Alpha (signal strength): {alpha:.4f}s")
    print(f"  Threshold: {threshold:.4f}s")
    print(f"  Method: Remove baseline, threshold on clean signal")
    print(f"\nNext steps:")
    print(f"  1. python receiver_noise_removal.py --seed {args.seed} --port 3334")
    print(f"  2. python sender.py --ip 127.0.0.1 --port 3334")

if __name__ == "__main__":
    main()
