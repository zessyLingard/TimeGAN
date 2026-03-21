# encoder_delta.py
"""
Covert Timing Channel Encoder - Delta Method (Theo ý tưởng của GV)
Sử dụng tính chất: output(z, c=1) - output(z, c=0) = CONSTANT
"""
import os
import argparse
import torch
import numpy as np
import pandas as pd
import json

import model_classes
from model_classes import (MCTimeGAN, Generator, Discriminator, Recovery, Embedder, ConditioningNetwork, Supervisor)

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

def generate_ipd_with_delta_method(model, bits, horizon, seed):
    """
    Encode bits using Delta Method.
    
    KEY INSIGHT: output(z, c=1) - output(z, c=0) = CONSTANT!
    So we can:
    1. Generate baseline with c=0
    2. For bit=1, add the delta
    """
    num_windows = (len(bits) + horizon - 1) // horizon
    
    print(f"\n[+] Generating IPD with Delta Method...")
    print(f"    Windows: {num_windows}")
    
    # Set seed
    set_shared_seed(seed)
    
    # Generate all baselines (c=0) and deltas
    all_ipd = []
    
    for window_idx in range(num_windows):
        # Save RNG state
        torch_state = torch.get_rng_state()
        np_state = np.random.get_state()
        
        # Generate baseline (c=0)
        cond_0 = np.zeros((1, horizon, 1), dtype=np.float32)
        with torch.no_grad():
            out_0 = model.transform(cond_0.shape, cond=cond_0)
        if out_0.ndim == 2:
            out_0 = out_0[:, :, np.newaxis]
        out_0 = (out_0 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
        
        # Restore RNG and generate with c=1
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        
        cond_1 = np.ones((1, horizon, 1), dtype=np.float32)
        with torch.no_grad():
            out_1 = model.transform(cond_1.shape, cond=cond_1)
        if out_1.ndim == 2:
            out_1 = out_1[:, :, np.newaxis]
        out_1 = (out_1 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
        
        # Calculate delta
        delta = out_1 - out_0
        
        # Encode bits in this window
        start_idx = window_idx * horizon
        end_idx = min(start_idx + horizon, len(bits))
        window_bits = bits[start_idx:end_idx]
        
        for i, bit in enumerate(window_bits):
            if bit == 0:
                ipd = out_0[i]  # Use baseline
            else:
                ipd = out_0[i] + delta[i]  # Add delta
            
            all_ipd.append(ipd)
    
    return np.array(all_ipd)

def main():
    parser = argparse.ArgumentParser(description="Encode file with Delta Method")
    parser.add_argument("--file", required=True, help="Secret file")
    parser.add_argument("--output", default="covert_ipd.csv")
    parser.add_argument("--metadata", default="covert_metadata.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("COVERT TIMING CHANNEL ENCODER (Delta Method)")
    print(f"{'='*60}")
    
    # Setup
    device = torch.device("cpu")
    model_classes.device = device
    
    # Load model
    print(f"\n[1] Loading model...")
    model = torch.load(args.model, map_location=device, weights_only=False)
    model.eval()
    print(f"    ✓ Model loaded")
    
    # Read file
    print(f"\n[2] Reading file...")
    bits, file_size = file_to_bits(args.file)
    
    # Encode
    print(f"\n[3] Encoding with Delta Method (seed={args.seed})...")
    ipd_sequence = generate_ipd_with_delta_method(model, bits, args.horizon, args.seed)
    
    print(f"    ✓ Encoded {len(ipd_sequence)} IPDs")
    print(f"    Range: [{ipd_sequence.min():.4f}, {ipd_sequence.max():.4f}]")
    
    # Save
    print(f"\n[4] Saving...")
    pd.DataFrame({'IPD': ipd_sequence}).to_csv(args.output, index=False)
    
    metadata = {
        "filename": os.path.basename(args.file),
        "num_bits": len(bits),
        "num_packets": len(ipd_sequence),
        "seed": args.seed,
        "horizon": args.horizon,
        "method": "delta",
        "min_val": MIN_VAL,
        "max_val": MAX_VAL
    }
    
    with open(args.metadata, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"    ✓ {args.output}")
    print(f"    ✓ {args.metadata}")
    
    print(f"\n{'='*60}")
    print("✅ ENCODING COMPLETE (Delta Method)")
    print(f"    Method: Remove noise with shared seed")
    print(f"    Seed: {args.seed}")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. python receiver_delta.py --seed {args.seed} --port 3334")
    print(f"  2. python sender.py --ip 127.0.0.1 --port 3334")

if __name__ == "__main__":
    main()