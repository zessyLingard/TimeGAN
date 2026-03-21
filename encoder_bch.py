"""
Covert Timing Channel Encoder with BCH - STEALTH VERSION
Pipeline: file → [len+payload+CRC] → BCH → GAN → IPD

Message format: [2 bytes length][payload][4 bytes CRC32]
No metadata file needed - all params pre-shared with receiver
"""
import os
import sys
import argparse
import struct
import zlib
import torch
import numpy as np
import pandas as pd
import galois

sys.path.append(os.getcwd())
import model_classes
from model_classes import (MCTimeGAN, Generator, Discriminator, Recovery, Embedder, ConditioningNetwork, Supervisor)

# ============================================================
# PRE-SHARED PARAMETERS - MUST MATCH RECEIVER
# ============================================================
MODEL_PATH = "helper/models/mctimegan_model.pth"
MIN_VAL = 0.0615832379381183
MAX_VAL = 0.2239434066266675
HORIZON = 24
SEED = 2025

# BCH parameters
BCH_N = 255
BCH_K = 131

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def bytes_to_bits(data: bytes) -> list:
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    return bits

def bch_encode(bch, data_bits: list) -> list:
    n, k = bch.n, bch.k
    num_blocks = (len(data_bits) + k - 1) // k
    padded = data_bits + [0] * (num_blocks * k - len(data_bits))
    
    encoded = []
    for b in range(num_blocks):
        block = galois.GF2(padded[b*k:(b+1)*k])
        encoded.extend(bch.encode(block).tolist())
    
    return encoded, num_blocks

def gan_encode(model, bits: list) -> np.ndarray:
    num_windows = (len(bits) + HORIZON - 1) // HORIZON
    set_seed(SEED)
    
    ipds = []
    alpha_sum = 0
    
    for w in range(num_windows):
        ts, ns = torch.get_rng_state(), np.random.get_state()
        
        cond_0 = np.zeros((1, HORIZON, 1), dtype=np.float32)
        with torch.no_grad():
            out_0 = model.transform(cond_0.shape, cond=cond_0)
        if out_0.ndim == 2:
            out_0 = out_0[:, :, np.newaxis]
        baseline = (out_0 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
        
        torch.set_rng_state(ts)
        np.random.set_state(ns)
        
        cond_1 = np.ones((1, HORIZON, 1), dtype=np.float32)
        with torch.no_grad():
            out_1 = model.transform(cond_1.shape, cond=cond_1)
        if out_1.ndim == 2:
            out_1 = out_1[:, :, np.newaxis]
        out_1_scaled = (out_1 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
        
        alpha = (out_1_scaled - baseline).mean()
        alpha_sum += alpha
        
        for i in range(HORIZON):
            idx = w * HORIZON + i
            if idx >= len(bits):
                break
            ipds.append(baseline[i] if bits[idx] == 0 else baseline[i] + alpha)
    
    return np.array(ipds), alpha_sum / num_windows

def main():
    parser = argparse.ArgumentParser(description="Stealth encoder with BCH + GAN")
    parser.add_argument("--file", required=True, help="File to encode")
    parser.add_argument("--output", default="covert_ipd.csv", help="Output IPD file")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("COVERT TIMING CHANNEL ENCODER (STEALTH)")
    print(f"{'='*60}")
    print(f"Pre-shared: SEED={SEED}, BCH({BCH_N},{BCH_K})")
    
    # Load model
    print(f"\n[1] Loading GAN model...")
    device = torch.device("cpu")
    model_classes.device = device
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    print(f"    OK")
    
    # Create BCH
    print(f"\n[2] Creating BCH codec...")
    bch = galois.BCH(BCH_N, BCH_K)
    print(f"    BCH({bch.n}, {bch.k}), t={bch.t}")
    
    # Read file
    print(f"\n[3] Reading file...")
    with open(args.file, 'rb') as f:
        payload = f.read()
    print(f"    File: {args.file}")
    print(f"    Size: {len(payload)} bytes")
    
    # Create packet: [2 bytes len][payload][4 bytes CRC32]
    print(f"\n[4] Creating packet...")
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    packet = struct.pack('<H', len(payload)) + payload + struct.pack('<I', crc)
    print(f"    Packet: {len(packet)} bytes (len + payload + CRC)")
    
    # Convert to bits
    data_bits = bytes_to_bits(packet)
    print(f"    Bits: {len(data_bits)}")
    
    # BCH encode
    print(f"\n[5] BCH encoding...")
    encoded_bits, num_blocks = bch_encode(bch, data_bits)
    print(f"    Blocks: {num_blocks}")
    print(f"    Encoded: {len(encoded_bits)} bits")
    print(f"    Overhead: {len(encoded_bits)/len(data_bits):.2f}x")
    
    # GAN encode
    print(f"\n[6] GAN encoding...")
    ipds, alpha = gan_encode(model, encoded_bits)
    print(f"    IPDs: {len(ipds)}")
    print(f"    Alpha: {alpha*1000:.2f} ms")
    
    # Save IPD only (no metadata!)
    print(f"\n[7] Saving...")
    pd.DataFrame({'IPD': ipds}).to_csv(args.output, index=False)
    print(f"    {args.output}")
    
    # Summary
    total_time = ipds.sum()
    print(f"\n{'='*60}")
    print("ENCODING COMPLETE")
    print(f"{'='*60}")
    print(f"\nSummary:")
    print(f"  Payload:   {len(payload)} bytes")
    print(f"  BCH bits:  {len(encoded_bits)} ({num_blocks} blocks)")
    print(f"  IPDs:      {len(ipds)}")
    print(f"  Est. time: {total_time:.1f}s")
    print(f"\nTo send:")
    print(f"  python sender.py --ip <target> --port 3334")

if __name__ == "__main__":
    main()
