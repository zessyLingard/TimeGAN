"""
End-to-End Pipeline Test
Shows output at each step: File → AES → BCH → GAN → IPD

This test verifies the complete pipeline with C BCH integration.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import bch_wrapper

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

sys.path.append(os.getcwd())
import model_classes
from model_classes import (MCTimeGAN, Generator, Discriminator, Recovery, 
                           Embedder, ConditioningNetwork, Supervisor)

# Parameters
MODEL_PATH = "helper/models/mctimegan_model.pth"
KEY_FILE = "aes_key.txt"
MIN_VAL = 0.0615832379381183
MAX_VAL = 0.2239434066266675
HORIZON = 24
SEED = 2025
BCH_N = 255
BCH_K = 191
BCH_T = 8

def bytes_to_bits(data):
    """Convert bytes to bits (LSB first)"""
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    return bits

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main():
    print("=" * 80)
    print("END-TO-END PIPELINE TEST")
    print("=" * 80)
    print()
    
    # ========================================
    # STEP 1: Read File
    # ========================================
    print("[STEP 1] File Input")
    print("-" * 80)
    test_file = "data/msg_001.txt"
    with open(test_file, 'rb') as f:
        plaintext = f.read()
    
    print(f"File: {test_file}")
    print(f"Size: {len(plaintext)} bytes")
    print(f"Content preview: {plaintext[:100]}")
    print(f"✓ Step 1 complete\n")
    
    # ========================================
    # STEP 2: AES-256 Encryption
    # ========================================
    print("[STEP 2] AES-256 Encryption")
    print("-" * 80)
    
    # Load key
    with open(KEY_FILE, 'r') as f:
        key_hex = f.read().strip()
    key = bytes.fromhex(key_hex)
    
    # Encrypt
    iv = os.urandom(16)
    padder = padding.PKCS7(128).padder()
    padded = padder.update(plaintext) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = iv + encryptor.update(padded) + encryptor.finalize()
    
    print(f"Input: {len(plaintext)} bytes plaintext")
    print(f"Output: {len(ciphertext)} bytes ciphertext (includes 16-byte IV)")
    print(f"Ciphertext (hex): {ciphertext[:32].hex()}...")
    print(f"✓ Step 2 complete\n")
    
    # ========================================
    # STEP 3: BCH Encoding (C Library)
    # ========================================
    print("[STEP 3] BCH Encoding (C Library)")
    print("-" * 80)
    
    # Convert to bits
    data_bits = bytes_to_bits(ciphertext)
    print(f"Ciphertext as bits: {len(data_bits)} bits")
    print(f"First 32 bits: {data_bits[:32]}")
    
    # BCH encode
    k = bch_wrapper._BCH_K
    num_blocks = (len(data_bits) + k - 1) // k
    padded_bits = data_bits + [0] * (num_blocks * k - len(data_bits))
    
    print(f"\nBCH Parameters: BCH({BCH_N}, {k}, t={BCH_T})")
    print(f"Data bits: {len(data_bits)} bits")
    print(f"Padded to: {len(padded_bits)} bits ({num_blocks} blocks of {k} bits)")
    
    encoded_bits = []
    for b in range(num_blocks):
        block = padded_bits[b*k:(b+1)*k]
        codeword = bch_wrapper.encode(block)
        encoded_bits.extend(codeword)
    
    print(f"Output: {len(encoded_bits)} bits ({num_blocks} blocks of {BCH_N} bits)")
    print(f"Overhead: {len(encoded_bits) - len(data_bits)} bits ({BCH_N - k} parity bits per block)")
    print(f"First codeword (32 bits): {encoded_bits[:32]}")
    print(f"✓ Step 3 complete\n")
    
    # ========================================
    # STEP 4: GAN Encoding
    # ========================================
    print("[STEP 4] GAN Encoding (Bits → IPDs)")
    print("-" * 80)
    
    # Load model
    device = torch.device("cpu")
    model_classes.device = device
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    print(f"GAN model loaded from {MODEL_PATH}")
    
    # Encode bits to IPDs
    set_seed(SEED)
    ipds = []
    num_windows = (len(encoded_bits) + HORIZON - 1) // HORIZON
    
    print(f"Encoding {len(encoded_bits)} bits in {num_windows} windows of {HORIZON} bits")
    
    for w in range(num_windows):
        # Generate baseline (bit=0)
        ts, ns = torch.get_rng_state(), np.random.get_state()
        cond_0 = np.zeros((1, HORIZON, 1), dtype=np.float32)
        with torch.no_grad():
            out_0 = model.transform(cond_0.shape, cond=cond_0)
        if out_0.ndim == 2:
            out_0 = out_0[:, :, np.newaxis]
        baseline = (out_0 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
        
        # Generate alpha (bit=1 offset)
        torch.set_rng_state(ts)
        np.random.set_state(ns)
        cond_1 = np.ones((1, HORIZON, 1), dtype=np.float32)
        with torch.no_grad():
            out_1 = model.transform(cond_1.shape, cond=cond_1)
        if out_1.ndim == 2:
            out_1 = out_1[:, :, np.newaxis]
        out_1_scaled = (out_1 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
        alpha = (out_1_scaled - baseline).mean()
        
        # Encode bits
        for i in range(HORIZON):
            idx = w * HORIZON + i
            if idx >= len(encoded_bits):
                break
            ipds.append(baseline[i] if encoded_bits[idx] == 0 else baseline[i] + alpha)
    
    print(f"Output: {len(ipds)} IPD values")
    print(f"IPD range: [{min(ipds):.6f}, {max(ipds):.6f}] seconds")
    print(f"Mean IPD: {np.mean(ipds):.6f} seconds")
    print(f"First 10 IPDs: {[f'{x:.6f}' for x in ipds[:10]]}")
    print(f"✓ Step 4 complete\n")
    
    # ========================================
    # STEP 5: Save to CSV
    # ========================================
    print("[STEP 5] Save to CSV")
    print("-" * 80)
    
    output_file = "test_pipeline_output.csv"
    pd.DataFrame({'IPDs': ipds}).to_csv(output_file, index=False)
    
    print(f"Saved to: {output_file}")
    print(f"Total transmission time: {sum(ipds):.2f} seconds")
    print(f"Throughput: {len(plaintext) * 8 / sum(ipds):.2f} bits/second")
    print(f"✓ Step 5 complete\n")
    
    # ========================================
    # Summary
    # ========================================
    print("=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"1. File:        {len(plaintext):,} bytes")
    print(f"2. AES:         {len(ciphertext):,} bytes")
    print(f"3. BCH:         {len(encoded_bits):,} bits ({num_blocks} blocks)")
    print(f"4. GAN:         {len(ipds):,} IPDs")
    print(f"5. CSV:         {output_file}")
    print()
    print(f"Encoding efficiency: {len(plaintext) * 8 / len(encoded_bits) * 100:.2f}%")
    print(f"Transmission time: {sum(ipds):.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
