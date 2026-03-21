"""
Round-Trip Decode Test
Tests: CSV → GAN Decode → BCH Decode → AES Decrypt → File

Verifies the complete decoding pipeline with C BCH integration.
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

# Parameters (must match encoder)
MODEL_PATH = "helper/models/mctimegan_model.pth"
KEY_FILE = "aes_key.txt"
MIN_VAL = 0.0615832379381183
MAX_VAL = 0.2239434066266675
HORIZON = 24
SEED = 2025
BCH_N = 255
BCH_K = 191
BCH_T = 8

def bits_to_bytes(bits):
    """Convert bits to bytes (LSB first)"""
    result = bytearray()
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            result.append(sum(bits[i+j] << j for j in range(8)))
    return bytes(result)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def gan_decode(model, received_ipds):
    """Decode IPDs back to bits using noise removal"""
    set_seed(SEED)
    
    # Calculate threshold
    cond_0 = np.zeros((1, HORIZON, 1), dtype=np.float32)
    cond_1 = np.ones((1, HORIZON, 1), dtype=np.float32)
    ts, ns = torch.get_rng_state(), np.random.get_state()
    with torch.no_grad():
        out_0 = model.transform(cond_0.shape, cond=cond_0)
    torch.set_rng_state(ts)
    np.random.set_state(ns)
    with torch.no_grad():
        out_1 = model.transform(cond_1.shape, cond=cond_1)
    
    if out_0.ndim == 2:
        out_0 = out_0[:, :, np.newaxis]
    if out_1.ndim == 2:
        out_1 = out_1[:, :, np.newaxis]
    
    baseline_sample = (out_0 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
    out_1_scaled = (out_1 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
    threshold = (out_1_scaled - baseline_sample).mean() / 2
    
    # Decode using noise removal
    set_seed(SEED)
    decoded_bits = []
    ipd_idx = 0
    num_windows = (len(received_ipds) + HORIZON - 1) // HORIZON
    
    for _ in range(num_windows):
        cond_0 = np.zeros((1, HORIZON, 1), dtype=np.float32)
        with torch.no_grad():
            out_0 = model.transform(cond_0.shape, cond=cond_0)
        if out_0.ndim == 2:
            out_0 = out_0[:, :, np.newaxis]
        baseline = (out_0 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
        
        for i in range(HORIZON):
            if ipd_idx >= len(received_ipds):
                break
            # Noise removal: clean = received - baseline
            clean = received_ipds[ipd_idx] - baseline[i]
            decoded_bits.append(0 if clean < threshold else 1)
            ipd_idx += 1
    
    return decoded_bits, threshold

def bch_decode(encoded_bits):
    """Decode BCH encoded bits using C library"""
    n, k = BCH_N, bch_wrapper._BCH_K
    num_blocks = (len(encoded_bits) + n - 1) // n
    decoded_bits = []
    total_errors = 0
    failed_blocks = 0
    
    for b in range(num_blocks):
        start = b * n
        block = encoded_bits[start:start + n]
        if len(block) < n:
            block = block + [0] * (n - len(block))
        
        decoded, errors = bch_wrapper.decode(block)
        
        if errors >= 0:
            decoded_bits.extend(decoded)
            total_errors += errors
        else:
            print(f"    ⚠ BCH block {b+1}/{num_blocks} uncorrectable")
            decoded_bits.extend([0] * k)
            failed_blocks += 1
    
    return decoded_bits, total_errors, failed_blocks

def main():
    print("=" * 80)
    print("ROUND-TRIP DECODE TEST")
    print("=" * 80)
    print()
    
    # ========================================
    # STEP 1: Load CSV
    # ========================================
    print("[STEP 1] Load CSV")
    print("-" * 80)
    csv_file = "test_pipeline_output.csv"
    df = pd.read_csv(csv_file)
    ipds = df['IPDs'].values
    
    print(f"File: {csv_file}")
    print(f"IPDs loaded: {len(ipds)}")
    print(f"IPD range: [{ipds.min():.6f}, {ipds.max():.6f}]")
    print(f"✓ Step 1 complete\n")
    
    # ========================================
    # STEP 2: GAN Decode
    # ========================================
    print("[STEP 2] GAN Decode (IPDs → Bits)")
    print("-" * 80)
    
    # Load model
    device = torch.device("cpu")
    model_classes.device = device
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    
    bits, threshold = gan_decode(model, ipds)
    
    print(f"Input: {len(ipds)} IPDs")
    print(f"Output: {len(bits)} bits")
    print(f"Threshold: {threshold:.6f}")
    print(f"First 32 bits: {bits[:32]}")
    print(f"✓ Step 2 complete\n")
    
    # ========================================
    # STEP 3: BCH Decode
    # ========================================
    print("[STEP 3] BCH Decode (C Library)")
    print("-" * 80)
    
    corrected, errors, failed = bch_decode(bits)
    
    print(f"Input: {len(bits)} bits")
    print(f"Output: {len(corrected)} bits")
    print(f"Errors corrected: {errors}")
    print(f"Failed blocks: {failed}")
    print(f"First 32 bits: {corrected[:32]}")
    print(f"✓ Step 3 complete\n")
    
    # ========================================
    # STEP 4: AES Decrypt
    # ========================================
    print("[STEP 4] AES-256 Decryption")
    print("-" * 80)
    
    # Convert bits to bytes
    ciphertext_raw = bits_to_bytes(corrected)
    
    # Truncate to multiple of 16 (AES block size)
    ciphertext_len = (len(ciphertext_raw) // 16) * 16
    ciphertext = ciphertext_raw[:ciphertext_len]
    
    print(f"Bits to bytes: {len(ciphertext_raw)} bytes")
    print(f"Truncated to: {len(ciphertext)} bytes")
    
    # Load key
    with open(KEY_FILE, 'r') as f:
        key_hex = f.read().strip()
    key = bytes.fromhex(key_hex)
    
    # Decrypt
    try:
        iv = ciphertext[:16]
        ct = ciphertext[16:]
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded = decryptor.update(ct) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded) + unpadder.finalize()
        
        print(f"Decrypted: {len(plaintext)} bytes")
        print(f"✓ Step 4 complete\n")
        success = True
    except Exception as e:
        print(f"✗ Decryption failed: {e}")
        plaintext = ciphertext
        success = False
    
    # ========================================
    # STEP 5: Verify
    # ========================================
    print("[STEP 5] Verification")
    print("-" * 80)
    
    # Load original
    with open("data/msg_001.txt", 'rb') as f:
        original = f.read()
    
    print(f"Original: {len(original)} bytes")
    print(f"Decoded:  {len(plaintext)} bytes")
    
    if plaintext == original:
        print(f"✓ MATCH! Decoding successful!")
        print(f"\nOriginal content:")
        print(original.decode('utf-8')[:200])
    else:
        print(f"✗ MISMATCH!")
        print(f"\nOriginal (first 100 bytes):")
        print(original[:100])
        print(f"\nDecoded (first 100 bytes):")
        print(plaintext[:100])
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 80)
    print("DECODE SUMMARY")
    print("=" * 80)
    print(f"1. CSV:         {len(ipds)} IPDs")
    print(f"2. GAN Decode:  {len(bits)} bits")
    print(f"3. BCH Decode:  {len(corrected)} bits ({errors} errors corrected)")
    print(f"4. AES Decrypt: {len(plaintext)} bytes")
    print(f"5. Verification: {'✓ SUCCESS' if plaintext == original else '✗ FAILED'}")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
