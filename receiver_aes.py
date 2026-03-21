"""
CTC-GAN Offline Decoder (AES)
Pipeline: IPD logs (from results/) -> GAN decode -> BCH decode -> AES decrypt -> file

Reads AES key from aes_key.txt
Uses noise removal method for robust decoding
"""
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import bch_wrapper

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

sys.path.append(os.getcwd())
import model_classes
from model_classes import (MCTimeGAN, Generator, Discriminator, Recovery, 
                           Embedder, ConditioningNetwork, Supervisor)

# ============================================================
# PRE-SHARED PARAMETERS
# ============================================================
MODEL_PATH = "helper/models/mctimegan_model.pth"
KEY_FILE = "aes_key.txt"
MIN_VAL = 0.0615832379381183  # Must match encoder!
MAX_VAL = 0.2239434066266675  # Must match encoder!
HORIZON = 24
SEED = 2025
BCH_N = 255
BCH_K = 191  # Updated to match C BCH implementation
BCH_T = 8

RESULTS_DIR = "results"
DECODED_DIR = "decoded"

def load_key():
    with open(KEY_FILE, 'r') as f:
        key_hex = f.read().strip()
    return bytes.fromhex(key_hex)

def bits_to_bytes(bits):
    result = bytearray()
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            result.append(sum(bits[i+j] << j for j in range(8)))
    return bytes(result)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def gan_decode(model, received_ipds):
    """Decode bits from IPDs using noise removal method"""
    set_seed(SEED)
    
    # Get threshold by computing alpha
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
            # DO NOT use corrupted data - mark block as failed
            print(f"    ⚠ BCH block {b+1}/{num_blocks} uncorrectable (>t={BCH_T} errors)")
            decoded_bits.extend([0] * k)  # Fill with zeros
            failed_blocks += 1
    
    if failed_blocks > 0:
        print(f"    ⚠ {failed_blocks} block(s) had uncorrectable errors")
    
    return decoded_bits, total_errors

def decode_single_log(log_path, model, aes_key, output_path):
    """Decode a single IPD log file"""
    
    # Load IPDs
    df = pd.read_csv(log_path)
    ipds = df['IPD'].values
    print(f"    IPDs: {len(ipds)}, range: [{ipds.min():.4f}, {ipds.max():.4f}]")
    
    # GAN decode with noise removal
    bits, threshold = gan_decode(model, ipds)
    print(f"    GAN decoded: {len(bits)} bits (threshold={threshold:.6f})")
    
    # BCH decode
    corrected, errors = bch_decode(bits)
    print(f"    BCH decoded: {len(corrected)} bits, {errors} errors corrected")
    
    # To bytes
    ciphertext_raw = bits_to_bytes(corrected)
    print(f"    Ciphertext: {len(ciphertext_raw)} bytes")
    
    # AES decrypt (CTR mode - no padding needed)
    try:
        nonce = ciphertext_raw[:16]
        ct = ciphertext_raw[16:]
        cipher = Cipher(algorithms.AES(aes_key), modes.CTR(nonce), backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ct) + decryptor.finalize()
        success = True
    except Exception as e:
        print(f"    ✗ Decryption failed: {e}")
        plaintext = ciphertext_raw
        success = False
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(plaintext)
    
    return success, errors, plaintext

def main():
    parser = argparse.ArgumentParser(description="CTC-GAN Offline Decoder (AES)")
    parser.add_argument("--input", default=RESULTS_DIR, help="Input folder with log CSVs")
    parser.add_argument("--output", default=DECODED_DIR, help="Output folder for decoded files")
    parser.add_argument("--file", help="Decode single log file instead of folder")
    
    args = parser.parse_args()
    
    print("")
    print("=" * 60)
    print("CTC-GAN OFFLINE DECODER (AES + BCH + NOISE REMOVAL)")
    print("=" * 60)
    
    # Load key
    aes_key = load_key()
    print(f"[1] AES key loaded from {KEY_FILE}")
    
    # Load model
    device = torch.device("cpu")
    model_classes.device = device
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    print("[2] GAN model loaded")
    
    # Initialize C BCH
    k = bch_wrapper._BCH_K
    print(f"[3] BCH({BCH_N},{k}) ready, t={BCH_T}")
    
    # Single file mode
    if args.file:
        print(f"\n[DECODE] {args.file}")
        os.makedirs(args.output, exist_ok=True)
        basename = os.path.splitext(os.path.basename(args.file))[0]
        out_path = os.path.join(args.output, f"{basename}_decoded.txt")
        
        success, errors, plaintext = decode_single_log(args.file, model, aes_key, out_path)
        
        if success:
            print(f"\n✓ SUCCESS: {out_path} ({len(plaintext)} bytes)")
            try:
                print(f"  Content: {plaintext[:200].decode('utf-8')}")
            except:
                print(f"  Content: (binary data)")
        else:
            print(f"\n✗ FAILED: Saved raw to {out_path}")
        return
    
    # Batch mode - decode all logs in folder
    if not os.path.exists(args.input):
        print(f"\nERROR: Input folder not found: {args.input}")
        return
    
    log_files = sorted([f for f in os.listdir(args.input) if f.startswith("log_") and f.endswith(".csv")])
    
    if not log_files:
        print(f"\nNo log files found in {args.input}/")
        return
    
    print(f"\n[4] Found {len(log_files)} log files in {args.input}/")
    os.makedirs(args.output, exist_ok=True)
    
    success_count = 0
    total_errors = 0
    
    for log_file in log_files:
        log_path = os.path.join(args.input, log_file)
        msg_num = log_file.replace("log_", "").replace(".csv", "")
        out_path = os.path.join(args.output, f"decoded_{msg_num}.txt")
        
        print(f"\n[{msg_num}] {log_file}")
        success, errors, plaintext = decode_single_log(log_path, model, aes_key, out_path)
        total_errors += errors
        
        if success:
            print(f"    ✓ Saved: {out_path} ({len(plaintext)} bytes)")
            success_count += 1
        else:
            print(f"    ✗ Failed, raw saved: {out_path}")
    
    print("")
    print("=" * 60)
    print(f"COMPLETE: {success_count}/{len(log_files)} decoded successfully")
    print(f"Total BCH errors corrected: {total_errors}")
    print(f"Output: {args.output}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
