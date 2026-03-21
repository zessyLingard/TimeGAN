"""
CTC-GAN Encoder
Pipeline: file → AES-256 → BCH → GAN → IPD (CSV)

Reads AES key from aes_key.txt
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
MIN_VAL = 0.1
MAX_VAL = 0.4
HORIZON = 24
SEED = 2025
BCH_N = 255
BCH_K = 191  
BCH_T = 8

def load_key():
    """Load AES key from file"""
    with open(KEY_FILE, 'r') as f:
        key_hex = f.read().strip()
    return bytes.fromhex(key_hex)

def aes_encrypt(plaintext, key):
    """AES-256-CTR encrypt (no padding needed)"""
    nonce = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    ciphertext = cipher.encryptor().update(plaintext) + cipher.encryptor().finalize()
    return nonce + ciphertext

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="File to encode")
    parser.add_argument("--output", default="covert_ipd.csv", help="Output CSV")
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print("CTC-GAN ENCODER")
    print(f"{'='*50}")
    
    # Load key
    key = load_key()
    print(f"[1] AES key loaded from {KEY_FILE}")
    
    # Load model
    device = torch.device("cpu")
    model_classes.device = device
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    print(f"[2] GAN model loaded")
    
    # Initialize C BCH
    k = bch_wrapper._BCH_K
    print(f"[3] BCH({BCH_N},{k}) ready, t={BCH_T}")
    
    # Read file
    with open(args.file, 'rb') as f:
        plaintext = f.read()
    print(f"[4] Read {len(plaintext)} bytes: {args.file}")
    
    # AES encrypt (CTR mode - no padding needed)
    nonce = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = nonce + encryptor.update(plaintext) + encryptor.finalize()
    print(f"[5] AES encrypted: {len(ciphertext)} bytes")
    
    # To bits
    data_bits = bytes_to_bits(ciphertext)
    
    # BCH encode
    k = bch_wrapper._BCH_K
    num_blocks = (len(data_bits) + k - 1) // k
    padded_bits = data_bits + [0] * (num_blocks * k - len(data_bits))
    encoded_bits = []
    for b in range(num_blocks):
        block = padded_bits[b*k:(b+1)*k]
        codeword = bch_wrapper.encode(block)
        encoded_bits.extend(codeword)
    print(f"[6] BCH encoded: {len(encoded_bits)} bits ({num_blocks} blocks)")
    
    # GAN encode
    set_seed(SEED)
    ipds = []
    num_windows = (len(encoded_bits) + HORIZON - 1) // HORIZON
    
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
        
        for i in range(HORIZON):
            idx = w * HORIZON + i
            if idx >= len(encoded_bits):
                break
            ipds.append(baseline[i] if encoded_bits[idx] == 0 else baseline[i] + alpha)
    
    print(f"[7] GAN encoded: {len(ipds)} IPDs")
    
    # Save
    pd.DataFrame({'IPDs': ipds}).to_csv(args.output, index=False)
    print(f"[8] Saved: {args.output}")
    
    print(f"\n{'='*50}")
    print(f"Done! {len(plaintext)} bytes → {len(ipds)} IPDs")
    print(f"Est time: {sum(ipds):.1f}s")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
