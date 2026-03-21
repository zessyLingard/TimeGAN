"""
Covert Timing Channel Receiver with BCH - STEALTH VERSION
NO metadata expected - all parameters pre-shared
Waits for packets, decodes on timeout
"""
import os
import sys
import argparse
import socket
import struct
import zlib
import time
import torch
import numpy as np
import galois

sys.path.append(os.getcwd())
import model_classes
from model_classes import (MCTimeGAN, Generator, Discriminator, Recovery, Embedder, ConditioningNetwork, Supervisor)

# ============================================================
# PRE-SHARED PARAMETERS - MUST MATCH ENCODER
# ============================================================
MODEL_PATH = "helper/models/mctimegan_model.pth"
MIN_VAL = 0.0615832379381183
MAX_VAL = 0.2239434066266675
HORIZON = 24
SEED = 2025

# BCH parameters - MUST MATCH ENCODER
BCH_N = 255
BCH_K = 131

# Receiver settings
DEFAULT_PORT = 3334
DEFAULT_OUTPUT_DIR = "received_files"
PACKET_TIMEOUT = 2.0  # Seconds without packet = transmission done

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def bits_to_bytes(bits):
    result = bytearray()
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            result.append(sum(bits[i+j] << j for j in range(8)))
    return bytes(result)

def decode_ipd_with_gan(model, received_ipds, num_bits):
    """GAN decode with noise removal - pre-shared parameters"""
    set_seed(SEED)
    
    # Calculate threshold from first window
    cond_0 = np.zeros((1, HORIZON, 1), dtype=np.float32)
    cond_1 = np.ones((1, HORIZON, 1), dtype=np.float32)
    
    ts, ns = torch.get_rng_state(), np.random.get_state()
    with torch.no_grad():
        out_0 = model.transform(cond_0.shape, cond=cond_0)
    torch.set_rng_state(ts)
    np.random.set_state(ns)
    with torch.no_grad():
        out_1 = model.transform(cond_1.shape, cond=cond_1)
    
    baseline_sample = (out_0 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
    out_1_scaled = (out_1 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
    threshold = (out_1_scaled - baseline_sample).mean() / 2
    
    # Reset seed and decode
    set_seed(SEED)
    decoded_bits = []
    ipd_idx = 0
    
    num_windows = (num_bits + HORIZON - 1) // HORIZON
    
    for window_idx in range(num_windows):
        cond_0 = np.zeros((1, HORIZON, 1), dtype=np.float32)
        with torch.no_grad():
            out_0 = model.transform(cond_0.shape, cond=cond_0)
        baseline = (out_0 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
        
        for i in range(HORIZON):
            if ipd_idx >= len(received_ipds) or ipd_idx >= num_bits:
                break
            
            clean_signal = received_ipds[ipd_idx] - baseline[i]
            decoded_bits.append(0 if clean_signal < threshold else 1)
            ipd_idx += 1
    
    return decoded_bits, threshold

def bch_decode_bits(bch, encoded_bits):
    """BCH decode with pre-shared parameters"""
    n, k = bch.n, bch.k
    num_blocks = (len(encoded_bits) + n - 1) // n
    
    decoded_bits = []
    total_errors = 0
    
    for block_idx in range(num_blocks):
        start = block_idx * n
        block = encoded_bits[start:start + n]
        if len(block) < n:
            block = block + [0] * (n - len(block))
        
        try:
            decoded, errors = bch.decode(galois.GF2(block), errors=True)
            decoded_bits.extend(decoded.tolist())
            total_errors += errors
        except:
            # Uncorrectable - use raw data bits
            decoded_bits.extend(block[n-k:])
    
    return decoded_bits, total_errors

def run_receiver(listen_port, output_dir, timeout=300):
    print(f"\n{'='*60}")
    print("COVERT TIMING CHANNEL RECEIVER (STEALTH)")
    print(f"{'='*60}")
    print(f"Pre-shared: SEED={SEED}, BCH({BCH_N},{BCH_K})")
    
    # Load model
    print(f"\n[1] Loading GAN model...")
    device = torch.device("cpu")
    model_classes.device = device
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    print(f"    OK")
    
    # Create BCH decoder
    bch = galois.BCH(BCH_N, BCH_K)
    print(f"    BCH: t={bch.t} errors/block")
    
    # Listen
    print(f"\n[2] Waiting for packets on port {listen_port}...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", listen_port))
    sock.settimeout(timeout)
    
    try:
        # Wait for first packet
        _, addr = sock.recvfrom(65535)
        print(f"    Connection from {addr[0]}")
        
        # Collect packets until timeout
        timestamps = [time.time()]
        sock.settimeout(PACKET_TIMEOUT)
        
        while True:
            try:
                sock.recvfrom(65535)
                timestamps.append(time.time())
                if len(timestamps) % 100 == 0:
                    print(f"\r    Received: {len(timestamps)}", end='', flush=True)
            except socket.timeout:
                break
        
        print(f"\n    Total packets: {len(timestamps)}")
        
        if len(timestamps) < 2:
            print("    Error: Not enough packets")
            return
        
        # Calculate IPDs
        received_ipds = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        print(f"    IPDs: {len(received_ipds)}")
        print(f"    Range: [{min(received_ipds)*1000:.2f}, {max(received_ipds)*1000:.2f}] ms")
        
        # GAN decode
        print(f"\n[3] GAN decoding...")
        decoded_bits, threshold = decode_ipd_with_gan(model, received_ipds, len(received_ipds))
        print(f"    Decoded: {len(decoded_bits)} bits")
        print(f"    Threshold: {threshold*1000:.2f} ms")
        
        # BCH decode
        print(f"\n[4] BCH decoding...")
        corrected_bits, errors = bch_decode_bits(bch, decoded_bits)
        print(f"    Errors corrected: {errors}")
        print(f"    Data bits: {len(corrected_bits)}")
        
        # Convert to bytes
        raw_bytes = bits_to_bytes(corrected_bits)
        
        # Parse message format: [2 bytes len][payload][4 bytes CRC]
        print(f"\n[5] Parsing message...")
        if len(raw_bytes) < 6:
            print(f"    Error: Data too short ({len(raw_bytes)} bytes)")
            return
        
        msg_len = struct.unpack('<H', raw_bytes[:2])[0]
        print(f"    Message length: {msg_len} bytes")
        
        if msg_len > len(raw_bytes) - 6:
            print(f"    Warning: Declared length exceeds data")
            msg_len = min(msg_len, len(raw_bytes) - 6)
        
        message = raw_bytes[2:2+msg_len]
        
        # Verify CRC
        if len(raw_bytes) >= 2 + msg_len + 4:
            expected_crc = struct.unpack('<I', raw_bytes[2+msg_len:2+msg_len+4])[0]
            actual_crc = zlib.crc32(message) & 0xFFFFFFFF
            crc_valid = (expected_crc == actual_crc)
            print(f"    CRC: {'VALID' if crc_valid else 'INVALID'}")
        else:
            crc_valid = False
            print(f"    CRC: Not found")
        
        # Save
        print(f"\n[6] Saving...")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "received_message.bin")
        
        with open(output_path, 'wb') as f:
            f.write(message)
        
        print(f"    Saved: {output_path}")
        print(f"    Size: {len(message)} bytes")
        
        # Try to display as text
        try:
            text = message.decode('utf-8')
            print(f"\n    Content: {text[:200]}{'...' if len(text) > 200 else ''}")
        except:
            print(f"    Content: (binary data)")
        
        print(f"\n{'='*60}")
        print(f"RECEIVE COMPLETE {'(CRC OK)' if crc_valid else '(CRC FAIL - may be corrupted)'}")
        print(f"{'='*60}")
        
    except socket.timeout:
        print(f"\n    Timeout waiting for packets")
    except KeyboardInterrupt:
        print(f"\n    Interrupted")
    except Exception as e:
        print(f"\n    Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sock.close()

def main():
    parser = argparse.ArgumentParser(description="Stealth receiver with BCH")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=300)
    
    args = parser.parse_args()
    run_receiver(args.port, args.output, args.timeout)

if __name__ == "__main__":
    main()
