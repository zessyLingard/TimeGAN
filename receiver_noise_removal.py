"""
Covert Timing Channel Receiver - TRUE Noise Removal Method
Remove baseline (noise), threshold on clean signal
"""
import os
import argparse
import socket
import json
import time
import torch
import numpy as np

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
DEFAULT_OUTPUT_DIR = "received_files"
DEFAULT_SEED = 2025
DEFAULT_HORIZON = 24

MIN_VAL = 0.0615832379381183
MAX_VAL = 0.2239434066266675

def set_shared_seed(seed):
    """Set seed - MUST match encoder!"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def bits_to_file(bits, output_path):
    """Convert bits to binary file"""
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte_bits = bits[i:i+8]
            byte_val = int(''.join(str(b) for b in byte_bits), 2)
            byte_array.append(byte_val)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(byte_array)
    
    return len(byte_array)

def decode_with_noise_removal(model, received_ipds, num_bits, horizon, seed, threshold):
    """
    TRUE Noise Removal Decoding
    
    Nguyên lý:
      1. Generate CÙNG baseline như encoder: baseline = f(z) + β
      2. Remove baseline từ received IPD: clean = received - baseline
      3. Threshold trên clean signal:
         - clean < threshold → bit = 0
         - clean >= threshold → bit = 1
    
    Điểm khác với Reference Comparison:
      - Không so sánh khoảng cách
      - TRỪ baseline ra (noise removal!)
      - Dùng threshold cố định trên clean signal
    """
    num_windows = (num_bits + horizon - 1) // horizon
    
    print(f"\n[+] Decoding with TRUE Noise Removal...")
    print(f"    Threshold: {threshold:.4f}s")
    
    # Set SAME seed as encoder
    set_shared_seed(seed)
    
    decoded_bits = []
    ipd_idx = 0
    
    debug_info = []
    
    for window_idx in range(num_windows):
        # Generate SAME baseline as encoder
        cond_0 = np.zeros((1, horizon, 1), dtype=np.float32)
        with torch.no_grad():
            out_0 = model.transform(cond_0.shape, cond=cond_0)
        if out_0.ndim == 2:
            out_0 = out_0[:, :, np.newaxis]
        baseline = (out_0 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
        
        # Decode bits in this window
        start_idx = window_idx * horizon
        end_idx = min(start_idx + horizon, num_bits)
        
        for i in range(end_idx - start_idx):
            if ipd_idx >= len(received_ipds):
                break
            
            received = received_ipds[ipd_idx]
            
            # KEY STEP: Remove baseline (noise removal!)
            clean_signal = received - baseline[i]
            
            # Threshold on clean signal
            if clean_signal < threshold:
                decoded_bit = 0
            else:
                decoded_bit = 1
            
            decoded_bits.append(decoded_bit)
            
            # Debug first 5
            if ipd_idx < 5:
                debug_info.append({
                    'idx': ipd_idx,
                    'received': received,
                    'baseline': baseline[i],
                    'clean': clean_signal,
                    'threshold': threshold,
                    'decoded': decoded_bit
                })
            
            ipd_idx += 1
    
    # Print debug info
    print(f"\n    Debug (first 5 positions):")
    for info in debug_info:
        print(f"    [{info['idx']}] received={info['received']:.4f}, baseline={info['baseline']:.4f}, "
              f"clean={info['clean']:.4f} {'<' if info['clean'] < info['threshold'] else '>='} "
              f"threshold={info['threshold']:.4f} → bit={info['decoded']}")
    
    return decoded_bits

def run_receiver(listen_port, output_dir, model_path, seed, horizon):
    print(f"\n{'='*60}")
    print("COVERT TIMING CHANNEL RECEIVER")
    print("Method: TRUE Noise Removal")
    print(f"{'='*60}")
    
    # Setup
    device = torch.device("cpu")
    model_classes.device = device
    
    # Load model
    print(f"\n[1] Loading model...")
    if not os.path.exists(model_path):
        print(f"    ❌ Model not found: {model_path}")
        return
    
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    print(f"    ✓ Model loaded (seed={seed})")
    
    # Listen
    print(f"\n[2] Waiting for metadata on port {listen_port}...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", listen_port))
    sock.settimeout(60)
    
    try:
        # Receive metadata
        data, addr = sock.recvfrom(65535)
        metadata = json.loads(data.decode('utf-8'))
        
        print(f"    ✓ Metadata from {addr[0]}")
        print(f"    → File: {metadata['filename']}")
        print(f"    → Bits: {metadata['num_bits']}")
        print(f"    → Method: {metadata.get('method', 'unknown')}")
        
        if metadata.get('method') != 'noise_removal':
            print(f"    ⚠️  WARNING: Expected 'noise_removal', got '{metadata.get('method')}'")
        
        num_bits = metadata['num_bits']
        num_packets = metadata['num_packets']
        filename = metadata['filename']
        threshold = metadata.get('threshold', 0.03)
        alpha = metadata.get('alpha', 0.06)
        
        print(f"    → Alpha (signal): {alpha:.4f}s")
        print(f"    → Threshold: {threshold:.4f}s")
        
        # Receive packets (+ 1 dummy)
        total_packets = num_packets + 1
        print(f"\n[3] Receiving {total_packets} packets (1 dummy + {num_packets} data)...")
        sock.settimeout(10)
        
        timestamps = []
        while len(timestamps) < total_packets:
            try:
                data, _ = sock.recvfrom(65535)
                timestamps.append(time.time())
                
                if len(timestamps) % 50 == 0 or len(timestamps) == total_packets:
                    print(f"\r    {len(timestamps)}/{total_packets}", end='', flush=True)
            except socket.timeout:
                print(f"\n    ⚠️ Timeout! Got {len(timestamps)}/{total_packets}")
                break
        
        print(f"\n    ✓ Received {len(timestamps)} packets")
        
        if len(timestamps) < total_packets:
            print(f"    ❌ Missing {total_packets - len(timestamps)} packets!")
        
        # Calculate IPD (skip first dummy packet)
        received_ipds = []
        for i in range(1, len(timestamps)):
            ipd = timestamps[i] - timestamps[i-1]
            received_ipds.append(ipd)
        
        print(f"    ✓ Calculated {len(received_ipds)} IPDs")
        print(f"    IPD range: [{min(received_ipds):.4f}, {max(received_ipds):.4f}]")
        print(f"    IPD mean: {np.mean(received_ipds):.4f}")
        
        # Decode with noise removal
        print(f"\n[4] Decoding...")
        decoded_bits = decode_with_noise_removal(model, received_ipds, num_bits, horizon, seed, threshold)
        
        print(f"    ✓ Decoded {len(decoded_bits)} bits")
        
        if len(decoded_bits) != num_bits:
            print(f"    ⚠️ Bit count mismatch: expected {num_bits}, got {len(decoded_bits)}")
            if len(decoded_bits) < num_bits:
                decoded_bits.extend([0] * (num_bits - len(decoded_bits)))
            else:
                decoded_bits = decoded_bits[:num_bits]
        
        # Save
        print(f"\n[5] Saving file...")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        file_size = bits_to_file(decoded_bits, output_path)
        
        print(f"    ✓ {output_path}")
        print(f"    ✓ Size: {file_size} bytes")
        
        print(f"\n{'='*60}")
        print("✅ DECODING COMPLETE")
        print(f"{'='*60}")
        print(f"\nVerify: fc /b {metadata['filename']} {output_path}")
        
    except socket.timeout:
        print(f"\n    ❌ Timeout waiting for metadata!")
    except KeyboardInterrupt:
        print(f"\n    ⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receive with TRUE Noise Removal Method")
    parser.add_argument("--port", type=int, default=3334, help="Listen port")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shared seed (MUST match encoder)")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Time horizon")
    
    args = parser.parse_args()
    
    run_receiver(args.port, args.output, args.model, args.seed, args.horizon)
