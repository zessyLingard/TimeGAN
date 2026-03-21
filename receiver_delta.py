# receiver_delta.py
"""
Covert Timing Channel Receiver - Delta Method
Decode by removing noise using shared seed
"""
import os
import argparse
import socket
import json
import time
import torch
import numpy as np

import model_classes
from model_classes import (MCTimeGAN, Generator, Discriminator, Recovery, Embedder, ConditioningNetwork, Supervisor)

# Constants (giống encoder)
DEFAULT_MODEL = "helper/models/mctimegan_model.pth"
DEFAULT_OUTPUT_DIR = "received_files"
DEFAULT_SEED = 2025
DEFAULT_HORIZON = 24

MIN_VAL = 0.0615832379381183
MAX_VAL = 0.2239434066266675

def set_shared_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def bits_to_file(bits, output_path):
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

def decode_with_delta_method(model, received_ipds, num_bits, horizon, seed):
    """
    Decode using Delta Method.
    
    For each received IPD:
    1. Generate baseline (c=0) with same seed
    2. Calculate delta = output(c=1) - output(c=0)
    3. Compare received to baseline and baseline+delta
    4. Choose closer one
    """
    num_windows = (num_bits + horizon - 1) // horizon
    
    print(f"\n[+] Decoding with Delta Method...")
    
    # Set seed (SAME as encoder!)
    set_shared_seed(seed)
    
    decoded_bits = []
    ipd_idx = 0
    
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
        
        # Restore and generate with c=1
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
        
        # Decode bits in this window
        start_idx = window_idx * horizon
        end_idx = min(start_idx + horizon, num_bits)
        
        for i in range(end_idx - start_idx):
            if ipd_idx >= len(received_ipds):
                break
            
            received = received_ipds[ipd_idx]
            
            # Compare distances
            dist_to_0 = abs(received - out_0[i])
            dist_to_1 = abs(received - (out_0[i] + delta[i]))
            
            if dist_to_0 <= dist_to_1:
                decoded_bits.append(0)
            else:
                decoded_bits.append(1)
            
            ipd_idx += 1
    
    return decoded_bits

def run_receiver(listen_port, output_dir, model_path, seed, horizon):
    print(f"\n{'='*60}")
    print("COVERT TIMING CHANNEL RECEIVER (Delta Method)")
    print(f"{'='*60}")
    
    # Setup
    device = torch.device("cpu")
    model_classes.device = device
    
    # Load model
    print(f"\n[1] Loading model...")
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
        
        num_bits = metadata['num_bits']
        num_packets = metadata['num_packets']
        filename = metadata['filename']
        
        # Receive packets (+ 1 dummy packet đầu tiên)
        total_packets = num_packets + 1
        print(f"\n[3] Receiving {total_packets} packets (1 dummy + {num_packets} data)...")
        sock.settimeout(5)
        
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
        
        # Calculate IPD (từ packet thứ 2, bỏ qua dummy packet đầu)
        received_ipds = []
        for i in range(1, len(timestamps)):
            ipd = timestamps[i] - timestamps[i-1]
            received_ipds.append(ipd)
        
        print(f"    IPD range: [{min(received_ipds):.4f}, {max(received_ipds):.4f}]")
        
        # Save received IPDs to CSV for analysis
        import pandas as pd
        received_ipd_path = "received_ipd.csv"
        pd.DataFrame({'IPD': received_ipds}).to_csv(received_ipd_path, index=False)
        print(f"    ✓ Saved received IPDs to: {received_ipd_path}")
        
        # Decode
        print(f"\n[4] Decoding with Delta Method...")
        decoded_bits = decode_with_delta_method(model, received_ipds, num_bits, horizon, seed)
        
        print(f"    ✓ Decoded {len(decoded_bits)} bits")
        
        # Save
        print(f"\n[5] Saving file...")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        file_size = bits_to_file(decoded_bits, output_path)
        
        print(f"    ✓ {output_path} ({file_size} bytes)")
        
        print(f"\n{'='*60}")
        print("✅ DECODING COMPLETE (Delta Method)")
        print(f"{'='*60}")
        
        # Compare with sent IPDs if available
        sent_ipd_path = "covert_ipd.csv"
        if os.path.exists(sent_ipd_path):
            print(f"\n[ANALYSIS] Comparing sent vs received IPDs...")
            import pandas as pd
            
            df_sent = pd.read_csv(sent_ipd_path)
            sent_ipds = df_sent['IPD'].values[:len(received_ipds)]
            
            # Calculate errors
            errors = np.array(received_ipds) - sent_ipds
            
            print(f"    Sent IPDs    : [{sent_ipds.min():.4f}, {sent_ipds.max():.4f}], mean={sent_ipds.mean():.4f}")
            print(f"    Received IPDs: [{min(received_ipds):.4f}, {max(received_ipds):.4f}], mean={np.mean(received_ipds):.4f}")
            print(f"    Errors       : [{errors.min():.4f}, {errors.max():.4f}], mean={errors.mean():.4f}, std={errors.std():.4f}")
            
            # Count how many IPDs are significantly different
            threshold = 0.01  # 10ms
            large_errors = np.abs(errors) > threshold
            print(f"    Large errors (>{threshold}s): {large_errors.sum()}/{len(errors)} ({large_errors.sum()/len(errors)*100:.1f}%)")
            
            # Save comparison
            comparison_path = "ipd_comparison.csv"
            pd.DataFrame({
                'Sent_IPD': sent_ipds,
                'Received_IPD': received_ipds,
                'Error': errors,
                'Abs_Error': np.abs(errors)
            }).to_csv(comparison_path, index=False)
            print(f"    ✓ Comparison saved to: {comparison_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receive with Delta Method")
    parser.add_argument("--port", type=int, default=3334)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    
    args = parser.parse_args()
    
    run_receiver(args.port, args.output, args.model, args.seed, args.horizon)