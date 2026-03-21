"""
TCP Packet Sender for Covert Timing Channel
Uses TCP for reliable delivery while maintaining precise timing.
"""
import argparse
import socket
import time
import pandas as pd
import os

DEFAULT_CSV = "covert_ipd.csv"
DEFAULT_IP = "127.0.0.1"
DEFAULT_PORT = 3334

def precise_sleep(duration):
    """Precise sleep using busy-wait."""
    end_time = time.perf_counter() + duration
    while time.perf_counter() < end_time:
        pass

def send_covert_traffic(csv_path, target_ip, target_port):
    print(f"\n{'='*60}")
    print("COVERT TIMING CHANNEL SENDER (TCP)")
    print(f"{'='*60}")
    
    # Load IPD sequence
    print(f"\n[1] Loading IPD sequence...")
    if not os.path.exists(csv_path):
        print(f"    Error: {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    ipd_sequence = df['IPDs'].values
    
    print(f"    IPDs: {len(ipd_sequence)}")
    print(f"    Range: [{ipd_sequence.min():.4f}, {ipd_sequence.max():.4f}]")
    print(f"    Est. time: {ipd_sequence.sum():.1f}s")
    
    # Setup TCP socket
    print(f"\n[2] Connecting to {target_ip}:{target_port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Match C implementation: TCP_NODELAY (disable Nagle's algorithm)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    # Match C implementation: Increase send buffer to 256KB
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
    
    try:
        sock.connect((target_ip, target_port))
        print(f"    Connected!")
    except Exception as e:
        print(f"    Connection failed: {e}")
        return
    
    print(f"\n[3] Sending {len(ipd_sequence)} packets...")
    
    # Send first packet (timing reference)
    sock.send(b'\x00')
    
    start_time = time.perf_counter()
    for i, ipd in enumerate(ipd_sequence):
        precise_sleep(float(ipd))
        sock.send(b'\x00')
        
        if (i + 1) % 100 == 0 or (i + 1) == len(ipd_sequence):
            elapsed = time.perf_counter() - start_time
            print(f"\r    {i+1}/{len(ipd_sequence)} ({elapsed:.1f}s)", end='', flush=True)
    
    total_time = time.perf_counter() - start_time
    
    print(f"\n    Done! Total: {total_time:.1f}s")
    sock.close()
    
    print(f"\n{'='*60}")
    print("SENDING COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCP covert traffic sender")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="IPD CSV file")
    parser.add_argument("--ip", default=DEFAULT_IP, help="Target IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Target port")
    
    args = parser.parse_args()
    send_covert_traffic(args.csv, args.ip, args.port)
