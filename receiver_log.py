"""
CTC-GAN Receiver for LAN Testing
Logs inter-arrival times to results folder, then decodes offline
"""
import os
import sys
import argparse
import socket
import time
import signal
import pandas as pd

# Settings
DEFAULT_PORT = 3334
RESULTS_DIR = "results"
PACKET_TIMEOUT = 3.0  # Seconds without packet = message complete

# Global flag for Ctrl+C
stop_flag = False

def signal_handler(sig, frame):
    global stop_flag
    print("\n    Ctrl+C pressed, stopping...")
    stop_flag = True

signal.signal(signal.SIGINT, signal_handler)

def receive_and_log(port, message_id, timeout=300):
    """Receive packets, log IPDs to CSV, return for offline decoding"""
    global stop_flag
    stop_flag = False
    
    print(f"\n[MSG {message_id}] Waiting on port {port}... (Ctrl+C to stop)")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(1.0)  # Short timeout for Ctrl+C to work
    
    timestamps = []
    waited = 0
    
    try:
        # Wait for first packet
        while waited < timeout and not stop_flag:
            try:
                _, addr = sock.recvfrom(65535)
                timestamps.append(time.time())
                print(f"    Connected from {addr[0]}")
                break
            except socket.timeout:
                waited += 1
                continue
        
        if stop_flag:
            sock.close()
            return None
            
        if waited >= timeout:
            print("    Timeout waiting for packets")
            sock.close()
            return None
        
        # Collect packets until timeout
        sock.settimeout(PACKET_TIMEOUT)
        
        while not stop_flag:
            try:
                sock.recvfrom(65535)
                timestamps.append(time.time())
                if len(timestamps) % 50 == 0:
                    print(f"\r    Packets: {len(timestamps)}", end='', flush=True)
            except socket.timeout:
                break
        
        print(f"\n    Total: {len(timestamps)} packets")
        
        if len(timestamps) < 2:
            print("    ERROR: Not enough packets")
            return None
        
        # Calculate IPDs
        ipds = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
        # Save to results folder
        os.makedirs(RESULTS_DIR, exist_ok=True)
        log_path = os.path.join(RESULTS_DIR, f"log_{message_id:03d}.csv")
        pd.DataFrame({'IPD': ipds}).to_csv(log_path, index=False)
        print(f"    Saved: {log_path} ({len(ipds)} IPDs)")
        
        return log_path
        
    except Exception as e:
        print(f"    Error: {e}")
        return None
    finally:
        sock.close()

def main():
    parser = argparse.ArgumentParser(description="CTC-GAN Receiver - Log Mode")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--count", type=int, default=100, help="Number of messages to receive")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per message")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("CTC-GAN RECEIVER - LOG MODE")
    print(f"{'='*60}")
    print(f"Will receive {args.count} messages on port {args.port}")
    print(f"Logs saved to: {RESULTS_DIR}/")
    print(f"{'='*60}\n")
    
    received = 0
    for i in range(1, args.count + 1):
        if stop_flag:
            break
            
        log_path = receive_and_log(args.port, i, args.timeout)
        if log_path:
            received += 1
        
        if i < args.count and not stop_flag:
            print(f"    Ready for next message... (Ctrl+C to stop)")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: Received {received}/{args.count} messages")
    print(f"Logs in: {RESULTS_DIR}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
