"""
TCP Receiver for Covert Timing Channel
Logs inter-arrival times to results folder for offline decoding.
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

# Global flag for Ctrl+C
stop_flag = False

def signal_handler(sig, frame):
    global stop_flag
    print("\n    Ctrl+C pressed, stopping...")
    stop_flag = True

signal.signal(signal.SIGINT, signal_handler)

def receive_and_log(port, message_id, timeout=300):
    """Receive packets via TCP, log IPDs to CSV"""
    global stop_flag
    stop_flag = False
    
    print(f"\n[MSG {message_id}] Waiting on port {port}... (Ctrl+C to stop)")
    
    # Create TCP server socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Match C implementation: TCP_NODELAY and buffer sizes
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)  # 256KB
    
    server.bind(("0.0.0.0", port))
    server.listen(1)
    server.settimeout(1.0)
    
    timestamps = []
    waited = 0
    
    try:
        # Accept connection and apply socket options
        while waited < timeout and not stop_flag:
            try:
                conn, addr = server.accept()
                # Apply options to connection socket too (matches C server.c line 220-225)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
                print(f"    Connected from {addr[0]}")
                break
            except socket.timeout:
                waited += 1
                continue
        
        if stop_flag or waited >= timeout:
            server.close()
            return None
        
        # Receive packets until sender closes connection
        conn.settimeout(None)  # No timeout - wait for connection close
        
        while not stop_flag:
            try:
                data = conn.recv(1)
                if not data:
                    print("\n    ✓ Sender closed connection - transmission complete")
                    break
                timestamps.append(time.time())
                if len(timestamps) % 50 == 0:
                    print(f"\r    Packets: {len(timestamps)}", end='', flush=True)
            except Exception as e:
                print(f"\n    Error: {e}")
                break
        
        print(f"\n    Total: {len(timestamps)} packets")
        
        conn.close()
        
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
        server.close()

def main():
    parser = argparse.ArgumentParser(description="TCP Receiver - Log Mode")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--count", type=int, default=100, help="Number of messages")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per message")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("CTC-GAN RECEIVER - TCP MODE")
    print(f"{'='*60}")
    print(f"Will receive {args.count} messages on port {args.port}")
    print(f"{'='*60}\n")
    
    received = 0
    for i in range(1, args.count + 1):
        if stop_flag:
            break
            
        log_path = receive_and_log(args.port, i, args.timeout)
        if log_path:
            received += 1
        
        if i < args.count and not stop_flag:
            print(f"    Ready for next message...")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: Received {received}/{args.count} messages")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
