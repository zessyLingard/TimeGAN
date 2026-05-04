import socket
import time
import argparse
import struct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ip", help="Destination Server IP")
    parser.add_argument("timing_file", help="File containing the intended float IPDs")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    with open(args.timing_file, 'r') as f:
        raw_data = f.read().replace(',', '\n').split()
    
    delays = [float(val) for val in raw_data if val.strip()]
    print(f"[*] Loaded {len(delays)} delays for TCP transmission.")

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Disable Nagle's Algorithm to force immediate transmission to the OS buffer
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    print(f"[*] Connecting to {args.ip}:{args.port}...")
    try:
        s.connect((args.ip, args.port))
    except ConnectionRefusedError:
        print("[!] Connection refused. Is the server running?")
        return

    print("[*] Connection established. Starting precise timing injection...")

    for i, delay in enumerate(delays):
        if delay > 0:
            time.sleep(delay)
        
        # Inject sequence number and padding (64 bytes total)
        payload = struct.pack("!I", i) + b"X" * 60
        s.sendall(payload)
        
        if (i+1) % 1000 == 0:
            print(f"    -> Sent {i+1}/{len(delays)} TCP segments")

    print("[*] Transmission complete.")
    s.close()

if __name__ == "__main__":
    main()
