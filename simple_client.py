import socket
import time
import argparse
import struct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ip", help="Destination Server IP")
    parser.add_argument("timing_file", help="File containing the exact float IPDs to sleep for")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    with open(args.timing_file, 'r') as f:
        raw_data = f.read().replace(',', '\n').split()
    
    delays = [float(val) for val in raw_data if val.strip()]
    print(f"[*] Loaded {len(delays)} packets to send.")

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print(f"[*] Sending Sequenced UDP traffic to {args.ip}:{args.port}...")

    for i, delay in enumerate(delays):
        if delay > 0:
            time.sleep(delay)
        
        # Inject the sequence number directly into the payload!
        payload = struct.pack("!I", i) + b"X" * 60
        s.sendto(payload, (args.ip, args.port))
        
        if (i+1) % 1000 == 0:
            print(f"    -> Sent {i+1}/{len(delays)} packets")

    print("[*] Transmission complete.")
    s.close()

if __name__ == "__main__":
    main()
