import socket
import time
import argparse
import struct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output", default="server_log.txt")
    parser.add_argument("--packets", type=int, default=9180)
    args = parser.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('0.0.0.0', args.port))

    print(f"[*] Sequenced UDP Server listening on port {args.port}...")
    
    ipds = [0.0] * args.packets
    
    print("[*] Waiting for traffic...")
    data, addr = s.recvfrom(1024)
    last_time = time.perf_counter()
    packets_received = 1

    try:
        while True:
            s.settimeout(15.0)
            data, addr = s.recvfrom(1024)
            current_time = time.perf_counter()
            
            if len(data) >= 4:
                seq = struct.unpack("!I", data[:4])[0]
                if seq < args.packets:
                    # If a packet drops, this seamlessly merges the time into the next packet!
                    ipds[seq] = current_time - last_time
                    last_time = current_time
                    packets_received += 1
            
    except socket.timeout:
        print("\n[!] Timed out. Transmission ended.")
    except KeyboardInterrupt:
        pass
    finally:
        s.close()

    drops = args.packets - packets_received
    print(f"\n[*] Transmission complete.")
    print(f"[*] Received: {packets_received} | Dropped: {drops} ({(drops/args.packets)*100:.2f}%)")
    
    with open(args.output, 'w') as f:
        for ipd in ipds:
            f.write(f"{ipd:.9f}\n")
            
    print(f"[*] Saved perfectly aligned float timings to {args.output}")

if __name__ == "__main__":
    main()
