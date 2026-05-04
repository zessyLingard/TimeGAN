#!/bin/bash
# Usage: ./udp_bulk_client.sh [SERVER_IP] [PORT]
SERVER_IP=${1:-34.142.140.125}
PORT=${2:-5000}
DATA_DIR="data"
TIMING_DIR="timings_udp_v2"

mkdir -p "$TIMING_DIR"
echo "[*] Starting UDP Bulk Send to $SERVER_IP:$PORT"

total_files=$(ls -1q "$DATA_DIR" | wc -l)
current=0

for file in "$DATA_DIR"/*; do
    [ -f "$file" ] || continue
    current=$((current + 1))
    
    base=$(basename "$file")
    timing_file="$TIMING_DIR/${base}_timing.txt"

    echo "[*] Processing $current/$total_files: $base"
    
    # 1. Encode payload into timings
    python3 gan_encoder.py "$file" "$timing_file"
    
    # 2. Transmit via UDP
    python3 simple_client.py "$SERVER_IP" "$timing_file" --port "$PORT"
    
    # 3. Mandatory cooldown: Let VPN buffers flush and give the Server time to restart
    sleep 5 
done

echo "[*] Client batch completed."
