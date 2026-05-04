#!/bin/bash

# Usage: ./tcp_bulk_client.sh <SERVER_IP> <PORT>
SERVER_IP=${1:-127.0.0.1}
PORT=${2:-5000}

mkdir -p timings

echo "======================================================"
echo " STARTING TCP BULK CLIENT TO $SERVER_IP:$PORT"
echo "======================================================"

# Start the timer!
START_TIME=$SECONDS
FILE_COUNT=0

for file in data/*; do
    [ -f "$file" ] || continue
    
    base=$(basename "$file")
    timing_file="timings/${base}_timing.txt"

    echo "------------------------------------------------"
    echo "[*] Processing File: $file"
    
    # 1. Encode
    python3 gan_encoder.py "$file" "$timing_file"

    # 2. Transmit
    python3 tcp_client.py "$SERVER_IP" "$timing_file" --port "$PORT"
    
    FILE_COUNT=$((FILE_COUNT + 1))
    
    # Let the server reset
    sleep 5
done

# Calculate total time
ELAPSED_TIME=$(($SECONDS - $START_TIME))

echo "======================================================"
echo " DONE! Transmitted $FILE_COUNT files."
echo " Total Time Elapsed: $ELAPSED_TIME seconds."
echo "======================================================"
