import socket
import time
import sys

if len(sys.argv) != 2:
    print("Sử dụng: python3 baseline_client.py <IP_SERVER>")
    sys.exit(1)

SERVER_IP = sys.argv[1]
PORT = 5000
TARGET_DELAY = 0.200  # 200ms
PACKETS_TO_SEND = 1000

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"[*] Bắt đầu bắn {PACKETS_TO_SEND} gói tin sang {SERVER_IP}:{PORT}")
print(f"[*] Nhịp điệu: {TARGET_DELAY}s / gói...")

for i in range(PACKETS_TO_SEND):
    time.sleep(TARGET_DELAY)
    s.sendto(b"PING", (SERVER_IP, PORT))
    
    if (i+1) % 200 == 0:
        print(f"    -> Đã bắn {i+1}/{PACKETS_TO_SEND} gói")

print("[*] Hoàn tất!")
s.close()

