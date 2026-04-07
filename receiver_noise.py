"""
CTC-GAN Receiver (Covert Timing Channel)
Pipeline: IPD (Network) → Shared Seed GAN → Bits → BCH Decode → AES Decrypt → File
"""
import os
import argparse
import socket
import json
import time
import torch
import torch.nn as nn
import numpy as np
import bch_wrapper

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# ============================================================
# KIẾN TRÚC MODEL TỪ ĐỒ ÁN (BẮT BUỘC GIỐNG ENCODER)
# ============================================================
def get_rnn_class(module_name):
    return nn.GRU if module_name == "gru" else nn.LSTM

class NoiseGenerator(nn.Module):
    def __init__(self, module_name, input_dim, hidden_dim, num_layers, noise_scale=0.15):
        super().__init__()
        rnn_class = get_rnn_class(module_name)
        self.rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        self.noise_scale = noise_scale

    def forward(self, z, c_scaled):
        x = torch.cat([z, c_scaled], dim=-1)
        seq, _ = self.rnn(x)
        delta_t = self.noise_head(seq) * self.noise_scale
        final_ipd_scaled = c_scaled + delta_t
        return final_ipd_scaled, delta_t

# ============================================================
# PRE-SHARED PARAMETERS
# ============================================================
DEFAULT_MODEL = "covert_channel_generator.pth"
DEFAULT_OUTPUT_DIR = "received_files"
KEY_FILE = "aes_key.txt"

# Hằng số vật lý
PHYS_MIN = 0.0
PHYS_MAX = 2.5
BIT0_TIME = 0.5
BIT1_TIME = 1.0

BIT0_SCALED = (BIT0_TIME - PHYS_MIN) / (PHYS_MAX - PHYS_MIN)
BIT1_SCALED = (BIT1_TIME - PHYS_MIN) / (PHYS_MAX - PHYS_MIN)

SEQ_LEN = 24
HIDDEN_DIM = 24
NUM_LAYERS = 3
MODULE_NAME = "gru"
NOISE_SCALE = 0.4
DEFAULT_SEED = 2025

BCH_N = 255
BCH_K = 191
BCH_T = 8

def inverse_scale(data_scaled):
    return data_scaled * (PHYS_MAX - PHYS_MIN) + PHYS_MIN

def set_shared_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_key():
    with open(KEY_FILE, 'r') as f:
        key_hex = f.read().strip()
    return bytes.fromhex(key_hex)

def bits_to_bytes(bits):
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte_bits = bits[i:i+8]
            byte_val = 0
            for j, b in enumerate(byte_bits):
                byte_val |= (b << j)
            byte_array.append(byte_val)
    return bytes(byte_array)

def decode_shared_seed(model, received_ipds, num_bits, seq_len, seed, device):
    """
    Giải mã bằng Shared Seed Testing (Đúng chuẩn Cell 7)
    """
    print(f"\n[+] Giải mã bằng Shared Seed Hypothesis Testing...")
    set_shared_seed(seed)
    
    num_windows = (num_bits + seq_len - 1) // seq_len
    decoded_bits = []
    
    # Pre-create tensors cho 2 giả thuyết
    c_scaled_0 = torch.full((1, seq_len, 1), BIT0_SCALED, device=device)
    c_scaled_1 = torch.full((1, seq_len, 1), BIT1_SCALED, device=device)
    
    ipd_idx = 0
    for w in range(num_windows):
        z_tensor = torch.rand(1, seq_len, 1, device=device)
        
        with torch.no_grad():
            _, expected_delta_0 = model(z_tensor, c_scaled_0)
            _, expected_delta_1 = model(z_tensor, c_scaled_1)
            
        # Tính IPD lý thuyết
        expected_ipd_0 = inverse_scale(BIT0_SCALED + expected_delta_0.cpu().numpy().flatten())
        expected_ipd_1 = inverse_scale(BIT1_SCALED + expected_delta_1.cpu().numpy().flatten())
        
        # So sánh từng nhịp trong cửa sổ
        for i in range(seq_len):
            if len(decoded_bits) >= num_bits:
                break
                
            # Đề phòng trường hợp mất gói (mặc dù UDP local ít mất)
            if ipd_idx >= len(received_ipds):
                print(f"⚠️ Hết IPD nhận được, đoán bừa bit còn lại.")
                decoded_bits.append(0)
                continue
                
            received = received_ipds[ipd_idx]
            
            # Tính khoảng cách
            dist_0 = abs(received - expected_ipd_0[i])
            dist_1 = abs(received - expected_ipd_1[i])
            
            # Chốt bit
            decoded_bits.append(0 if dist_0 < dist_1 else 1)
            ipd_idx += 1
            
    return decoded_bits

def run_receiver(listen_port, output_dir, model_path, seed):
    print(f"\n{'='*60}")
    print("CTC-GAN RECEIVER (SHARED SEED HYPOTHESIS TESTING)")
    print(f"{'='*60}")
    
    device = torch.device("cpu")
    key = load_key()
    
    # 1. Load model
    print(f"\n[1] Đang nạp AI Model...")
    model = NoiseGenerator(MODULE_NAME, input_dim=2, hidden_dim=HIDDEN_DIM, 
                           num_layers=NUM_LAYERS, noise_scale=NOISE_SCALE).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        full_dict = torch.load('covert_channel_full_model.pth', map_location=device)
        model.load_state_dict(full_dict['generator'])
    model.eval()
    print(f"    ✓ Model loaded (seed={seed})")
    
    # 2. Lắng nghe UDP
    print(f"\n[2] Đang chờ metadata trên cổng {listen_port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", listen_port))
    sock.settimeout(120)
    
    try:
        data, addr = sock.recvfrom(65535)
        metadata = json.loads(data.decode('utf-8'))
        
        filename = metadata['filename']
        num_bits = metadata['num_bits']
        num_packets = metadata['num_packets']
        
        print(f"    ✓ Đã nhận Metadata từ {addr[0]}")
        print(f"    → File: {filename}")
        print(f"    → Số bit cần giải mã: {num_bits}")
        
        # 3. Nhận gói tin
        total_packets = num_packets + 1
        print(f"\n[3] Bắt đầu hứng {total_packets} gói tin...")
        sock.settimeout(5)
        
        timestamps = []
        while len(timestamps) < total_packets:
            try:
                data, _ = sock.recvfrom(65535)
                timestamps.append(time.time())
                if len(timestamps) % 50 == 0:
                    print(f"\r    {len(timestamps)}/{total_packets}", end='', flush=True)
            except socket.timeout:
                print(f"\n    ⚠️ Timeout! Chỉ nhận được {len(timestamps)}/{total_packets}")
                break
                
        print(f"\n    ✓ Đã nhận {len(timestamps)} gói")
        
        # 4. Tính IPD
        received_ipds = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
        # 5. Giải mã GAN
        decoded_bits = decode_shared_seed(model, received_ipds, num_bits, SEQ_LEN, seed, device)
        print(f"    ✓ AI đã giải mã xong {len(decoded_bits)} bits")
        
        # 6. Giải mã BCH
        print(f"\n[4] Sửa lỗi bằng BCH...")
        k = bch_wrapper._BCH_K
        n = bch_wrapper._BCH_N
        num_blocks = len(decoded_bits) // n
        
        corrected_bits = []
        total_errors = 0
        for b in range(num_blocks):
            block = decoded_bits[b*n : (b+1)*n]
            corrected, errs = bch_wrapper.decode(block)
            corrected_bits.extend(corrected[:k])
            if errs > 0:
                total_errors += errs
                
        print(f"    ✓ Đã sửa {total_errors} lỗi trên đường truyền")
        
        # 7. Giải mã AES
        print(f"\n[5] Giải mã AES-256-CTR...")
        ciphertext_bytes = bits_to_bytes(corrected_bits)
        
        # Tách nonce và ciphertext
        nonce = ciphertext_bytes[:16]
        actual_ciphertext = ciphertext_bytes[16:]
        
        cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(actual_ciphertext) + decryptor.finalize()
        
        # 8. Lưu file
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'wb') as f:
            f.write(plaintext)
            
        print(f"\n[6] 🎉 HOÀN THÀNH! File đã lưu tại: {output_path}")

    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receive with CTC-GAN")
    parser.add_argument("--port", type=int, default=3334)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    
    args = parser.parse_args()
    run_receiver(args.port, args.output, args.model, args.seed)