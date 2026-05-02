import os
import sys
import subprocess

def main():
    print("==================================================")
    print("      OFFLINE PIPELINE TEST (NO INTERNET)         ")
    print("==================================================")

    # 1. Use existing input file instead of dummy secret
    input_file = "data/part_aa"

    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        sys.exit(1)

    print(f"[1] Using input file: {input_file}")

    # 2. Run Encoder
    print("\n[2] Running gan_encoder.py...")
    subprocess.run([
        sys.executable,
        "gan_encoder.py",
        input_file,
        "client_ipds.csv",
        "--weights",
        "stealth_ctc_generator.pth"
    ], check=True)

    # 3. Simulate the Internet (Client -> Server)
    print("\n[3] Simulating Network Transmission...")
    with open("client_ipds.csv", "r") as f:
        # Replace commas with newlines just in case, then split by whitespace/newlines
        raw_data = f.read().replace(',', '\n').split()

    with open("simulated_server_log.txt", "w") as f:
        for ms in raw_data:
            if ms.strip():
                seconds = float(ms.strip())
                f.write(f"{seconds:.9f}\n")

    print("    -> Converted client_ipds.csv into simulated_server_log.txt")

    # 4. Run Decoder
    print("\n[4] Running gan_decoder.py on simulated server log...")
    subprocess.run([
        sys.executable,
        "gan_decoder.py",
        "simulated_server_log.txt"
    ], check=True)

    print("\n==================================================")
    print("If you see your decoded output above, the OFFLINE test is a SUCCESS!")
    print("==================================================")

if __name__ == "__main__":
    main()