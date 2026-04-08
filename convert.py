#!/usr/bin/env python3

import sys

precision = 3  # number of decimal places

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <input_iat_ms.txt> <output_timing.txt>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file) as f:
    data = f.read().strip()

values = [v.strip() for v in data.split(",") if v.strip()]
seconds = [f"{float(v) / 1000:.{precision}f}" for v in values]

with open(output_file, "w") as f:
    f.write("\n".join(seconds))
