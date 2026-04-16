# convert_csv_to_lines.py

import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r') as f:
    data = f.read().strip()

values = data.split(',')

with open(output_file, 'w') as f:
    for v in values:
        f.write(v.strip() + '\n')

print(f"Converted {input_file} -> {output_file}")