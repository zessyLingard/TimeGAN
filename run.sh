#!/bin/bash

mkdir -p timing

for file in data/part_*; do
    base=$(basename "$file")

    iat_file="data/${base}_encoded.csv"

    timing_file="timing/${base}_timing.csv"

    echo "Processing $file ..."

    # Step 1: Encode
    python3 encoder_noise.py --file "$file" --output "$iat_file"

    # Step 2: Convert (ẩn output)
    python3 convert.py "$iat_file" "$timing_file" >/dev/null 2>&1

done

output_file="final_output.csv"

cat timing/* > "$output_file"

echo "Done! Output saved to $output_file"
