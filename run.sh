#!/bin/bash

mkdir -p timing

for file in data/part_*; do
    base=$(basename "$file")

    tmp_encoded=$(mktemp)
    timing_file="timing/${base}_timing.csv"

    echo "Processing $file ..."

    # Step 1: Encode
    python3 encoder_noise.py --file "$file" --output "$tmp_encoded"

    # Step 2: Convert
    python3 convert.py "$tmp_encoded" "$timing_file" >/dev/null 2>&1

    rm -f "$tmp_encoded"

done

output_file="final_output.csv"

# FIX newline khi concat
for f in timing/*; do
    cat "$f"
    echo
done > "$output_file"

echo "Done! Output saved to $output_file"
