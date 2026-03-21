#!/bin/bash
# Compile BCH library for Linux
# Run this on the Linux receiver machine

echo "Compiling BCH library for Linux..."

gcc -shared -fPIC -o bch_lib.so bch_lib.c -O2

if [ $? -eq 0 ]; then
    echo "✓ Successfully compiled bch_lib.so"
    echo ""
    echo "File info:"
    ls -lh bch_lib.so
    echo ""
    echo "You can now run:"
    echo "  python3 receiver_aes.py --file results/log_001.csv"
else
    echo "✗ Compilation failed"
    exit 1
fi
