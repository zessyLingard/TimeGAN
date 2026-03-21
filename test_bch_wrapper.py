"""
Unit tests for BCH wrapper
Run this to verify the C library integration works correctly
"""
import bch_wrapper

def test_init():
    """Test BCH initialization"""
    k = bch_wrapper._BCH_K
    assert k == 191, f"Expected k=191, got {k}"
    print(f"✓ BCH initialized: k={k}")

def test_encode_decode_no_errors():
    """Test round-trip encoding/decoding without errors"""
    # Create test data (191 bits)
    data = [1, 0, 1, 1, 0] * 38 + [1]
    assert len(data) == 191
    
    # Encode
    codeword = bch_wrapper.encode(data)
    assert len(codeword) == 255, f"Expected 255 bits, got {len(codeword)}"
    
    # Decode
    decoded, errors = bch_wrapper.decode(codeword)
    assert decoded == data, "Decoded data doesn't match original"
    assert errors == 0, f"Expected 0 errors, got {errors}"
    
    print(f"✓ Round-trip encode/decode works (no errors)")

def test_error_correction():
    """Test error correction capability"""
    data = [1, 0, 1, 1, 0] * 38 + [1]
    codeword = bch_wrapper.encode(data)
    
    # Flip 5 bits (should be correctable with t=8)
    corrupted = codeword.copy()
    for i in [10, 50, 100, 150, 200]:
        corrupted[i] ^= 1
    
    # Decode
    decoded, errors = bch_wrapper.decode(corrupted)
    assert decoded == data, "Failed to correct errors"
    assert errors == 5, f"Expected 5 errors corrected, got {errors}"
    
    print(f"✓ Error correction works (5 errors corrected)")

def test_uncorrectable():
    """Test detection of uncorrectable errors"""
    data = [1, 0, 1, 1, 0] * 38 + [1]
    codeword = bch_wrapper.encode(data)
    
    # Flip 10 bits (exceeds t=8, should fail)
    corrupted = codeword.copy()
    for i in range(10, 20):
        corrupted[i] ^= 1
    
    # Decode
    decoded, errors = bch_wrapper.decode(corrupted)
    assert errors == -1, f"Should detect uncorrectable error, got {errors}"
    
    print(f"✓ Uncorrectable error detected correctly")

def test_multiple_blocks():
    """Test encoding/decoding multiple blocks"""
    num_blocks = 5
    all_data = []
    all_codewords = []
    
    for i in range(num_blocks):
        # Create unique data for each block
        data = [(i + j) % 2 for j in range(191)]
        codeword = bch_wrapper.encode(data)
        all_data.append(data)
        all_codewords.append(codeword)
    
    # Decode all blocks
    for i in range(num_blocks):
        decoded, errors = bch_wrapper.decode(all_codewords[i])
        assert decoded == all_data[i], f"Block {i} failed"
        assert errors == 0, f"Block {i} had unexpected errors"
    
    print(f"✓ Multiple blocks ({num_blocks}) encoded/decoded correctly")

if __name__ == "__main__":
    print("=" * 60)
    print("BCH Wrapper Unit Tests")
    print("=" * 60)
    print()
    
    try:
        test_init()
        test_encode_decode_no_errors()
        test_error_correction()
        test_uncorrectable()
        test_multiple_blocks()
        
        print()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
