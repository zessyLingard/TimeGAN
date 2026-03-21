"""
Python wrapper for C BCH library
Provides clean interface matching galois API for easy integration
"""
import ctypes
import platform
from pathlib import Path

# Load compiled library (auto-detect OS)
if platform.system() == "Windows":
    _lib_path = Path(__file__).parent / "bch_lib.dll"
else:  # Linux, macOS, etc.
    _lib_path = Path(__file__).parent / "bch_lib.so"

_lib = ctypes.CDLL(str(_lib_path))

# Define function signatures
_lib.bch_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.bch_init.restype = ctypes.c_int

_lib.bch_encode_bits.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
_lib.bch_encode_bits.restype = ctypes.c_int

_lib.bch_decode_bits.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
_lib.bch_decode_bits.restype = ctypes.c_int

# BCH parameters
BCH_M = 8
BCH_N = 255
BCH_T = 8
_BCH_K = None  # Will be set by init()

def init():
    """Initialize BCH(255, k, 8). Returns k."""
    global _BCH_K
    _BCH_K = _lib.bch_init(BCH_M, BCH_N, BCH_T)
    if _BCH_K < 0:
        raise RuntimeError(f"Failed to initialize BCH({BCH_M}, {BCH_N}, {BCH_T})")
    return _BCH_K

def encode(data_bits):
    """
    Encode data bits using BCH.
    
    Args:
        data_bits: list or array of k bits (0 or 1)
    
    Returns:
        list of n encoded bits
    """
    if _BCH_K is None:
        raise RuntimeError("Call init() first")
    
    if len(data_bits) != _BCH_K:
        raise ValueError(f"Expected {_BCH_K} bits, got {len(data_bits)}")
    
    # Convert to C array
    data_arr = (ctypes.c_int * _BCH_K)(*data_bits)
    codeword_arr = (ctypes.c_int * BCH_N)()
    
    # Call C function
    n = _lib.bch_encode_bits(data_arr, codeword_arr)
    
    # Convert back to Python list
    return list(codeword_arr[:n])

def decode(received_bits):
    """
    Decode received bits using BCH.
    
    Args:
        received_bits: list or array of n bits (0 or 1)
    
    Returns:
        tuple: (decoded_bits, errors_corrected)
        decoded_bits: list of k bits
        errors_corrected: int, or -1 if uncorrectable
    """
    if _BCH_K is None:
        raise RuntimeError("Call init() first")
    
    if len(received_bits) != BCH_N:
        raise ValueError(f"Expected {BCH_N} bits, got {len(received_bits)}")
    
    # Convert to C array
    received_arr = (ctypes.c_int * BCH_N)(*received_bits)
    data_arr = (ctypes.c_int * _BCH_K)()
    
    # Call C function
    errors = _lib.bch_decode_bits(received_arr, data_arr)
    
    if errors < 0:
        # Return zeros on failure (matching current behavior)
        return [0] * _BCH_K, -1
    
    # Convert back to Python list
    return list(data_arr[:_BCH_K]), errors

# Initialize on import
try:
    k = init()
    print(f"BCH wrapper initialized: BCH({BCH_N}, {k}, t={BCH_T})")
except Exception as e:
    print(f"Warning: Failed to initialize BCH wrapper: {e}")
    print(f"Make sure bch_lib.dll is in the same directory as this file")
