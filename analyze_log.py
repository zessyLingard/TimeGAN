"""
Log Analysis Script for Covert Timing Channel
Analyzes IPD log files to extract timing statistics and detect clusters.
"""
import pandas as pd
import numpy as np
import sys
from collections import Counter

def analyze_log(filepath="log.csv"):
    """Analyze IPD log file and report statistics."""
    
    print("=" * 60)
    print("COVERT CHANNEL LOG ANALYSIS")
    print("=" * 60)
    
    # Load data
    try:
        df = pd.read_csv(filepath, header=None, names=['IPDs'])
    except FileNotFoundError:
        # Try with header
        df = pd.read_csv(filepath)
        if 'IPDs' not in df.columns:
            df.columns = ['IPDs']
    
    ipds = df['IPDs'].values
    print(f"\nFile: {filepath}")
    print(f"Total packets: {len(ipds)}")
    
    # Basic statistics
    print("\n--- BASIC STATISTICS ---")
    print(f"Mean:   {np.mean(ipds)*1000:.1f} ms")
    print(f"Std:    {np.std(ipds)*1000:.1f} ms")
    print(f"Min:    {np.min(ipds)*1000:.1f} ms")
    print(f"Max:    {np.max(ipds)*1000:.1f} ms")
    print(f"Median: {np.median(ipds)*1000:.1f} ms")
    
    # Zero analysis
    print("\n--- ZERO ANALYSIS (Buffering) ---")
    zeros = np.sum(ipds == 0)
    near_zeros = np.sum(ipds < 0.01)
    print(f"Exact zeros (0.000):     {zeros} ({100*zeros/len(ipds):.2f}%)")
    print(f"Near zeros (<10ms):      {near_zeros} ({100*near_zeros/len(ipds):.2f}%)")
    
    # Find zero indices
    zero_indices = np.where(ipds == 0)[0]
    if len(zero_indices) > 0:
        print(f"First zero at index:     {zero_indices[0]}")
        
        # Check for consecutive zeros (VPN drops)
        consecutive = 1
        max_consecutive = 1
        for i in range(1, len(zero_indices)):
            if zero_indices[i] == zero_indices[i-1] + 1:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 1
        print(f"Max consecutive zeros:   {max_consecutive}")
    
    # Cluster detection
    print("\n--- CLUSTER ANALYSIS ---")
    
    # Define clusters based on common timing schemes
    clusters = [
        ("< 100ms", 0, 0.1),
        ("100-300ms", 0.1, 0.3),
        ("300-450ms", 0.3, 0.45),
        ("450-550ms (bit0?)", 0.45, 0.55),
        ("550-700ms", 0.55, 0.7),
        ("700-850ms", 0.7, 0.85),
        ("850-1100ms (bit1?)", 0.85, 1.1),
        ("> 1100ms (spikes)", 1.1, float('inf')),
    ]
    
    print(f"{'Cluster':<25} {'Count':>8} {'Percent':>10} {'Mean (ms)':>12}")
    print("-" * 55)
    
    for name, low, high in clusters:
        mask = (ipds >= low) & (ipds < high)
        count = np.sum(mask)
        if count > 0:
            mean_val = np.mean(ipds[mask]) * 1000
            print(f"{name:<25} {count:>8} {100*count/len(ipds):>9.1f}% {mean_val:>11.1f}")
    
    # Auto-detect two main clusters (bit 0 and bit 1)
    print("\n--- AUTO-DETECTED BIT CLUSTERS ---")
    non_zero = ipds[ipds > 0.1]  # Exclude zeros and near-zeros
    
    if len(non_zero) > 10:
        # Use median to split
        median = np.median(non_zero)
        low_cluster = non_zero[non_zero < median]
        high_cluster = non_zero[non_zero >= median]
        
        if len(low_cluster) > 0 and len(high_cluster) > 0:
            bit0_mean = np.mean(low_cluster) * 1000
            bit0_std = np.std(low_cluster) * 1000
            bit1_mean = np.mean(high_cluster) * 1000
            bit1_std = np.std(high_cluster) * 1000
            gap = bit1_mean - bit0_mean
            
            print(f"Bit 0 (low):  {bit0_mean:.1f} ± {bit0_std:.1f} ms (n={len(low_cluster)})")
            print(f"Bit 1 (high): {bit1_mean:.1f} ± {bit1_std:.1f} ms (n={len(high_cluster)})")
            print(f"Gap:          {gap:.1f} ms")
            print(f"Separation:   {gap / (bit0_std + bit1_std):.2f} σ")
    
    # Spike analysis
    print("\n--- SPIKE ANALYSIS (VPN Issues) ---")
    spikes = ipds[ipds > 1.5]
    if len(spikes) > 0:
        print(f"Spikes (>1.5s): {len(spikes)}")
        print(f"Spike values:   {[f'{x:.2f}s' for x in sorted(spikes)[:10]]}")
        if len(spikes) > 10:
            print(f"                ... and {len(spikes)-10} more")
    else:
        print("No spikes detected (>1.5s)")
    
    # Quality assessment
    print("\n--- QUALITY ASSESSMENT ---")
    quality_score = 100
    issues = []
    
    if zeros / len(ipds) > 0.05:
        quality_score -= 30
        issues.append(f"High buffer rate ({zeros} zeros, {100*zeros/len(ipds):.1f}%)")
    elif zeros / len(ipds) > 0.01:
        quality_score -= 15
        issues.append(f"Moderate buffer rate ({zeros} zeros)")
    
    if len(spikes) > 5:
        quality_score -= 20
        issues.append(f"VPN instability ({len(spikes)} spikes)")
    
    if 'gap' in dir() and gap < 300:
        quality_score -= 20
        issues.append(f"Small bit gap ({gap:.0f}ms)")
    
    if issues:
        print("Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No major issues detected")
    
    print(f"\nQuality Score: {max(0, quality_score)}/100")
    
    if quality_score >= 80:
        print("Status: ✅ GOOD - Should decode successfully")
    elif quality_score >= 50:
        print("Status: ⚠️ FAIR - May have some decoding errors")
    else:
        print("Status: ❌ POOR - Likely decoding failures")
    
    print("\n" + "=" * 60)
    
    return {
        'total': len(ipds),
        'zeros': zeros,
        'zero_pct': 100*zeros/len(ipds),
        'quality': quality_score
    }

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "log.csv"
    analyze_log(filepath)
