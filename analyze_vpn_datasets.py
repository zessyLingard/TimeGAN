"""
VPN Traffic Analysis for Covert Timing Channel
===============================================

This script analyzes each VPN pcap file from the ISCX VPN-nonVPN 2016 dataset
to scientifically justify which traffic type to use for training our GAN model.

Output: A comprehensive report explaining:
1. What each VPN traffic type looks like
2. Statistical characteristics of each
3. WHY we choose specific traffic for worst-case conditions
4. Recommended parameters for covert timing channel
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from scapy.all import PcapReader, conf
conf.use_pcap = True


def extract_ipd_from_pcap(pcap_file):
    """Extract Inter-Packet Delays (IPDs) from a pcap file."""
    timestamps = []
    
    try:
        with PcapReader(str(pcap_file)) as reader:
            for packet in reader:
                timestamps.append(float(packet.time))
    except Exception as e:
        return None, str(e)
    
    if len(timestamps) < 2:
        return None, "Too few packets"
    
    timestamps = np.array(timestamps)
    ipds = np.diff(timestamps) * 1000  # Convert to milliseconds
    ipds = ipds[ipds > 0]  # Remove negative values
    
    return ipds, None


def analyze_traffic(ipds, name):
    """Compute comprehensive statistics for a traffic type."""
    # Filter extreme outliers for main statistics
    ipds_clean = ipds[ipds < 10000]  # Remove > 10 seconds
    
    stats = {
        'name': name,
        'total_packets': len(ipds),
        'mean_ipd_ms': np.mean(ipds_clean),
        'median_ipd_ms': np.median(ipds_clean),
        'std_ipd_ms': np.std(ipds_clean),
        'min_ipd_ms': np.min(ipds),
        'max_ipd_ms': np.max(ipds),
        'p5': np.percentile(ipds_clean, 5),
        'p25': np.percentile(ipds_clean, 25),
        'p50': np.percentile(ipds_clean, 50),
        'p75': np.percentile(ipds_clean, 75),
        'p95': np.percentile(ipds_clean, 95),
        'p99': np.percentile(ipds_clean, 99),
    }
    
    # Calculate jitter in different ranges (to understand network vs app)
    for max_val in [50, 100, 200, 500]:
        subset = ipds_clean[ipds_clean < max_val]
        if len(subset) > 100:
            stats[f'jitter_under_{max_val}ms'] = np.std(subset)
            stats[f'pct_under_{max_val}ms'] = len(subset) / len(ipds_clean) * 100
    
    return stats


def categorize_traffic(filename):
    """Categorize traffic type based on filename."""
    name = filename.lower()
    if 'audio' in name or 'voip' in name:
        return 'VoIP/Audio', 'Regular intervals, low latency requirement'
    elif 'chat' in name or 'icq' in name:
        return 'Chat/Messaging', 'Interactive, sporadic, user-driven'
    elif 'youtube' in name or 'netflix' in name or 'vimeo' in name or 'spotify' in name:
        return 'Streaming', 'Bursty, buffered, variable delays'
    elif 'sftp' in name or 'file' in name:
        return 'File Transfer', 'Bulk transfer, high throughput, large pauses'
    else:
        return 'Other', 'Unknown pattern'


def generate_report(all_stats):
    """Generate a comprehensive, beginner-friendly report."""
    
    report = []
    report.append("=" * 80)
    report.append("VPN TRAFFIC ANALYSIS REPORT")
    report.append("For Covert Timing Channel Training Data Selection")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: ISCX VPN-nonVPN 2016")
    report.append(f"Total files analyzed: {len(all_stats)}")
    
    # Section 1: What is this analysis?
    report.append("\n" + "=" * 80)
    report.append("SECTION 1: WHAT IS THIS ANALYSIS?")
    report.append("=" * 80)
    report.append("""
For our covert timing channel, we encode data in the TIMING between packets.
The challenge: Network conditions add "jitter" (random delay variations).

We need to answer:
1. How much jitter does VPN traffic have?
2. Which traffic type has the WORST jitter? (for robust design)
3. What timing parameters should we use?

This analysis examines real VPN traffic to find these answers.
""")
    
    # Section 2: Key Terms
    report.append("\n" + "=" * 80)
    report.append("SECTION 2: KEY TERMS (FOR BEGINNERS)")
    report.append("=" * 80)
    report.append("""
IPD (Inter-Packet Delay): Time between consecutive packets (in milliseconds)
    - Small IPD (< 50ms): Packets sent close together (burst)
    - Large IPD (> 500ms): Long pause between packets (user activity)

Jitter (σ): Standard deviation of IPD - measures timing VARIABILITY
    - Low jitter (< 20ms): Consistent timing, easier to decode
    - High jitter (> 100ms): Highly variable, harder to decode

For our covert channel:
    - We send packets with specific delays to encode bits
    - Bit 0 = shorter delay, Bit 1 = longer delay
    - The "gap" between bit-0 and bit-1 delays must be > jitter
""")
    
    # Section 3: Per-file analysis
    report.append("\n" + "=" * 80)
    report.append("SECTION 3: ANALYSIS OF EACH VPN TRAFFIC TYPE")
    report.append("=" * 80)
    
    # Group by category
    categories = {}
    for stats in all_stats:
        cat, desc = categorize_traffic(stats['name'])
        if cat not in categories:
            categories[cat] = {'desc': desc, 'files': []}
        categories[cat]['files'].append(stats)
    
    for cat, data in sorted(categories.items()):
        report.append(f"\n--- {cat.upper()} ---")
        report.append(f"Description: {data['desc']}\n")
        
        for stats in data['files']:
            report.append(f"  File: {stats['name']}")
            report.append(f"    Packets analyzed: {stats['total_packets']:,}")
            report.append(f"    Mean IPD: {stats['mean_ipd_ms']:.2f} ms")
            report.append(f"    Median IPD: {stats['median_ipd_ms']:.2f} ms")
            report.append(f"    Jitter (σ): {stats['std_ipd_ms']:.2f} ms  ← KEY METRIC")
            report.append(f"    Range: {stats['min_ipd_ms']:.4f} - {stats['max_ipd_ms']:.2f} ms")
            report.append(f"    Percentiles: P5={stats['p5']:.1f}, P50={stats['p50']:.1f}, P95={stats['p95']:.1f} ms")
            report.append("")
    
    # Section 4: Comparison table
    report.append("\n" + "=" * 80)
    report.append("SECTION 4: COMPARISON TABLE")
    report.append("=" * 80)
    report.append("\nRanked by JITTER (highest = worst case):\n")
    
    sorted_stats = sorted(all_stats, key=lambda x: x['std_ipd_ms'], reverse=True)
    
    report.append(f"{'Rank':<5} {'Traffic Type':<40} {'Jitter (σ)':<15} {'Category':<15}")
    report.append("-" * 75)
    
    for i, stats in enumerate(sorted_stats, 1):
        cat, _ = categorize_traffic(stats['name'])
        name_short = stats['name'].replace('vpn_', '').replace('.pcap', '')
        report.append(f"{i:<5} {name_short:<40} {stats['std_ipd_ms']:<15.2f} {cat:<15}")
    
    # Section 5: Worst-case analysis
    worst = sorted_stats[0]
    best = sorted_stats[-1]
    
    report.append("\n" + "=" * 80)
    report.append("SECTION 5: WORST-CASE ANALYSIS")
    report.append("=" * 80)
    report.append(f"""
HIGHEST JITTER (Worst Case):
    Traffic: {worst['name']}
    Jitter: {worst['std_ipd_ms']:.2f} ms
    Category: {categorize_traffic(worst['name'])[0]}

LOWEST JITTER (Best Case):
    Traffic: {best['name']}  
    Jitter: {best['std_ipd_ms']:.2f} ms
    Category: {categorize_traffic(best['name'])[0]}

For robust covert channel design, we should handle the WORST case.
""")
    
    # Section 6: Recommended parameters
    worst_jitter = worst['std_ipd_ms']
    gap_3sigma = 3 * worst_jitter
    gap_4sigma = 4 * worst_jitter
    
    report.append("\n" + "=" * 80)
    report.append("SECTION 6: RECOMMENDED TRAINING PARAMETERS")
    report.append("=" * 80)
    report.append(f"""
Based on worst-case jitter of {worst_jitter:.1f} ms:

MINIMUM GAP CALCULATION:
    For 99% reliability:   Gap = 3σ = {gap_3sigma:.0f} ms
    For 99.9% reliability: Gap = 4σ = {gap_4sigma:.0f} ms

RECOMMENDED CONFIGURATION (Conservative):
    Bit 0 center: 150 ms
    Bit 1 center: {150 + gap_4sigma:.0f} ms
    Threshold: {(150 + 150 + gap_4sigma) / 2:.0f} ms
    Gap: {gap_4sigma:.0f} ms

THROUGHPUT ESTIMATE:
    Average delay per bit: {(150 + 150 + gap_4sigma) / 2:.0f} ms
    Bits per second: ~{1000 / ((150 + 150 + gap_4sigma) / 2):.1f} bits/sec
    Time for 100 bytes (800 bits): ~{800 / (1000 / ((150 + 150 + gap_4sigma) / 2)):.0f} seconds
""")
    
    # Section 7: Why we choose this dataset
    report.append("\n" + "=" * 80)
    report.append("SECTION 7: SCIENTIFIC JUSTIFICATION")
    report.append("=" * 80)
    report.append(f"""
WHY WE USE THE ISCX VPN-nonVPN 2016 DATASET:

1. WIDELY CITED: Used in 100+ academic papers on network traffic analysis
   
2. REALISTIC TRAFFIC: Contains actual VPN-tunneled application traffic
   (not synthetic or simulated)
   
3. DIVERSE APPLICATIONS: Includes various traffic types:
   - Streaming (YouTube, Netflix, Vimeo, Spotify)
   - VoIP (Skype, VoIPBuster)
   - Chat (ICQ, Skype chat)
   - File transfer (SFTP)
   
4. WORST-CASE CONDITIONS: By analyzing all traffic types and choosing
   the one with HIGHEST jitter, we design for worst-case conditions.

WHY WE CHOOSE "{worst['name']}" FOR TRAINING:

1. HIGHEST JITTER ({worst_jitter:.1f} ms): This represents the most 
   challenging network conditions for timing-based communication.
   
2. CONSERVATIVE DESIGN: If our covert channel works under these 
   conditions, it will work under any normal VPN scenario.
   
3. Traffic category "{categorize_traffic(worst['name'])[0]}": 
   {categorize_traffic(worst['name'])[1]}

CITATION:
    Gerard Draper-Gil, Arash Habibi Lashkari, Mohammad Saiful Islam Mamun,
    and Ali A. Ghorbani, "Characterization of Encrypted and VPN Traffic 
    using Time-related Features", ICISSP 2016.
""")
    
    # Section 8: For your paper
    report.append("\n" + "=" * 80)
    report.append("SECTION 8: TEXT FOR YOUR PAPER")
    report.append("=" * 80)
    report.append(f'''
Copy this into your methodology section:

"We utilize the ISCX VPN-nonVPN 2016 dataset [Draper-Gil et al., 2016], 
a widely-cited benchmark for VPN traffic analysis containing diverse 
application types tunneled through VPN connections.

We analyzed {len(all_stats)} VPN traffic captures to characterize timing 
jitter under various conditions. Table X shows the jitter (standard 
deviation of inter-packet delays) for each traffic type. 

For robust covert channel design, we select {worst['name'].replace("vpn_", "").replace(".pcap", "")} 
traffic, which exhibits the highest jitter (σ = {worst_jitter:.1f} ms) among 
all traffic types, representing worst-case network conditions.

Using the 4σ rule for 99.9% reliability, we derive a minimum gap of 
{gap_4sigma:.0f} ms between bit-0 and bit-1 distributions. We configure 
our training parameters as: Bit-0 centered at 150ms, Bit-1 centered 
at {150 + gap_4sigma:.0f}ms, with a decision threshold of {(150 + 150 + gap_4sigma) / 2:.0f}ms."
''')
    
    return "\n".join(report)


def main():
    print("=" * 60)
    print("VPN TRAFFIC ANALYSIS FOR COVERT TIMING CHANNEL")
    print("=" * 60)
    
    vpn_dir = Path("VPN-PCAPs")
    pcap_files = sorted(vpn_dir.glob("*.pcap"))
    
    print(f"\n[*] Found {len(pcap_files)} VPN pcap files")
    print("[*] Analyzing each file (this may take a few minutes)...\n")
    
    all_stats = []
    
    for pcap_file in pcap_files:
        print(f"  Processing: {pcap_file.name}...", end=" ", flush=True)
        
        ipds, error = extract_ipd_from_pcap(pcap_file)
        
        if error:
            print(f"Error: {error}")
            continue
        
        stats = analyze_traffic(ipds, pcap_file.name)
        all_stats.append(stats)
        
        print(f"OK ({stats['total_packets']:,} packets, jitter={stats['std_ipd_ms']:.1f}ms)")
    
    print(f"\n[*] Successfully analyzed {len(all_stats)} files")
    
    # Generate report
    print("[*] Generating report...")
    report = generate_report(all_stats)
    
    # Save report
    output_file = "VPN_ANALYSIS_REPORT.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n[+] Report saved to: {output_file}")
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    
    # Quick summary
    sorted_stats = sorted(all_stats, key=lambda x: x['std_ipd_ms'], reverse=True)
    worst = sorted_stats[0]
    
    print(f"\nWorst-case traffic: {worst['name']}")
    print(f"Worst-case jitter: {worst['std_ipd_ms']:.1f} ms")
    print(f"Recommended gap (4σ): {4 * worst['std_ipd_ms']:.0f} ms")
    print(f"\nSee {output_file} for full analysis and paper text.")
    
    # Also save CSV for data
    df = pd.DataFrame(all_stats)
    df.to_csv("VPN_ANALYSIS_DATA.csv", index=False)
    print(f"[+] Raw data saved to: VPN_ANALYSIS_DATA.csv")


if __name__ == "__main__":
    main()
