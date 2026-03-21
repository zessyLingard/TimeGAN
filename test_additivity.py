import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.getcwd())
import model_classes
from model_classes import (MCTimeGAN, Generator, Discriminator, Recovery, Embedder, ConditioningNetwork, Supervisor)

# Constants
MODEL_PATH = "helper/models/mctimegan_model.pth"
MIN_VAL = 0.0615832379381183
MAX_VAL = 0.2239434066266675
HORIZON = 24
NUM_SEEDS = 100  # Test with 100 different random seeds

def explain_architecture():
    """Print architecture explanation"""
    print("\n" + "="*70)
    print("MC-TimeGAN ARCHITECTURE & ADDITIVITY")
    print("="*70)
    
    print("""
The model processes inputs as:
    
    ┌─────────────────────────────────────────────────────────┐
    │  z (random)     c (condition)                           │
    │      │              │                                    │
    │      └──────┬───────┘                                    │
    │             ↓                                            │
    │      [z; c] CONCAT ← This looks non-linear!             │
    │             ↓                                            │
    │           RNN                                            │
    │             ↓                                            │
    │         output(z,c)                                      │
    └─────────────────────────────────────────────────────────┘
    
But the RNN CAN learn approximately additive processing:

    Hidden neurons group 1: processes z → f(z)
    Hidden neurons group 2: processes c → g(c)
    Output: ≈ f(z) + g(c)
    
This is called EMERGENT ADDITIVITY - not designed, but learned!
    
WHY THIS HAPPENS:
- z and c are statistically independent (random noise vs discrete label)
- Training doesn't encourage z-c interaction
- Simplest solution: process separately and combine
    """)

def test_additivity():
    explain_architecture()
    
    print("\n" + "="*70)
    print("EMPIRICAL TEST: Measuring Additivity")
    print("="*70)
    
    # Setup
    device = torch.device("cpu")
    model_classes.device = device
    
    # Load model
    print(f"\n[1] Loading model...")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    print(f"    ✓ Loaded")
    
    # Test with different seeds
    print(f"\n[2] Testing {NUM_SEEDS} random seeds...")
    
    all_deltas = []
    
    for seed in range(NUM_SEEDS):
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Save RNG state
        torch_state = torch.get_rng_state()
        np_state = np.random.get_state()
        
        # Generate with c=0
        cond_0 = np.zeros((1, HORIZON, 1), dtype=np.float32)
        with torch.no_grad():
            out_0 = model.transform(cond_0.shape, cond=cond_0)
        out_0 = (out_0 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
        
        # Restore and generate with c=1 (SAME z!)
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        
        cond_1 = np.ones((1, HORIZON, 1), dtype=np.float32)
        with torch.no_grad():
            out_1 = model.transform(cond_1.shape, cond=cond_1)
        out_1 = (out_1 * (MAX_VAL - MIN_VAL) + MIN_VAL).flatten()
        
        # Calculate delta = g(1) - g(0)
        delta = out_1 - out_0
        all_deltas.append(delta)
        
        if (seed + 1) % 20 == 0:
            print(f"    Tested {seed + 1}/{NUM_SEEDS} seeds")
    
    all_deltas = np.array(all_deltas)  # Shape: (NUM_SEEDS, HORIZON)
    
    # Analysis
    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    mean_delta = all_deltas.mean()
    std_delta = all_deltas.std()
    
    # Coefficient of variation (CV)
    cv = (std_delta / abs(mean_delta)) * 100 if mean_delta != 0 else 0
    
    # Additivity score
    max_range = MAX_VAL - MIN_VAL
    additivity_score = 1 - (std_delta / max_range)
    
    # SNR
    snr = abs(mean_delta) / std_delta if std_delta > 0 else float('inf')
    
    print(f"\nKEY METRICS:")
    print(f"  Mean δ = g(1) - g(0):  {mean_delta*1000:.3f} ms")
    print(f"  Std(δ) across seeds:   {std_delta*1000:.3f} ms")
    print(f"  Coefficient Variation: {cv:.2f}%")
    print(f"  Additivity Score:      {additivity_score:.4f} (1.0 = perfect)")
    print(f"  SNR:                   {snr:.1f}")
    
    print(f"\nINTERPRETATION:")
    if cv < 5 and additivity_score > 0.95:
        print("  ✅ EXCELLENT: Model exhibits strong additivity")
        print("     → Noise removal method is highly reliable")
    elif cv < 15 and additivity_score > 0.85:
        print("  ✓ GOOD: Model is approximately additive")
        print("     → Noise removal should work well")
    elif cv < 30:
        print("  ⚠️ MODERATE: Some additive structure")
        print("     → Noise removal may work, delta method safer")
    else:
        print("  ❌ POOR: Model is not additive")
        print("     → Use delta method instead")
    
    print(f"\nWHY DOES THIS MATTER?")
    print(f"""
  If output(z,c) = f(z) + g(c), then:
    - f(z) is the "noise" (depends on random seed)
    - g(c) is the "signal" (depends on bit value)
    
  Noise Removal Method:
    received_IPD - f(z) = g(c)  ← clean signal!
    
  This ONLY works if additivity holds.
  
  Your result: δ varies by only {std_delta*1000:.2f}ms (CV={cv:.1f}%)
  → Additivity is {'STRONG' if cv < 10 else 'moderate'}
  → Noise removal is {'recommended' if cv < 15 else 'risky'}
    """)
    
    # Visualization
    print(f"\n[3] Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Delta distribution
    ax1 = axes[0, 0]
    ax1.hist(all_deltas.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(mean_delta, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_delta:.4f}')
    ax1.set_xlabel('Delta (seconds)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'δ = output(c=1) - output(c=0)\n({NUM_SEEDS} seeds × {HORIZON} positions)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Delta per position
    ax2 = axes[0, 1]
    positions = list(range(HORIZON))
    means_per_pos = all_deltas.mean(axis=0)
    stds_per_pos = all_deltas.std(axis=0)
    ax2.errorbar(positions, means_per_pos, yerr=stds_per_pos, fmt='o-', capsize=3, alpha=0.7)
    ax2.axhline(mean_delta, color='red', linestyle='--', label='Global mean')
    ax2.set_xlabel('Position in sequence', fontsize=11)
    ax2.set_ylabel('Delta (seconds)', fontsize=11)
    ax2.set_title('Delta consistency across positions', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Delta per seed
    ax3 = axes[1, 0]
    seed_means = all_deltas.mean(axis=1)
    ax3.plot(range(NUM_SEEDS), seed_means, 'o', markersize=3, alpha=0.6)
    ax3.axhline(mean_delta, color='red', linestyle='--', linewidth=2, label='Global mean')
    ax3.fill_between(range(NUM_SEEDS), 
                      mean_delta - std_delta, 
                      mean_delta + std_delta, 
                      alpha=0.2, color='red', label='±1 std')
    ax3.set_xlabel('Seed (different z)', fontsize=11)
    ax3.set_ylabel('Mean Delta', fontsize=11)
    ax3.set_title('Delta consistency across different z', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary = f"""
ADDITIVITY TEST SUMMARY
{'='*45}

Hypothesis: output(z,c) ≈ f(z) + g(c)

Test:
  • {NUM_SEEDS} different random seeds (z)
  • Each: δ = output(z,c=1) - output(z,c=0)
  
If additive: δ should be constant ∀z

Results:
  Mean(δ):      {mean_delta*1000:.3f} ms
  Std(δ):       {std_delta*1000:.3f} ms
  CV:           {cv:.2f}%
  Add. Score:   {additivity_score:.4f}
  SNR:          {snr:.1f}

Verdict: {'ADDITIVE ✓' if cv < 15 else 'NON-ADDITIVE ✗'}

Implication for Covert Channel:
  → {'Noise removal VALID' if cv < 15 else 'Use delta method'}
  → Signal = {mean_delta*1000:.2f}±{std_delta*1000:.2f} ms
    """
    
    ax4.text(0.05, 0.5, summary, fontsize=9, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    os.makedirs("helper/img", exist_ok=True)
    plt.savefig("helper/img/additivity_test.png", dpi=150, bbox_inches='tight')
    print(f"    Saved: helper/img/additivity_test.png")
    
    # Save data
    df = pd.DataFrame(all_deltas, columns=[f'pos_{i}' for i in range(HORIZON)])
    df['seed'] = range(NUM_SEEDS)
    df.to_csv("additivity_results.csv", index=False)
    print(f"    Saved: additivity_results.csv")
    
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}\n")
    
    return {
        'mean_delta': mean_delta,
        'std_delta': std_delta,
        'cv': cv,
        'additivity_score': additivity_score,
        'snr': snr
    }

if __name__ == "__main__":
    results = test_additivity()
