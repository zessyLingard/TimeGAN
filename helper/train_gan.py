"""train_gan.py - Train MCTimeGAN với dữ liệu IPD covert channel"""

import numpy as np
import pandas as pd
import torch
import os
import sys

# Import MCTimeGAN
sys.path.insert(0, 'helper')
from mctimegan import MCTimeGAN

print("="*60)
print("TRAINING MCTIMEGAN FOR COVERT CHANNEL")
print("="*60)

# ==========================================
# 1. LOAD DỮ LIỆU TỪ CSV
# ==========================================
print("\n[1] Loading data from CSV...")
try:
    ipd_data = pd.read_csv("helper/data/raw/heartbeat_ipd.csv")["IPDs"].values
    labels_data = pd.read_csv("helper/data/raw_labels/heartbeat_labels.csv")["Bit_Label"].values
    print(f"    ✓ IPD data shape: {ipd_data.shape}")
    print(f"    ✓ Labels shape: {labels_data.shape}")
except FileNotFoundError:
    print("    ✗ CSV files not found! Run create_train_data.py first.")
    sys.exit(1)

# ==========================================
# 2. RESHAPE THÀNH SEQUENCES
# ==========================================
print("\n[2] Reshaping data to sequences...")
seq_len = 100
step = 50

sequences = []
conditions = []

for i in range(0, len(ipd_data) - seq_len, step):
    seq = ipd_data[i:i+seq_len].reshape(-1, 1)  # Shape: (seq_len, 1)
    sequences.append(seq)
    
    # Lấy label dominant trong sequence
    labels_in_seq = labels_data[i:i+seq_len]
    dominant_label = np.bincount(labels_in_seq.astype(int)).argmax()
    # FIXED: Repeat condition qua sequence length để match x dimensions
    cond_seq = np.full((seq_len, 1), dominant_label, dtype=np.float32)
    conditions.append(cond_seq)

sequences = np.array(sequences, dtype=np.float32)
conditions = np.array(conditions, dtype=np.float32)  # Shape: (num_seq, seq_len, 1)

print(f"    ✓ Sequences shape: {sequences.shape} (num_seq, seq_len, features)")
print(f"    ✓ Conditions shape: {conditions.shape} (num_seq, seq_len, 1)")

# ==========================================
# 3. NORMALIZE DỮ LIỆU
# ==========================================
print("\n[3] Normalizing data to [0, 1]...")
min_val = sequences.min()
max_val = sequences.max()
sequences_norm = (sequences - min_val) / (max_val - min_val)

print(f"    ✓ Original range: [{min_val:.4f}, {max_val:.4f}]")
print(f"    ✓ Normalized range: [{sequences_norm.min():.4f}, {sequences_norm.max():.4f}]")

# ==========================================
# 4. KHỞI TẠO MCTIMEGAN
# ==========================================
print("\n[4] Initializing MCTimeGAN...")
gan = MCTimeGAN(
    module_name="gru",
    input_features=1,
    input_conditions=1,
    hidden_dim=24,              # Balanced
    num_layers=3,
    epochs=150,                 # Balanced
    batch_size=24,              # Balanced
    learning_rate=1e-3          # Giữ nguyên
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================
print("\n[5] Training MCTimeGAN...")
print("-" * 60)

gan.fit(
    sequences_norm,
    bit_label=conditions  # Pass conditions
)

print("-" * 60)
print(f"    ✓ Training completed in {gan.fitting_time}s")

# ==========================================
# 6. LƯU MODEL
# ==========================================
print("\n[6] Saving model...")
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "mctimegan_model.pth")

torch.save(gan.state_dict(), model_path)
print(f"    ✓ Model saved to: {model_path}")

# ==========================================
# 7. SINH DỮ LIỆU TỔN HỢP TEST
# ==========================================
print("\n[7] Generating synthetic test data...")

num_sequences_gen = 100
# Tạo điều kiện xen kẽ: 0, 1, 0, 1, ...
conditions_gen = np.array([[i % 2] for i in range(num_sequences_gen)], dtype=np.float32)

synthetic_data = gan.transform(
    data_shape=(num_sequences_gen, seq_len, 1),
    bit_label=conditions_gen
)

print(f"    ✓ Synthetic data shape: {synthetic_data.shape}")

# Denormalize
synthetic_data_denorm = synthetic_data * (max_val - min_val) + min_val
synthetic_ipds = synthetic_data_denorm.reshape(-1)

print(f"    ✓ Synthetic IPD range: [{synthetic_ipds.min():.4f}, {synthetic_ipds.max():.4f}]")

# ==========================================
# 8. LƯU DỮ LIỆU TỔN HỢP
# ==========================================
print("\n[8] Saving synthetic data...")
synthetic_df = pd.DataFrame(synthetic_ipds, columns=["IPD_Synthetic"])
synthetic_path = "helper/data/synthetic_ipd.csv"
synthetic_df.to_csv(synthetic_path, index=False)
print(f"    ✓ Synthetic data saved to: {synthetic_path}")

# ==========================================
# 9. SUMMARY
# ==========================================
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"  Model path:        {model_path}")
print(f"  Synthetic data:    {synthetic_path}")
print(f"  Training time:     {gan.fitting_time}s")
print(f"  Total epochs:      300 (100 embedding + 100 supervised + 100 joint)")
print(f"  Real IPD range:    [{min_val:.4f}, {max_val:.4f}]")
print(f"  Synthetic range:   [{synthetic_ipds.min():.4f}, {synthetic_ipds.max():.4f}]")
print("="*60)
print("\n[✓] Pipeline completed successfully!")