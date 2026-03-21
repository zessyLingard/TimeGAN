"""__main__.py"""

import os
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

# Import các module từ helper
from helper.mctimegan import MCTimeGAN
from helper.data_processing import loading, preparing
from helper.metrics import visualization

def parse_arguments():
    parser = argparse.ArgumentParser(description="MC-TimeGAN Training Script")
    parser.add_argument(
        "--data",
        default=r"helper/data/raw/custom_traffic_data.csv",
        type=str,
        help="Path to the data file",
    )
    parser.add_argument(
        "--labels",
        default=r"helper/data/raw_labels/custom_traffic_labels.csv",
        type=str,
        help="Path to the labels file",
    )
    parser.add_argument(
        "--horizon", 
        default=24, 
        type=int, 
        help="Horizon for sequence slicing"
    )
    parser.add_argument(
        "--hidden_dim", 
        default=32, 
        type=int, 
        help="Hidden dimension size for the model"
    )
    parser.add_argument(
        "--num_layers", 
        default=3, 
        type=int, 
        help="Number of layers in the model"
    )
    parser.add_argument(
        "--epochs", 
        default=2000, 
        type=int, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        default=128, 
        type=int, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", 
        default=5e-4,  
        type=float, 
        help="Learning rate for training"
    )
    parser.add_argument(
        "--csv_filename",
        default=r"helper/synthetic_data/main_traffic_synthetic_data.csv",
        type=str,
        help="Filename for the exported CSV of synthetic data",
    )
    parser.add_argument(
        "--model_save_path",
        default=r"helper/models/ctc_model.pth",
        type=str,
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Skip training and only run visualization with pre-trained model",
    )
    parser.add_argument(
        "--skip_visualization",
        action="store_true",
        help="Skip visualization to speed up training",
    )
    return parser.parse_args()


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[-] Đang sử dụng thiết bị: {device}")

    # 1. Load Data
    print("[-] Đang đọc dữ liệu...")
    data, labels = loading(args.data, args.labels)
    
    print(f"    Shape Data gốc: {data.shape}")
    print(f"    Shape Label gốc: {labels.shape}")

    # 2. Preprocessing & Windowing
    print("[-] Đang xử lý cửa sổ trượt (Sliding Window)...")
    data_train, max_val, min_val, labels_train = preparing(
        (data, True), 
        (labels, False), 
        horizon=args.horizon, 
        shuffle_stack=True
    )
    
    # Convert to float for calculations
    max_val = float(max_val)
    min_val = float(min_val)
    
    print(f"    Shape Train (Batch): {data_train.shape}")
    print(f"    Value range: [{min_val:.4f}, {max_val:.4f}]")

    # 3. Initialize or Load Model
    if args.evaluate_only:
        print(f"[-] Đang tải model từ: {args.model_save_path}")
        if not os.path.exists(args.model_save_path):
            raise FileNotFoundError(f"Model file not found: {args.model_save_path}")
        model = torch.load(args.model_save_path, map_location=device, weights_only=False)
        model.eval()
        print("[+] Model đã được tải thành công!")
    else:
        model = MCTimeGAN(
            input_features=data_train.shape[-1],
            input_conditions=labels_train.shape[-1],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        ).to(device)

        # 4. Train
        print("[-] Bắt đầu Training...")
        print(f"    Epochs: {args.epochs}")
        print(f"    Batch size: {args.batch_size}")
        print(f"    Learning rate: {args.learning_rate}")
        print(f"    Hidden dim: {args.hidden_dim}")
        
        model.fit(data_train, cond=labels_train) 

        # Save model
        print(f"[-] Đang lưu model vào: {args.model_save_path}")
        os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
        torch.save(model, args.model_save_path)
        print("[+] Model saved successfully!")
    
    # 5. Quick Validation
    print("\n[-] Running quick validation...")
    validate_model(model, max_val, min_val, args.horizon)
    
    # 6. Skip visualization if requested
    if args.skip_visualization:
        print("\n[+] Training complete! (Skipped visualization)")
        return
    
    # 7. Synthesis & Evaluate
    print("[-] Đang sinh dữ liệu thử nghiệm để đánh giá...")
    with torch.no_grad():
        data_gen = model.transform(data_train.shape, cond=labels_train)

    # Ensure 3D shape
    if data_gen.ndim == 2:
        data_gen = data_gen[:, :, np.newaxis]
    
    # Rescale back
    data_gen_final = data_gen * (max_val - min_val) + min_val
    data_ori_final = data_train * (max_val - min_val) + min_val

    # Export CSV
    data_gen_reshaped = data_gen_final.reshape(data_gen_final.shape[0], -1)
    os.makedirs(os.path.dirname(args.csv_filename), exist_ok=True)
    pd.DataFrame(data_gen_reshaped).to_csv(args.csv_filename, index=False)
    print(f"[+] Đã lưu dữ liệu sinh mẫu vào: {args.csv_filename}")

    # Visualize
    print("[-] Đang vẽ biểu đồ đánh giá...")
    try:
        visualize_sequences(data_gen_final, data_ori_final, labels_train, args.horizon)
    except Exception as e:
        print(f"[!] Lỗi vẽ Sequence: {e}")
    
    print("\n=> HOÀN TẤT QUÁ TRÌNH HUẤN LUYỆN!")


def validate_model(model, max_val, min_val, horizon):
    """Quick validation to check if conditioning works."""
    THRESHOLD = 0.125
    n_test = 100
    
    # Generate with pure Bit 0
    cond_0 = np.zeros((n_test, horizon, 1))
    with torch.no_grad():
        gen_0 = model.transform((n_test, horizon, 1), cond=cond_0)
    if gen_0.ndim == 2:
        gen_0 = gen_0[:, :, np.newaxis]
    gen_0 = gen_0 * (max_val - min_val) + min_val
    
    # Generate with pure Bit 1
    cond_1 = np.ones((n_test, horizon, 1))
    with torch.no_grad():
        gen_1 = model.transform((n_test, horizon, 1), cond=cond_1)
    if gen_1.ndim == 2:
        gen_1 = gen_1[:, :, np.newaxis]
    gen_1 = gen_1 * (max_val - min_val) + min_val
    
    mean_0 = gen_0.mean()
    mean_1 = gen_1.mean()
    
    acc_0 = (gen_0.flatten() < THRESHOLD).mean() * 100
    acc_1 = (gen_1.flatten() >= THRESHOLD).mean() * 100
    
    print(f"    Bit 0: Mean={mean_0:.4f}s, Accuracy={acc_0:.1f}%")
    print(f"    Bit 1: Mean={mean_1:.4f}s, Accuracy={acc_1:.1f}%")
    print(f"    Overall: {(acc_0 + acc_1) / 2:.1f}%")
    
    if mean_0 < THRESHOLD and mean_1 > THRESHOLD:
        print("    ✓ Conditioning is working!")
    else:
        print("    ⚠ Conditioning may need improvement")

def visualize_sequences(data_gen, data_ori, labels_train, horizon):
    """Visualize generated vs real sequences."""
    # Ensure 3D
    if data_gen.ndim == 2:
        data_gen = data_gen[:, :, np.newaxis]
    if data_ori.ndim == 2:
        data_ori = data_ori[:, :, np.newaxis]
    
    _, ax = plt.subplots(figsize=(10, 5))
    ax_label = ax.twinx()
    
    for i in range(min(2, len(data_gen))):
        ax.plot(data_gen[i, :, 0], label=f"Synthetic (Sample {i})", linestyle='--')
    
    for i in range(min(1, len(data_ori))):
        ax.plot(data_ori[i, :, 0], label="Real Traffic (Sample 0)", linewidth=2, color='black', alpha=0.7)
    
    # Labels
    for i in range(min(1, len(labels_train))):
        lbl = labels_train[i]
        if lbl.ndim > 1:
            lbl = lbl[:, 0]
        ax_label.scatter(range(horizon), lbl, color='red', marker='x', label='Bit Label', s=20)

    ax.axhline(y=0.125, color='green', linestyle='--', alpha=0.5, label='Threshold')
    ax.legend(loc='upper left')
    ax.set_title("So sánh Traffic Sinh ra vs Traffic Thật (kèm Nhãn)")
    ax.set_xlim(0, horizon)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("IPD (seconds)")
    
    ax_label.set_ylabel("Bit (0/1)")
    ax_label.set_yticks([0, 1])
    
    plt.tight_layout()
    os.makedirs("helper/synthetic_data", exist_ok=True)
    plt.savefig("helper/synthetic_data/comparison_plot.png")
    print("[+] Saved: helper/synthetic_data/comparison_plot.png")
    plt.close()


def visualize_data(data_train, data_gen):
    pass

def main():
    args = parse_arguments()
    train_model(args)

if __name__ == "__main__":
    main()