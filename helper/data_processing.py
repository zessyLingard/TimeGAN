"""data_processing.py"""

import os
import json
import datetime
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


# Function to save model state and metadata
def save_model(model: nn.Module, data: Dict, name: Optional[str] = None) -> None:
    """
    Save the model state and metadata.

    Args:
    - model (nn.Module): The neural network model to be saved.
    - data (dict): Metadata associated with the training.
    - name (str): Optional custom name for the saved files.

    Returns:
    - None
    """
    model_dir = os.path.join("helper", "models")
    date_str = str(datetime.date.today())
    name = f"{date_str}_{name}" if name is not None else f"{date_str}_MC-TimeGAN"
    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)
    # Save metadata as JSON file
    metadata_path = os.path.join(model_dir, name + ".json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(data, metadata_file, ensure_ascii=False, indent=4)
    # Save model state as .pth file
    model_path = os.path.join(model_dir, name + ".pth")
    torch.save(model.state_dict(), model_path)


# Function to load CSV files into pandas DataFrames
def loading(*files: str) -> pd.DataFrame:
    """
    Load CSV files into pandas DataFrames.

    Args:
    - files (str): Names of the CSV files to be loaded (without extension).

    Returns:
    - pd.DataFrame or tuple of pd.DataFrame: Loaded data.
    """
    print(
        """
    ___  ________     _____ _                _____   ___   _   _ 
    |  \/  /  __ \   |_   _(_)              |  __ \ / _ \ | \ | |
    | .  . | /  \/_____| |  _ _ __ ___   ___| |  \// /_\ \|  \| |
    | |\/| | |  |______| | | | '_ ` _ \ / _ \ | __ |  _  || . ` |
    | |  | | \__/\     | | | | | | | | |  __/ |_\ \| | | || |\  |
    \_|  |_/\____/     \_/ |_|_| |_| |_|\___|\____/\_| |_/\_| \_/"""
    )
    return_list = list()
    for file_name in files:
        data = pd.read_csv(file_name)
        print('Shape of "' + file_name + '":', data.shape)
        return_list.append(data)

    return return_list.pop() if len(return_list) == 1 else tuple(return_list)


# Function to prepare data with scaling and sliding window
# Function to prepare data with scaling and sliding window
def preparing(
    *inputs: Tuple,
    horizon: int,
    shuffle_stack: bool = True,
    random_state: Optional[int] = None,
):
    """
    Prepare data by scaling and creating sequences with a sliding window.
    Fix: Synchronized shuffling for Data and Labels.
    """
    if len(inputs) > 2:
        raise ValueError(
            "Only one input (data) or two inputs (data and labels) are allowed"
        )

    processed_stacks = []
    scaling_info = []  # Lưu lại max/min nếu có scale

    # --- BƯỚC 1: Xử lý từng Input (Scale + Cắt cửa sổ) ---
    for data, bool_scale in inputs:
        # Scaling
        if bool_scale:
            scaler = MinMaxScaler().fit(data)
            data_transformed = scaler.transform(data)
            max_val = scaler.data_max_
            min_val = scaler.data_min_
            scaling_info.append((max_val, min_val))
        else:
            data_transformed = data
            scaling_info.append(None)

        # Create sequences (Sliding Window)
        # Cắt thành các đoạn dài 'horizon'
        stack = np.stack(
            [data_transformed[i : i + horizon] for i in range(len(data_transformed) - horizon)]
        )
        processed_stacks.append(stack)

    # --- BƯỚC 2: Xáo trộn ĐỒNG BỘ (Synchronized Shuffling) ---
    # Đây là bước quan trọng nhất: Xáo trộn cả X và y cùng một hoán vị
    if shuffle_stack:
        if len(processed_stacks) > 1:
            # sklearn.shuffle sẽ xáo trộn các mảng đầu vào theo cùng 1 thứ tự
            processed_stacks = shuffle(*processed_stacks, random_state=random_state)
            # Lưu ý: shuffle trả về tuple nếu input là nhiều mảng
        else:
            processed_stacks = [shuffle(processed_stacks[0], random_state=random_state)]

    # --- BƯỚC 3: Đóng gói kết quả trả về ---
    return_list = []
    
    # Input 1: Data (Thường là có scale)
    return_list.append(processed_stacks[0])
    if scaling_info[0] is not None:
        return_list.append(scaling_info[0][0]) # max
        return_list.append(scaling_info[0][1]) # min
        
    # Input 2: Labels (Nếu có)
    if len(processed_stacks) > 1:
        return_list.append(processed_stacks[1])
        if scaling_info[1] is not None:
            return_list.append(scaling_info[1][0])
            return_list.append(scaling_info[1][1])

    return return_list