"""
Model Classes cho MC-TimeGAN
Được tách ra để dùng chung cho encoder.py, receiver.py, test_localhost.py

QUAN TRỌNG: Các class này phải GIỐNG HỆT với lúc train trên Kaggle
để torch.load() có thể deserialize đúng model.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = torch.device("cpu")

class ConditioningNetwork(nn.Module):
    def __init__(self, input_size, condition_size):
        super().__init__()
        hidden_size = max(16, condition_size * 4)
        self.condition = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, condition_size),
            nn.Tanh()
        )
    def forward(self, conds):
        return self.condition(conds) if conds is not None else None

class Embedder(nn.Module):
    def __init__(self, module_name, input_features, hidden_dim, num_layers):
        super().__init__()
        rnn_class = nn.GRU if module_name == "gru" else nn.LSTM
        self.rnn = rnn_class(input_features, hidden_dim, num_layers, batch_first=True)
        self.model = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
    def forward(self, x, c=None):
        if c is not None: x = torch.cat([x, c], dim=-1)
        seq, _ = self.rnn(x)
        return self.model(seq)

class Recovery(nn.Module):
    def __init__(self, module_name, input_features, hidden_dim, num_layers):
        super().__init__()
        rnn_class = nn.GRU if module_name == "gru" else nn.LSTM
        self.rnn = rnn_class(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.model = nn.Sequential(nn.Linear(hidden_dim, input_features), nn.Sigmoid())
    def forward(self, x, c=None):
        if c is not None: x = torch.cat([x, c], dim=-1)
        seq, _ = self.rnn(x)
        return self.model(seq)

class Generator(nn.Module):
    def __init__(self, module_name, input_features, hidden_dim, num_layers):
        super().__init__()
        rnn_class = nn.GRU if module_name == "gru" else nn.LSTM
        self.rnn = rnn_class(input_features, hidden_dim, num_layers, batch_first=True)
        self.model = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
    def forward(self, x, c=None):
        if c is not None: x = torch.cat([x, c], dim=-1)
        seq, _ = self.rnn(x)
        return self.model(seq)

class Supervisor(nn.Module):
    def __init__(self, module_name, input_features, hidden_dim, num_layers):
        super().__init__()
        rnn_class = nn.GRU if module_name == "gru" else nn.LSTM
        self.rnn = rnn_class(input_features, hidden_dim, max(1, num_layers - 1), batch_first=True)
        self.model = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
    def forward(self, x, c=None):
        if c is not None: x = torch.cat([x, c], dim=-1)
        seq, _ = self.rnn(x)
        return self.model(seq)

class Discriminator(nn.Module):
    def __init__(self, module_name, hidden_dim, num_layers):
        super().__init__()
        rnn_class = nn.GRU if module_name == "gru" else nn.LSTM
        self.rnn = rnn_class(hidden_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.model = nn.Linear(2 * hidden_dim, 1)
    def forward(self, x, c=None):
        if c is not None: x = torch.cat([x, c], dim=-1)
        seq, _ = self.rnn(x)
        return self.model(seq)

class MCTimeGAN(nn.Module):
    def __init__(self, module_name="gru", input_features=1, input_conditions=None,
                 hidden_dim=8, num_layers=3, epochs=100, batch_size=128, learning_rate=1e-3):
        super().__init__()
        self.module_name = module_name
        self.input_features = input_features
        self.input_conditions = input_conditions
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cond_size = max(8, hidden_dim // 4)
        self.losses = []
        self.fitting_time = None
        
        if input_conditions is not None:
            self.condnet = ConditioningNetwork(input_conditions, self.cond_size)
            self.embedder = Embedder(module_name, input_features + self.cond_size, hidden_dim, num_layers)
            self.recovery = Recovery(module_name, input_features, hidden_dim + self.cond_size, num_layers)
            self.generator = Generator(module_name, input_features + self.cond_size, hidden_dim, num_layers)
            self.supervisor = Supervisor(module_name, hidden_dim + self.cond_size, hidden_dim, num_layers)
            self.discriminator = Discriminator(module_name, hidden_dim + self.cond_size, num_layers)

    def transform(self, data_shape, **kwargs):
        data_z = torch.rand(size=data_shape, dtype=torch.float32, device=device)
        conditions = torch.tensor(
            np.concatenate([c for c in kwargs.values()], axis=-1), 
            dtype=torch.float32, device=device) if kwargs else None

        dataset = TensorDataset(data_z, conditions) if kwargs else TensorDataset(data_z)
        batches = DataLoader(dataset, batch_size=self.batch_size)
        
        generated = []
        self.eval()
        with torch.no_grad():
            for batch in batches:
                z, c = batch if kwargs else (*batch, None)
                conds = self.condnet(c) if c is not None else None
                e_hat = self.generator(z, conds)
                h_hat = self.supervisor(e_hat, conds)
                x_hat = self.recovery(h_hat, conds)
                generated.append(x_hat.cpu().numpy())
        
        return np.concatenate(generated, axis=0)