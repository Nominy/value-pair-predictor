import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib

from data.preprocess import load

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def preprocess_data(file_path, seq_length):
    prices = np.array(load(file_path)).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    prices_normalized = scaler.fit_transform(prices)
    X, y = create_sequences(prices_normalized, seq_length)
    return torch.FloatTensor(X), torch.FloatTensor(y), scaler

def train_model(model, train_loader, criterion, optimizer, num_epochs, scheduler=None):
    model.train()
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        if scheduler:
            scheduler.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

def get_data_loader(X, y, batch_size):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # Parameters
    seq_length = 5
    batch_size = 64
    num_epochs = 60
    learning_rate = 0.005
    dropout = 0.89
    bidirectional = True

    # Model and training setup
    input_dim = 1
    hidden_dim = 100
    num_layers = 2
    output_dim = 1

    # File paths
    data_file_path = './data/data.csv'
    model_path = 'currency_pair_prediction_model.pth'
    scaler_path = 'currency_pair_scaler.save'

    X, y, scaler = preprocess_data(data_file_path, seq_length)
    train_loader = get_data_loader(X, y, batch_size)

    model = LSTMModel(input_dim, hidden_dim, num_layers=2, output_dim=output_dim, dropout=dropout)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    train_model(model, train_loader, criterion, optimizer, num_epochs, scheduler)

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
