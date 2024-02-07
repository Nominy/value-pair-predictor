import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from main import LSTMModel, create_sequences
from data.preprocess import load

def load_scaler(scaler_path='currency_pair_scaler.save'):
    return joblib.load(scaler_path)

def preprocess_data(data_path, scaler, seq_length=50):
    eval_prices = np.array(load(data_path)).reshape(-1, 1)
    normalized_eval_prices = scaler.transform(eval_prices)
    X_eval, _ = create_sequences(normalized_eval_prices, seq_length)
    return torch.FloatTensor(X_eval), eval_prices

def load_model(model_path='currency_pair_prediction_model.pth', input_dim=1, hidden_dim=100, num_layers=2, output_dim=1):
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout=0.89, bidirectional=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, X_eval_tensor, scaler):
    with torch.no_grad():
        predictions_normalized = model(X_eval_tensor)
    return scaler.inverse_transform(predictions_normalized.detach().numpy())

def plot_results(eval_prices, predictions):
    last_eval_price = eval_prices[-1]
    predictions_with_continuity = np.insert(predictions.flatten(), 0, last_eval_price)

    eval_prices_plot = eval_prices.flatten()
    predictions_plot = predictions_with_continuity.flatten()

    time_steps = np.arange(len(eval_prices_plot) + len(predictions_plot))

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps[:len(eval_prices_plot)], eval_prices_plot, label='Evaluation Prices', color='blue')
    plt.plot(time_steps[len(eval_prices_plot):len(eval_prices_plot)+len(predictions_plot)], predictions_plot, label='Predictions', color='red', linestyle='--')

    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.title('Evaluation Prices and Predictions')
    plt.legend()

    plt.show()

def main():
    seq_length=5

    scaler_path = 'currency_pair_scaler.save'
    scaler = load_scaler(scaler_path)

    model_path = 'currency_pair_prediction_model.pth'
    model = load_model(model_path)

    data_path = './data/eval_data.csv'
    X_eval_tensor, eval_prices = preprocess_data(data_path, scaler, seq_length)

    predictions = predict(model, X_eval_tensor, scaler)

    plot_results(eval_prices, predictions)

if __name__ == "__main__":
    main()
