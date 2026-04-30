from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend

from dataset import DatasetManager
from lstm import StockLSTM
from model_evaluation import train_model, evaluate_model


# Configurações para download e pré-processamento dos dados
ticker = 'AAPL'
start_date = '2010-01-01'
#end_date = '2024-01-01'

# Configurações do modelo LSTM
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 64
num_layers = 2
dropout = 0.2

prediction_period = 1
lookback_period = 30

batch_size = 32
epochs = 100

dataset = DatasetManager(ticker=ticker, start_date=start_date)

df = dataset.download_data()
features_to_remove = ['Open', 'High', 'Low', 'Volume']
#df = dataset.preprocess_data(df)
df = dataset.remove_features(df, features_to_remove)

features = dataset.get_features(df)
input_size = len(features)

scaled_data = dataset.normalize_data(df)

x_train, y_train, x_test, y_test = dataset.split_data(scaled_data, features, 0.8 ,lookback_period, device)

model = StockLSTM(
    input_size=input_size, 
    hidden_size=hidden_size, 
    num_layers=num_layers, 
    output_size=prediction_period, #número de dias a serem previstos
    dropout=dropout
).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_model(model, optimizer, criterion, epochs, x_train, y_train)

y_test_pred, y_test, test_rmse, test_mae, test_mape = evaluate_model(df, model, ticker, dataset, x_test, y_test)

fig = plt.figure(figsize=(12,10))
gs = fig.add_gridspec(5,1)
ax1 = fig.add_subplot(gs[:3, 0])
ax1.plot(df.iloc[-len(y_test):].index, y_test, color='blue', label='Actual Price')
ax1.plot(df.iloc[-len(y_test):].index, y_test_pred, color='green', label='Predicted Price')
ax1.legend()
plt.title(f"{ticker} price prediction")
plt.xlabel('Date')
plt.ylabel('Price')

ax2 = fig.add_subplot(gs[4,0])
ax2.axhline(test_rmse, color='blue', linestyle='--', label='RMSE')
ax2.axhline(test_mae, color='green', linestyle='--', label='MAE')
ax2.axhline(test_mape, color='pink', linestyle='--', label='MAPE')
ax2.plot(df[-len(y_test):].index, abs(y_test - y_test_pred), 'r', label = 'Prediction Error')
ax2.legend()
plt.title('Prediction Error')
plt.xlabel('Date')
plt.ylabel('Error')

plt.show()