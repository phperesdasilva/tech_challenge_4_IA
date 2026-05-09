from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import torch

from dataset import DatasetManager
from lstm import StockLSTM
from model_evaluation import train_model, evaluate_model

ticker = 'PETR4.SA'
years = 5

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 1
hidden_size = 32
num_layers = 2
dropout = 0.2

prediction_period = 15
lookback_period = 60

batch_size = 32
epochs = 200

dataset = DatasetManager(ticker=ticker, years=years)

df = dataset.download_data()

features_to_keep = ['Date', 'Close']
df = dataset.filter_features(df, features_to_keep)

features = dataset.get_features(df)
input_size = len(features)

df_train, y_train, df_test, y_test = dataset.split_data(df, 'Close', 0.8)

y_train_scaled, y_test_scaled = dataset.normalize_data(y_train, y_test)

x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = dataset.create_train_test_sequences(lookback_period, prediction_period, y_train_scaled, y_test_scaled)

model = StockLSTM(
    input_size=input_size, 
    hidden_size=hidden_size, 
    num_layers=num_layers, 
    output_size=prediction_period, #número de dias a serem previstos
    dropout=dropout
).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


