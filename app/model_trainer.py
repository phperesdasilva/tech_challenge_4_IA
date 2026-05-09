from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import torch

import plotly.express as px
import pandas as pd

from dataset import DatasetManager
from lstm import StockLSTM
from torch.utils.data import TensorDataset,DataLoader

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

df_train, y_train, df_test, y_test, split = dataset.split_data(df, 'Close', 0.8)

y_train_scaled, y_test_scaled = dataset.normalize_data(y_train, y_test)

x_train, y_train, x_test, y_test = dataset.create_train_test_sequences(lookback_period, prediction_period, y_train_scaled, y_test_scaled)


x_train_tensor = torch.FloatTensor(x_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
x_test_tensor = torch.FloatTensor(x_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

model = StockLSTM(
    input_size=input_size, 
    hidden_size=hidden_size, 
    num_layers=num_layers, 
    output_size=prediction_period, #número de dias a serem previstos
    dropout=dropout
).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(200):
    epoch_loss = 0.0
    num_batches = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()
        num_batches +=1
    epoch_loss_avg = epoch_loss / num_batches
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss_avg:.4f}')

model.eval()

with torch.no_grad():
    y_pred_test = model(x_test_tensor.to(device))

y_pred_test = y_pred_test.cpu().numpy()

y_train_real = dataset.inverse_transform(y_train)
y_pred_real = dataset.inverse_transform(y_pred_test)
y_test_real = dataset.inverse_transform(y_test)

mape = mean_absolute_percentage_error(y_test_real, y_pred_real)
print(f'MAPE: {mape:.4f}')

mae = mean_absolute_error(y_test_real, y_pred_real)
print(f'MAE: {mae:.4f}')

rmse = root_mean_squared_error(y_test_real, y_pred_real)
print(f'RMSE: {rmse:.4f}')

model_path = 'lstm_model_weights.pth'
torch.save(model.state_dict(),model_path)
print(f'Model saved to {model_path}')

#####################
# parte não testada #
#####################

# Alinhar datas com sequências
train_start_idx = lookback_period
train_end_idx = split - prediction_period + 1
test_start_idx = split + lookback_period

test_dates = df['Date'].iloc[test_start_idx:test_start_idx + len(y_pred_real)].values

train_dates = df['Date'].iloc[train_start_idx:train_end_idx].values

# Criar DataFrames com valores REAIS (desnormalizados)
df_train_full = pd.DataFrame({
    'Date': train_dates,
    'Price': y_train_real[:,0].flatten(),
    'Type': 'Treino'
})

df_test_full = pd.DataFrame({
    'Date': test_dates,
    'Price': y_test_real[:,0].flatten(),
    'Type': 'Real (Teste)'
})

df_pred_full = pd.DataFrame({
    'Date': test_dates,
    'Price': y_pred_real[:,0].flatten(),
    'Type': 'Previsto'
})

# Preencher a lacuna no gráfico SEM viés no modelo
gap_start_idx = split
gap_end_idx = split + lookback_period

gap_dates = df['Date'].iloc[gap_start_idx:gap_end_idx].values
gap_prices = df['Close'].iloc[gap_start_idx:gap_end_idx].values

df_gap = pd.DataFrame({
    'Date': gap_dates,
    'Price': gap_prices,
    'Type': 'Real (Teste)'
})

df_plot = pd.concat([df_train_full, df_gap, df_test_full, df_pred_full], ignore_index=True)

fig = px.line(df_plot, x='Date', y='Price', color='Type',
              title='LSTM: Previsão de Preço de Ação',
              labels={'Price': 'Preço (R$)', 'Date': 'Data', 'Type': 'Tipo'})

fig.update_layout(
    xaxis_title='Data',
    yaxis_title='Preço (R$)',
    hovermode='x unified',
    template='plotly_white',
    height=600
)

fig.show()
