import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

#Abri csv
data = pd.read_csv('BBAS3_historico.csv')
df = pd.DataFrame(data)
df = df[['Date','Close']]

#Separação em treino e teste:
split = int(0.8*len(df['Close']))
df_train = df.iloc[:split]
df_teste = df.iloc[split:]
y_train = df_train['Close'].values
y_test = df_teste['Close'].values

#Normalizar os dados:
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(y_train.reshape(-1, 1))
y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()

#Separar em sequências para o LSTM - Treino
X_train_list = []
y_train_list = []
lookback =180
horizonte = 15
for i in range(len(y_train_scaled)-lookback-horizonte+1):
    X_train_list.append(y_train_scaled[i:(i+lookback)])
    y_train_list.append(y_train_scaled[i+lookback:i+lookback+horizonte])

#Separar em sequências para o LSTM - Teste
X_test_list = []
y_test_list = []
lookback = 180
horizonte = 15
for i in range(len(y_test_scaled)-lookback-horizonte+1):
    X_test_list.append(y_test_scaled[i:(i+lookback)])
    y_test_list.append(y_test_scaled[i+lookback:i+lookback+horizonte])

# Converter para numpy arrays
X_train = np.array(X_train_list).reshape(-1,180 , 1)
y_train = np.array(y_train_list)
X_test = np.array(X_test_list).reshape(-1,180, 1)
y_test = np.array(y_test_list)

#Converter para tensor PyTorch:
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

#Criar DataLoader:
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=2, output_size=15):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTM()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(300):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / num_batches:.4f}')

model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)

y_pred_test = y_pred_test.numpy()

y_train_real = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_pred_real = scaler.inverse_transform(y_pred_test.reshape(-1, 1)).reshape(y_pred_test.shape)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# Calcular MAPE
mape = mean_absolute_percentage_error(y_test_real, y_pred_real)
print(f'MAPE: {mape:.4f}')

#Salvar modelo:

PATH = 'modelo_lstm.pth'
torch.save(model.state_dict(),PATH)
print('modelo salvo com sucesso')

# Alinhar datas com sequências
train_start_idx = lookback
train_end_idx = split - horizonte + 1
test_start_idx = split + lookback

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
gap_end_idx = split + lookback

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

