from pathlib import Path
import sys

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import torch
import mlflow

from model.dataset import DatasetManager
from model.lstm import StockLSTM
from torch.utils.data import TensorDataset,DataLoader

from api.global_params import params

mlflow.set_tracking_uri(params['mlflow_tracking_uri'])

tickers = ['PETR4.SA']
years = 5

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DatasetManager(tickers=tickers, years=years)

df = dataset.download_data()

features_to_keep = ['Close']
df = dataset.filter_features(df, features_to_keep)

features = dataset.get_features(df)
input_size = len(features)

df_train, y_train, df_test, y_test, split = dataset.split_data(df, 'Close', 0.8)

y_train_scaled, y_test_scaled = dataset.normalize_data(y_train, y_test)

x_train, y_train, x_test, y_test = dataset.create_train_test_sequences(params['lookback_period'], params['prediction_period'], y_train_scaled, y_test_scaled)

x_train_tensor = torch.FloatTensor(x_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
x_test_tensor = torch.FloatTensor(x_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

model = StockLSTM(
    input_size=input_size, 
    hidden_size=params['hidden_size'], 
    num_layers=params['num_layers'], 
    output_size=params['prediction_period'], #número de dias a serem previstos
    dropout=params['dropout']
).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

mlflow.enable_system_metrics_logging()

with mlflow.start_run(run_name='Stock Prediction') as run:

    mlflow.log_params({"Learning Rate": params['learning_rate'], 
                       "Batch Size": params['batch_size'], 
                       "Epochs": params['epochs'],
                       "Hidden Size": params['hidden_size'],
                       "Number of Layers": params['num_layers'],
                       "Dropout": params['dropout']})

    for epoch in range(params['epochs']):
        epoch_loss = 0.0
        num_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss +=loss.item()
            num_batches +=1

        epoch_loss_avg = epoch_loss / num_batches

        if epoch % 10 == 0 :
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss_avg:.4f}')

        mlflow.log_metric("Loss", loss.item(), step=epoch)

    run_id = run.info.run_id
    print(f"Run ID: {run_id}\n\n\n")

    with open("last_run_id.txt", "w") as f:
        f.write(run_id)

    model_uri = f"runs:/{run_id}/{params['model_name']}"
    
    model.eval()

    with torch.no_grad():
        y_pred_test = model(x_test_tensor.to(device))

    y_pred_test = y_pred_test.cpu().numpy()

    y_pred_real = dataset.inverse_transform(y_pred_test)
    y_test_real = dataset.inverse_transform(y_test)

    mape = mean_absolute_percentage_error(y_test_real, y_pred_real)
    print(f'\n\n\nMAPE: {mape:.4f}')

    mae = mean_absolute_error(y_test_real, y_pred_real)
    print(f'MAE: {mae:.4f}')

    rmse = root_mean_squared_error(y_test_real, y_pred_real)
    print(f'RMSE: {rmse:.4f}')

    mlflow.log_metrics({"MAPE": mape,"MAE": mae,"RMSE": rmse})

    torch.save(model.state_dict(), params['model_path'])
    print(f'Model saved to {params['model_path']}')
    
    mlflow.pytorch.log_model(model, name=params['model_name'])                                    
