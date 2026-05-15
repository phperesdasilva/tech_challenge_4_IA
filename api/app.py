from mlflow.runs import RUN_ID
from datetime import datetime, timedelta
import torch
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from flasgger import Swagger
import mlflow
from api.global_params import params
import joblib
from model.lstm import StockLSTM
from model.model_trainer import train_model
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

swagger = Swagger(app, template={
    "info": {
        "title": "Stock Prediction API",
        "description": "API para previsão de preços de ações usando LSTM (Tech Challenge 4)",
        "version": "1.0.0"
    }
})

LAST_RUN_DATE_FILE = "last_run_date.txt"
RETRAIN_AFTER_DAYS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mlflow_info():
  with open("last_run_id.txt", "r", encoding="utf-8") as arquivo:
      RUN_ID = arquivo.read()

  MODEL_URI = f"runs:/{RUN_ID}/{params['model_name']}"
  print(f'\n\nModel URI: {MODEL_URI}\n\n')

  mlflow.set_tracking_uri(params['mlflow_tracking_uri'])
  print(f"MLFlow Tracking URI configurado para: {params['mlflow_tracking_uri']}")
  
  return RUN_ID, MODEL_URI

def load_resources(ticker):
    # try:
    #   model = mlflow.pytorch.load_model(MODEL_URI, map_location=device)
    #   print("Modelo carregado do servidor MLFlow.")
    # except Exception as e:
    #   print(f"Erro ao carregar o modelo do servidor MLFlow: {e}")
    model = StockLSTM(
        input_size=1, 
        hidden_size=params['hidden_size'], 
        num_layers=params['num_layers'], 
        output_size=params['prediction_period'],
        dropout=params['dropout']
    ).to(device)
    model_file = f'model_files/{ticker}_{params['model_path']}'
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    print("Modelo carregado localmente.")
    scales_file = f'scales_files/{ticker}_{params['scaler_path']}'
    scaler = joblib.load(scales_file)
    return model, scaler

def should_retrain_model(ticker):
    for model in os.listdir('model_files'):
        file_ticker = model.split('_')[0]
        real_ticker = ticker.split('.')[0]

        if file_ticker == real_ticker:
            print('Model found in model_files')
            return False
    return True    

def retrain_if_needed(ticker):
    global model, scaler

    if not should_retrain_model(ticker = ticker):
        return False, None

    train_model(ticker)
    model, scaler = load_resources(ticker)
    run_id, _ = get_mlflow_info()
    return True, run_id

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    health_status = {
        "status": "healthy",
        "device": str(device),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }
    
    if not model or not scaler:
        health_status["status"] = "unhealthy"
        return jsonify(health_status), 503
        
    return jsonify(health_status), 200
    
@app.route('/predict', methods=['POST'])
def predict_next_days():
    global ticker
    
    try:
        content = request.json
        ticker = content.get('ticker')

        retrained, run_id = retrain_if_needed(ticker)

        if not retrained:
            model, scaler = load_resources(ticker)
        
        if not ticker:
            return jsonify({"error": "Ticker não fornecido"}), 400

        data = yf.download(ticker, period="5y")
        if data.empty:
            return jsonify({"error": "Ticker não encontrado ou sem dados"}), 404

        df_input = data[['Close']].tail(params['lookback_period']).values
        
        scaled_data = scaler.transform(df_input)
        
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(input_tensor)
            predictions_scaled = y_pred.cpu().numpy()

        inverse_predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))

        response = {
            "ticker": ticker,
            "predictions": inverse_predictions.flatten().tolist()
        }

        if retrained:
            response["model_retrained"] = True
            response["run_id"] = run_id
            response["retrain_message"] = f"Modelo retreinado automaticamente: passaram-se {diff_days} dias desde o último treino."
            response["mlflow_link"] = f"http://localhost:3050/#/experiments/0/runs/{run_id}"
            
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)