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
    """Lê o último `run_id` salvo e retorna a URI do modelo no MLflow.

    Abre o arquivo `last_run_id.txt`, lê o identificador do último run
    e monta a URI do modelo no formato aceito pelo MLflow.

    Returns:
        tuple[str, str]: `(run_id, model_uri)`
    """
    with open("last_run_id.txt", "r", encoding="utf-8") as arquivo:
        RUN_ID = arquivo.read()

    MODEL_URI = f"runs:/{RUN_ID}/{params['model_name']}"
    print(f'\n\nModel URI: {MODEL_URI}\n\n')

    mlflow.set_tracking_uri(params['mlflow_tracking_uri'])
    print(f"MLFlow Tracking URI configurado para: {params['mlflow_tracking_uri']}")
    
    return RUN_ID, MODEL_URI

def load_resources(ticker):
    """Carrega o modelo e o scaler para o `ticker` fornecido.

    Args:
        ticker (str): Símbolo da ação (p.ex. 'AAPL' ou 'PETR4.SA').

    Returns:
        tuple: `(model, scaler)` prontos para inferência.
    """
    if '.' in ticker:
        ticker = ticker.replace('.', '_')

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
    scaler_file = f'scaler_files/{ticker}_{params['scaler_path']}'
    scaler = joblib.load(scaler_file)
    return model, scaler

def should_retrain_model(ticker):
    """Verifica se já existe um modelo treinado para o `ticker`.

    Procura por arquivos em `model_files` cujo prefixo corresponda ao
    ticker solicitado.

    Args:
        ticker (str): Símbolo da ação.

    Returns:
        bool: `False` se o modelo existir, `True` caso contrário.
    """
    for model in os.listdir('model_files'):
        file_ticker = model.split('_')[0]
        real_ticker = ticker.split('.')[0]

        if file_ticker == real_ticker:
            print('Model found in model_files')
            return False
        
    print('Model not found in model_files')
    return True    

def retrain_if_needed(ticker):
    """Re-treina o modelo para o `ticker` se não houver um modelo local.

    Se já existir um modelo em `model_files`, apenas carrega os recursos.
    Caso contrário, executa `train_model` e retorna o `run_id` criado.

    Args:
        ticker (str): Símbolo da ação.

    Returns:
        tuple: `(retrained (bool), run_id_or_None, model, scaler)`.
    """
    if not should_retrain_model(ticker = ticker):
        model, scaler = load_resources(ticker)
        return False, None, model, scaler

    train_model(ticker)
    model, scaler = load_resources(ticker)
    run_id, _ = get_mlflow_info()
    return True, run_id, model, scaler

@app.route('/')
def home():
    """Rota raiz que renderiza a página inicial do frontend."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    health_status = {
        "status": "healthy",
        "device": str(device),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }
        
    return jsonify(health_status), 200
    
"""Rota de saúde da API que retorna o status e artefatos carregados."""
    
@app.route('/predict', methods=['POST'])
def predict_next_days():
    global ticker
    """Endpoint `/predict` que recebe um JSON com o ticker e retorna previsões.

    Espera um payload JSON com a chave `ticker`. Garante que o modelo
    esteja disponível (re-treinando se necessário), baixa dados históricos,
    prepara os tensores, realiza inferência e devolve as previsões desscaladas.
    """

    try:
        model, scaler = load_resources(ticker)
    except Exception as e:
        print(f"Erro ao carregar modelo ou scaler: {e}")
    
    try:

        content = request.json
        ticker = content.get('ticker')

        retrained, run_id, model, scaler = retrain_if_needed(ticker)
        
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
            response["mlflow_link"] = f"http://localhost:3050/#/experiments/0/runs/{run_id}"
            
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)