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

def get_mlflow_info():
  with open("last_run_id.txt", "r", encoding="utf-8") as arquivo:
      RUN_ID = arquivo.read()

  MODEL_URI = f"runs:/{RUN_ID}/{params['model_name']}"
  print(f'\n\nModel URI: {MODEL_URI}\n\n')

  mlflow.set_tracking_uri(params['mlflow_tracking_uri'])
  print(f"MLFlow Tracking URI configurado para: {params['mlflow_tracking_uri']}")
  
  return RUN_ID, MODEL_URI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def should_retrain_model(days_threshold=RETRAIN_AFTER_DAYS):
    try:
        with open(LAST_RUN_DATE_FILE, "r", encoding="utf-8") as file:
            last_run_date_str = file.read().strip()
    except OSError:
        print("Arquivo de data de treino nao encontrado. Treinando modelo.")
        return True

    if not last_run_date_str:
        print("Arquivo de data de treino vazio. Treinando modelo.")
        return True

    try:
        last_run_date = datetime.fromisoformat(last_run_date_str)
    except ValueError:
        print("Data de treino invalida. Treinando modelo.")
        return True

    return datetime.now() - last_run_date >= timedelta(days=days_threshold)

def load_resources():
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
    model.load_state_dict(torch.load(params['model_path'], map_location=device))
    model.to(device)
    model.eval()
    print("Modelo carregado localmente.")
    scaler = joblib.load(params['scaler_path'])
    return model, scaler


def retrain_if_needed():
    global model, scaler

    if not should_retrain_model():
        return False, None

    train_model()
    model, scaler = load_resources()
    run_id, _ = get_mlflow_info()
    return True, run_id

model, scaler = load_resources()

@app.route('/')
def home():
  
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """
    Endpoint de verificação de saúde da API.
    ---
    responses:
      200:
        description: API está online e recursos carregados.
        schema:
          properties:
            status:
              type: string
              example: "healthy"
            device:
              type: string
              example: "cuda"
            model_loaded:
              type: boolean
            scaler_loaded:
              type: boolean
    """
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
    """
    Realiza a previsão do preço de fechamento para os próximos 15 dias.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - ticker
          properties:
            ticker:
              type: string
              example: "PETR4.SA"
              description: O símbolo da ação no Yahoo Finance.
    responses:
      200:
        description: Previsões geradas com sucesso.
        schema:
          properties:
            ticker:
              type: string
            predictions:
              type: array
              items:
                type: number
              description: Lista com os preços previstos para os próximos 15 dias.
      400:
        description: Erro na requisição (ex. ticker faltando).
      404:
        description: Ticker não encontrado no Yahoo Finance.
      500:
        description: Erro interno no processamento ou carregamento do modelo.
    """
    try:
        retrained, run_id = retrain_if_needed()

        content = request.json
        ticker = content.get('ticker')
        
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

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['GET'])
def train():
    """
    Endpoint para treinar o modelo LSTM com os dados mais recentes.
    ---
    responses:
      200:
        description: Treinamento iniciado com sucesso.
      403:
        description: Erro interno durante o processo de treinamento.
    """
    try:
        global model, scaler
        train_model()
        model, scaler = load_resources()
        RUN_ID, _ = get_mlflow_info()
        mlflow_link = f"http://localhost:3050/#/experiments/0/runs/{RUN_ID}"
        return jsonify({"message": f"Treinamento finalizado com sucesso. Log: {mlflow_link}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 403

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)