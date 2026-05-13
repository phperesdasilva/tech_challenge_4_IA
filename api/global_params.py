params = {
    'hidden_size': 32,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'prediction_period': 15,
    'lookback_period': 60,
    'batch_size': 32,
    'epochs': 200,
    'model_path': 'lstm_model_weights.pt2',
    'scaler_path': 'lstm_scaler.pkl',
    'mlflow_tracking_uri': 'http://localhost:3050',
    'model_name': 'Stock Prediction LSTM',
}