import torch.nn as nn

class StockLSTM(nn.Module):
    """Modelo LSTM simples para previsão de séries temporais de preços.

    Args:
        input_size (int): Número de features de entrada por timestep.
        hidden_size (int): Tamanho do estado oculto da LSTM.
        num_layers (int): Número de camadas LSTM empilhadas.
        output_size (int): Dimensão da saída (número de passos previstos).
        dropout (float): Dropout entre camadas LSTM.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=10, dropout=0.2):
        super(StockLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """Propaga os dados de entrada pela LSTM e retorna as previsões.

        Args:
            x (torch.Tensor): Tensor de entrada com shape `(batch, seq_len, input_size)`.

        Returns:
            torch.Tensor: Saída da camada fully-connected com shape `(batch, output_size)`.
        """
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out