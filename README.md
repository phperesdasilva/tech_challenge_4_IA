# Stock Prediction API - Tech Challenge 4

## 📊 Descrição do Projeto

Sistema de previsão de preços de ações usando redes neurais LSTM (Long Short-Term Memory). O projeto implementa uma API REST que fornece previsões de preços para os próximos 15 dias, baseado em análise de dados históricos da ação PETR4.SA dos últimos 5 anos.

**Fase:** Deep Learning - Fase 4 | **PosTech ML Engineering**

---

## 🎯 Objetivo

Desenvolver um modelo de deep learning capaz de:
- Processar séries temporais de preços de ações
- Treinar uma rede LSTM para capturar padrões temporais
- Fazer previsões de preços futuros
- Expor as previsões através de uma API REST documentada
- Rastrear experimentos e modelos com MLflow

---

## 🏗️ Arquitetura

### Componentes Principais

```
┌─────────────────────────────────────┐
│     Flask API (Port 5000)           │
│  - Endpoints de Previsão            │
│  - Documentação Swagger             │
│  - Retreinamento Automático         │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       │               │
┌──────▼────────┐  ┌──▼─────────────┐
│  LSTM Model   │  │  MinMaxScaler  │
│  (32 neurons) │  │  (Normalized)  │
└───────────────┘  └────────────────┘
       ▲
       │
┌──────┴──────────────┐
│  Data Processing    │
│  - yfinance         │
│  - Sequences        │
│  - Train/Test Split │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  MLflow Tracking    │
│  (Port 3050)        │
│  - Params Logging   │
│  - Metrics Logging  │
│  - Model Registry   │
└─────────────────────┘
```

### Stack Tecnológico

- **Deep Learning:** PyTorch, LSTM
- **API:** Flask, Flasgger (Swagger), Flask-CORS
- **Processamento de Dados:** Pandas, NumPy, scikit-learn
- **Dados:** yfinance
- **ML Ops:** MLflow
- **Containerização:** Docker, Docker Compose
- **Deployment:** Gunicorn

---

## 📋 Requisitos

### Dependências Principais

```
torch>=2.0.0
fastapi==0.136.1
Flask==3.0.0
Flask-CORS==6.0.2
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
yfinance>=0.2.32
mlflow>=2.10.0
joblib>=1.3.0
gunicorn>=25.0.0
```

Para a lista completa, consulte [requirements.txt](requirements.txt)

---

## 🚀 Instalação e Uso

### Opção 1: Execução Local com Ambiente Virtual

#### 1. Clonar o Repositório
```bash
git clone <repo-url>
cd tech_challenge_4_IA
```

#### 2. Criar Ambiente Virtual
```bash
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

#### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```

#### 4. Treinar o Modelo (Opcional)
```bash
python -m model.model_trainer
```

#### 5. Iniciar a API
```bash
python -m api.app
```

A API estará disponível em `http://localhost:5000`

---

### Opção 2: Execução com Docker Compose (Recomendado)

#### 1. Pré-requisitos
- Docker Desktop instalado
- Docker Compose (incluído no Docker Desktop)

#### 2. Iniciar os Serviços
```bash
docker-compose up --build
```

Isso iniciará:
- **API Flask:** http://localhost:5000
- **MLflow Server:** http://localhost:3050

#### 3. Parar os Serviços
```bash
docker-compose down
```

---

## 📡 API Endpoints

### Documentação Interativa
- **Swagger UI:** http://localhost:5000/apidocs
- **ReDoc:** http://localhost:5000/redoc

### Endpoints Disponíveis

#### 1. **Home / Interface Web**
```
GET /
```
Retorna a página HTML interativa para testes.

#### 2. **Previsão de Preços**
```
POST /predict
Content-Type: application/json

Payload:
{
    "ticker": "PETR4.SA"
}

Response (200):
{
    "ticker": "PETR4.SA",
    "predictions": [103.45, 104.12, 103.89, ...],
    "timestamp": "2026-05-13T10:30:00"
}
```

---

## 🧠 Modelo LSTM

### Arquitetura

```
Input Layer: 1 (preço de fechamento)
    ↓
LSTM Layer 1: 32 neurons + Dropout(0.2)
    ↓
LSTM Layer 2: 32 neurons + Dropout(0.2)
    ↓
Dense Layer: 15 neurons (15 dias de previsão)
    ↓
Output: Vetor de previsões para os próximos 15 dias
```

### Hiperparâmetros

| Parâmetro | Valor |
|-----------|-------|
| Hidden Size | 32 |
| Número de Camadas LSTM | 2 |
| Dropout | 0.2 |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 200 |
| Lookback Period | 60 dias |
| Prediction Period | 15 dias |

### Processamento de Dados

1. **Download:** Dados históricos de 5 anos via yfinance
2. **Filtragem:** Mantém apenas a coluna 'Close' (preço de fechamento)
3. **Normalização:** MinMaxScaler (range 0-1)
4. **Sequências:** Cria janelas móveis de 60 dias para prever 15 dias
5. **Split:** 80% treino, 20% teste

---

## 📊 Retreinamento Automático

O modelo é automaticamente retreinado se:
- **Condição:** Passaram mais de 1 dia desde o último treinamento
- **Configuração:** Arquivo `last_run_date.txt` rastreia data do último treino
- **Acionamento:** Quando a API recebe uma requisição POST para `/predict`

### Arquivos de Controle

- `last_run_id.txt` - ID da última execução MLflow
- `last_run_date.txt` - Data do último treinamento
- `lstm_model_weights.pt2` - Pesos do modelo treinado
- `lstm_scaler.pkl` - Scaler para normalização dos dados

---

## 📁 Estrutura do Projeto

```
tech_challenge_4_IA/
├── api/
│   ├── __init__.py
│   ├── app.py                    # Aplicação Flask principal
│   ├── global_params.py          # Configurações globais
│   ├── mlartifacts/              # Artefatos MLflow
│   └── templates/
│       └── index.html            # Interface web
├── model/
│   ├── __init__.py
│   ├── lstm.py                   # Arquitetura LSTM
│   ├── dataset.py                # Gerenciador de dados
│   └── model_trainer.py          # Script de treinamento
├── Dockerfile                    # Configuração Docker
├── docker-compose.yml            # Orquestração Docker
├── requirements.txt              # Dependências Python
├── api_test.py                   # Script de teste da API
├── README.md                     # Este arquivo
└── mlartifacts/                  # Artefatos e modelos MLflow
```

---

## 🧪 Testando a API

### Teste Local (Python)
```bash
python api_test.py
```

Isso:
1. Envia uma requisição de previsão para a API
2. Exibe as previsões para os próximos 15 dias
3. Gera um gráfico com as previsões

### Teste com cURL
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA"}'
```

### Teste com Swagger UI
1. Acesse http://localhost:5000/apidocs
2. Clique em "Try it out" no endpoint `/predict`
3. Digite `PETR4.SA` no campo ticker
4. Clique "Execute"

---

## 🔍 Monitoramento com MLflow

### Acessar MLflow
1. Abra http://localhost:3050
2. Navegue até o experimento "Stock Prediction"
3. Visualize:
   - Parâmetros do modelo
   - Métricas de perda durante treinamento
   - Timestamp e duração da execução

### Logs Capturados

```
Parâmetros:
- Learning Rate
- Batch Size
- Epochs
- Hidden Size
- Number of Layers
- Dropout

Métricas:
- Loss (por epoch)
- System metrics (CPU, memória, etc.)
```

---

## 🐳 Variáveis de Ambiente (Docker)

```bash
FLASK_ENV=development
FLASK_DEBUG=1
MLFLOW_TRACKING_URI=http://mlflow:3050
PYTHONPATH=/tech_challenge_4_IA
```

Para uso local, configure:
```bash
export MLFLOW_TRACKING_URI=http://localhost:3050
```

---

## 🔧 Troubleshooting

### Erro: "Modelo não encontrado"
```
Solução: Execute o treinamento: python -m model.model_trainer
```

### Erro: "Conexão recusada em localhost:3050"
```
Solução: Certifique-se que MLflow está rodando:
- Docker: docker-compose up
- Local: mlflow server --host 0.0.0.0 --port 3050
```

### Erro: "CUDA não disponível"
```
O modelo automaticamente cai para CPU.
Verifique com: python -c "import torch; print(torch.cuda.is_available())"
```

### Erro: "Arquivo de dados não encontrado"
```
Solução: Verifique a conexão com internet (yfinance precisa baixar dados)
```

---

## 📈 Métricas de Desempenho

O modelo é avaliado usando:
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **RMSE** (Root Mean Squared Error)

Consulte MLflow para valores específicos das execuções.

---

## 📝 Exemplo de Uso Completo

```python
import requests
import json

# URL da API
API_URL = "http://localhost:5000/predict"

# Fazer previsão
response = requests.post(API_URL, json={"ticker": "PETR4.SA"})

if response.status_code == 200:
    data = response.json()
    print(f"Ticker: {data['ticker']}")
    print(f"Previsões (próximos 15 dias):")
    for i, price in enumerate(data['predictions'], 1):
        print(f"  Dia {i:02d}: R$ {price:.2f}")
else:
    print(f"Erro: {response.status_code}")
    print(response.text)
```

---

## 🚀 Próximas Melhorias

- [ ] Suporte para múltiplos tickers
- [ ] Previsões com intervalo de confiança
- [ ] Modelo ensemble (múltiplos LSTM)
- [ ] Validação cruzada
- [ ] Dashboard interativo (Grafana/Plotly)
- [ ] API assíncrona com FastAPI
- [ ] Testes unitários e integração
- [ ] CI/CD pipeline (GitHub Actions)

---

## 👥 Autores

**Tech Challenge 4 - PosTech ML Engineering - Fase 4 (Deep Learning)**

Desenvolvimento em: Maio de 2026

---

## 📄 Licença

Este projeto é parte do programa PosTech e está protegido sob suas políticas.

---

## 📞 Suporte

Para questões ou problemas:
1. Verifique a seção [Troubleshooting](#-troubleshooting)
2. Consulte os logs: `docker-compose logs -f api`
3. Acesse MLflow para análise de experimentos: http://localhost:3050

---

**Status do Projeto:** ✅ Em Produção | **Última Atualização:** Maio 2026
