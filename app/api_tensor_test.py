from datetime import date, timedelta
from main import device, lookback_period
from dataset import DatasetManager

ticker = 'MSFT'

lookback_date = str(date.today() - timedelta(days=lookback_period))

dataset = DatasetManager(ticker, start_date=lookback_date)
input_tensor = dataset.get_api_tensor(lookback=lookback_period, device=device)