import requests
import pandas as pd
from binance_historical_data import BinanceDataDumper
import datetime
start_date = datetime.date(year=2023, month=1, day=1)
end_date = datetime.date(year=2023, month=10, day=10)
data_dumper = BinanceDataDumper(
    path_dir_where_to_dump=".",
    asset_class="spot",  # spot, um, cm
    data_type="klines",  # aggTrades, klines, trades
    data_frequency="15m",
)
data_dumper.dump_data(
    tickers=['SOLUSDT'],
    date_start=start_date,
    date_end=end_date,
    is_to_update_existing=False,
    tickers_to_exclude=[],
)
import os
columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
df = pd.DataFrame(columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
## Merging
month_path = './spot/monthly/klines/SOLUSDT/15m'
daily_path = './spot/daily/klines/SOLUSDT/15m'
for file in os.listdir(month_path):
    df1 = pd.read_csv(os.path.join(month_path, file))
    df1.columns = columns
    df = pd.concat([df1, df], axis=0, ignore_index=True)
for file in os.listdir(daily_path):
    df1 = pd.read_csv(os.path.join(daily_path, file))
    df1.columns = columns
    df = pd.concat([df1, df], axis=0, ignore_index=True)
df.to_csv('data1.csv')
'''
# Set the API endpoint and parameters
url = 'https://api.binance.com/api/v3/klines?symbol=' + \
            'SOLUSDT' + '&interval=' + '15m' + '&limit='  + str(5000) + '&startTime=' + '2023-08-01'
params = {
    "symbol": "SOLUSDT",
    "interval": "15min",
    "startTime": "2023-01-01",
    "endTime": "2023-10-10"
}

# Make a GET request to the API endpoint
df2 = pd.read_json(url)

# Convert the response to a Pandas DataFrame
df = pd.DataFrame(response.json(), columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignored"])

# Convert the timestamp to a readable date format
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# Set the timestamp as the DataFrame index
df.set_index("timestamp", inplace=True)

df.to_csv('data1.csv')
# Print the DataFrame
print(df)
'''