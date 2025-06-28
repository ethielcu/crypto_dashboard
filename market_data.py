import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time


class MarketDataFetcher:
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        
    def get_top_cryptos(self, limit: int = 50) -> pd.DataFrame:
        try:
            url = f"{self.base_url}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': False
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            df = df[['id', 'symbol', 'name', 'current_price', 'market_cap', 
                    'price_change_percentage_24h', 'total_volume', 'circulating_supply']]
            
            df.columns = ['ID', 'Symbol', 'Name', 'Price (USD)', 'Market Cap', 
                         '24h Change (%)', 'Volume (24h)', 'Circulating Supply']
            
            return df
            
        except Exception as e:
            print(f"Error fetching top cryptos: {e}")
            return pd.DataFrame()
    
    def get_price_history(self, coin_id: str, days: int = 30) -> pd.DataFrame:
        try:
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily' if days > 90 else 'hourly'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            if volumes:
                volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                volume_df.set_index('timestamp', inplace=True)
                df = df.join(volume_df)
            
            return df
            
        except Exception as e:
            print(f"Error fetching price history for {coin_id}: {e}")
            return pd.DataFrame()
    
    def get_fear_greed_index(self) -> Dict:
        try:
            url = "https://api.alternative.me/fng/"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('data'):
                latest = data['data'][0]
                return {
                    'value': int(latest['value']),
                    'classification': latest['value_classification'],
                    'timestamp': latest['timestamp']
                }
            
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
            
        return {'value': 50, 'classification': 'Neutral', 'timestamp': str(int(time.time()))}
    
    def get_global_market_data(self) -> Dict:
        try:
            url = f"{self.base_url}/global"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()['data']
            
            return {
                'total_market_cap': data.get('total_market_cap', {}).get('usd', 0),
                'total_volume': data.get('total_volume', {}).get('usd', 0),
                'market_cap_percentage': data.get('market_cap_percentage', {}),
                'active_cryptocurrencies': data.get('active_cryptocurrencies', 0),
                'markets': data.get('markets', 0)
            }
            
        except Exception as e:
            print(f"Error fetching global market data: {e}")
            return {}


def get_stock_data(symbol: str, period: str = "1mo") -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    fetcher = MarketDataFetcher()
    
    print("Testing market data fetcher...")
    
    top_cryptos = fetcher.get_top_cryptos(10)
    print(f"Top cryptos shape: {top_cryptos.shape}")
    
    btc_history = fetcher.get_price_history('bitcoin', 7)
    print(f"BTC history shape: {btc_history.shape}")
    
    fg_index = fetcher.get_fear_greed_index()
    print(f"Fear & Greed Index: {fg_index}")
    
    global_data = fetcher.get_global_market_data()
    print(f"Global market data keys: {list(global_data.keys())}")