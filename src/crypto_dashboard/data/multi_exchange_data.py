import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Union, Tuple
import time
from datetime import datetime, timedelta
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed


class ExchangeAPI:
    def __init__(self, name: str, base_url: str, rate_limit: float = 1.0):
        self.name = name
        self.base_url = base_url.rstrip('/')
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = requests.Session()
    
    def _rate_limit_wait(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        self._rate_limit_wait()
        
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching from {self.name}: {e}")
            return None
    
    def get_price(self, symbol: str) -> Optional[float]:
        raise NotImplementedError
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        raise NotImplementedError
    
    def get_order_book(self, symbol: str, depth: int = 100) -> Optional[Dict]:
        raise NotImplementedError


class CoinGeckoAPI(ExchangeAPI):
    def __init__(self):
        super().__init__("CoinGecko", "https://api.coingecko.com/api/v3", 1.0)
    
    def get_price(self, symbol: str) -> Optional[float]:
        params = {'ids': symbol.lower(), 'vs_currencies': 'usd'}
        data = self._make_request('simple/price', params)
        
        if data and symbol.lower() in data:
            return data[symbol.lower()].get('usd')
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        params = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}
        data = self._make_request(f'coins/{symbol.lower()}/market_chart', params)
        
        if data and 'prices' in data:
            prices = data['prices']
            volumes = data.get('total_volumes', [])
            
            df_data = []
            for i, (timestamp, price) in enumerate(prices):
                volume = volumes[i][1] if i < len(volumes) else 0
                df_data.append({
                    'timestamp': pd.to_datetime(timestamp, unit='ms'),
                    'close': price,
                    'volume': volume
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            return df
        
        return None


class BinanceAPI(ExchangeAPI):
    def __init__(self):
        super().__init__("Binance", "https://api.binance.com/api/v3", 0.1)
    
    def _convert_symbol(self, symbol: str) -> str:
        symbol_map = {
            'bitcoin': 'BTCUSDT',
            'ethereum': 'ETHUSDT',
            'binancecoin': 'BNBUSDT',
            'cardano': 'ADAUSDT',
            'solana': 'SOLUSDT',
            'polkadot': 'DOTUSDT',
            'dogecoin': 'DOGEUSDT',
            'avalanche-2': 'AVAXUSDT',
            'chainlink': 'LINKUSDT',
            'polygon': 'MATICUSDT'
        }
        return symbol_map.get(symbol.lower(), f"{symbol.upper()}USDT")
    
    def get_price(self, symbol: str) -> Optional[float]:
        binance_symbol = self._convert_symbol(symbol)
        params = {'symbol': binance_symbol}
        data = self._make_request('ticker/price', params)
        
        if data and 'price' in data:
            return float(data['price'])
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        binance_symbol = self._convert_symbol(symbol)
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        params = {
            'symbol': binance_symbol,
            'interval': '1d',
            'startTime': start_time,
            'endTime': end_time,
            'limit': days
        }
        
        data = self._make_request('klines', params)
        
        if data:
            df_data = []
            for kline in data:
                df_data.append({
                    'timestamp': pd.to_datetime(int(kline[0]), unit='ms'),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            return df
        
        return None
    
    def get_order_book(self, symbol: str, depth: int = 100) -> Optional[Dict]:
        binance_symbol = self._convert_symbol(symbol)
        params = {'symbol': binance_symbol, 'limit': min(depth, 5000)}
        data = self._make_request('depth', params)
        
        if data:
            return {
                'bids': [(float(price), float(qty)) for price, qty in data['bids']],
                'asks': [(float(price), float(qty)) for price, qty in data['asks']],
                'timestamp': datetime.now()
            }
        return None


class CoinbaseAPI(ExchangeAPI):
    def __init__(self):
        super().__init__("Coinbase", "https://api.exchange.coinbase.com", 0.1)
    
    def _convert_symbol(self, symbol: str) -> str:
        symbol_map = {
            'bitcoin': 'BTC-USD',
            'ethereum': 'ETH-USD',
            'cardano': 'ADA-USD',
            'solana': 'SOL-USD',
            'polkadot': 'DOT-USD',
            'dogecoin': 'DOGE-USD',
            'avalanche-2': 'AVAX-USD',
            'chainlink': 'LINK-USD',
            'polygon': 'MATIC-USD'
        }
        return symbol_map.get(symbol.lower(), f"{symbol.upper()}-USD")
    
    def get_price(self, symbol: str) -> Optional[float]:
        coinbase_symbol = self._convert_symbol(symbol)
        data = self._make_request(f'products/{coinbase_symbol}/ticker')
        
        if data and 'price' in data:
            return float(data['price'])
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        coinbase_symbol = self._convert_symbol(symbol)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        params = {
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'granularity': 86400
        }
        
        data = self._make_request(f'products/{coinbase_symbol}/candles', params)
        
        if data:
            df_data = []
            for candle in reversed(data):
                df_data.append({
                    'timestamp': pd.to_datetime(candle[0], unit='s'),
                    'low': float(candle[1]),
                    'high': float(candle[2]),
                    'open': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            return df
        
        return None


class KrakenAPI(ExchangeAPI):
    def __init__(self):
        super().__init__("Kraken", "https://api.kraken.com/0/public", 0.5)
    
    def _convert_symbol(self, symbol: str) -> str:
        symbol_map = {
            'bitcoin': 'XBTUSD',
            'ethereum': 'ETHUSD',
            'cardano': 'ADAUSD',
            'solana': 'SOLUSD',
            'polkadot': 'DOTUSD',
            'dogecoin': 'XDGUSD',
            'avalanche-2': 'AVAXUSD',
            'chainlink': 'LINKUSD'
        }
        return symbol_map.get(symbol.lower(), f"{symbol.upper()}USD")
    
    def get_price(self, symbol: str) -> Optional[float]:
        kraken_symbol = self._convert_symbol(symbol)
        params = {'pair': kraken_symbol}
        data = self._make_request('Ticker', params)
        
        if data and 'result' in data:
            for pair_data in data['result'].values():
                if 'c' in pair_data:
                    return float(pair_data['c'][0])
        return None


class MultiExchangeDataManager:
    def __init__(self, db_path: str = "crypto_data.db"):
        self.exchanges = {
            'coingecko': CoinGeckoAPI(),
            'binance': BinanceAPI(),
            'coinbase': CoinbaseAPI(),
            'kraken': KrakenAPI()
        }
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL NOT NULL,
                    volume REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(exchange, symbol, timestamp)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS current_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(exchange, symbol)
                )
            ''')
            
            conn.commit()
    
    def get_multi_exchange_prices(self, symbols: List[str]) -> pd.DataFrame:
        results = []
        
        with ThreadPoolExecutor(max_workers=len(self.exchanges)) as executor:
            futures = []
            
            for exchange_name, exchange_api in self.exchanges.items():
                for symbol in symbols:
                    future = executor.submit(self._get_price_with_exchange, 
                                           exchange_name, exchange_api, symbol)
                    futures.append(future)
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        if results:
            df = pd.DataFrame(results)
            pivot_df = df.pivot(index='symbol', columns='exchange', values='price')
            return pivot_df
        
        return pd.DataFrame()
    
    def _get_price_with_exchange(self, exchange_name: str, exchange_api: ExchangeAPI, 
                               symbol: str) -> Optional[Dict]:
        try:
            price = exchange_api.get_price(symbol)
            if price:
                self._store_current_price(exchange_name, symbol, price)
                return {
                    'exchange': exchange_name,
                    'symbol': symbol,
                    'price': price,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            print(f"Error getting price for {symbol} from {exchange_name}: {e}")
        
        return None
    
    def _store_current_price(self, exchange: str, symbol: str, price: float):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO current_prices 
                (exchange, symbol, price, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (exchange, symbol, price, datetime.now()))
            conn.commit()
    
    def get_historical_data_multi_exchange(self, symbol: str, days: int = 30) -> Dict[str, pd.DataFrame]:
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self.exchanges)) as executor:
            futures = {
                executor.submit(exchange_api.get_historical_data, symbol, days): exchange_name
                for exchange_name, exchange_api in self.exchanges.items()
            }
            
            for future in as_completed(futures):
                exchange_name = futures[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[exchange_name] = data
                        self._store_historical_data(exchange_name, symbol, data)
                except Exception as e:
                    print(f"Error getting historical data for {symbol} from {exchange_name}: {e}")
        
        return results
    
    def _store_historical_data(self, exchange: str, symbol: str, data: pd.DataFrame):
        with sqlite3.connect(self.db_path) as conn:
            for timestamp, row in data.iterrows():
                try:
                    conn.execute('''
                        INSERT OR REPLACE INTO price_data 
                        (exchange, symbol, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        exchange, symbol, timestamp,
                        row.get('open'), row.get('high'), row.get('low'),
                        row['close'], row.get('volume')
                    ))
                except Exception as e:
                    print(f"Error storing data point: {e}")
            
            conn.commit()
    
    def get_price_differences(self, symbols: List[str]) -> pd.DataFrame:
        prices_df = self.get_multi_exchange_prices(symbols)
        
        if prices_df.empty:
            return pd.DataFrame()
        
        differences = pd.DataFrame()
        
        for symbol in prices_df.index:
            symbol_prices = prices_df.loc[symbol].dropna()
            
            if len(symbol_prices) > 1:
                max_price = symbol_prices.max()
                min_price = symbol_prices.min()
                
                differences.loc[symbol, 'max_price'] = max_price
                differences.loc[symbol, 'min_price'] = min_price
                differences.loc[symbol, 'price_diff'] = max_price - min_price
                differences.loc[symbol, 'price_diff_pct'] = ((max_price - min_price) / min_price) * 100
                differences.loc[symbol, 'exchanges_count'] = len(symbol_prices)
        
        return differences.sort_values('price_diff_pct', ascending=False)
    
    def get_arbitrage_opportunities(self, symbols: List[str], min_diff_pct: float = 0.5) -> pd.DataFrame:
        differences = self.get_price_differences(symbols)
        
        opportunities = differences[differences['price_diff_pct'] >= min_diff_pct].copy()
        
        if not opportunities.empty:
            prices_df = self.get_multi_exchange_prices(symbols)
            
            for symbol in opportunities.index:
                symbol_prices = prices_df.loc[symbol].dropna()
                
                buy_exchange = symbol_prices.idxmin()
                sell_exchange = symbol_prices.idxmax()
                
                opportunities.loc[symbol, 'buy_exchange'] = buy_exchange
                opportunities.loc[symbol, 'sell_exchange'] = sell_exchange
                opportunities.loc[symbol, 'buy_price'] = symbol_prices.min()
                opportunities.loc[symbol, 'sell_price'] = symbol_prices.max()
        
        return opportunities
    
    def get_exchange_reliability_stats(self, days: int = 7) -> pd.DataFrame:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT exchange, 
                       COUNT(*) as total_requests,
                       COUNT(DISTINCT symbol) as symbols_covered,
                       AVG(CASE WHEN price IS NOT NULL THEN 1 ELSE 0 END) as success_rate
                FROM current_prices 
                WHERE created_at >= ?
                GROUP BY exchange
            '''
            
            df = pd.read_sql_query(query, conn, params=(cutoff_date,))
            return df
    
    def export_data(self, filepath: str, symbols: List[str], days: int = 30, 
                   format: str = 'csv'):
        all_data = []
        
        for symbol in symbols:
            historical_data = self.get_historical_data_multi_exchange(symbol, days)
            
            for exchange, data in historical_data.items():
                for timestamp, row in data.iterrows():
                    all_data.append({
                        'symbol': symbol,
                        'exchange': exchange,
                        'timestamp': timestamp,
                        'open': row.get('open'),
                        'high': row.get('high'),
                        'low': row.get('low'),
                        'close': row['close'],
                        'volume': row.get('volume')
                    })
        
        df = pd.DataFrame(all_data)
        
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False)
        elif format.lower() == 'json':
            df.to_json(filepath, orient='records', date_format='iso')
        elif format.lower() == 'parquet':
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError("Unsupported format. Use 'csv', 'json', or 'parquet'")
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM price_data WHERE created_at < ?', (cutoff_date,))
            cursor.execute('DELETE FROM current_prices WHERE created_at < ?', (cutoff_date,))
            
            conn.commit()
            
            deleted_price_data = cursor.rowcount
            print(f"Cleaned up {deleted_price_data} old records")