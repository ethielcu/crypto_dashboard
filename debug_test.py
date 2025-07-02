#!/usr/bin/env python3

import sys
sys.path.append('src')

from crypto_dashboard.data.market_data import MarketDataFetcher

def test_fetcher():
    fetcher = MarketDataFetcher()
    
    print("Testing top cryptos...")
    top_cryptos = fetcher.get_top_cryptos(10)
    print(f"Top cryptos shape: {top_cryptos.shape}")
    if not top_cryptos.empty:
        print("First 3 coins:")
        print(top_cryptos[['ID', 'Name']].head(3))
        
        print("\nTesting price history for each...")
        for _, row in top_cryptos.head(3).iterrows():
            coin_id = row['ID']
            coin_name = row['Name']
            print(f"\nTesting {coin_name} (ID: {coin_id})")
            
            price_data = fetcher.get_price_history(coin_id, 7)
            print(f"  Result shape: {price_data.shape}")
            print(f"  Empty: {price_data.empty}")
            
            if not price_data.empty:
                print(f"  Price range: ${price_data['price'].min():.2f} - ${price_data['price'].max():.2f}")
            else:
                print(f"  ERROR: No data returned for {coin_id}")

if __name__ == "__main__":
    test_fetcher()