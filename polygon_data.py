"""
Polygon API Data Fetcher
By Josephina Kim (JK)

Fetches real market data from Polygon API for testing the OU strategy.
Works with both equities and crypto.
"""

import requests
import numpy as np
from datetime import datetime, timedelta


class PolygonDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
    
    def get_stock_data(self, ticker, days=500, multiplier=1, timespan="day"):
        """
        Fetch stock/equity price data from Polygon API.
        
        Parameters:
        - ticker: Stock symbol (e.g., "AAPL", "MSFT", "TSLA")
        - days: Number of days of data to fetch
        - multiplier: Size of timespan (1 = 1 day, 1 hour, etc.)
        - timespan: "day", "hour", "minute"
        
        Returns:
        - prices: numpy array of closing prices
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") not in ["OK", "DELAYED"]:
            error_msg = data.get('error', data.get('statusMessage', 'Unknown error'))
            raise ValueError(f"API Error: {error_msg}")
        
        if "results" not in data or not data["results"]:
            raise ValueError(f"API Error: No data returned for {ticker}")
        
        prices = [result["c"] for result in data["results"]]  # 'c' = close price
        return np.array(prices)
    
    def get_crypto_data(self, ticker, days=500, multiplier=1, timespan="day"):
        """
        Fetch cryptocurrency price data from Polygon API.
        
        Parameters:
        - ticker: Crypto pair (e.g., "X:BTCUSD", "X:ETHUSD")
        - days: Number of days of data to fetch
        - multiplier: Size of timespan
        - timespan: "day", "hour", "minute"
        
        Returns:
        - prices: numpy array of closing prices
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") not in ["OK", "DELAYED"]:
            error_msg = data.get('error', data.get('statusMessage', 'Unknown error'))
            raise ValueError(f"API Error: {error_msg}")
        
        if "results" not in data or not data["results"]:
            raise ValueError(f"API Error: No data returned for {ticker}")
        
        prices = [result["c"] for result in data["results"]]  # 'c' = close price
        return np.array(prices)
    
    def get_spread_data(self, ticker1, ticker2, days=500, multiplier=1, timespan="day"):
        """
        Calculate spread between two assets (for pairs trading).
        
        Parameters:
        - ticker1: First asset (e.g., "AAPL")
        - ticker2: Second asset (e.g., "MSFT")
        - days: Number of days of data
        - multiplier: Size of timespan
        - timespan: "day", "hour", "minute"
        
        Returns:
        - spread: numpy array of price differences
        """
        prices1 = self.get_stock_data(ticker1, days, multiplier, timespan)
        prices2 = self.get_stock_data(ticker2, days, multiplier, timespan)
        
        min_len = min(len(prices1), len(prices2))
        spread = prices1[:min_len] - prices2[:min_len]
        
        return spread


def test_real_data():
    """
    Test function to fetch real market data and run OU strategy.
    """
    api_key = "phX3UGTSILWy8uHUdxQauDRZF578YwRL"
    fetcher = PolygonDataFetcher(api_key)
    
    print("="*70)
    print("TESTING WITH REAL MARKET DATA")
    print("="*70)
    print()
    
    # Example: Test with Apple stock
    print("Fetching AAPL stock data...")
    try:
        prices = fetcher.get_stock_data("AAPL", days=500)
        print(f"✓ Fetched {len(prices)} data points")
        print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print()
        
        # Run OU strategy on real data
        from trading_strategy import MeanReversionStrategy
        from ou_estimator import OUEstimator
        
        print("Estimating OU parameters from real data...")
        strategy = MeanReversionStrategy(threshold_sigma=2.0)
        strategy.fit(prices, dt=1.0)
        
        estimated_params = strategy.get_parameters()
        print(f"  Estimated θ: {estimated_params['theta']:.4f}")
        print(f"  Estimated μ: ${estimated_params['mu']:.2f}")
        print(f"  Estimated σ: ${estimated_params['sigma']:.2f}")
        print()
        
        # Generate current signal
        current_price = prices[-1]
        signal, signal_type = strategy.generate_signal(current_price)
        deviation = current_price - estimated_params['mu']
        print(f"Current price: ${current_price:.2f}")
        print(f"Deviation from mean: ${deviation:+.2f}")
        print(f"Signal: {signal_type}")
        print()
        
        return prices, strategy
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None


if __name__ == "__main__":
    test_real_data()

