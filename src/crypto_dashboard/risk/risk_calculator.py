import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import math


class RiskCalculator:
    
    def __init__(self):
        self.risk_free_rate = 0.02
        
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        return prices.pct_change().dropna()
    
    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(365)
        return vol
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> float:
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        excess_returns = returns.mean() * 365 - risk_free_rate
        volatility = self.calculate_volatility(returns)
        
        if volatility == 0:
            return 0
        
        return excess_returns / volatility
    
    def calculate_max_drawdown(self, prices: pd.Series) -> Dict[str, float]:
        rolling_max = prices.expanding().max()
        
        drawdown = (prices - rolling_max) / rolling_max
        
        max_drawdown = drawdown.min()
        
        max_dd_date = drawdown.idxmin()
        peak_date = rolling_max.loc[:max_dd_date].idxmax()
        
        return {
            'max_drawdown': abs(max_drawdown),
            'max_drawdown_pct': abs(max_drawdown) * 100,
            'peak_date': peak_date,
            'trough_date': max_dd_date,
            'current_drawdown': abs(drawdown.iloc[-1]) * 100 if len(drawdown) > 0 else 0
        }
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        if len(returns) == 0:
            return 0
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
            return 1.0
            
        aligned_data = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_data) < 2:
            return 1.0
            
        covariance = aligned_data['asset'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()
        
        if market_variance == 0:
            return 1.0
            
        return covariance / market_variance
    
    def calculate_portfolio_metrics(self, portfolio_data: Dict[str, pd.Series], 
                                  weights: Dict[str, float]) -> Dict[str, float]:
        if not portfolio_data or not weights:
            return {}
        
        total_weight = sum(weights.values())
        if total_weight != 1.0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        portfolio_returns = pd.Series(0, index=list(portfolio_data.values())[0].index)
        
        for asset, returns in portfolio_data.items():
            if asset in weights:
                portfolio_returns += returns * weights[asset]
        
        metrics = {
            'portfolio_volatility': self.calculate_volatility(portfolio_returns),
            'portfolio_sharpe': self.calculate_sharpe_ratio(portfolio_returns),
            'portfolio_var_5': abs(self.calculate_var(portfolio_returns, 0.05)),
            'portfolio_cvar_5': abs(self.calculate_cvar(portfolio_returns, 0.05)),
            'portfolio_return_annual': portfolio_returns.mean() * 365,
            'portfolio_return_total': (1 + portfolio_returns).prod() - 1
        }
        
        return metrics
    
    def calculate_position_sizing(self, portfolio_value: float, risk_per_trade: float,
                                entry_price: float, stop_loss_price: float) -> Dict[str, float]:
        if entry_price <= 0 or stop_loss_price <= 0:
            return {'position_size': 0, 'shares': 0, 'risk_amount': 0}
        
        risk_per_share = abs(entry_price - stop_loss_price)
        
        max_risk_amount = portfolio_value * (risk_per_trade / 100)
        
        if risk_per_share > 0:
            shares = max_risk_amount / risk_per_share
            position_size = shares * entry_price
        else:
            shares = 0
            position_size = 0
        
        return {
            'position_size': position_size,
            'shares': shares,
            'risk_amount': max_risk_amount,
            'risk_per_share': risk_per_share,
            'position_pct': (position_size / portfolio_value) * 100 if portfolio_value > 0 else 0
        }
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0
        
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        return min(max(kelly_fraction, 0), 0.25)
    
    def calculate_correlation_matrix(self, returns_data: Dict[str, pd.Series]) -> pd.DataFrame:
        if not returns_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(returns_data)
        return df.corr()
    
    def monte_carlo_simulation(self, initial_price: float, returns: pd.Series, 
                              days: int = 30, simulations: int = 1000) -> Dict[str, float]:
        if len(returns) == 0:
            return {}
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        random_returns = np.random.normal(mean_return, std_return, (simulations, days))
        
        price_paths = np.zeros((simulations, days + 1))
        price_paths[:, 0] = initial_price
        
        for i in range(days):
            price_paths[:, i + 1] = price_paths[:, i] * (1 + random_returns[:, i])
        
        final_prices = price_paths[:, -1]
        
        return {
            'mean_final_price': np.mean(final_prices),
            'median_final_price': np.median(final_prices),
            'std_final_price': np.std(final_prices),
            'percentile_5': np.percentile(final_prices, 5),
            'percentile_95': np.percentile(final_prices, 95),
            'probability_profit': np.sum(final_prices > initial_price) / simulations * 100,
            'max_simulated_price': np.max(final_prices),
            'min_simulated_price': np.min(final_prices)
        }
    
    def calculate_risk_adjusted_returns(self, returns: pd.Series) -> Dict[str, float]:
        if len(returns) == 0:
            return {}
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 else 0
        
        annual_return = returns.mean() * 365
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        prices = (1 + returns).cumprod()
        max_dd_info = self.calculate_max_drawdown(prices)
        calmar_ratio = annual_return / max_dd_info['max_drawdown'] if max_dd_info['max_drawdown'] > 0 else 0
        
        return {
            'annual_return': annual_return,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'downside_deviation': downside_std,
            'upside_capture': len(returns[returns > 0]) / len(returns) * 100 if len(returns) > 0 else 0
        }


class PortfolioOptimizer:
    
    def __init__(self):
        self.risk_calculator = RiskCalculator()
    
    def calculate_efficient_frontier_point(self, returns_data: Dict[str, pd.Series], 
                                         target_return: float) -> Dict[str, float]:
        
        n_assets = len(returns_data)
        if n_assets == 0:
            return {}
        
        weights = {asset: 1/n_assets for asset in returns_data.keys()}
        
        portfolio_metrics = self.risk_calculator.calculate_portfolio_metrics(returns_data, weights)
        
        return {
            'weights': weights,
            'expected_return': portfolio_metrics.get('portfolio_return_annual', 0),
            'volatility': portfolio_metrics.get('portfolio_volatility', 0),
            'sharpe_ratio': portfolio_metrics.get('portfolio_sharpe', 0)
        }


if __name__ == "__main__":
    calculator = RiskCalculator()
    
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
    returns = calculator.calculate_returns(prices)
    
    print("Testing Risk Calculator...")
    
    vol = calculator.calculate_volatility(returns)
    sharpe = calculator.calculate_sharpe_ratio(returns)
    max_dd = calculator.calculate_max_drawdown(prices)
    var = calculator.calculate_var(returns)
    
    print(f"Volatility: {vol:.4f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {max_dd['max_drawdown_pct']:.2f}%")
    print(f"VaR (5%): {var:.4f}")
    
    position = calculator.calculate_position_sizing(10000, 2, 50, 45)
    print(f"Position size: ${position['position_size']:.2f}")
    
    print("Risk calculator tests completed successfully!")