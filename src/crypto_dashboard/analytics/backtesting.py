import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


class Strategy:
    def __init__(self, name: str):
        self.name = name
        self.signals = None
        self.parameters = {}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Subclasses must implement generate_signals method")


class SimpleMovingAverageStrategy(Strategy):
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__("Simple Moving Average Crossover")
        self.short_window = short_window
        self.long_window = long_window
        self.parameters = {'short_window': short_window, 'long_window': long_window}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()
        
        signals[short_ma > long_ma] = 1
        signals[short_ma < long_ma] = -1
        
        return signals


class RSIStrategy(Strategy):
    def __init__(self, rsi_window: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSI Mean Reversion")
        self.rsi_window = rsi_window
        self.oversold = oversold
        self.overbought = overbought
        self.parameters = {
            'rsi_window': rsi_window,
            'oversold': oversold,
            'overbought': overbought
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        from .technical_indicators import TechnicalIndicators
        
        signals = pd.Series(0, index=data.index)
        rsi = TechnicalIndicators.rsi(data['close'], self.rsi_window)
        
        signals[rsi < self.oversold] = 1
        signals[rsi > self.overbought] = -1
        
        return signals


class MACDStrategy(Strategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD Crossover")
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.parameters = {'fast': fast, 'slow': slow, 'signal': signal}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        from .technical_indicators import TechnicalIndicators
        
        signals = pd.Series(0, index=data.index)
        macd_data = TechnicalIndicators.macd(data['close'], self.fast, self.slow, self.signal)
        
        signals[macd_data['macd'] > macd_data['signal']] = 1
        signals[macd_data['macd'] < macd_data['signal']] = -1
        
        return signals


class BacktestResult:
    def __init__(self, strategy_name: str, returns: pd.Series, positions: pd.Series,
                 trades: List[Dict], metrics: Dict[str, float]):
        self.strategy_name = strategy_name
        self.returns = returns
        self.positions = positions
        self.trades = trades
        self.metrics = metrics


class BacktestEngine:
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
    
    def run_backtest(self, data: pd.DataFrame, strategy: Strategy,
                    start_date: Optional[str] = None, end_date: Optional[str] = None) -> BacktestResult:
        
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        signals = strategy.generate_signals(data)
        positions = self._calculate_positions(signals)
        returns, trades = self._calculate_returns(data, positions)
        metrics = self._calculate_metrics(returns, data, trades)
        
        result = BacktestResult(
            strategy_name=strategy.name,
            returns=returns,
            positions=positions,
            trades=trades,
            metrics=metrics
        )
        
        self.results[strategy.name] = result
        return result
    
    def _calculate_positions(self, signals: pd.Series) -> pd.Series:
        positions = pd.Series(0, index=signals.index)
        current_position = 0
        
        for i in range(len(signals)):
            if signals.iloc[i] != 0:
                current_position = signals.iloc[i]
            positions.iloc[i] = current_position
        
        return positions
    
    def _calculate_returns(self, data: pd.DataFrame, positions: pd.Series) -> Tuple[pd.Series, List[Dict]]:
        price_changes = data['close'].pct_change()
        position_changes = positions.diff()
        
        portfolio_returns = positions.shift(1) * price_changes
        commission_costs = abs(position_changes) * self.commission
        net_returns = portfolio_returns - commission_costs
        
        trades = self._extract_trades(data, positions)
        
        return net_returns.fillna(0), trades
    
    def _extract_trades(self, data: pd.DataFrame, positions: pd.Series) -> List[Dict]:
        trades = []
        position_changes = positions.diff()
        
        for i in range(1, len(positions)):
            if position_changes.iloc[i] != 0:
                if positions.iloc[i-1] == 0:
                    entry_price = data['close'].iloc[i]
                    entry_date = data.index[i]
                    trade_type = 'long' if positions.iloc[i] > 0 else 'short'
                    
                    exit_idx = None
                    for j in range(i+1, len(positions)):
                        if positions.iloc[j] == 0 or np.sign(positions.iloc[j]) != np.sign(positions.iloc[i]):
                            exit_idx = j
                            break
                    
                    if exit_idx:
                        exit_price = data['close'].iloc[exit_idx]
                        exit_date = data.index[exit_idx]
                        
                        if trade_type == 'long':
                            pnl = (exit_price - entry_price) / entry_price
                        else:
                            pnl = (entry_price - exit_price) / entry_price
                        
                        pnl -= 2 * self.commission
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'type': trade_type,
                            'pnl': pnl,
                            'duration': (exit_date - entry_date).days
                        })
        
        return trades
    
    def _calculate_metrics(self, returns: pd.Series, data: pd.DataFrame, trades: List[Dict]) -> Dict[str, float]:
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        buy_hold_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'avg_trade_duration': np.mean([t['duration'] for t in trades]) if trades else 0,
            'buy_hold_return': buy_hold_return
        }
    
    def optimize_strategy(self, data: pd.DataFrame, strategy_class: type,
                         param_grid: Dict[str, List], metric: str = 'sharpe_ratio') -> Dict:
        
        best_params = None
        best_score = float('-inf')
        results = []
        
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations:
            strategy = strategy_class(**params)
            result = self.run_backtest(data, strategy)
            
            score = result.metrics[metric]
            results.append({
                'parameters': params,
                'metrics': result.metrics,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        from itertools import product
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = []
        
        for combination in product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def compare_strategies(self, data: pd.DataFrame, strategies: List[Strategy]) -> pd.DataFrame:
        comparison_data = []
        
        for strategy in strategies:
            result = self.run_backtest(data, strategy)
            comparison_data.append({
                'Strategy': strategy.name,
                **result.metrics
            })
        
        return pd.DataFrame(comparison_data)
    
    def create_performance_chart(self, result: BacktestResult, data: pd.DataFrame) -> go.Figure:
        cumulative_returns = (1 + result.returns).cumprod()
        buy_hold = data['close'] / data['close'].iloc[0]
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=['Cumulative Returns', 'Positions', 'Drawdown'],
            row_width=[0.4, 0.2, 0.4]
        )
        
        fig.add_trace(
            go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values,
                      name=result.strategy_name, line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=buy_hold.index, y=buy_hold.values,
                      name='Buy & Hold', line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=result.positions.index, y=result.positions.values,
                      name='Position', mode='lines', line=dict(color='green')),
            row=2, col=1
        )
        
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                      name='Drawdown', fill='tonexty', line=dict(color='red')),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f"Backtest Results: {result.strategy_name}",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_metrics_table(self, results: List[BacktestResult]) -> go.Figure:
        metrics_data = []
        
        for result in results:
            metrics_data.append([
                result.strategy_name,
                f"{result.metrics['total_return']:.2%}",
                f"{result.metrics['annual_return']:.2%}",
                f"{result.metrics['annual_volatility']:.2%}",
                f"{result.metrics['sharpe_ratio']:.2f}",
                f"{result.metrics['max_drawdown']:.2%}",
                f"{result.metrics['win_rate']:.2%}",
                f"{result.metrics['total_trades']:.0f}"
            ])
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Strategy', 'Total Return', 'Annual Return', 'Volatility',
                       'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Total Trades'],
                fill_color='lightblue',
                align='center'
            ),
            cells=dict(
                values=list(zip(*metrics_data)),
                fill_color='white',
                align='center'
            )
        )])
        
        fig.update_layout(title="Strategy Comparison Metrics")
        return fig