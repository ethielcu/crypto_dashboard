import unittest
import pandas as pd
import numpy as np
from src.crypto_dashboard.risk.risk_calculator import RiskCalculator


class TestRiskCalculator(unittest.TestCase):
    
    def setUp(self):
        self.calculator = RiskCalculator()
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        self.prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
        self.returns = self.calculator.calculate_returns(self.prices)
    
    def test_calculate_returns(self):
        returns = self.calculator.calculate_returns(self.prices)
        self.assertEqual(len(returns), len(self.prices) - 1)
        self.assertFalse(returns.isna().any())
    
    def test_calculate_volatility(self):
        vol = self.calculator.calculate_volatility(self.returns)
        self.assertIsInstance(vol, float)
        self.assertGreater(vol, 0)
    
    def test_calculate_sharpe_ratio(self):
        sharpe = self.calculator.calculate_sharpe_ratio(self.returns)
        self.assertIsInstance(sharpe, float)
    
    def test_calculate_var(self):
        var = self.calculator.calculate_var(self.returns)
        self.assertIsInstance(var, float)
        self.assertLess(var, 0)
    
    def test_position_sizing(self):
        position = self.calculator.calculate_position_sizing(10000, 2, 50, 45)
        self.assertIn('position_size', position)
        self.assertIn('shares', position)
        self.assertIn('risk_amount', position)
        self.assertGreater(position['position_size'], 0)


if __name__ == '__main__':
    unittest.main()