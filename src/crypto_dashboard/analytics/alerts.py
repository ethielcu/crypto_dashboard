import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Union, Any
from datetime import datetime, timedelta
import json


class Alert:
    def __init__(self, alert_id: str, asset: str, condition: str, threshold: float,
                 message: str, active: bool = True, created_at: Optional[datetime] = None):
        self.alert_id = alert_id
        self.asset = asset
        self.condition = condition
        self.threshold = threshold
        self.message = message
        self.active = active
        self.created_at = created_at or datetime.now()
        self.triggered_at = None
        self.trigger_count = 0
    
    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'asset': self.asset,
            'condition': self.condition,
            'threshold': self.threshold,
            'message': self.message,
            'active': self.active,
            'created_at': self.created_at.isoformat(),
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'trigger_count': self.trigger_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Alert':
        alert = cls(
            alert_id=data['alert_id'],
            asset=data['asset'],
            condition=data['condition'],
            threshold=data['threshold'],
            message=data['message'],
            active=data['active'],
            created_at=datetime.fromisoformat(data['created_at'])
        )
        if data['triggered_at']:
            alert.triggered_at = datetime.fromisoformat(data['triggered_at'])
        alert.trigger_count = data['trigger_count']
        return alert


class AlertSystem:
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.triggered_alerts: List[Alert] = []
        self.conditions = {
            'price_above': self._price_above,
            'price_below': self._price_below,
            'price_change_percent': self._price_change_percent,
            'volume_above': self._volume_above,
            'volume_below': self._volume_below,
            'rsi_above': self._rsi_above,
            'rsi_below': self._rsi_below,
            'macd_bullish_crossover': self._macd_bullish_crossover,
            'macd_bearish_crossover': self._macd_bearish_crossover,
            'bollinger_upper_breach': self._bollinger_upper_breach,
            'bollinger_lower_breach': self._bollinger_lower_breach,
            'moving_average_crossover': self._moving_average_crossover
        }
    
    def create_alert(self, asset: str, condition: str, threshold: float,
                    message: Optional[str] = None) -> str:
        alert_id = f"{asset}_{condition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if message is None:
            message = f"{asset} {condition} {threshold}"
        
        alert = Alert(
            alert_id=alert_id,
            asset=asset,
            condition=condition,
            threshold=threshold,
            message=message
        )
        
        self.alerts[alert_id] = alert
        return alert_id
    
    def remove_alert(self, alert_id: str) -> bool:
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            return True
        return False
    
    def activate_alert(self, alert_id: str) -> bool:
        if alert_id in self.alerts:
            self.alerts[alert_id].active = True
            return True
        return False
    
    def deactivate_alert(self, alert_id: str) -> bool:
        if alert_id in self.alerts:
            self.alerts[alert_id].active = False
            return True
        return False
    
    def check_alerts(self, market_data: Dict[str, pd.DataFrame]) -> List[Alert]:
        triggered = []
        
        for alert in self.alerts.values():
            if not alert.active:
                continue
            
            if alert.asset not in market_data:
                continue
            
            data = market_data[alert.asset]
            if self._evaluate_condition(alert, data):
                alert.triggered_at = datetime.now()
                alert.trigger_count += 1
                triggered.append(alert)
                self.triggered_alerts.append(alert)
        
        return triggered
    
    def _evaluate_condition(self, alert: Alert, data: pd.DataFrame) -> bool:
        if alert.condition not in self.conditions:
            return False
        
        return self.conditions[alert.condition](data, alert.threshold)
    
    def _price_above(self, data: pd.DataFrame, threshold: float) -> bool:
        if 'close' not in data.columns or data.empty:
            return False
        return data['close'].iloc[-1] > threshold
    
    def _price_below(self, data: pd.DataFrame, threshold: float) -> bool:
        if 'close' not in data.columns or data.empty:
            return False
        return data['close'].iloc[-1] < threshold
    
    def _price_change_percent(self, data: pd.DataFrame, threshold: float) -> bool:
        if 'close' not in data.columns or len(data) < 2:
            return False
        
        current_price = data['close'].iloc[-1]
        previous_price = data['close'].iloc[-2]
        change_percent = ((current_price - previous_price) / previous_price) * 100
        
        return abs(change_percent) > threshold
    
    def _volume_above(self, data: pd.DataFrame, threshold: float) -> bool:
        if 'volume' not in data.columns or data.empty:
            return False
        return data['volume'].iloc[-1] > threshold
    
    def _volume_below(self, data: pd.DataFrame, threshold: float) -> bool:
        if 'volume' not in data.columns or data.empty:
            return False
        return data['volume'].iloc[-1] < threshold
    
    def _rsi_above(self, data: pd.DataFrame, threshold: float) -> bool:
        from .technical_indicators import TechnicalIndicators
        
        if 'close' not in data.columns or len(data) < 15:
            return False
        
        rsi = TechnicalIndicators.rsi(data['close'])
        return not pd.isna(rsi.iloc[-1]) and rsi.iloc[-1] > threshold
    
    def _rsi_below(self, data: pd.DataFrame, threshold: float) -> bool:
        from .technical_indicators import TechnicalIndicators
        
        if 'close' not in data.columns or len(data) < 15:
            return False
        
        rsi = TechnicalIndicators.rsi(data['close'])
        return not pd.isna(rsi.iloc[-1]) and rsi.iloc[-1] < threshold
    
    def _macd_bullish_crossover(self, data: pd.DataFrame, threshold: float) -> bool:
        from .technical_indicators import TechnicalIndicators
        
        if 'close' not in data.columns or len(data) < 35:
            return False
        
        macd_data = TechnicalIndicators.macd(data['close'])
        
        if len(macd_data['macd']) < 2:
            return False
        
        current_macd = macd_data['macd'].iloc[-1]
        current_signal = macd_data['signal'].iloc[-1]
        prev_macd = macd_data['macd'].iloc[-2]
        prev_signal = macd_data['signal'].iloc[-2]
        
        return (prev_macd <= prev_signal and current_macd > current_signal)
    
    def _macd_bearish_crossover(self, data: pd.DataFrame, threshold: float) -> bool:
        from .technical_indicators import TechnicalIndicators
        
        if 'close' not in data.columns or len(data) < 35:
            return False
        
        macd_data = TechnicalIndicators.macd(data['close'])
        
        if len(macd_data['macd']) < 2:
            return False
        
        current_macd = macd_data['macd'].iloc[-1]
        current_signal = macd_data['signal'].iloc[-1]
        prev_macd = macd_data['macd'].iloc[-2]
        prev_signal = macd_data['signal'].iloc[-2]
        
        return (prev_macd >= prev_signal and current_macd < current_signal)
    
    def _bollinger_upper_breach(self, data: pd.DataFrame, threshold: float) -> bool:
        from .technical_indicators import TechnicalIndicators
        
        if 'close' not in data.columns or len(data) < 21:
            return False
        
        bb_data = TechnicalIndicators.bollinger_bands(data['close'])
        current_price = data['close'].iloc[-1]
        upper_band = bb_data['upper'].iloc[-1]
        
        return current_price > upper_band
    
    def _bollinger_lower_breach(self, data: pd.DataFrame, threshold: float) -> bool:
        from .technical_indicators import TechnicalIndicators
        
        if 'close' not in data.columns or len(data) < 21:
            return False
        
        bb_data = TechnicalIndicators.bollinger_bands(data['close'])
        current_price = data['close'].iloc[-1]
        lower_band = bb_data['lower'].iloc[-1]
        
        return current_price < lower_band
    
    def _moving_average_crossover(self, data: pd.DataFrame, threshold: float) -> bool:
        from .technical_indicators import TechnicalIndicators
        
        if 'close' not in data.columns or len(data) < 51:
            return False
        
        short_ma = TechnicalIndicators.sma(data['close'], 20)
        long_ma = TechnicalIndicators.sma(data['close'], 50)
        
        if len(short_ma) < 2 or len(long_ma) < 2:
            return False
        
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]
        
        bullish_cross = prev_short <= prev_long and current_short > current_long
        bearish_cross = prev_short >= prev_long and current_short < current_long
        
        return bullish_cross or bearish_cross
    
    def get_active_alerts(self) -> List[Alert]:
        return [alert for alert in self.alerts.values() if alert.active]
    
    def get_triggered_alerts(self, hours: int = 24) -> List[Alert]:
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.triggered_alerts 
                if alert.triggered_at and alert.triggered_at > cutoff_time]
    
    def clear_triggered_alerts(self):
        self.triggered_alerts.clear()
    
    def save_alerts(self, filepath: str):
        alerts_data = {alert_id: alert.to_dict() 
                      for alert_id, alert in self.alerts.items()}
        
        with open(filepath, 'w') as f:
            json.dump(alerts_data, f, indent=2)
    
    def load_alerts(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                alerts_data = json.load(f)
            
            self.alerts = {alert_id: Alert.from_dict(data) 
                          for alert_id, data in alerts_data.items()}
        except FileNotFoundError:
            pass
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        total_alerts = len(self.alerts)
        active_alerts = len(self.get_active_alerts())
        triggered_today = len(self.get_triggered_alerts(24))
        
        condition_counts = {}
        for alert in self.alerts.values():
            condition_counts[alert.condition] = condition_counts.get(alert.condition, 0) + 1
        
        asset_counts = {}
        for alert in self.alerts.values():
            asset_counts[alert.asset] = asset_counts.get(alert.asset, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'triggered_today': triggered_today,
            'condition_distribution': condition_counts,
            'asset_distribution': asset_counts,
            'available_conditions': list(self.conditions.keys())
        }


class NotificationManager:
    def __init__(self):
        self.notification_methods = []
    
    def add_notification_method(self, method: Callable[[Alert], None]):
        self.notification_methods.append(method)
    
    def send_notifications(self, triggered_alerts: List[Alert]):
        for alert in triggered_alerts:
            for method in self.notification_methods:
                try:
                    method(alert)
                except Exception as e:
                    print(f"Failed to send notification for alert {alert.alert_id}: {e}")
    
    @staticmethod
    def console_notification(alert: Alert):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ALERT TRIGGERED: {alert.message}")
    
    @staticmethod
    def log_notification(alert: Alert, log_file: str = "alerts.log"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] ALERT: {alert.alert_id} - {alert.message}\n"
        
        with open(log_file, "a") as f:
            f.write(log_entry)