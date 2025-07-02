import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px


class SentimentAnalyzer:
    def __init__(self):
        self.fear_greed_cache = None
        self.cache_timestamp = None
        self.cache_duration = 3600
    
    def get_fear_greed_index(self, use_cache: bool = True) -> Dict:
        if (use_cache and self.fear_greed_cache and 
            self.cache_timestamp and 
            time.time() - self.cache_timestamp < self.cache_duration):
            return self.fear_greed_cache
        
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                result = {
                    'value': int(data['data'][0]['value']),
                    'classification': data['data'][0]['value_classification'],
                    'timestamp': data['data'][0]['timestamp'],
                    'time_until_update': data['data'][0].get('time_until_update')
                }
                
                self.fear_greed_cache = result
                self.cache_timestamp = time.time()
                return result
            
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
        
        return {
            'value': 50,
            'classification': 'Neutral',
            'timestamp': str(int(time.time())),
            'time_until_update': None
        }
    
    def get_fear_greed_history(self, limit: int = 30) -> pd.DataFrame:
        try:
            url = f"https://api.alternative.me/fng/?limit={limit}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data:
                records = []
                for entry in data['data']:
                    records.append({
                        'date': pd.to_datetime(entry['timestamp'], unit='s'),
                        'value': int(entry['value']),
                        'classification': entry['value_classification']
                    })
                
                df = pd.DataFrame(records)
                df = df.sort_values('date').reset_index(drop=True)
                return df
            
        except Exception as e:
            print(f"Error fetching Fear & Greed history: {e}")
        
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='D')
        return pd.DataFrame({
            'date': dates,
            'value': np.random.randint(20, 80, limit),
            'classification': ['Neutral'] * limit
        })
    
    def analyze_sentiment_trend(self, history_df: pd.DataFrame) -> Dict:
        if history_df.empty:
            return {'trend': 'neutral', 'strength': 0, 'volatility': 0}
        
        values = history_df['value'].values
        
        if len(values) < 2:
            return {'trend': 'neutral', 'strength': 0, 'volatility': 0}
        
        recent_avg = np.mean(values[-7:]) if len(values) >= 7 else np.mean(values)
        older_avg = np.mean(values[:-7]) if len(values) >= 14 else np.mean(values)
        
        trend_strength = abs(recent_avg - older_avg)
        
        if recent_avg > older_avg + 5:
            trend = 'improving'
        elif recent_avg < older_avg - 5:
            trend = 'deteriorating'
        else:
            trend = 'neutral'
        
        volatility = np.std(values)
        
        return {
            'trend': trend,
            'strength': trend_strength,
            'volatility': volatility,
            'current_level': values[-1] if len(values) > 0 else 50,
            'recent_average': recent_avg,
            'overall_average': np.mean(values)
        }
    
    def get_sentiment_signals(self, current_value: int) -> List[str]:
        signals = []
        
        if current_value <= 25:
            signals.append("Extreme Fear - Potential buying opportunity")
        elif current_value <= 45:
            signals.append("Fear - Market may be oversold")
        elif current_value >= 75:
            signals.append("Extreme Greed - Potential selling opportunity")
        elif current_value >= 55:
            signals.append("Greed - Market may be overbought")
        else:
            signals.append("Neutral sentiment - No clear signal")
        
        return signals
    
    def create_fear_greed_gauge(self, current_value: int, classification: str) -> go.Figure:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = current_value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Fear & Greed Index<br><span style='font-size:0.8em;color:gray'>{classification}</span>"},
            delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 25], 'color': 'red'},
                    {'range': [25, 45], 'color': 'orange'},
                    {'range': [45, 55], 'color': 'yellow'},
                    {'range': [55, 75], 'color': 'lightgreen'},
                    {'range': [75, 100], 'color': 'green'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': current_value
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="white",
            font={'color': "darkblue", 'family': "Arial"},
            height=400
        )
        
        return fig
    
    def create_fear_greed_history_chart(self, history_df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        
        colors = []
        for value in history_df['value']:
            if value <= 25:
                colors.append('red')
            elif value <= 45:
                colors.append('orange')
            elif value <= 55:
                colors.append('yellow')
            elif value <= 75:
                colors.append('lightgreen')
            else:
                colors.append('green')
        
        fig.add_trace(go.Scatter(
            x=history_df['date'],
            y=history_df['value'],
            mode='lines+markers',
            name='Fear & Greed Index',
            line=dict(width=2, color='blue'),
            marker=dict(size=6, color=colors)
        ))
        
        fig.add_hline(y=25, line_dash="dash", line_color="red", opacity=0.7, 
                     annotation_text="Extreme Fear")
        fig.add_hline(y=75, line_dash="dash", line_color="green", opacity=0.7,
                     annotation_text="Extreme Greed")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5,
                     annotation_text="Neutral")
        
        fig.update_layout(
            title="Fear & Greed Index History",
            xaxis_title="Date",
            yaxis_title="Index Value",
            yaxis=dict(range=[0, 100]),
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def get_social_media_keywords(self) -> Dict[str, List[str]]:
        return {
            'bullish': [
                'moon', 'rocket', 'bull', 'bullish', 'pump', 'surge', 'rally',
                'breakout', 'ath', 'all time high', 'hodl', 'diamond hands',
                'buy the dip', 'accumulate', 'long', 'calls'
            ],
            'bearish': [
                'bear', 'bearish', 'dump', 'crash', 'correction', 'dip',
                'sell', 'short', 'puts', 'paper hands', 'capitulation',
                'blood bath', 'rekt', 'liquidation'
            ],
            'neutral': [
                'sideways', 'consolidation', 'range', 'support', 'resistance',
                'technical analysis', 'fundamental', 'hodl', 'waiting'
            ]
        }
    
    def simulate_social_sentiment(self, asset: str) -> Dict:
        np.random.seed(hash(asset + str(datetime.now().date())) % 2**32)
        
        base_sentiment = np.random.choice(['bullish', 'bearish', 'neutral'], 
                                        p=[0.4, 0.3, 0.3])
        
        volume = np.random.randint(100, 10000)
        
        if base_sentiment == 'bullish':
            score = np.random.uniform(0.6, 1.0)
        elif base_sentiment == 'bearish':
            score = np.random.uniform(0.0, 0.4)
        else:
            score = np.random.uniform(0.4, 0.6)
        
        engagement = np.random.randint(50, 500)
        
        return {
            'asset': asset,
            'sentiment': base_sentiment,
            'score': score,
            'volume': volume,
            'engagement': engagement,
            'timestamp': datetime.now()
        }
    
    def create_sentiment_summary_chart(self, sentiment_data: List[Dict]) -> go.Figure:
        if not sentiment_data:
            return go.Figure()
        
        df = pd.DataFrame(sentiment_data)
        
        fig = go.Figure()
        
        for sentiment in ['bullish', 'bearish', 'neutral']:
            sentiment_df = df[df['sentiment'] == sentiment]
            if not sentiment_df.empty:
                color = {'bullish': 'green', 'bearish': 'red', 'neutral': 'gray'}[sentiment]
                fig.add_trace(go.Bar(
                    x=sentiment_df['asset'],
                    y=sentiment_df['volume'],
                    name=sentiment.capitalize(),
                    marker_color=color,
                    opacity=0.7
                ))
        
        fig.update_layout(
            title="Social Media Sentiment Volume by Asset",
            xaxis_title="Cryptocurrency",
            yaxis_title="Discussion Volume",
            barmode='stack',
            showlegend=True
        )
        
        return fig
    
    def get_sentiment_insights(self, fear_greed_data: Dict, 
                             social_data: List[Dict]) -> Dict[str, str]:
        insights = {}
        
        fg_value = fear_greed_data['value']
        fg_classification = fear_greed_data['classification']
        
        insights['market_sentiment'] = f"Overall market sentiment is {fg_classification.lower()} with Fear & Greed Index at {fg_value}"
        
        if fg_value <= 25:
            insights['recommendation'] = "Extreme fear often presents buying opportunities for long-term investors"
        elif fg_value >= 75:
            insights['recommendation'] = "Extreme greed suggests caution and potential profit-taking opportunities"
        else:
            insights['recommendation'] = "Neutral sentiment suggests waiting for clearer directional signals"
        
        if social_data:
            bullish_count = sum(1 for item in social_data if item['sentiment'] == 'bullish')
            bearish_count = sum(1 for item in social_data if item['sentiment'] == 'bearish')
            
            if bullish_count > bearish_count:
                insights['social_trend'] = "Social media discussions are predominantly bullish"
            elif bearish_count > bullish_count:
                insights['social_trend'] = "Social media discussions are predominantly bearish"
            else:
                insights['social_trend'] = "Social media sentiment is mixed"
        
        return insights