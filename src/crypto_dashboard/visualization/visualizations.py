import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class CryptoVisualizer:
    
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8'
        }
    
    def create_price_chart(self, df: pd.DataFrame, title: str = "Price Chart") -> go.Figure:
        fig = go.Figure()
        
        if df.empty:
            fig.add_annotation(text="No data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['price'],
            mode='lines',
            name='Price',
            line=dict(color=self.colors['primary'], width=2),
            hovertemplate='<b>%{x}</b><br>Price: $%{y:,.2f}<extra></extra>'
        ))
        
        if 'volume' in df.columns:
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                yaxis='y2',
                opacity=0.3,
                marker_color=self.colors['secondary'],
                hovertemplate='<b>%{x}</b><br>Volume: $%{y:,.0f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            yaxis2=dict(
                title="Volume (USD)",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_market_overview(self, df: pd.DataFrame) -> go.Figure:
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No market data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Market Cap Distribution', 'Price Changes (24h)', 
                          'Volume vs Market Cap', 'Top 10 by Market Cap'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        top_10 = df.head(10).copy()
        
        fig.add_trace(go.Pie(
            labels=top_10['Name'],
            values=top_10['Market Cap'],
            name="Market Cap",
            hovertemplate='<b>%{label}</b><br>Market Cap: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        ), row=1, col=1)
        
        colors = ['red' if x < 0 else 'green' for x in top_10['24h Change (%)']]
        fig.add_trace(go.Bar(
            x=top_10['Symbol'],
            y=top_10['24h Change (%)'],
            name="24h Change",
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>24h Change: %{y:.2f}%<extra></extra>'
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=top_10['Market Cap'],
            y=top_10['Volume (24h)'],
            mode='markers+text',
            text=top_10['Symbol'],
            textposition='top center',
            name="Volume vs Market Cap",
            marker=dict(size=10, color=self.colors['primary']),
            hovertemplate='<b>%{text}</b><br>Market Cap: $%{x:,.0f}<br>Volume: $%{y:,.0f}<extra></extra>'
        ), row=2, col=1)
        
        fig.add_trace(go.Bar(
            x=top_10['Symbol'],
            y=top_10['Market Cap'],
            name="Market Cap",
            marker_color=self.colors['info'],
            hovertemplate='<b>%{x}</b><br>Market Cap: $%{y:,.0f}<extra></extra>'
        ), row=2, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=False,
            template='plotly_white',
            title_text="Cryptocurrency Market Overview"
        )
        
        return fig
    
    def create_fear_greed_gauge(self, value: int, classification: str) -> go.Figure:
        if value <= 25:
            color = "red"
        elif value <= 45:
            color = "orange" 
        elif value <= 55:
            color = "yellow"
        elif value <= 75:
            color = "lightgreen"
        else:
            color = "green"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Fear & Greed Index<br><span style='font-size:16px'>{classification}</span>"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 25], 'color': "lightcoral"},
                    {'range': [25, 45], 'color': "lightsalmon"},
                    {'range': [45, 55], 'color': "lightgoldenrodyellow"},
                    {'range': [55, 75], 'color': "lightgreen"},
                    {'range': [75, 100], 'color': "darkgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame, coins: List[str]) -> go.Figure:
        if df.empty or len(coins) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for correlation analysis", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        correlation_matrix = df[coins].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Cryptocurrency Price Correlation Matrix",
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def create_portfolio_pie(self, portfolio: Dict[str, float]) -> go.Figure:
        if not portfolio:
            fig = go.Figure()
            fig.add_annotation(text="No portfolio data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        labels = list(portfolio.keys())
        values = list(portfolio.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hovertemplate='<b>%{label}</b><br>Allocation: %{percent}<br>Value: $%{value:,.2f}<extra></extra>',
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_risk_metrics_chart(self, metrics: Dict[str, float]) -> go.Figure:
        if not metrics:
            fig = go.Figure()
            fig.add_annotation(text="No risk metrics available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        colors = []
        for value in metric_values:
            if 'volatility' in str(value).lower() or 'var' in str(value).lower():
                colors.append('red' if value > 0.3 else 'orange' if value > 0.2 else 'green')
            else:
                colors.append(self.colors['primary'])
        
        fig = go.Figure(data=[go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Risk Metrics Overview",
            xaxis_title="Metrics",
            yaxis_title="Value",
            template='plotly_white',
            height=400
        )
        
        return fig


if __name__ == "__main__":
    visualizer = CryptoVisualizer()
    
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    test_df = pd.DataFrame({
        'price': np.random.randn(30).cumsum() + 50000,
        'volume': np.random.randint(1000000, 10000000, 30)
    }, index=dates)
    
    price_chart = visualizer.create_price_chart(test_df, "Test BTC Price")
    print("Price chart created successfully")
    
    fg_gauge = visualizer.create_fear_greed_gauge(75, "Greed")
    print("Fear & Greed gauge created successfully")