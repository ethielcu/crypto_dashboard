import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .modern_theme import ModernTheme


class ModernCharts:
    
    CHART_COLORS = {
        'background': '#2B3139',
        'paper': '#1E2329',
        'text': '#FAFAFA',
        'grid': '#373D47',
        'candlestick_up': '#00D4AA',
        'candlestick_down': '#F84960',
        'volume': '#F0B90B',
        'line_primary': '#F0B90B',
        'line_secondary': '#FF6B35',
        'area_fill': 'rgba(240, 185, 11, 0.1)'
    }
    
    @classmethod
    def get_chart_layout(cls, title: str = "", height: int = 500) -> Dict:
        return {
            'title': {
                'text': title,
                'font': {'color': cls.CHART_COLORS['text'], 'size': 18, 'family': 'Inter'},
                'x': 0.5,
                'xanchor': 'center'
            },
            'paper_bgcolor': cls.CHART_COLORS['paper'],
            'plot_bgcolor': cls.CHART_COLORS['background'],
            'font': {'color': cls.CHART_COLORS['text'], 'family': 'Inter'},
            'height': height,
            'margin': {'l': 40, 'r': 40, 't': 60, 'b': 40},
            'xaxis': {
                'gridcolor': cls.CHART_COLORS['grid'],
                'showgrid': True,
                'zeroline': False,
                'color': cls.CHART_COLORS['text']
            },
            'yaxis': {
                'gridcolor': cls.CHART_COLORS['grid'],
                'showgrid': True,
                'zeroline': False,
                'color': cls.CHART_COLORS['text']
            },
            'legend': {
                'bgcolor': 'rgba(0,0,0,0)',
                'bordercolor': cls.CHART_COLORS['grid'],
                'borderwidth': 1,
                'font': {'color': cls.CHART_COLORS['text']}
            }
        }
    
    @classmethod
    def candlestick_chart(cls, df: pd.DataFrame, title: str = "Price Chart") -> go.Figure:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        candlestick = go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing_line_color=cls.CHART_COLORS['candlestick_up'],
            decreasing_line_color=cls.CHART_COLORS['candlestick_down'],
            increasing_fillcolor=cls.CHART_COLORS['candlestick_up'],
            decreasing_fillcolor=cls.CHART_COLORS['candlestick_down'],
            name="Price"
        )
        
        fig.add_trace(candlestick, row=1, col=1)
        
        if 'volume' in df.columns:
            volume_colors = [cls.CHART_COLORS['candlestick_up'] if close >= open 
                           else cls.CHART_COLORS['candlestick_down'] 
                           for close, open in zip(df['close'], df['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    marker_color=volume_colors,
                    name="Volume",
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        layout = cls.get_chart_layout(title, 600)
        layout['xaxis2'] = layout['xaxis'].copy()
        layout['yaxis2'] = layout['yaxis'].copy()
        layout['yaxis2']['title'] = 'Volume'
        
        fig.update_layout(layout)
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        return fig
    
    @classmethod
    def line_chart_with_volume(cls, df: pd.DataFrame, title: str = "Price Analysis") -> go.Figure:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'] if 'close' in df.columns else df['price'],
                mode='lines',
                name='Price',
                line=dict(color=cls.CHART_COLORS['line_primary'], width=2),
                fill='tonexty',
                fillcolor=cls.CHART_COLORS['area_fill']
            ),
            row=1, col=1
        )
        
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    marker_color=cls.CHART_COLORS['volume'],
                    name="Volume",
                    opacity=0.6
                ),
                row=2, col=1
            )
        
        layout = cls.get_chart_layout(title, 600)
        fig.update_layout(layout)
        
        return fig
    
    @classmethod
    def technical_indicators_chart(cls, df: pd.DataFrame, indicators: List[str]) -> go.Figure:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('Price with Indicators', 'RSI', 'MACD'),
            row_width=[0.5, 0.25, 0.25]
        )
        
        price_col = 'close' if 'close' in df.columns else 'price'
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[price_col],
                mode='lines',
                name='Price',
                line=dict(color=cls.CHART_COLORS['line_primary'], width=2)
            ),
            row=1, col=1
        )
        
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color=cls.CHART_COLORS['line_secondary'], width=1)
                ),
                row=1, col=1
            )
        
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(255, 255, 255, 0.3)', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_lower'],
                    mode='lines',
                    name='Bollinger Bands',
                    line=dict(color='rgba(255, 255, 255, 0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(255, 255, 255, 0.05)'
                ),
                row=1, col=1
            )
        
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color=cls.CHART_COLORS['volume'], width=2)
                ),
                row=2, col=1
            )
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
        
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd'],
                    mode='lines',
                    name='MACD',
                    line=dict(color=cls.CHART_COLORS['line_primary'], width=2)
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd_signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color=cls.CHART_COLORS['line_secondary'], width=2)
                ),
                row=3, col=1
            )
            
            if 'macd_histogram' in df.columns:
                colors = ['green' if val >= 0 else 'red' for val in df['macd_histogram']]
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['macd_histogram'],
                        marker_color=colors,
                        name="Histogram",
                        opacity=0.6
                    ),
                    row=3, col=1
                )
        
        layout = cls.get_chart_layout("Technical Analysis", 800)
        fig.update_layout(layout)
        
        return fig
    
    @classmethod
    def correlation_heatmap(cls, correlation_matrix: pd.DataFrame, title: str = "Asset Correlation") -> go.Figure:
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale=[
                [0, cls.CHART_COLORS['candlestick_down']],
                [0.5, '#FFFFFF'],
                [1, cls.CHART_COLORS['candlestick_up']]
            ],
            zmin=-1,
            zmax=1,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 12, "color": "white"},
            hoverongaps=False
        ))
        
        layout = cls.get_chart_layout(title, 600)
        layout['xaxis']['side'] = 'bottom'
        fig.update_layout(layout)
        
        return fig
    
    @classmethod
    def portfolio_performance_chart(cls, returns: pd.Series, benchmark_returns: pd.Series = None, 
                                  title: str = "Portfolio Performance") -> go.Figure:
        cumulative_returns = (1 + returns).cumprod()
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode='lines',
                name='Portfolio',
                line=dict(color=cls.CHART_COLORS['line_primary'], width=3),
                fill='tonexty',
                fillcolor=cls.CHART_COLORS['area_fill']
            )
        )
        
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color=cls.CHART_COLORS['line_secondary'], width=2, dash='dash')
                )
            )
        
        layout = cls.get_chart_layout(title, 500)
        layout['yaxis']['title'] = 'Cumulative Returns'
        fig.update_layout(layout)
        
        return fig
    
    @classmethod
    def market_overview_treemap(cls, market_data: pd.DataFrame) -> go.Figure:
        if market_data.empty:
            return go.Figure()
        
        market_data = market_data.head(20)
        
        colors = [cls.CHART_COLORS['candlestick_up'] if change >= 0 
                 else cls.CHART_COLORS['candlestick_down'] 
                 for change in market_data['24h Change (%)']]
        
        fig = go.Figure(go.Treemap(
            labels=market_data['Symbol'],
            values=market_data['Market Cap'],
            parents=[""] * len(market_data),
            marker=dict(
                colors=colors,
                line=dict(width=2, color='white')
            ),
            textinfo="label+value",
            textfont=dict(size=12, color='white'),
            hovertemplate='<b>%{label}</b><br>Market Cap: $%{value:,.0f}<extra></extra>'
        ))
        
        layout = cls.get_chart_layout("Market Cap Distribution", 600)
        fig.update_layout(layout)
        
        return fig
    
    @classmethod
    def fear_greed_gauge(cls, value: int, classification: str) -> go.Figure:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Fear & Greed Index<br><span style='font-size:0.8em'>{classification}</span>"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': cls.CHART_COLORS['text']},
                'bar': {'color': cls.CHART_COLORS['line_primary']},
                'bgcolor': cls.CHART_COLORS['background'],
                'borderwidth': 2,
                'bordercolor': cls.CHART_COLORS['grid'],
                'steps': [
                    {'range': [0, 25], 'color': cls.CHART_COLORS['candlestick_down']},
                    {'range': [25, 45], 'color': '#FF6B35'},
                    {'range': [45, 55], 'color': '#FFA500'},
                    {'range': [55, 75], 'color': '#90EE90'},
                    {'range': [75, 100], 'color': cls.CHART_COLORS['candlestick_up']}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        
        layout = cls.get_chart_layout("", 400)
        layout['paper_bgcolor'] = cls.CHART_COLORS['background']
        fig.update_layout(layout)
        
        return fig
    
    @classmethod
    def volume_profile_chart(cls, df: pd.DataFrame, bins: int = 50) -> go.Figure:
        if 'volume' not in df.columns:
            return go.Figure()
        
        price_col = 'close' if 'close' in df.columns else 'price'
        
        price_range = df[price_col].max() - df[price_col].min()
        bin_size = price_range / bins
        
        volume_profile = []
        for i in range(bins):
            price_level = df[price_col].min() + (i * bin_size)
            price_mask = ((df[price_col] >= price_level) & 
                         (df[price_col] < price_level + bin_size))
            volume_at_level = df.loc[price_mask, 'volume'].sum()
            volume_profile.append({'price': price_level, 'volume': volume_at_level})
        
        profile_df = pd.DataFrame(volume_profile)
        
        fig = make_subplots(
            rows=1, cols=2,
            shared_yaxes=True,
            horizontal_spacing=0.05,
            column_widths=[0.7, 0.3]
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[price_col],
                mode='lines',
                name='Price',
                line=dict(color=cls.CHART_COLORS['line_primary'], width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=profile_df['volume'],
                y=profile_df['price'],
                orientation='h',
                name='Volume Profile',
                marker_color=cls.CHART_COLORS['volume'],
                opacity=0.7
            ),
            row=1, col=2
        )
        
        layout = cls.get_chart_layout("Volume Profile Analysis", 600)
        fig.update_layout(layout)
        
        return fig