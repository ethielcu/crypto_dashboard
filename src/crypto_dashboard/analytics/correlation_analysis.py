import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class CorrelationAnalyzer:
    
    def __init__(self):
        self.correlation_matrix = None
        self.returns_data = None
    
    def calculate_returns(self, price_data: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
        if method == 'simple':
            returns = price_data.pct_change().dropna()
        elif method == 'log':
            returns = np.log(price_data / price_data.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")
        
        self.returns_data = returns
        return returns
    
    def calculate_correlation_matrix(self, returns_data: Optional[pd.DataFrame] = None, 
                                   method: str = 'pearson') -> pd.DataFrame:
        if returns_data is not None:
            self.returns_data = returns_data
        
        if self.returns_data is None:
            raise ValueError("Returns data must be provided")
        
        if method == 'pearson':
            correlation_matrix = self.returns_data.corr()
        elif method == 'spearman':
            correlation_matrix = self.returns_data.corr(method='spearman')
        elif method == 'kendall':
            correlation_matrix = self.returns_data.corr(method='kendall')
        else:
            raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
        
        self.correlation_matrix = correlation_matrix
        return correlation_matrix
    
    def rolling_correlation(self, asset1: str, asset2: str, window: int = 30) -> pd.Series:
        if self.returns_data is None:
            raise ValueError("Returns data must be calculated first")
        
        return self.returns_data[asset1].rolling(window=window).corr(self.returns_data[asset2])
    
    def cross_correlation_analysis(self, max_lags: int = 10) -> Dict[str, pd.DataFrame]:
        if self.returns_data is None:
            raise ValueError("Returns data must be calculated first")
        
        assets = self.returns_data.columns
        cross_corr_results = {}
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:
                    pair_name = f"{asset1}_{asset2}"
                    correlations = []
                    lags = list(range(-max_lags, max_lags + 1))
                    
                    for lag in lags:
                        if lag == 0:
                            corr = self.returns_data[asset1].corr(self.returns_data[asset2])
                        elif lag > 0:
                            corr = self.returns_data[asset1].corr(self.returns_data[asset2].shift(lag))
                        else:
                            corr = self.returns_data[asset1].shift(-lag).corr(self.returns_data[asset2])
                        
                        correlations.append(corr)
                    
                    cross_corr_results[pair_name] = pd.DataFrame({
                        'lag': lags,
                        'correlation': correlations
                    })
        
        return cross_corr_results
    
    def identify_correlation_clusters(self, threshold: float = 0.7) -> Dict[str, List[str]]:
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix must be calculated first")
        
        clusters = {}
        processed_assets = set()
        
        for asset in self.correlation_matrix.columns:
            if asset in processed_assets:
                continue
            
            highly_correlated = self.correlation_matrix[asset][
                (self.correlation_matrix[asset] >= threshold) & 
                (self.correlation_matrix[asset] < 1.0)
            ].index.tolist()
            
            if highly_correlated:
                cluster_name = f"cluster_{len(clusters) + 1}"
                clusters[cluster_name] = [asset] + highly_correlated
                processed_assets.update([asset] + highly_correlated)
        
        return clusters
    
    def calculate_diversification_ratio(self, weights: np.array) -> float:
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix must be calculated first")
        
        if self.returns_data is None:
            raise ValueError("Returns data must be calculated first")
        
        individual_vol = self.returns_data.std()
        weighted_avg_vol = np.sum(weights * individual_vol)
        
        portfolio_vol = np.sqrt(
            np.dot(weights, np.dot(self.correlation_matrix * 
                                 np.outer(individual_vol, individual_vol), weights))
        )
        
        return weighted_avg_vol / portfolio_vol
    
    def create_correlation_heatmap(self, correlation_matrix: Optional[pd.DataFrame] = None,
                                 title: str = "Asset Correlation Matrix") -> go.Figure:
        if correlation_matrix is not None:
            corr_matrix = correlation_matrix
        elif self.correlation_matrix is not None:
            corr_matrix = self.correlation_matrix
        else:
            raise ValueError("Correlation matrix must be provided")
        
        fig = px.imshow(
            corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title=title,
            zmin=-1,
            zmax=1
        )
        
        fig.update_traces(
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=10)
        )
        
        fig.update_layout(
            xaxis_title="Assets",
            yaxis_title="Assets",
            width=800,
            height=600
        )
        
        return fig
    
    def create_rolling_correlation_chart(self, asset1: str, asset2: str, 
                                       window: int = 30) -> go.Figure:
        rolling_corr = self.rolling_correlation(asset1, asset2, window)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr.values,
            mode='lines',
            name=f'Rolling Correlation ({window}d)',
            line=dict(width=2)
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
        fig.add_hline(y=0.5, line_dash="dot", line_color="red", opacity=0.7)
        fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.7)
        
        fig.update_layout(
            title=f"Rolling Correlation: {asset1} vs {asset2}",
            xaxis_title="Date",
            yaxis_title="Correlation",
            yaxis=dict(range=[-1, 1]),
            showlegend=True
        )
        
        return fig
    
    def create_cross_correlation_chart(self, asset1: str, asset2: str, 
                                     max_lags: int = 10) -> go.Figure:
        cross_corr = self.cross_correlation_analysis(max_lags)
        pair_name = f"{asset1}_{asset2}"
        
        if pair_name not in cross_corr:
            pair_name = f"{asset2}_{asset1}"
        
        if pair_name not in cross_corr:
            raise ValueError(f"Cross-correlation data not found for {asset1} and {asset2}")
        
        data = cross_corr[pair_name]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data['lag'],
            y=data['correlation'],
            name='Cross-Correlation',
            marker_color=['red' if x < 0 else 'blue' for x in data['correlation']]
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
        
        fig.update_layout(
            title=f"Cross-Correlation: {asset1} vs {asset2}",
            xaxis_title="Lag (days)",
            yaxis_title="Correlation",
            yaxis=dict(range=[-1, 1]),
            showlegend=True
        )
        
        return fig
    
    def create_correlation_network(self, threshold: float = 0.3) -> go.Figure:
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix must be calculated first")
        
        assets = list(self.correlation_matrix.columns)
        edges = []
        edge_weights = []
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:
                    corr = self.correlation_matrix.loc[asset1, asset2]
                    if abs(corr) >= threshold:
                        edges.append((i, j))
                        edge_weights.append(abs(corr))
        
        pos = {}
        n = len(assets)
        for i, asset in enumerate(assets):
            angle = 2 * np.pi * i / n
            pos[i] = (np.cos(angle), np.sin(angle))
        
        edge_trace = []
        for (i, j), weight in zip(edges, edge_weights):
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight*5, color='rgba(125,125,125,0.5)'),
                hoverinfo='none'
            ))
        
        node_trace = go.Scatter(
            x=[pos[i][0] for i in range(n)],
            y=[pos[i][1] for i in range(n)],
            mode='markers+text',
            marker=dict(size=20, color='lightblue', line=dict(width=2, color='black')),
            text=assets,
            textposition="middle center",
            hoverinfo='text',
            hovertext=assets
        )
        
        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(
            title="Asset Correlation Network",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=600
        )
        
        return fig