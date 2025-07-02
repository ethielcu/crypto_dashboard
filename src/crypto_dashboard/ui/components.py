import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import plotly.graph_objects as go
from .modern_theme import ModernTheme


class ModernComponents:
    
    @staticmethod
    def sidebar_navigation():
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #F0B90B; font-weight: 700; margin: 0; font-size: 1.5rem;">
                CryptoDash
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        pages = [
            "Market Overview",
            "Price Analysis", 
            "Risk Calculator",
            "Portfolio Optimizer",
            "Advanced Analytics",
            "Multi-Exchange Data",
            "Alerts & Monitoring"
        ]
        
        selected = st.selectbox(
            "Navigate to:",
            options=pages,
            label_visibility="collapsed"
        )
        
        return selected
    
    @staticmethod
    def price_ticker_header(price_data: List[Dict]):
        if not price_data:
            return
            
        cols = st.columns(min(len(price_data), 6))
        
        for i, crypto in enumerate(price_data[:6]):
            if i < len(cols):
                with cols[i]:
                    change_str = crypto.get('change', '0%')
                    try:
                        change_value = float(change_str.replace('%', '').replace('+', ''))
                        change_positive = change_value >= 0
                        arrow = "▲" if change_positive else "▼"
                        color = "#00D4AA" if change_positive else "#F84960"
                        
                        st.markdown(f"""
                        <div style="background: #2B3139; border: 1px solid #373D47; border-radius: 8px; 
                                   padding: 12px; text-align: center; margin-bottom: 8px;">
                            <div style="color: #FAFAFA; font-weight: 600; font-size: 14px;">
                                {crypto['symbol'].upper()}
                            </div>
                            <div style="color: #FAFAFA; font-weight: 700; font-size: 16px; margin: 4px 0;">
                                {crypto['price']}
                            </div>
                            <div style="color: {color}; font-weight: 600; font-size: 12px;">
                                {arrow} {change_str}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    except:
                        st.markdown(f"""
                        <div style="background: #2B3139; border: 1px solid #373D47; border-radius: 8px; 
                                   padding: 12px; text-align: center; margin-bottom: 8px;">
                            <div style="color: #FAFAFA; font-weight: 600;">{crypto['symbol'].upper()}</div>
                            <div style="color: #FAFAFA; font-weight: 700;">{crypto['price']}</div>
                            <div style="color: #B7BDC6;">{crypto.get('change', 'N/A')}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    @staticmethod
    def modern_metric_row(metrics: List[Dict], columns: int = 4):
        if not metrics:
            return
        
        rows_needed = (len(metrics) + columns - 1) // columns
        
        for row in range(rows_needed):
            cols = st.columns(columns)
            start_idx = row * columns
            end_idx = min(start_idx + columns, len(metrics))
            
            for i, metric_idx in enumerate(range(start_idx, end_idx)):
                if metric_idx < len(metrics):
                    metric = metrics[metric_idx]
                    with cols[i]:
                        st.markdown(
                            ModernTheme.render_metric_card(
                                metric['value'],
                                metric['label'],
                                metric.get('delta'),
                                metric.get('delta_color', 'success')
                            ),
                            unsafe_allow_html=True
                        )
    
    @staticmethod
    def modern_card_container(content_func, title: Optional[str] = None, height: Optional[int] = None):
        card_style = "background: #2B3139; border: 1px solid #373D47; border-radius: 12px; padding: 24px; margin: 16px 0;"
        if height:
            card_style += f" height: {height}px; overflow-y: auto;"
        
        if title:
            st.markdown(f"""
            <div style="{card_style}">
                <h3 style="color: #FAFAFA; margin-bottom: 20px; font-weight: 600; font-size: 1.25rem;">
                    {title}
                </h3>
            </div>
            """, unsafe_allow_html=True)
        
        with st.container():
            content_func()
    
    @staticmethod
    def trading_pair_selector(pairs: List[str], default_index: int = 0):
        st.markdown("""
        <style>
        .trading-pair-container {
            background: #2B3139;
            border: 1px solid #373D47;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        return st.selectbox(
            "Trading Pair",
            options=pairs,
            index=default_index,
            help="Select cryptocurrency pair for analysis"
        )
    
    @staticmethod
    def timeframe_selector(timeframes: List[str] = None, default: str = "1D"):
        if timeframes is None:
            timeframes = ["5M", "15M", "1H", "4H", "1D", "1W", "1M"]
        
        cols = st.columns(len(timeframes))
        selected_timeframe = default
        
        for i, tf in enumerate(timeframes):
            with cols[i]:
                if st.button(tf, key=f"tf_{tf}", use_container_width=True):
                    selected_timeframe = tf
        
        return selected_timeframe
    
    @staticmethod
    def status_indicator(status: str, label: str):
        color_map = {
            'online': '#00D4AA',
            'offline': '#F84960',
            'warning': '#FF6B35',
            'neutral': '#B7BDC6'
        }
        
        color = color_map.get(status.lower(), '#B7BDC6')
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; 
                    background: {color}20; border: 1px solid {color}; border-radius: 6px;">
            <div style="width: 8px; height: 8px; background: {color}; border-radius: 50%;"></div>
            <span style="color: {color}; font-weight: 500; font-size: 0.875rem;">{label}</span>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def progress_bar(value: float, max_value: float = 100, label: str = "", 
                    color: str = "#F0B90B", height: int = 8):
        percentage = (value / max_value) * 100
        
        st.markdown(f"""
        <div style="margin: 12px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #FAFAFA; font-size: 0.875rem; font-weight: 500;">{label}</span>
                <span style="color: #B7BDC6; font-size: 0.875rem;">{percentage:.1f}%</span>
            </div>
            <div style="background: #373D47; border-radius: {height//2}px; height: {height}px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, {color}, {color}80); 
                           width: {percentage}%; height: 100%; border-radius: {height//2}px;
                           transition: width 0.3s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def alert_badge(message: str, alert_type: str = "info", dismissible: bool = False):
        type_styles = {
            'success': {'bg': '#00D4AA20', 'border': '#00D4AA', 'color': '#00D4AA'},
            'danger': {'bg': '#F8496020', 'border': '#F84960', 'color': '#F84960'},
            'warning': {'bg': '#FF6B3520', 'border': '#FF6B35', 'color': '#FF6B35'},
            'info': {'bg': '#F0B90B20', 'border': '#F0B90B', 'color': '#F0B90B'}
        }
        
        style = type_styles.get(alert_type, type_styles['info'])
        dismiss_btn = "❌" if dismissible else ""
        
        st.markdown(f"""
        <div style="background: {style['bg']}; border: 1px solid {style['border']}; 
                    border-radius: 8px; padding: 12px 16px; margin: 8px 0;
                    display: flex; justify-content: space-between; align-items: center;">
            <span style="color: {style['color']}; font-weight: 500;">{message}</span>
            <span style="cursor: pointer; color: {style['color']};">{dismiss_btn}</span>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def modern_table(df: pd.DataFrame, max_height: int = 400):
        if df.empty:
            st.markdown("""
            <div style="text-align: center; padding: 40px; color: #848E9C;">
                <div style="font-size: 1.1rem;">No data available</div>
            </div>
            """, unsafe_allow_html=True)
            return
        
        st.dataframe(
            df,
            use_container_width=True,
            height=min(len(df) * 40 + 40, max_height)
        )
    
    @staticmethod
    def loading_spinner(text: str = "Loading..."):
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: center; 
                    padding: 40px; color: #B7BDC6;">
            <div class="loading-spinner" style="margin-right: 12px;"></div>
            <span style="font-weight: 500;">{text}</span>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def feature_highlight_card(title: str, description: str, 
                             action_text: str = "Learn More", action_key: str = ""):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #2B3139, #1E2329); 
                    border: 1px solid #373D47; border-radius: 16px; 
                    padding: 24px; margin: 16px 0; text-align: center;
                    transition: all 0.3s ease; position: relative; overflow: hidden;">
            <h3 style="color: #FAFAFA; margin-bottom: 12px; font-weight: 600;">{title}</h3>
            <p style="color: #B7BDC6; margin-bottom: 20px; line-height: 1.5;">{description}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if action_text and st.button(action_text, key=action_key, use_container_width=True):
            return True
        return False
    
    @staticmethod
    def trading_view_widget():
        st.markdown("""
        <div style="background: #2B3139; border: 1px solid #373D47; border-radius: 12px; 
                    padding: 16px; margin: 16px 0; min-height: 400px; 
                    display: flex; align-items: center; justify-content: center;">
            <div style="text-align: center; color: #B7BDC6;">
                <div style="font-size: 1.1rem; font-weight: 500;">TradingView Chart</div>
                <div style="font-size: 0.9rem; margin-top: 8px;">Advanced charting coming soon</div>
            </div>
        </div>
        """, unsafe_allow_html=True)