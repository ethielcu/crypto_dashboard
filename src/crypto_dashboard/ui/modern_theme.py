import streamlit as st
from typing import Dict, List, Optional


class ModernTheme:
    
    COLORS = {
        'primary_bg': '#0B0E11',
        'secondary_bg': '#1E2329',
        'card_bg': '#2B3139',
        'accent_gold': '#F0B90B',
        'accent_orange': '#FF6B35',
        'success_green': '#00D4AA',
        'danger_red': '#F84960',
        'text_primary': '#FAFAFA',
        'text_secondary': '#B7BDC6',
        'text_muted': '#848E9C',
        'border': '#373D47',
        'gradient_start': '#F0B90B',
        'gradient_end': '#FF6B35'
    }
    
    @classmethod
    def get_custom_css(cls) -> str:
        return f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp {{
            background: linear-gradient(135deg, {cls.COLORS['primary_bg']} 0%, #151A23 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }}
        
        /* Sidebar Styling */
        .css-1d391kg {{
            background: linear-gradient(180deg, {cls.COLORS['secondary_bg']} 0%, {cls.COLORS['primary_bg']} 100%);
            border-right: 1px solid {cls.COLORS['border']};
        }}
        
        .css-1d391kg .css-1y4p8pa {{
            background: transparent;
        }}
        
        /* Modern Cards */
        .modern-card {{
            background: {cls.COLORS['card_bg']};
            border: 1px solid {cls.COLORS['border']};
            border-radius: 12px;
            padding: 24px;
            margin: 16px 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .modern-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, {cls.COLORS['gradient_start']}, {cls.COLORS['gradient_end']});
        }}
        
        .modern-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(240, 185, 11, 0.15);
            border-color: {cls.COLORS['accent_gold']};
        }}
        
        /* Price Ticker */
        .price-ticker {{
            background: {cls.COLORS['secondary_bg']};
            border: 1px solid {cls.COLORS['border']};
            border-radius: 8px;
            padding: 12px 16px;
            margin: 8px 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.3s ease;
        }}
        
        .price-ticker:hover {{
            background: {cls.COLORS['card_bg']};
            border-color: {cls.COLORS['accent_gold']};
        }}
        
        .crypto-symbol {{
            font-weight: 600;
            color: {cls.COLORS['text_primary']};
            font-size: 14px;
        }}
        
        .crypto-price {{
            font-weight: 700;
            color: {cls.COLORS['text_primary']};
            font-size: 16px;
        }}
        
        .price-change-positive {{
            color: {cls.COLORS['success_green']};
            font-weight: 600;
        }}
        
        .price-change-negative {{
            color: {cls.COLORS['danger_red']};
            font-weight: 600;
        }}
        
        /* Modern Metrics */
        .metric-container {{
            background: {cls.COLORS['card_bg']};
            border: 1px solid {cls.COLORS['border']};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .metric-container::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, {cls.COLORS['gradient_start']}, {cls.COLORS['gradient_end']});
        }}
        
        .metric-container:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(240, 185, 11, 0.2);
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {cls.COLORS['text_primary']};
            margin-bottom: 8px;
            background: linear-gradient(135deg, {cls.COLORS['gradient_start']}, {cls.COLORS['gradient_end']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .metric-label {{
            font-size: 0.875rem;
            color: {cls.COLORS['text_secondary']};
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .metric-delta {{
            font-size: 1rem;
            font-weight: 600;
            margin-top: 4px;
        }}
        
        /* Navigation Icons */
        .nav-item {{
            display: flex;
            align-items: center;
            padding: 12px 16px;
            margin: 4px 0;
            border-radius: 8px;
            transition: all 0.3s ease;
            color: {cls.COLORS['text_secondary']};
            text-decoration: none;
            cursor: pointer;
        }}
        
        .nav-item:hover {{
            background: {cls.COLORS['card_bg']};
            color: {cls.COLORS['accent_gold']};
            transform: translateX(4px);
        }}
        
        .nav-item.active {{
            background: linear-gradient(135deg, {cls.COLORS['accent_gold']}20, {cls.COLORS['accent_orange']}20);
            color: {cls.COLORS['accent_gold']};
            border-left: 3px solid {cls.COLORS['accent_gold']};
        }}
        
        .nav-icon {{
            margin-right: 12px;
            font-size: 1.2rem;
        }}
        
        /* Chart Styling */
        .js-plotly-plot {{
            background: {cls.COLORS['card_bg']} !important;
            border-radius: 12px;
            border: 1px solid {cls.COLORS['border']};
        }}
        
        /* Header Styling */
        .main-header {{
            background: {cls.COLORS['secondary_bg']};
            border: 1px solid {cls.COLORS['border']};
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
        }}
        
        .main-header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, {cls.COLORS['gradient_start']}, {cls.COLORS['gradient_end']});
        }}
        
        .main-title {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, {cls.COLORS['gradient_start']}, {cls.COLORS['gradient_end']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            text-align: center;
        }}
        
        .main-subtitle {{
            color: {cls.COLORS['text_secondary']};
            text-align: center;
            margin-top: 8px;
            font-size: 1.1rem;
            font-weight: 400;
        }}
        
        /* Table Styling */
        .dataframe {{
            background: {cls.COLORS['card_bg']};
            border: 1px solid {cls.COLORS['border']};
            border-radius: 12px;
            overflow: hidden;
        }}
        
        .dataframe th {{
            background: {cls.COLORS['secondary_bg']};
            color: {cls.COLORS['text_primary']};
            font-weight: 600;
            padding: 12px 16px;
            border-bottom: 2px solid {cls.COLORS['accent_gold']};
        }}
        
        .dataframe td {{
            background: {cls.COLORS['card_bg']};
            color: {cls.COLORS['text_primary']};
            padding: 12px 16px;
            border-bottom: 1px solid {cls.COLORS['border']};
        }}
        
        .dataframe tr:hover {{
            background: {cls.COLORS['secondary_bg']};
        }}
        
        /* Button Styling */
        .stButton > button {{
            background: linear-gradient(135deg, {cls.COLORS['gradient_start']}, {cls.COLORS['gradient_end']});
            color: {cls.COLORS['primary_bg']};
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(240, 185, 11, 0.4);
        }}
        
        /* Select Box Styling */
        .stSelectbox > div > div {{
            background: {cls.COLORS['card_bg']};
            border: 1px solid {cls.COLORS['border']};
            border-radius: 8px;
            color: {cls.COLORS['text_primary']};
        }}
        
        /* Input Styling */
        .stTextInput > div > div > input {{
            background: {cls.COLORS['card_bg']};
            border: 1px solid {cls.COLORS['border']};
            border-radius: 8px;
            color: {cls.COLORS['text_primary']};
        }}
        
        .stNumberInput > div > div > input {{
            background: {cls.COLORS['card_bg']};
            border: 1px solid {cls.COLORS['border']};
            border-radius: 8px;
            color: {cls.COLORS['text_primary']};
        }}
        
        /* Slider Styling */
        .stSlider > div > div > div > div {{
            background: {cls.COLORS['accent_gold']};
        }}
        
        /* Alert Styling */
        .alert-success {{
            background: linear-gradient(135deg, {cls.COLORS['success_green']}20, {cls.COLORS['success_green']}10);
            border: 1px solid {cls.COLORS['success_green']};
            border-radius: 8px;
            padding: 16px;
            color: {cls.COLORS['success_green']};
        }}
        
        .alert-danger {{
            background: linear-gradient(135deg, {cls.COLORS['danger_red']}20, {cls.COLORS['danger_red']}10);
            border: 1px solid {cls.COLORS['danger_red']};
            border-radius: 8px;
            padding: 16px;
            color: {cls.COLORS['danger_red']};
        }}
        
        .alert-warning {{
            background: linear-gradient(135deg, {cls.COLORS['accent_orange']}20, {cls.COLORS['accent_orange']}10);
            border: 1px solid {cls.COLORS['accent_orange']};
            border-radius: 8px;
            padding: 16px;
            color: {cls.COLORS['accent_orange']};
        }}
        
        /* Loading Animation */
        .loading-spinner {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid {cls.COLORS['border']};
            border-radius: 50%;
            border-top-color: {cls.COLORS['accent_gold']};
            animation: spin 1s ease-in-out infinite;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        
        /* Scrollbar Styling */
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {cls.COLORS['primary_bg']};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {cls.COLORS['accent_gold']};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {cls.COLORS['accent_orange']};
        }}
        
        /* Hide Streamlit Branding */
        .css-1rs6os {{
            display: none;
        }}
        
        .css-164nlkn {{
            display: none;
        }}
        
        #MainMenu {{
            display: none;
        }}
        
        .css-1y4p8pa {{
            padding-top: 2rem;
        }}
        </style>
        """
    
    @classmethod
    def render_modern_card(cls, content: str, title: Optional[str] = None) -> str:
        title_html = f"<h3 style='color: {cls.COLORS['text_primary']}; margin-bottom: 16px; font-weight: 600;'>{title}</h3>" if title else ""
        return f"""
        <div class="modern-card">
            {title_html}
            {content}
        </div>
        """
    
    @classmethod
    def render_metric_card(cls, value: str, label: str, delta: Optional[str] = None, 
                          delta_color: str = 'success') -> str:
        delta_class = f"price-change-{'positive' if delta_color == 'success' else 'negative'}"
        delta_html = f"<div class='metric-delta {delta_class}'>{delta}</div>" if delta else ""
        
        return f"""
        <div class="metric-container">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            {delta_html}
        </div>
        """
    
    @classmethod
    def render_price_ticker(cls, symbol: str, price: str, change: str, 
                           change_positive: bool = True) -> str:
        change_class = "price-change-positive" if change_positive else "price-change-negative"
        arrow = "▲" if change_positive else "▼"
        
        return f"""
        <div class="price-ticker">
            <span class="crypto-symbol">{symbol}</span>
            <span class="crypto-price">{price}</span>
            <span class="{change_class}">{arrow} {change}</span>
        </div>
        """
    
    @classmethod
    def render_main_header(cls, title: str, subtitle: Optional[str] = None) -> str:
        subtitle_html = f"<div class='main-subtitle'>{subtitle}</div>" if subtitle else ""
        
        return f"""
        <div class="main-header">
            <h1 class="main-title">{title}</h1>
            {subtitle_html}
        </div>
        """


def apply_modern_theme():
    st.markdown(ModernTheme.get_custom_css(), unsafe_allow_html=True)


def create_metric_grid(metrics: List[Dict[str, str]], cols: int = 4):
    columns = st.columns(cols)
    for i, metric in enumerate(metrics):
        with columns[i % cols]:
            st.markdown(
                ModernTheme.render_metric_card(
                    metric['value'], 
                    metric['label'], 
                    metric.get('delta'),
                    metric.get('delta_color', 'success')
                ), 
                unsafe_allow_html=True
            )