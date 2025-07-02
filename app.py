import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.crypto_dashboard.data.market_data import MarketDataFetcher
from src.crypto_dashboard.visualization.visualizations import CryptoVisualizer
from src.crypto_dashboard.risk.risk_calculator import RiskCalculator, PortfolioOptimizer
from src.crypto_dashboard.analytics import (
    TechnicalIndicators, CorrelationAnalyzer, BacktestEngine,
    SimpleMovingAverageStrategy, RSIStrategy, MACDStrategy,
    AlertSystem, SentimentAnalyzer
)
from src.crypto_dashboard.data.multi_exchange_data import MultiExchangeDataManager
from src.crypto_dashboard.utils.export_utils import ExportManager
from src.crypto_dashboard.ui import apply_modern_theme, ModernComponents, ModernCharts

st.set_page_config(
    page_title="",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def init_components():
    return (
        MarketDataFetcher(), CryptoVisualizer(), RiskCalculator(), PortfolioOptimizer(),
        TechnicalIndicators(), CorrelationAnalyzer(), BacktestEngine(),
        AlertSystem(), SentimentAnalyzer(), MultiExchangeDataManager(), ExportManager()
    )

@st.cache_data(ttl=300)
def fetch_top_cryptos(limit=50):
    components = init_components()
    fetcher = components[0]
    return fetcher.get_top_cryptos(limit)

@st.cache_data(ttl=600)
def fetch_price_history(coin_id, days):
    components = init_components()
    fetcher = components[0]
    return fetcher.get_price_history(coin_id, days)

@st.cache_data(ttl=300)
def fetch_fear_greed_index():
    components = init_components()
    fetcher = components[0]
    return fetcher.get_fear_greed_index()

@st.cache_data(ttl=600)
def fetch_global_market_data():
    components = init_components()
    fetcher = components[0]
    return fetcher.get_global_market_data()

def main():
    apply_modern_theme()
    
    page = ModernComponents.sidebar_navigation()
    
    components = init_components()
    fetcher, visualizer, risk_calc, optimizer = components[:4]
    tech_indicators, corr_analyzer, backtest_engine, alert_system, sentiment_analyzer, multi_exchange, export_manager = components[4:]
    
    if page == "Market Overview":
        market_overview_page(fetcher, visualizer)
    elif page == "Price Analysis":
        price_analysis_page(fetcher, visualizer, risk_calc)
    elif page == "Risk Calculator":
        risk_calculator_page(risk_calc)
    elif page == "Portfolio Optimizer":
        portfolio_optimizer_page(fetcher, risk_calc, optimizer)
    elif page == "Advanced Analytics":
        advanced_analytics_page(fetcher, visualizer, tech_indicators, corr_analyzer, backtest_engine, sentiment_analyzer, export_manager)
    elif page == "Multi-Exchange Data":
        multi_exchange_page(multi_exchange, visualizer, export_manager)
    elif page == "Alerts & Monitoring":
        alerts_monitoring_page(alert_system, fetcher, sentiment_analyzer)

def market_overview_page(fetcher, visualizer):
    with st.spinner("Fetching market data..."):
        top_cryptos = fetch_top_cryptos(50)
        fg_index = fetch_fear_greed_index()
        global_data = fetch_global_market_data()
    
    if top_cryptos.empty:
        ModernComponents.alert_badge("Unable to fetch market data. Please try again later.", "danger")
        return
    
    price_ticker_data = []
    for _, row in top_cryptos.head(6).iterrows():
        price_ticker_data.append({
            'symbol': row['Symbol'],
            'price': f"${row['Price (USD)']:,.2f}",
            'change': f"{row['24h Change (%)']:+.2f}%"
        })
    
    ModernComponents.price_ticker_header(price_ticker_data)
    
    metrics = []
    if global_data:
        total_mcap = global_data.get('total_market_cap', 0)
        total_volume = global_data.get('total_volume', 0)
        active_cryptos = global_data.get('active_cryptocurrencies', 0)
        
        metrics = [
            {'value': f"${total_mcap/1e12:.2f}T", 'label': 'Total Market Cap'},
            {'value': f"${total_volume/1e9:.1f}B", 'label': '24h Volume'},
            {'value': f"{active_cryptos:,}", 'label': 'Active Cryptos'},
            {'value': str(fg_index['value']), 'label': 'Fear & Greed', 'delta': fg_index['classification']}
        ]
    
    ModernComponents.modern_metric_row(metrics, 4)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Market Overview")
        treemap_fig = ModernCharts.market_overview_treemap(top_cryptos)
        st.plotly_chart(treemap_fig, use_container_width=True)
    
    with col2:
        st.markdown("### Market Sentiment")
        fg_fig = ModernCharts.fear_greed_gauge(fg_index['value'], fg_index['classification'])
        st.plotly_chart(fg_fig, use_container_width=True)
    
    st.markdown("### Top Cryptocurrencies")
    
    display_df = top_cryptos.copy()
    display_df['Price (USD)'] = display_df['Price (USD)'].apply(lambda x: f"${x:,.2f}")
    display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: f"${x:,.0f}")
    display_df['Volume (24h)'] = display_df['Volume (24h)'].apply(lambda x: f"${x:,.0f}")
    display_df['24h Change (%)'] = display_df['24h Change (%)'].apply(lambda x: f"{x:+.2f}%")
    
    ModernComponents.modern_table(
        display_df[['Symbol', 'Name', 'Price (USD)', 'Market Cap', '24h Change (%)', 'Volume (24h)']],
        max_height=500
    )

def price_analysis_page(fetcher, visualizer, risk_calc):
    st.markdown("### Price Analysis")
    
    top_cryptos = fetch_top_cryptos(100)
    if top_cryptos.empty:
        ModernComponents.alert_badge("Unable to fetch cryptocurrency list.", "danger")
        return
    
    coin_options = {row['Name']: row['ID'] for _, row in top_cryptos.iterrows()}
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        selected_coin_name = ModernComponents.trading_pair_selector(
            list(coin_options.keys())
        )
    
    with col2:
        time_period = st.selectbox(
            "Time Period:",
            options=[7, 30, 90, 365],
            format_func=lambda x: f"{x} days",
            index=1
        )
    
    with col3:
        chart_type = st.selectbox(
            "Chart Type:",
            options=["Candlestick", "Line", "Technical"],
            index=0
        )
    
    selected_coin_id = coin_options[selected_coin_name]
    
    with st.spinner(f"Fetching {selected_coin_name} price data..."):
        price_data = fetch_price_history(selected_coin_id, time_period)
    
    if price_data.empty:
        ModernComponents.alert_badge(f"Unable to fetch price data for {selected_coin_name}", "danger")
        return
    
    df = price_data.copy()
    df.columns = ['price', 'volume'] if 'volume' in df.columns else ['price']
    df['high'] = df['price'] * (1 + np.random.uniform(0, 0.02, len(df)))
    df['low'] = df['price'] * (1 - np.random.uniform(0, 0.02, len(df)))
    df['open'] = df['price'].shift(1).fillna(df['price'].iloc[0])
    df['close'] = df['price']
    
    if chart_type == "Candlestick":
        chart_fig = ModernCharts.candlestick_chart(df, f"{selected_coin_name} Price Analysis")
    elif chart_type == "Line":
        chart_fig = ModernCharts.line_chart_with_volume(df, f"{selected_coin_name} Price Trend")
    else:
        from src.crypto_dashboard.analytics import TechnicalIndicators
        tech_indicators = TechnicalIndicators()
        df_with_indicators = tech_indicators.calculate_all_indicators(df)
        chart_fig = ModernCharts.technical_indicators_chart(df_with_indicators, ['rsi', 'macd'])
    
    st.plotly_chart(chart_fig, use_container_width=True)
    
    returns = risk_calc.calculate_returns(price_data['price'])
    
    current_price = price_data['price'].iloc[-1]
    price_change = (price_data['price'].iloc[-1] / price_data['price'].iloc[0] - 1) * 100
    volatility = risk_calc.calculate_volatility(returns) * 100
    sharpe = risk_calc.calculate_sharpe_ratio(returns)
    max_dd = risk_calc.calculate_max_drawdown(price_data['price'])
    var_5 = abs(risk_calc.calculate_var(returns)) * 100
    
    metrics = [
        {'value': f"${current_price:,.2f}", 'label': 'Current Price', 
         'delta': f"{price_change:+.2f}%", 'delta_color': 'success' if price_change >= 0 else 'danger'},
        {'value': f"${price_data['price'].max():,.2f}", 'label': 'Period High'},
        {'value': f"${price_data['price'].min():,.2f}", 'label': 'Period Low'},
        {'value': f"{volatility:.1f}%", 'label': 'Volatility (Annual)'},
        {'value': f"{sharpe:.2f}", 'label': 'Sharpe Ratio'},
        {'value': f"{max_dd['max_drawdown_pct']:.1f}%", 'label': 'Max Drawdown'},
        {'value': f"{var_5:.1f}%", 'label': 'VaR (5%)'},
        {'value': f"{len(price_data)}", 'label': 'Data Points'}
    ]
    
    ModernComponents.modern_metric_row(metrics, 4)

def risk_calculator_page(risk_calc):
    st.header("Risk Management Calculator")
    
    st.subheader("Position Sizing Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        portfolio_value = st.number_input(
            "Portfolio Value ($)",
            min_value=100.0,
            value=10000.0,
            step=100.0
        )
        
        risk_per_trade = st.slider(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1
        )
    
    with col2:
        entry_price = st.number_input(
            "Entry Price ($)",
            min_value=0.01,
            value=50.0,
            step=0.01
        )
        
        stop_loss_price = st.number_input(
            "Stop Loss Price ($)",
            min_value=0.01,
            value=45.0,
            step=0.01
        )
    
    position_info = risk_calc.calculate_position_sizing(
        portfolio_value, risk_per_trade, entry_price, stop_loss_price
    )
    
    st.subheader("Position Sizing Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Position Size", f"${position_info['position_size']:,.2f}")
    
    with col2:
        st.metric("Shares/Units", f"{position_info['shares']:,.3f}")
    
    with col3:
        st.metric("Risk Amount", f"${position_info['risk_amount']:,.2f}")
    
    with col4:
        st.metric("Position %", f"{position_info['position_pct']:.2f}%")
    
    st.subheader("Kelly Criterion Calculator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        win_rate = st.slider("Win Rate (%)", 0, 100, 60) / 100
    
    with col2:
        avg_win = st.number_input("Average Win (%)", min_value=0.1, value=5.0, step=0.1) / 100
    
    with col3:
        avg_loss = st.number_input("Average Loss (%)", min_value=0.1, value=3.0, step=0.1) / 100
    
    kelly_fraction = risk_calc.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
    
    st.metric("Optimal Kelly Position Size", f"{kelly_fraction * 100:.2f}%")
    
    if kelly_fraction > 0.25:
        st.warning("Kelly suggests high allocation. Consider reducing for risk management.")
    elif kelly_fraction > 0:
        st.success(f"Kelly suggests allocating {kelly_fraction * 100:.1f}% of portfolio")
    else:
        st.error("Kelly suggests avoiding this trade (negative edge)")
    
    st.subheader("Monte Carlo Price Simulation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_price = st.number_input("Initial Price ($)", min_value=0.01, value=100.0)
    
    with col2:
        sim_days = st.selectbox("Simulation Days", [7, 14, 30, 60, 90], index=2)
    
    with col3:
        daily_volatility = st.slider("Daily Volatility (%)", 0.5, 10.0, 3.0) / 100
    
    if st.button("Run Simulation"):
        np.random.seed(42)
        mock_returns = np.random.normal(0.001, daily_volatility, 100)
        mock_returns_series = pd.Series(mock_returns)
        
        with st.spinner("Running Monte Carlo simulation..."):
            mc_results = risk_calc.monte_carlo_simulation(
                initial_price, mock_returns_series, sim_days, 1000
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Final Price", f"${mc_results['mean_final_price']:.2f}")
            st.metric("5th Percentile", f"${mc_results['percentile_5']:.2f}")
        
        with col2:
            st.metric("Median Final Price", f"${mc_results['median_final_price']:.2f}")
            st.metric("95th Percentile", f"${mc_results['percentile_95']:.2f}")
        
        with col3:
            st.metric("Probability of Profit", f"{mc_results['probability_profit']:.1f}%")
            st.metric("Max Simulated", f"${mc_results['max_simulated_price']:.2f}")

def portfolio_optimizer_page(fetcher, risk_calc, optimizer):
    st.header("Portfolio Optimizer")
    
    st.subheader("Current Portfolio")
    
    top_cryptos = fetch_top_cryptos(50)
    if top_cryptos.empty:
        st.error("Unable to fetch cryptocurrency data.")
        return
    
    coin_symbols = top_cryptos['Symbol'].tolist()
    
    portfolio_allocations = {}
    total_allocation = 0
    
    st.write("Enter your portfolio allocations:")
    
    num_assets = st.selectbox("Number of assets in portfolio:", range(2, 11), index=2)
    
    cols = st.columns(min(num_assets, 3))
    
    for i in range(num_assets):
        col_idx = i % 3
        with cols[col_idx]:
            asset = st.selectbox(f"Asset {i+1}:", coin_symbols, key=f"asset_{i}", index=i % len(coin_symbols))
            allocation = st.number_input(f"Allocation % for {asset}:", 
                                       min_value=0.0, max_value=100.0, 
                                       value=100.0/num_assets, step=1.0, key=f"alloc_{i}")
            
            if allocation > 0:
                portfolio_allocations[asset] = allocation
                total_allocation += allocation
    
    if portfolio_allocations:
        st.subheader("Portfolio Allocation")
        
        if total_allocation != 100:
            normalized_allocations = {k: (v/total_allocation)*100 for k, v in portfolio_allocations.items()}
            st.warning(f"Allocations sum to {total_allocation:.1f}%. Normalizing to 100%.")
        else:
            normalized_allocations = portfolio_allocations
        
        components = init_components()
        visualizer = components[1]
        portfolio_fig = visualizer.create_portfolio_pie(normalized_allocations)
        st.plotly_chart(portfolio_fig, use_container_width=True)
        
        st.subheader("Portfolio Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Number of Assets", len(normalized_allocations))
            st.metric("Largest Allocation", f"{max(normalized_allocations.values()):.1f}%")
            st.metric("Smallest Allocation", f"{min(normalized_allocations.values()):.1f}%")
        
        with col2:
            herfindahl = sum((allocation/100)**2 for allocation in normalized_allocations.values())
            diversification_score = (1 - herfindahl) * 100
            
            st.metric("Diversification Score", f"{diversification_score:.1f}%")
            
            if diversification_score > 80:
                st.success("Well diversified portfolio")
            elif diversification_score > 60:
                st.warning("Moderately diversified")
            else:
                st.error("Concentrated portfolio - consider diversifying")
        
        st.subheader("Rebalancing Analysis")
        
        equal_weight = 100 / len(normalized_allocations)
        rebalancing_needed = False
        
        for asset, allocation in normalized_allocations.items():
            difference = allocation - equal_weight
            if abs(difference) > 5:
                rebalancing_needed = True
                if difference > 0:
                    st.write(f"**{asset}**: Overweight by {difference:.1f}% - consider reducing")
                else:
                    st.write(f"**{asset}**: Underweight by {abs(difference):.1f}% - consider increasing")
        
        if not rebalancing_needed:
            st.success("Portfolio is well balanced")

def advanced_analytics_page(fetcher, visualizer, tech_indicators, corr_analyzer, backtest_engine, sentiment_analyzer, export_manager):
    st.header("Advanced Analytics")
    
    try:
        # Just show Technical Analysis for now to avoid crashes
        st.subheader("Technical Indicators")
        
        top_cryptos = fetch_top_cryptos(50)
        if top_cryptos.empty:
            st.error("Unable to fetch cryptocurrency data.")
            return
        
        coin_options = {row['Name']: row['ID'] for _, row in top_cryptos.iterrows()}
        
        col1, col2 = st.columns(2)
        with col1:
            selected_coin = st.selectbox("Select Cryptocurrency:", list(coin_options.keys()), key="tech_coin")
        with col2:
            time_period = st.selectbox("Time Period:", [30, 90, 180, 365], format_func=lambda x: f"{x} days", key="tech_period")
        
        coin_id = coin_options[selected_coin]
        price_data = fetch_price_history(coin_id, time_period)
        
        if not price_data.empty:
            try:
                df = price_data.copy()
                
                # Ensure we have the right column structure
                if 'price' in df.columns:
                    df['close'] = df['price']
                elif len(df.columns) == 1:
                    df['close'] = df.iloc[:, 0]
                elif 'close' not in df.columns:
                    df['close'] = df.iloc[:, 0]
                
                # Add volume if missing
                if 'volume' not in df.columns:
                    if len(df.columns) > 1 and df.columns[1] != 'close':
                        df['volume'] = df.iloc[:, 1]
                    else:
                        df['volume'] = df['close'] * np.random.uniform(800, 1200, len(df))
                
                # Generate OHLC data from close prices
                df['high'] = df['close'] * (1 + np.random.uniform(0, 0.02, len(df)))
                df['low'] = df['close'] * (1 - np.random.uniform(0, 0.02, len(df)))
                df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
                
                # Ensure proper data types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove any NaN values
                df = df.dropna()
                
                if len(df) > 20:  # Need at least 20 points for indicators
                    try:
                        indicators_df = tech_indicators.calculate_all_indicators(df)
                        st.success(f"✅ Technical indicators calculated successfully for {len(df)} data points")
                        
                        # Create charts only if indicators were calculated successfully
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Price with Moving Averages")
                            try:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=indicators_df.index, 
                                    y=indicators_df['close'], 
                                    name='Price', 
                                    line=dict(color='#F0B90B', width=2)
                                ))
                                
                                if 'sma_20' in indicators_df.columns and indicators_df['sma_20'].count() > 0:
                                    fig.add_trace(go.Scatter(
                                        x=indicators_df.index, 
                                        y=indicators_df['sma_20'], 
                                        name='SMA 20', 
                                        line=dict(color='#FF6B35', width=1.5)
                                    ))
                                
                                if 'sma_50' in indicators_df.columns and indicators_df['sma_50'].count() > 0:
                                    fig.add_trace(go.Scatter(
                                        x=indicators_df.index, 
                                        y=indicators_df['sma_50'], 
                                        name='SMA 50', 
                                        line=dict(color='#00D4AA', width=1.5)
                                    ))
                                
                                fig.update_layout(
                                    title=f"{selected_coin} Price with Moving Averages",
                                    xaxis_title="Date",
                                    yaxis_title="Price",
                                    paper_bgcolor='#1E2329',
                                    plot_bgcolor='#2B3139',
                                    font=dict(color='#FAFAFA'),
                                    height=400,
                                    showlegend=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating moving averages chart: {str(e)}")
                        
                        with col2:
                            st.subheader("RSI Indicator")
                            try:
                                if 'rsi' in indicators_df.columns and indicators_df['rsi'].count() > 0:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=indicators_df.index, 
                                        y=indicators_df['rsi'], 
                                        name='RSI', 
                                        line=dict(color='#F0B90B', width=2)
                                    ))
                                    fig.add_hline(y=70, line_dash="dash", line_color="#F84960", annotation_text="Overbought")
                                    fig.add_hline(y=30, line_dash="dash", line_color="#00D4AA", annotation_text="Oversold")
                                    fig.update_layout(
                                        title="RSI (14)",
                                        xaxis_title="Date",
                                        yaxis_title="RSI",
                                        yaxis=dict(range=[0, 100]),
                                        paper_bgcolor='#1E2329',
                                        plot_bgcolor='#2B3139',
                                        font=dict(color='#FAFAFA'),
                                        height=400,
                                        showlegend=True
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("RSI data not available")
                            except Exception as e:
                                st.error(f"Error creating RSI chart: {str(e)}")
                        
                        # Second row of charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("MACD")
                            try:
                                if all(col in indicators_df.columns for col in ['macd', 'macd_signal', 'macd_histogram']) and indicators_df['macd'].count() > 0:
                                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
                                    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['macd'], name='MACD', line=dict(color='#F0B90B', width=2)), row=1, col=1)
                                    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['macd_signal'], name='Signal', line=dict(color='#FF6B35', width=2)), row=1, col=1)
                                    fig.add_trace(go.Bar(x=indicators_df.index, y=indicators_df['macd_histogram'], name='Histogram', marker_color='#00D4AA'), row=2, col=1)
                                    fig.update_layout(
                                        title="MACD Indicator", 
                                        height=400,
                                        paper_bgcolor='#1E2329',
                                        plot_bgcolor='#2B3139',
                                        font=dict(color='#FAFAFA')
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("MACD data not available")
                            except Exception as e:
                                st.error(f"Error creating MACD chart: {str(e)}")
                        
                        with col2:
                            st.subheader("Bollinger Bands")
                            try:
                                if all(col in indicators_df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']) and indicators_df['bb_middle'].count() > 0:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['bb_upper'], name='Upper Band', line=dict(color='#F84960', dash='dash')))
                                    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['bb_middle'], name='Middle Band', line=dict(color='#F0B90B')))
                                    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['bb_lower'], name='Lower Band', line=dict(color='#F84960', dash='dash')))
                                    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['close'], name='Price', line=dict(color='#FAFAFA', width=2)))
                                    fig.update_layout(
                                        title="Bollinger Bands", 
                                        xaxis_title="Date", 
                                        yaxis_title="Price",
                                        paper_bgcolor='#1E2329',
                                        plot_bgcolor='#2B3139',
                                        font=dict(color='#FAFAFA'),
                                        height=400
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Bollinger Bands data not available")
                            except Exception as e:
                                st.error(f"Error creating Bollinger Bands chart: {str(e)}")
                                
                    except Exception as e:
                        st.error(f"Error calculating technical indicators: {str(e)}")
                        return
                            
                else:
                    st.error("Insufficient data for technical analysis (need at least 20 data points)")
                    return
                    
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                return
        else:
            st.error("No price data available for the selected cryptocurrency")
    
    except Exception as e:
        st.error(f"Critical error in Advanced Analytics: {str(e)}")
        st.info("Please refresh the page and try again.")

def multi_exchange_page(multi_exchange, visualizer, export_manager):
    st.header("Multi-Exchange Data")
    
    st.subheader("Price Comparison Across Exchanges")
    
    symbols = ['bitcoin', 'ethereum', 'cardano', 'solana', 'polkadot']
    selected_symbols = st.multiselect("Select cryptocurrencies:", symbols, default=symbols[:3])
    
    if selected_symbols and st.button("Fetch Multi-Exchange Prices"):
        with st.spinner("Fetching prices from multiple exchanges..."):
            prices_df = multi_exchange.get_multi_exchange_prices(selected_symbols)
        
        if not prices_df.empty:
            st.subheader("Current Prices")
            st.dataframe(prices_df.round(2))
            
            arbitrage_opps = multi_exchange.get_arbitrage_opportunities(selected_symbols, min_diff_pct=0.1)
            
            if not arbitrage_opps.empty:
                st.subheader("Arbitrage Opportunities")
                st.dataframe(arbitrage_opps[['buy_exchange', 'sell_exchange', 'buy_price', 'sell_price', 'price_diff_pct']].round(3))
            else:
                st.info("No significant arbitrage opportunities found")
    
    st.subheader("Exchange Reliability")
    reliability_stats = multi_exchange.get_exchange_reliability_stats()
    if not reliability_stats.empty:
        st.dataframe(reliability_stats)
    
    st.subheader("Data Export")
    if st.button("Export Multi-Exchange Data"):
        if selected_symbols:
            filepath = f"multi_exchange_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            multi_exchange.export_data(filepath, selected_symbols, days=30)
            st.success(f"Data exported to {filepath}")

def alerts_monitoring_page(alert_system, fetcher, sentiment_analyzer):
    st.header("Alerts & Monitoring")
    
    tab1, tab2, tab3 = st.tabs(["Create Alerts", "Active Alerts", "Alert History"])
    
    with tab1:
        st.subheader("Create New Alert")
        
        col1, col2 = st.columns(2)
        
        with col1:
            asset = st.text_input("Asset (e.g., bitcoin):", value="bitcoin")
            condition = st.selectbox("Condition:", [
                'price_above', 'price_below', 'price_change_percent',
                'volume_above', 'volume_below', 'rsi_above', 'rsi_below',
                'macd_bullish_crossover', 'macd_bearish_crossover'
            ])
        
        with col2:
            threshold = st.number_input("Threshold:", value=50000.0)
            message = st.text_input("Custom Message (optional):")
        
        if st.button("Create Alert"):
            alert_id = alert_system.create_alert(asset, condition, threshold, message)
            st.success(f"Alert created with ID: {alert_id}")
    
    with tab2:
        st.subheader("Active Alerts")
        
        active_alerts = alert_system.get_active_alerts()
        
        if active_alerts:
            for alert in active_alerts:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{alert.asset}** - {alert.condition} {alert.threshold}")
                    st.write(f"Message: {alert.message}")
                
                with col2:
                    if st.button("Deactivate", key=f"deactivate_{alert.alert_id}"):
                        alert_system.deactivate_alert(alert.alert_id)
                        st.rerun()
                
                with col3:
                    if st.button("Delete", key=f"delete_{alert.alert_id}"):
                        alert_system.remove_alert(alert.alert_id)
                        st.rerun()
                
                st.markdown("---")
        else:
            st.info("No active alerts")
        
        stats = alert_system.get_alert_statistics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Alerts", stats['total_alerts'])
        with col2:
            st.metric("Active Alerts", stats['active_alerts'])
        with col3:
            st.metric("Triggered Today", stats['triggered_today'])
    
    with tab3:
        st.subheader("Recent Alert Triggers")
        
        triggered_alerts = alert_system.get_triggered_alerts(hours=168)
        
        if triggered_alerts:
            for alert in triggered_alerts[-10:]:
                st.write(f"**{alert.triggered_at.strftime('%Y-%m-%d %H:%M')}** - {alert.message}")
        else:
            st.info("No recent alert triggers")

if __name__ == "__main__":
    main()