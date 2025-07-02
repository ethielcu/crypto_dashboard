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

st.set_page_config(
    page_title="Crypto Dashboard & Risk Calculator",
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
    st.title("Cryptocurrency Dashboard & Risk Calculator")
    st.markdown("---")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Market Overview", "Price Analysis", "Risk Calculator", "Portfolio Optimizer", 
         "Advanced Analytics", "Multi-Exchange Data", "Alerts & Monitoring"]
    )
    
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
    st.header("Market Overview")
    
    with st.spinner("Fetching market data..."):
        top_cryptos = fetch_top_cryptos(50)
        fg_index = fetch_fear_greed_index()
        global_data = fetch_global_market_data()
    
    if top_cryptos.empty:
        st.error("Unable to fetch market data. Please try again later.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if global_data:
            total_mcap = global_data.get('total_market_cap', 0)
            st.metric("Total Market Cap", f"${total_mcap:,.0f}")
        else:
            st.metric("Total Market Cap", "N/A")
    
    with col2:
        if global_data:
            total_volume = global_data.get('total_volume', 0)
            st.metric("24h Volume", f"${total_volume:,.0f}")
        else:
            st.metric("24h Volume", "N/A")
    
    with col3:
        if global_data:
            active_cryptos = global_data.get('active_cryptocurrencies', 0)
            st.metric("Active Cryptocurrencies", f"{active_cryptos:,}")
        else:
            st.metric("Active Cryptocurrencies", "N/A")
    
    with col4:
        st.metric("Fear & Greed Index", f"{fg_index['value']}", fg_index['classification'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Market Analysis")
        market_fig = visualizer.create_market_overview(top_cryptos)
        st.plotly_chart(market_fig, use_container_width=True)
    
    with col2:
        st.subheader("Market Sentiment")
        fg_fig = visualizer.create_fear_greed_gauge(fg_index['value'], fg_index['classification'])
        st.plotly_chart(fg_fig, use_container_width=True)
    
    st.subheader("Top Cryptocurrencies")
    
    display_df = top_cryptos.copy()
    display_df['Price (USD)'] = display_df['Price (USD)'].apply(lambda x: f"${x:,.2f}")
    display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: f"${x:,.0f}")
    display_df['Volume (24h)'] = display_df['Volume (24h)'].apply(lambda x: f"${x:,.0f}")
    display_df['24h Change (%)'] = display_df['24h Change (%)'].apply(lambda x: f"{x:+.2f}%")
    
    st.dataframe(
        display_df[['Symbol', 'Name', 'Price (USD)', 'Market Cap', '24h Change (%)', 'Volume (24h)']],
        use_container_width=True,
        height=400
    )

def price_analysis_page(fetcher, visualizer, risk_calc):
    st.header("Price Analysis")
    
    top_cryptos = fetch_top_cryptos(100)
    if top_cryptos.empty:
        st.error("Unable to fetch cryptocurrency list.")
        return
    
    coin_options = {row['Name']: row['ID'] for _, row in top_cryptos.iterrows()}
    
    col1, col2 = st.columns(2)
    with col1:
        selected_coin_name = st.selectbox(
            "Select Cryptocurrency:",
            options=list(coin_options.keys()),
            index=0
        )
    
    with col2:
        time_period = st.selectbox(
            "Time Period:",
            options=[7, 30, 90, 365],
            format_func=lambda x: f"{x} days",
            index=1
        )
    
    selected_coin_id = coin_options[selected_coin_name]
    
    with st.spinner(f"Fetching {selected_coin_name} price data..."):
        price_data = fetch_price_history(selected_coin_id, time_period)
    
    if price_data.empty:
        st.error(f"Unable to fetch price data for {selected_coin_name}")
        return
    
    st.subheader(f"{selected_coin_name} Price Chart")
    price_fig = visualizer.create_price_chart(price_data, f"{selected_coin_name} Price")
    st.plotly_chart(price_fig, use_container_width=True)
    
    returns = risk_calc.calculate_returns(price_data['price'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Statistics")
        current_price = price_data['price'].iloc[-1]
        price_change = (price_data['price'].iloc[-1] / price_data['price'].iloc[0] - 1) * 100
        
        st.metric("Current Price", f"${current_price:,.2f}")
        st.metric("Period Return", f"{price_change:+.2f}%")
        st.metric("Highest Price", f"${price_data['price'].max():,.2f}")
        st.metric("Lowest Price", f"${price_data['price'].min():,.2f}")
    
    with col2:
        st.subheader("Risk Metrics")
        
        volatility = risk_calc.calculate_volatility(returns) * 100
        sharpe = risk_calc.calculate_sharpe_ratio(returns)
        max_dd = risk_calc.calculate_max_drawdown(price_data['price'])
        var_5 = abs(risk_calc.calculate_var(returns)) * 100
        
        st.metric("Volatility (Annual)", f"{volatility:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
        st.metric("Max Drawdown", f"{max_dd['max_drawdown_pct']:.2f}%")
        st.metric("VaR (5%)", f"{var_5:.2f}%")
    
    risk_metrics = {
        'Volatility': volatility / 100,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd['max_drawdown'],
        'VaR (5%)': abs(risk_calc.calculate_var(returns))
    }
    
    risk_fig = visualizer.create_risk_metrics_chart(risk_metrics)
    st.plotly_chart(risk_fig, use_container_width=True)

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
        
        _, visualizer, _, _ = init_components()
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
    
    tab1, tab2, tab3, tab4 = st.tabs(["Technical Analysis", "Correlation Analysis", "Backtesting", "Sentiment Analysis"])
    
    with tab1:
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
            df = price_data.copy()
            df.columns = ['close', 'volume']
            df['high'] = df['close'] * (1 + np.random.uniform(0, 0.02, len(df)))
            df['low'] = df['close'] * (1 - np.random.uniform(0, 0.02, len(df)))
            df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
            
            indicators_df = tech_indicators.calculate_all_indicators(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Price with Moving Averages")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['close'], name='Price', line=dict(color='blue')))
                if 'sma_20' in indicators_df:
                    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['sma_20'], name='SMA 20', line=dict(color='orange')))
                if 'sma_50' in indicators_df:
                    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['sma_50'], name='SMA 50', line=dict(color='red')))
                
                fig.update_layout(title=f"{selected_coin} Price with Moving Averages", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("RSI Indicator")
                if 'rsi' in indicators_df:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['rsi'], name='RSI', line=dict(color='purple')))
                    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig.update_layout(title="RSI (14)", xaxis_title="Date", yaxis_title="RSI", yaxis=dict(range=[0, 100]))
                    st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("MACD")
                if all(col in indicators_df for col in ['macd', 'macd_signal', 'macd_histogram']):
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
                    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['macd'], name='MACD', line=dict(color='blue')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['macd_signal'], name='Signal', line=dict(color='red')), row=1, col=1)
                    fig.add_trace(go.Bar(x=indicators_df.index, y=indicators_df['macd_histogram'], name='Histogram'), row=2, col=1)
                    fig.update_layout(title="MACD Indicator", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Bollinger Bands")
                if all(col in indicators_df for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['bb_upper'], name='Upper Band', line=dict(color='red', dash='dash')))
                    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['bb_middle'], name='Middle Band', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['bb_lower'], name='Lower Band', line=dict(color='red', dash='dash')))
                    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['close'], name='Price', line=dict(color='black')))
                    fig.update_layout(title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Correlation Analysis")
        
        selected_assets = st.multiselect(
            "Select assets for correlation analysis:",
            options=list(coin_options.keys())[:10],
            default=list(coin_options.keys())[:5],
            key="corr_assets"
        )
        
        if len(selected_assets) >= 2:
            price_data_dict = {}
            
            with st.spinner("Fetching price data for correlation analysis..."):
                for asset in selected_assets:
                    coin_id = coin_options[asset]
                    data = fetch_price_history(coin_id, 90)
                    if not data.empty:
                        price_data_dict[asset] = data['price']
            
            if price_data_dict:
                price_df = pd.DataFrame(price_data_dict)
                returns_df = corr_analyzer.calculate_returns(price_df)
                corr_matrix = corr_analyzer.calculate_correlation_matrix(returns_df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Correlation Matrix")
                    corr_fig = corr_analyzer.create_correlation_heatmap(corr_matrix)
                    st.plotly_chart(corr_fig, use_container_width=True)
                
                with col2:
                    st.subheader("Correlation Statistics")
                    st.dataframe(corr_matrix.round(3))
                
                clusters = corr_analyzer.identify_correlation_clusters(threshold=0.7)
                if clusters:
                    st.subheader("High Correlation Clusters")
                    for cluster_name, assets in clusters.items():
                        st.write(f"**{cluster_name}**: {', '.join(assets)}")
    
    with tab3:
        st.subheader("Strategy Backtesting")
        
        col1, col2 = st.columns(2)
        with col1:
            backtest_coin = st.selectbox("Select Asset:", list(coin_options.keys()), key="backtest_coin")
            strategy_type = st.selectbox("Strategy Type:", ["Simple Moving Average", "RSI", "MACD"])
        
        with col2:
            backtest_period = st.selectbox("Backtest Period:", [90, 180, 365], format_func=lambda x: f"{x} days")
            initial_capital = st.number_input("Initial Capital ($):", min_value=1000, value=10000, step=1000)
        
        if st.button("Run Backtest"):
            coin_id = coin_options[backtest_coin]
            price_data = fetch_price_history(coin_id, backtest_period)
            
            if not price_data.empty:
                df = price_data.copy()
                df.columns = ['close', 'volume']
                df['high'] = df['close'] * (1 + np.random.uniform(0, 0.02, len(df)))
                df['low'] = df['close'] * (1 - np.random.uniform(0, 0.02, len(df)))
                df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
                
                if strategy_type == "Simple Moving Average":
                    strategy = SimpleMovingAverageStrategy(20, 50)
                elif strategy_type == "RSI":
                    strategy = RSIStrategy()
                else:
                    strategy = MACDStrategy()
                
                backtest_engine.initial_capital = initial_capital
                result = backtest_engine.run_backtest(df, strategy)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Performance Metrics")
                    st.metric("Total Return", f"{result.metrics['total_return']:.2%}")
                    st.metric("Annual Return", f"{result.metrics['annual_return']:.2%}")
                    st.metric("Sharpe Ratio", f"{result.metrics['sharpe_ratio']:.3f}")
                    st.metric("Max Drawdown", f"{result.metrics['max_drawdown']:.2%}")
                
                with col2:
                    st.metric("Win Rate", f"{result.metrics['win_rate']:.2%}")
                    st.metric("Total Trades", f"{result.metrics['total_trades']:.0f}")
                    st.metric("Profit Factor", f"{result.metrics['profit_factor']:.2f}")
                    st.metric("Buy & Hold Return", f"{result.metrics['buy_hold_return']:.2%}")
                
                performance_fig = backtest_engine.create_performance_chart(result, df)
                st.plotly_chart(performance_fig, use_container_width=True)
    
    with tab4:
        st.subheader("Market Sentiment Analysis")
        
        fg_data = sentiment_analyzer.get_fear_greed_index()
        fg_history = sentiment_analyzer.get_fear_greed_history(30)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fg_gauge = sentiment_analyzer.create_fear_greed_gauge(fg_data['value'], fg_data['classification'])
            st.plotly_chart(fg_gauge, use_container_width=True)
        
        with col2:
            fg_history_chart = sentiment_analyzer.create_fear_greed_history_chart(fg_history)
            st.plotly_chart(fg_history_chart, use_container_width=True)
        
        sentiment_trend = sentiment_analyzer.analyze_sentiment_trend(fg_history)
        signals = sentiment_analyzer.get_sentiment_signals(fg_data['value'])
        
        st.subheader("Sentiment Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Sentiment", fg_data['classification'])
            st.metric("Trend", sentiment_trend['trend'].title())
            st.metric("Volatility", f"{sentiment_trend['volatility']:.1f}")
        
        with col2:
            st.subheader("Trading Signals")
            for signal in signals:
                st.write(f"• {signal}")

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