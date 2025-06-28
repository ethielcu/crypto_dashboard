import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from src.crypto_dashboard.data.market_data import MarketDataFetcher
from src.crypto_dashboard.visualization.visualizations import CryptoVisualizer
from src.crypto_dashboard.risk.risk_calculator import RiskCalculator, PortfolioOptimizer

st.set_page_config(
    page_title="Crypto Dashboard & Risk Calculator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def init_components():
    return MarketDataFetcher(), CryptoVisualizer(), RiskCalculator(), PortfolioOptimizer()

@st.cache_data(ttl=300)
def fetch_top_cryptos(limit=50):
    fetcher, _, _, _ = init_components()
    return fetcher.get_top_cryptos(limit)

@st.cache_data(ttl=600)
def fetch_price_history(coin_id, days):
    fetcher, _, _, _ = init_components()
    return fetcher.get_price_history(coin_id, days)

@st.cache_data(ttl=300)
def fetch_fear_greed_index():
    fetcher, _, _, _ = init_components()
    return fetcher.get_fear_greed_index()

@st.cache_data(ttl=600)
def fetch_global_market_data():
    fetcher, _, _, _ = init_components()
    return fetcher.get_global_market_data()

def main():
    st.title("Cryptocurrency Dashboard & Risk Calculator")
    st.markdown("---")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Market Overview", "Price Analysis", "Risk Calculator", "Portfolio Optimizer"]
    )
    
    fetcher, visualizer, risk_calc, optimizer = init_components()
    
    if page == "Market Overview":
        market_overview_page(fetcher, visualizer)
    elif page == "Price Analysis":
        price_analysis_page(fetcher, visualizer, risk_calc)
    elif page == "Risk Calculator":
        risk_calculator_page(risk_calc)
    elif page == "Portfolio Optimizer":
        portfolio_optimizer_page(fetcher, risk_calc, optimizer)

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

if __name__ == "__main__":
    main()