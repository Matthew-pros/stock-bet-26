import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from scipy.stats import norm
import numpy as np
from io import StringIO
import concurrent.futures

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quant Analyzer", layout="wide")

# Dark theme CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #262730;
        color: white;
        border: 1px solid #4e4f57;
    }
    .stButton > button:hover {
        border-color: #3b82f6;
        color: #3b82f6;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    /* Highlight the dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #3b82f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 2. DATA CONSTANTS
# -----------------------------------------------------------------------------
sector_multiples = {
    'Technology': {'current_pe': 64.15, 'forward_pe': 86.40, 'pb': 11.12, 'ps': 14.26, 'evebitda': 34.48, 'peg': 1.5, 'growth': 0.15, 'roe': 0.20},
    'Financial Services': {'current_pe': 35.16, 'forward_pe': 20.64, 'pb': 2.11, 'ps': 5.14, 'evebitda': 62.82, 'peg': 1.29, 'growth': 0.1537, 'roe': 0.1311},
    'Real Estate': {'current_pe': 44.63, 'forward_pe': 41.45, 'pb': 2.01, 'ps': 6.02, 'evebitda': 20.33, 'peg': 4.75, 'growth': 0.05, 'roe': 0.045},
    'Consumer Cyclical': {'current_pe': 28.81, 'forward_pe': 22.74, 'pb': 8.43, 'ps': 1.94, 'evebitda': 18.21, 'peg': 10.52, 'growth': 0.08, 'roe': 0.12},
    'Consumer Defensive': {'current_pe': 23.97, 'forward_pe': 22.81, 'pb': 2.18, 'ps': 1.35, 'evebitda': 11.17, 'peg': 2.04, 'growth': 0.06, 'roe': 0.10},
    'Healthcare': {'current_pe': 129.64, 'forward_pe': 18.72, 'pb': 5.70, 'ps': 4.84, 'evebitda': 15.37, 'peg': 2.61, 'growth': 0.12, 'roe': 0.15},
    'Utilities': {'current_pe': 19.19, 'forward_pe': 16.47, 'pb': 1.82, 'ps': 2.97, 'evebitda': 13.44, 'peg': 3.28, 'growth': 0.04, 'roe': 0.08},
    'Communication Services': {'current_pe': 74.81, 'forward_pe': 46.36, 'pb': 1.62, 'ps': 1.30, 'evebitda': 6.62, 'peg': 3.10, 'growth': 0.10, 'roe': 0.09},
    'Energy': {'current_pe': 9.09, 'forward_pe': 14.75, 'pb': 1.66, 'ps': 1.39, 'evebitda': 6.70, 'peg': 4.15, 'growth': 0.07, 'roe': 0.14},
    'Industrials': {'current_pe': 43.07, 'forward_pe': 21.97, 'pb': 4.27, 'ps': 2.87, 'evebitda': 15.35, 'peg': 1.92, 'growth': 0.09, 'roe': 0.13},
    'Basic Materials': {'current_pe': 15.75, 'forward_pe': 14.53, 'pb': 1.61, 'ps': 0.70, 'evebitda': 7.97, 'peg': 2.55, 'growth': 0.05, 'roe': 0.11},
    'Other': {'current_pe': 22.0, 'forward_pe': 18.0, 'pb': 3.0, 'ps': 2.76, 'evebitda': 11.0, 'peg': 2.0, 'growth': 0.1, 'roe': 0.12}
}

sector_mapping = {
    'Technology': 'Technology', 'Financial Services': 'Financial Services', 'Real Estate': 'Real Estate',
    'Consumer Cyclical': 'Consumer Cyclical', 'Consumer Defensive': 'Consumer Defensive', 'Healthcare': 'Healthcare',
    'Utilities': 'Utilities', 'Communication Services': 'Communication Services', 'Energy': 'Energy',
    'Industrials': 'Industrials', 'Basic Materials': 'Basic Materials', 'Other': 'Other'
}

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS (Data Fetching)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get('https://www.slickcharts.com/sp500', headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        tickers = [row.find_all('td')[2].find('a').text for row in soup.find('table', class_='table').find_all('tr')[1:]]
        return list(set(tickers))
    except Exception:
        # Fallback
        return ['AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','META','BRK.B','LLY','V']

@st.cache_data(ttl=86400)
def get_nasdaq_tickers():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get('https://www.slickcharts.com/nasdaq100', headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        tickers = [row.find_all('td')[2].find('a').text for row in soup.find('table', class_='table').find_all('tr')[1:]]
        return list(set(tickers))
    except Exception:
        return ['AAPL','MSFT','AMZN','NVDA','META','AVGO','GOOGL','COST','TSLA','AMD']

@st.cache_data(ttl=86400)
def get_hang_seng_tickers():
    # Simplified manual list for reliability if scrape fails
    return ['0005.HK','0700.HK','0939.HK','0941.HK','1299.HK','9988.HK','3690.HK','0388.HK','0001.HK']

@st.cache_data(ttl=86400)
def get_nikkei_tickers():
    # Simplified manual list
    return ['7203.T','6758.T','9984.T','8035.T','9983.T','6861.T','4063.T','6098.T','6501.T']

@st.cache_data(ttl=86400)
def get_all_tickers():
    return list(set(get_sp500_tickers() + get_nasdaq_tickers()))

def calculate_fair_price(ticker):
    """Main logic to calculate valuation for a single ticker."""
    try:
        stock = yf.Ticker(ticker)
        # Fast info retrieval
        info = stock.info
        
        if not info or 'regularMarketPrice' not in info:
            return None

        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        if current_price == 0: return None

        # Extract metrics safely
        sector = info.get('sector', 'Other')
        mapped_sector = sector_mapping.get(sector, 'Other')
        multiples = sector_multiples.get(mapped_sector, sector_multiples['Other'])
        
        eps = info.get('trailingEps', 0) or 0
        forward_eps = info.get('forwardEps', 0) or 0
        bvps = info.get('bookValue', 0) or 0
        revenue_ps = info.get('revenuePerShare', 0) or 0
        shares = info.get('sharesOutstanding', 1)
        ebitda = info.get('ebitda', 0) or 0 # yfinance key might be lower case in some versions
        
        # Fallback for debt calculation
        total_debt = info.get('totalDebt', 0) or 0
        total_cash = info.get('totalCash', 0) or 0
        net_debt = total_debt - total_cash
        
        # Valuation Models
        fair_current_pe = eps * multiples['current_pe'] if eps > 0 else 0
        fair_forward_pe = forward_eps * multiples['forward_pe'] if forward_eps > 0 else 0
        fair_pb = bvps * multiples['pb'] if bvps > 0 else 0
        fair_ps = revenue_ps * multiples['ps']
        
        fair_evebitda = 0
        if shares > 0 and ebitda > 0:
            fair_evebitda = (ebitda * multiples['evebitda'] - net_debt) / shares
            if fair_evebitda < 0: fair_evebitda = 0

        # DDM Approximation
        roe = info.get('returnOnEquity', 0) or 0.1
        payout = info.get('payoutRatio', 0) or 0
        beta = info.get('beta', 1)
        cost_equity = 0.03 + beta * 0.06
        growth = (1 - payout) * roe
        if growth <= 0: growth = multiples['growth']
        
        if cost_equity > growth > 0 and eps > 0:
            fair_ddm = (eps * (1 - payout)) / (cost_equity - growth)
        else:
            fair_ddm = (fair_current_pe + fair_forward_pe) / 2

        # Weighted Average Fair Value
        weights = [0.3, 0.25, 0.15, 0.15, 0.15, 0.05] # PE, FPE, PB, PS, EV, DDM
        components = [fair_current_pe, fair_forward_pe, fair_pb, fair_ps, fair_evebitda, fair_ddm]
        
        # Only use non-zero components
        valid_components = []
        valid_weights = []
        for c, w in zip(components, weights):
            if c > 0:
                valid_components.append(c)
                valid_weights.append(w)
        
        if not valid_components:
            return None

        fair_price = np.average(valid_components, weights=valid_weights)
        
        # PEG Adjustment
        peg = info.get('pegRatio', 0)
        if peg > 0 and multiples['peg'] > 0:
            # Mild adjustment based on PEG
            fair_price = fair_price / (multiples['peg'] if multiples['peg'] > 1 else 1)

        # Final Sanity Check & Margin of Safety
        fair_price = fair_price * 0.95 # 5% Margin of safety
        
        diff_nominal = fair_price - current_price
        diff_percent = ((fair_price - current_price) / current_price) * 100

        # Basic Probability Logic (Simplified for speed)
        beat_prob = 3
        if info.get('earningsGrowth', 0) > 0.1: beat_prob += 1
        if info.get('shortPercentOfFloat', 0) < 0.05: beat_prob += 1
        
        miss_prob = 2
        if info.get('earningsGrowth', 0) < 0: miss_prob += 1

        return {
            'Ticker': ticker,
            'Price': round(current_price, 2),
            'Fair Value': round(fair_price, 2),
            'Diff %': round(diff_percent, 2),
            'Action': 'BUY' if diff_percent > 15 else ('SELL' if diff_percent < -15 else 'HOLD'),
            'Sector': sector,
            'PE': round(info.get('trailingPE', 0) or 0, 2),
            'Fwd PE': round(info.get('forwardPE', 0) or 0, 2),
            'PEG': round(info.get('pegRatio', 0) or 0, 2),
            'Beta': round(info.get('beta', 0) or 0, 2),
            'Beat Prob': '🟢' * min(beat_prob, 5),
            'Miss Prob': '🔴' * min(miss_prob, 5)
        }
    except Exception:
        return None

def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def get_options_single(ticker):
    """Get options for a single ticker only when requested."""
    try:
        stock = yf.Ticker(ticker)
        current = stock.info.get('currentPrice', 0)
        if current == 0: return pd.DataFrame()
        
        # Estimate volatility from history
        hist = stock.history(period="3mo")
        sigma = np.std(hist['Close'].pct_change()) * np.sqrt(252)
        
        options_data = []
        exps = stock.options
        if not exps: return pd.DataFrame()
        
        # Look at first expiration
        chain = stock.option_chain(exps[0])
        exp_date = datetime.strptime(exps[0], '%Y-%m-%d')
        T = (exp_date - datetime.now()).days / 365
        if T <= 0: T = 0.001

        for df, opt_type in [(chain.calls, 'call'), (chain.puts, 'put')]:
            # Filter for near-the-money to save processing
            df = df[(df['strike'] > current * 0.8) & (df['strike'] < current * 1.2)]
            
            for _, row in df.iterrows():
                K = row['strike']
                market = row['lastPrice']
                theo = black_scholes(current, K, T, 0.04, sigma, opt_type)
                diff_pct = ((theo - market) / market * 100) if market > 0 else 0
                
                if market > 0.1: # Filter out penny options
                    options_data.append({
                        'Type': opt_type.upper(),
                        'Strike': K,
                        'Exp': exps[0],
                        'Market': market,
                        'Theo': round(theo, 2),
                        'Value %': round(diff_pct, 1)
                    })
        return pd.DataFrame(options_data).sort_values('Value %', ascending=False).head(20)
    except Exception:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 4. MAIN APP LOGIC
# -----------------------------------------------------------------------------

st.title('Advanced Quant Stock Analyzer')
st.markdown('**Instructions:** Click a button below to load data. Use column headers to sort. Use the text box to analyze specific options.')

# Session State Initialization
if 'stock_data' not in st.session_state:
    st.session_state['stock_data'] = pd.DataFrame()
if 'active_list' not in st.session_state:
    st.session_state['active_list'] = ""

# --- Input Section (Single Ticker) ---
with st.expander("Single Ticker Deep Dive", expanded=False):
    single_ticker = st.text_input('Enter Ticker (e.g., NVDA):').upper()
    if st.button("Analyze Single Ticker"):
        if single_ticker:
            with st.spinner(f"Analyzing {single_ticker}..."):
                res = calculate_fair_price(single_ticker)
                if res:
                    st.dataframe(pd.DataFrame([res]))
                
                st.subheader("Top Value Options")
                opt_df = get_options_single(single_ticker)
                if not opt_df.empty:
                    st.dataframe(opt_df, use_container_width=True)
                else:
                    st.warning("No options data found.")

st.divider()

# --- Bulk Data Loaders ---
st.subheader("Market Scanners")

col1, col2, col3, col4, col5 = st.columns(5)

def load_data(ticker_list, name):
    """Handles the bulk loading with progress bar and threading."""
    st.session_state['active_list'] = name
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    total = len(ticker_list)
    
    # Multithreading for speed
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ticker = {executor.submit(calculate_fair_price, t): t for t in ticker_list}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker)):
            res = future.result()
            if res:
                results.append(res)
            
            # Update progress bar
            progress = (i + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Processed {i+1}/{total} tickers...")
    
    progress_bar.empty()
    status_text.empty()
    
    if results:
        df = pd.DataFrame(results)
        # Ensure consistent column order
        cols = ['Ticker', 'Price', 'Fair Value', 'Diff %', 'Action', 'Sector', 'PE', 'PEG', 'Beat Prob']
        # Filter for existing columns only
        cols = [c for c in cols if c in df.columns] 
        st.session_state['stock_data'] = df[cols].sort_values('Diff %', ascending=False)
    else:
        st.error("No data returned. Check connection.")

# Buttons
with col1:
    if st.button('Load S&P500'):
        load_data(get_sp500_tickers(), "S&P 500")
with col2:
    if st.button('Load NASDAQ'):
        load_data(get_nasdaq_tickers(), "NASDAQ 100")
with col3:
    if st.button('Load Hang Seng'):
        load_data(get_hang_seng_tickers(), "Hang Seng")
with col4:
    if st.button('Load Nikkei'):
        load_data(get_nikkei_tickers(), "Nikkei 225")
with col5:
    if st.button('Clear Data'):
        st.session_state['stock_data'] = pd.DataFrame()
        st.session_state['active_list'] = ""

# --- Main Data Display ---
if not st.session_state['stock_data'].empty:
    st.success(f"Showing results for: {st.session_state['active_list']}")
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        min_diff = st.slider("Filter by Min Diff %", -50, 50, 0)
    with filter_col2:
        search_term = st.text_input("Filter by Ticker", "")

    # Apply Filters
    df_display = st.session_state['stock_data'].copy()
    df_display = df_display[df_display['Diff %'] >= min_diff]
    if search_term:
        df_display = df_display[df_display['Ticker'].str.contains(search_term.upper())]

    # Interactive Dataframe (Allows sorting and filtering within the table)
    st.data_editor(
        df_display,
        column_config={
            "Diff %": st.column_config.NumberColumn(
                "Undervalued %",
                help="Positive means undervalued (Good)",
                format="%.2f %%"
            ),
            "Price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
            "Fair Value": st.column_config.NumberColumn("Fair Price", format="$%.2f"),
        },
        use_container_width=True,
        hide_index=True,
        disabled=True # Make read-only
    )
    
    st.write(f"Count: {len(df_display)} stocks")
else:
    st.info("Click a button above to load market data.")

# Footer
st.markdown("---")
st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
