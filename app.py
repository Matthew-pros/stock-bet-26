import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from scipy.stats import norm
import numpy as np
import concurrent.futures

# -----------------------------------------------------------------------------
# 1. CONFIG & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quant Stock Analyzer", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton > button { width: 100%; border-radius: 5px; height: 3em; background-color: #1f2937; color: white; border: 1px solid #3b82f6; }
    .stButton > button:hover { background-color: #3b82f6; color: white; }
    .stTextInput > div > div > input { background-color: #1f2937; color: white; }
    h1, h2, h3 { color: #ffffff; }
    [data-testid="stDataFrame"] { border: 1px solid #4e4f57; }
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
# 3. TICKER FETCHING (Robust Methods)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    # Wikipedia is more reliable than slickcharts for scraping
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        tickers = df['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers] # Fix BRK.B -> BRK-B for Yahoo
    except Exception as e:
        st.error(f"S&P 500 Error: {e}")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B']

@st.cache_data(ttl=86400)
def get_nasdaq_tickers():
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        df = table[4] # Typically index 4 is the components table
        if 'Ticker' not in df.columns and 'Symbol' in df.columns:
            tickers = df['Symbol'].tolist()
        else:
            tickers = df['Ticker'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except Exception as e:
        # Backup method if table index changes
        try:
            for t in pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100'):
                if 'Ticker' in t.columns:
                    return [x.replace('.', '-') for x in t['Ticker'].tolist()]
        except:
            pass
        st.error(f"NASDAQ Error: {e}")
        return ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'AVGO', 'GOOGL', 'COST', 'TSLA', 'AMD']

@st.cache_data(ttl=86400)
def get_hang_seng_tickers():
    try:
        # Wikipedia for Hang Seng
        table = pd.read_html('https://en.wikipedia.org/wiki/Hang_Seng_Index')
        df = table[2] # Constituents table
        tickers = []
        for code in df['Ticker'].astype(str):
            # Format code to 4 digits and add .HK (e.g., 5 -> 0005.HK)
            clean_code = code.strip().zfill(4)
            tickers.append(f"{clean_code}.HK")
        return list(set(tickers))
    except Exception:
        return ['0005.HK','0700.HK','0939.HK','0941.HK','1299.HK','9988.HK','3690.HK','0388.HK','0001.HK']

@st.cache_data(ttl=86400)
def get_nikkei_tickers():
    try:
        # Getting Nikkei 225 from Wikipedia
        table = pd.read_html('https://en.wikipedia.org/wiki/Nikkei_225')
        df = table[3] # Constituents
        tickers = []
        for code in df['Symbol'].astype(str):
            tickers.append(f"{code}.T")
        return list(set(tickers))
    except Exception:
        return ['7203.T','6758.T','9984.T','8035.T','9983.T','6861.T','4063.T','6098.T','6501.T']

# -----------------------------------------------------------------------------
# 4. CORE ANALYSIS
# -----------------------------------------------------------------------------

def calculate_fair_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Using fast_info or info, prioritizing speed but ensuring data presence
        info = stock.info
        
        if not info or 'regularMarketPrice' not in info:
            return None

        # --- Gather All Metrics (from original code) ---
        metrics = {
            'Ticker': ticker,
            'Sector': info.get('sector', 'Other'),
            'Price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
            'Market Cap': info.get('marketCap', 0),
            'PE': info.get('trailingPE', 0),
            'Fwd PE': info.get('forwardPE', 0),
            'PEG': info.get('pegRatio', 0),
            'PS': info.get('priceToSalesTrailing12Months', 0),
            'PB': info.get('priceToBook', 0),
            'Beta': info.get('beta', 0),
            'EPS': info.get('trailingEps', 0),
            'ROE': info.get('returnOnEquity', 0),
            'Profit Margin': info.get('profitMargins', 0),
            'Gross Margin': info.get('grossMargins', 0),
            'Debt/Equity': info.get('debtToEquity', 0),
            'Rev Growth': info.get('revenueGrowth', 0),
            'Earnings Growth': info.get('earningsGrowth', 0),
            'Free Cashflow': info.get('freeCashflow', 0),
            'Target Price': info.get('targetMeanPrice', 0),
            'Short Float': info.get('shortPercentOfFloat', 0),
            'Div Yield': info.get('dividendYield', 0),
            '52w High': info.get('fiftyTwoWeekHigh', 0),
            '52w Low': info.get('fiftyTwoWeekLow', 0),
        }

        # --- Fair Value Logic ---
        sector = metrics['Sector']
        mapped_sector = sector_mapping.get(sector, 'Other')
        multiples = sector_multiples.get(mapped_sector, sector_multiples['Other'])

        # Calculations
        eps = metrics['EPS'] or 0
        fwd_eps = info.get('forwardEps', 0) or 0
        bvps = info.get('bookValue', 0) or 0
        rev_ps = info.get('revenuePerShare', 0) or 0
        ebitda = info.get('ebitda', 0) or 0
        shares = info.get('sharesOutstanding', 1)
        net_debt = (info.get('totalDebt', 0) or 0) - (info.get('totalCash', 0) or 0)

        # Valuation Models
        fair_pe = eps * multiples['current_pe'] if eps > 0 else 0
        fair_fwd_pe = fwd_eps * multiples['forward_pe'] if fwd_eps > 0 else 0
        fair_pb = bvps * multiples['pb'] if bvps > 0 else 0
        fair_ps = rev_ps * multiples['ps']
        fair_evebitda = max(0, (ebitda * multiples['evebitda'] - net_debt) / shares) if shares > 0 else 0
        
        # DDM
        cost_equity = 0.03 + (metrics['Beta'] or 1) * 0.06
        payout = info.get('payoutRatio', 0) or 0
        growth_rate = (1 - payout) * (metrics['ROE'] or 0.1)
        if growth_rate <= 0: growth_rate = multiples['growth']
        
        fair_ddm = 0
        if cost_equity > growth_rate > 0 and eps > 0:
            fair_ddm = (eps * (1 - payout)) / (cost_equity - growth_rate)
        else:
            fair_ddm = (fair_pe + fair_fwd_pe) / 2

        # Weighted Fair Value
        vals = [fair_pe, fair_fwd_pe, fair_pb, fair_ps, fair_evebitda, fair_ddm]
        weights = [0.3, 0.25, 0.15, 0.15, 0.15, 0.05]
        
        valid_vals = []
        valid_weights = []
        for v, w in zip(vals, weights):
            if v > 0:
                valid_vals.append(v)
                valid_weights.append(w)
        
        if not valid_vals:
            return None
            
        fair_price = np.average(valid_vals, weights=valid_weights)
        
        # PEG Adjustment
        if multiples['peg'] > 0 and metrics['PEG'] and metrics['PEG'] > 0:
             fair_price /= (multiples['peg'] if multiples['peg'] > 1.2 else 1)

        # Margin of Safety
        fair_price *= 0.95
        
        current_price = metrics['Price']
        diff_pct = ((fair_price - current_price) / current_price) * 100 if current_price else 0

        metrics['Fair Value'] = round(fair_price, 2)
        metrics['Diff %'] = round(diff_pct, 2)
        
        # Earnings Beat Probability
        score = 0
        if (metrics['Earnings Growth'] or 0) > 0.1: score += 1
        if (metrics['Short Float'] or 0) < 0.05: score += 1
        if (metrics['Rev Growth'] or 0) > 0.05: score += 1
        beat_prob = min(max(score + 3, 1), 5)
        metrics['Beat Prob'] = '🟢' * beat_prob
        
        return metrics

    except Exception:
        return None

def get_options_single(ticker):
    try:
        stock = yf.Ticker(ticker)
        current = stock.info.get('currentPrice', 0)
        if current == 0: return pd.DataFrame()
        
        hist = stock.history(period="3mo")
        sigma = np.std(hist['Close'].pct_change()) * np.sqrt(252) if len(hist) > 1 else 0.5
        
        options_data = []
        exps = stock.options
        if not exps: return pd.DataFrame()
        
        # Get first expiration that is at least 7 days out for better data
        exp_to_use = exps[0]
        for e in exps:
            days = (datetime.strptime(e, '%Y-%m-%d') - datetime.now()).days
            if days > 5:
                exp_to_use = e
                break

        chain = stock.option_chain(exp_to_use)
        T = (datetime.strptime(exp_to_use, '%Y-%m-%d') - datetime.now()).days / 365
        if T <= 0: T = 0.001

        for df, typ in [(chain.calls, 'call'), (chain.puts, 'put')]:
             # Filter range
            df = df[(df['strike'] > current * 0.75) & (df['strike'] < current * 1.25)]
            for _, row in df.iterrows():
                K = row['strike']
                market = row['lastPrice']
                if market < 0.05: continue
                
                d1 = (np.log(current/K) + (0.04 + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                if typ == 'call':
                    theo = current*norm.cdf(d1) - K*np.exp(-0.04*T)*norm.cdf(d2)
                else:
                    theo = K*np.exp(-0.04*T)*norm.cdf(-d2) - current*norm.cdf(-d1)
                
                val_diff = (theo - market)/market * 100
                if val_diff > 10: # Filter clutter
                    options_data.append({
                        'Type': typ.upper(), 'Strike': K, 'Exp': exp_to_use,
                        'Market': market, 'Theo': round(theo, 2), 'Value %': round(val_diff, 1)
                    })
        return pd.DataFrame(options_data).sort_values('Value %', ascending=False).head(15)
    except Exception:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 5. APP INTERFACE
# -----------------------------------------------------------------------------

st.title('Advanced Quant Analyzer - Full Market')
st.markdown('Loads real-time data from Yahoo Finance. **Note:** Loading 500+ tickers takes time. Please wait for the progress bar.')

# Session State
if 'stock_df' not in st.session_state:
    st.session_state['stock_df'] = pd.DataFrame()
if 'list_name' not in st.session_state:
    st.session_state['list_name'] = ""

# --- Single Ticker ---
with st.expander("🔎 Single Ticker Analysis", expanded=False):
    t_in = st.text_input("Ticker (e.g. TSLA):").upper()
    if st.button("Analyze One"):
        if t_in:
            with st.spinner('Calculating...'):
                data = calculate_fair_price(t_in)
                if data:
                    st.dataframe(pd.DataFrame([data]), use_container_width=True)
                    st.subheader("Options Value")
                    st.dataframe(get_options_single(t_in), use_container_width=True)
                else:
                    st.error("Ticker not found or no data.")

st.divider()

# --- Bulk Loader Function ---
def run_bulk_analysis(ticker_list, list_label):
    st.session_state['list_name'] = list_label
    st.session_state['stock_df'] = pd.DataFrame() # Clear previous
    
    prog_bar = st.progress(0)
    status_area = st.empty()
    results = []
    
    # Chunking to prevent overwhelming user UI, but processing all
    total = len(ticker_list)
    
    # Using ThreadPool for speed - set to 20 workers to be safe with Yahoo API
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ticker = {executor.submit(calculate_fair_price, t): t for t in ticker_list}
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            data = future.result()
            if data:
                results.append(data)
            completed += 1
            # Update UI every 5 tickers to reduce lag
            if completed % 5 == 0 or completed == total:
                prog_bar.progress(completed / total)
                status_area.text(f"Scanning {completed}/{total} stocks... ({len(results)} successful)")

    prog_bar.empty()
    status_area.empty()
    
    if results:
        df = pd.DataFrame(results)
        # Reorder columns nicely: Key info first, then the rest
        priority = ['Ticker', 'Price', 'Fair Value', 'Diff %', 'Beat Prob', 'Sector', 'PE', 'PEG']
        other_cols = [c for c in df.columns if c not in priority]
        final_cols = priority + other_cols
        
        st.session_state['stock_df'] = df[final_cols].sort_values('Diff %', ascending=False)
    else:
        st.error("No data loaded. Yahoo Finance might be blocking requests momentarily.")

# --- Buttons ---
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    if st.button(f"Load S&P 500"):
        ts = get_sp500_tickers()
        st.info(f"Found {len(ts)} tickers. Starting scan...")
        run_bulk_analysis(ts, "S&P 500")
with c2:
    if st.button("Load NASDAQ 100"):
        ts = get_nasdaq_tickers()
        st.info(f"Found {len(ts)} tickers. Starting scan...")
        run_bulk_analysis(ts, "NASDAQ 100")
with c3:
    if st.button("Load Hang Seng"):
        ts = get_hang_seng_tickers()
        run_bulk_analysis(ts, "Hang Seng")
with c4:
    if st.button("Load Nikkei 225"):
        ts = get_nikkei_tickers()
        st.info(f"Found {len(ts)} tickers. Starting scan...")
        run_bulk_analysis(ts, "Nikkei 225")
with c5:
    if st.button("Load Custom"):
        # Example small list
        custom = ['NVDA','AAPL','MSFT','TSLA','AMD','PLTR','COIN','MSTR']
        run_bulk_analysis(custom, "Tech Favorites")

# --- Display ---
if not st.session_state['stock_df'].empty:
    st.subheader(f"Results: {st.session_state['list_name']} ({len(st.session_state['stock_df'])} stocks)")
    
    # Interactive Data Editor - This allows sorting and filtering in the UI
    st.data_editor(
        st.session_state['stock_df'],
        column_config={
            "Diff %": st.column_config.NumberColumn(
                "Undervalued %",
                help="Positive Green = Undervalued",
                format="%.2f %%"
            ),
            "Market Cap": st.column_config.NumberColumn("Mkt Cap", format="$%d"),
            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "Fair Value": st.column_config.NumberColumn("Fair Val", format="$%.2f"),
        },
        height=600,
        use_container_width=True,
        disabled=True # Read only
    )
    
    # Download button
    csv = st.session_state['stock_df'].to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="stock_analysis.csv", mime="text/csv")

st.markdown(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
