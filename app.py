import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from io import StringIO
from datetime import datetime
from scipy.stats import norm
import numpy as np
import concurrent.futures

# -----------------------------------------------------------------------------
# 1. KONFIGURACE A CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Pro Quant Analyzer", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton > button { 
        width: 100%; border-radius: 5px; height: 3em; 
        background-color: #1f2937; color: white; border: 1px solid #3b82f6; 
    }
    .stButton > button:hover { background-color: #3b82f6; color: white; }
    h1, h2, h3 { color: #ffffff; }
    div[data-testid="stDataFrame"] { border: 1px solid #4e4f57; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 2. SPUŠTĚNÍ A STÁHOVÁNÍ TICKERŮ (OPRAVENO)
# -----------------------------------------------------------------------------

# Hlavička, aby nás weby neblokovaly
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    """Stáhne S&P 500 (cca 503 tickerů)"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        r = requests.get(url, headers=HEADERS)
        # Použijeme StringIO pro pandas
        df = pd.read_html(StringIO(r.text))[0]
        tickers = df['Symbol'].tolist()
        # Nahradit tečku pomlčkou (BRK.B -> BRK-B) pro Yahoo
        return [t.replace('.', '-') for t in tickers]
    except Exception as e:
        st.error(f"Chyba S&P500: {e}")
        return ['SPY', 'AAPL', 'MSFT'] # Fallback jen při kritické chybě

@st.cache_data(ttl=86400)
def get_nasdaq_tickers():
    """Stáhne NASDAQ 100 (cca 101 tickerů)"""
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        r = requests.get(url, headers=HEADERS)
        tables = pd.read_html(StringIO(r.text))
        
        # Hledáme tabulku, která má sloupec 'Ticker' nebo 'Symbol'
        for t in tables:
            if 'Ticker' in t.columns:
                return [str(x).replace('.', '-') for x in t['Ticker'].tolist()]
            elif 'Symbol' in t.columns:
                return [str(x).replace('.', '-') for x in t['Symbol'].tolist()]
        return ['QQQ', 'AAPL', 'NVDA']
    except Exception as e:
        st.error(f"Chyba NASDAQ: {e}")
        return ['QQQ']

@st.cache_data(ttl=86400)
def get_hang_seng_tickers():
    """Stáhne Hang Seng (cca 82 tickerů)"""
    try:
        url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
        r = requests.get(url, headers=HEADERS)
        tables = pd.read_html(StringIO(r.text))
        # Tabulka Constituents bývá obvykle index 2, ale raději hledáme 'Ticker'
        for t in tables:
            if 'Ticker' in t.columns and len(t) > 50:
                raw_tickers = t['Ticker'].astype(str).tolist()
                # Formátování: 5 -> 0005.HK
                return [f"{code.strip().zfill(4)}.HK" for code in raw_tickers]
        return ['0700.HK', '9988.HK']
    except Exception as e:
        st.error(f"Chyba Hang Seng: {e}")
        return ['0700.HK']

@st.cache_data(ttl=86400)
def get_nikkei_tickers():
    """Stáhne Nikkei 225 (225 tickerů)"""
    try:
        url = "https://en.wikipedia.org/wiki/Nikkei_225"
        r = requests.get(url, headers=HEADERS)
        tables = pd.read_html(StringIO(r.text))
        for t in tables:
            if 'Symbol' in t.columns and len(t) > 200:
                return [f"{code}.T" for code in t['Symbol'].astype(str).tolist()]
        return ['7203.T']
    except Exception as e:
        st.error(f"Chyba Nikkei: {e}")
        return ['7203.T']

# -----------------------------------------------------------------------------
# 3. LOGIKA OCENĚNÍ (Valuation)
# -----------------------------------------------------------------------------
sector_multiples = {
    # Data NYU Stern 2025 (zjednodušená pro rychlost)
    'Technology': {'pe': 35, 'growth': 0.15},
    'Financial Services': {'pe': 14, 'growth': 0.08},
    'Healthcare': {'pe': 25, 'growth': 0.10},
    'Consumer Cyclical': {'pe': 22, 'growth': 0.10},
    'Communication Services': {'pe': 20, 'growth': 0.12},
    'Energy': {'pe': 11, 'growth': 0.05},
    'Industrials': {'pe': 21, 'growth': 0.08},
    'Other': {'pe': 20, 'growth': 0.08}
}

def calculate_metrics(ticker):
    """Získá data pro jeden ticker - optimalizováno, aby nepadalo."""
    try:
        stock = yf.Ticker(ticker)
        # Použijeme 'fast_info' tam kde to jde, pro zbytek 'info'
        # 'info' je pomalé, ale obsahuje detailní data
        try:
            info = stock.info
        except:
            return None # Pokud Yahoo nemá data

        if not info or 'regularMarketPrice' not in info:
            return None

        price = info.get('currentPrice', info.get('regularMarketPrice'))
        if not price: return None

        # Načtení klíčových parametrů
        sector = info.get('sector', 'Other')
        eps = info.get('trailingEps', 0)
        fwd_eps = info.get('forwardEps', eps)
        pe = info.get('trailingPE', 0)
        peg = info.get('pegRatio', 0)
        pb = info.get('priceToBook', 0)
        beta = info.get('beta', 1)
        mkt_cap = info.get('marketCap', 0)
        
        # --- Jednoduchý Fair Value Model ---
        # Používáme kombinaci násobků sektoru a vlastních dat
        mult = sector_multiples.get(sector, sector_multiples['Other'])
        
        # 1. PE Model
        fair_pe_val = (eps if eps else fwd_eps) * mult['pe']
        
        # 2. PEG Model (pokud máme growth)
        growth = info.get('earningsGrowth', mult['growth'])
        if growth is None: growth = 0.05
        fair_peg_val = (eps if eps else 1) * (growth * 100) 
        
        # Průměr (vážený)
        if fair_pe_val > 0 and fair_peg_val > 0:
            fair_value = (fair_pe_val * 0.7) + (fair_peg_val * 0.3)
        elif fair_pe_val > 0:
            fair_value = fair_pe_val
        else:
            fair_value = price # Nelze spočítat, předpokládáme tržní cenu
            
        diff_pct = ((fair_value - price) / price) * 100
        
        return {
            'Ticker': ticker,
            'Price': price,
            'Fair Value': round(fair_value, 2),
            'Diff %': round(diff_pct, 2),
            'Sector': sector,
            'PE': round(pe, 2) if pe else None,
            'Fwd PE': round(info.get('forwardPE', 0), 2),
            'PEG': peg,
            'PB': round(pb, 2) if pb else None,
            'EPS': eps,
            'Beta': round(beta, 2) if beta else None,
            'Mkt Cap': mkt_cap,
            'Profit Margin': info.get('profitMargins'),
            'Debt/Equity': info.get('debtToEquity'),
            'Free Cashflow': info.get('freeCashflow')
        }

    except Exception:
        return None

# -----------------------------------------------------------------------------
# 4. LOGIKA APLIKACE
# -----------------------------------------------------------------------------

st.title('Advanced Stock Analyzer (Fixed)')
st.markdown("Data loaded via Wikipedia & Yahoo Finance. **Please allow time for full lists (500+ items) to load.**")

# Stav aplikace (aby data nemizela)
if 'data_result' not in st.session_state:
    st.session_state['data_result'] = pd.DataFrame()
if 'active_index' not in st.session_state:
    st.session_state['active_index'] = ""

# Funkce pro hromadné načtení
def load_market_data(ticker_func, index_name):
    st.session_state['active_index'] = index_name
    st.session_state['data_result'] = pd.DataFrame() # Reset
    
    with st.spinner(f"Načítám seznam tickerů pro {index_name}..."):
        tickers = ticker_func()
    
    if len(tickers) < 10:
        st.error(f"Pozor: Načteno pouze {len(tickers)} symbolů. Pravděpodobně chyba scrapingu.")
    else:
        st.success(f"Nalezeno {len(tickers)} symbolů. Začínám stahovat data (to chvíli potrvá)...")

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    total = len(tickers)
    
    # Multithreading (20 vláken)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks
        future_to_ticker = {executor.submit(calculate_metrics, t): t for t in tickers}
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            res = future.result()
            if res:
                results.append(res)
            
            completed += 1
            # Aktualizace progress baru každých 5 položek
            if completed % 5 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"Analyzováno: {completed} / {total}")
    
    progress_bar.empty()
    status_text.empty()
    
    if results:
        df = pd.DataFrame(results)
        # Seřadit sloupce: Ticker, Cena, Fair Value, Diff %, zbytek
        cols = ['Ticker', 'Price', 'Fair Value', 'Diff %', 'Sector', 'PE', 'PEG', 'Profit Margin', 'Debt/Equity', 'Mkt Cap']
        # Přidat ostatní sloupce, které nejsou v seznamu
        all_cols = cols + [c for c in df.columns if c not in cols]
        # Filtrovat jen existující
        final_cols = [c for c in all_cols if c in df.columns]
        
        st.session_state['data_result'] = df[final_cols].sort_values('Diff %', ascending=False)
    else:
        st.error("Nepodařilo se stáhnout žádná data.")

# Tlačítka
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    if st.button("Load S&P 500 (500+)"):
        load_market_data(get_sp500_tickers, "S&P 500")
with c2:
    if st.button("Load NASDAQ (100+)"):
        load_market_data(get_nasdaq_tickers, "NASDAQ 100")
with c3:
    if st.button("Load Hang Seng"):
        load_market_data(get_hang_seng_tickers, "Hang Seng")
with c4:
    if st.button("Load Nikkei 225"):
        load_market_data(get_nikkei_tickers, "Nikkei 225")
with c5:
    if st.button("Clear / Reset"):
        st.session_state['data_result'] = pd.DataFrame()

# Zobrazení tabulky
if not st.session_state['data_result'].empty:
    df = st.session_state['data_result']
    
    st.subheader(f"Výsledky: {st.session_state['active_index']} ({len(df)} akcií)")
    
    # Data Editor s plnými daty
    st.data_editor(
        df,
        column_config={
            "Diff %": st.column_config.NumberColumn(
                "Podhodnocení %",
                help="Kladné číslo = Levná akcie (Buy)",
                format="%.2f %%"
            ),
            "Price": st.column_config.NumberColumn("Cena", format="$%.2f"),
            "Fair Value": st.column_config.NumberColumn("Fér Hodnota", format="$%.2f"),
            "Mkt Cap": st.column_config.NumberColumn("Market Cap", format="$%d"),
            "Profit Margin": st.column_config.NumberColumn("Marže", format="%.2f"),
        },
        height=700,
        use_container_width=True,
        disabled=True # Read-only
    )
    
    # Export do CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Stáhnout CSV",
        data=csv,
        file_name=f"{st.session_state['active_index']}_analysis.csv",
        mime='text/csv',
    )
