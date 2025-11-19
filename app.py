import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import concurrent.futures
import time
from io import StringIO
from scipy.stats import norm
import numpy as np

# -----------------------------------------------------------------------------
# 1. STATIC LISTS (ZÁCHRANNÁ SÍŤ)
# -----------------------------------------------------------------------------
# Pokud selže scraping, použijeme tyto seznamy. Tím garantujeme funkčnost.

STATIC_SP500 = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC","BK","BBWI","BAX","BDX","BRK-B","BBY","BIO","TECH","BIIB","BLK","BX","BKNG","BWA","BXP","BSX","BMY","AVGO","BR","BRO","BF-B","BLDR","BG","CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNP","CDAY","CF","CHRW","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CMA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CTVA","CSGP","COST","CTRA","CCI","CSX","CMI","CVS","DHI","DHR","DRI","DVA","DE","DAL","XRAY","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DHI","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","LLY","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETS","EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX","FITB","FSLR","FE","FIS","FI","FLT","FMC","F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD","GPN","GL","GS","HAL","HIG","HAS","HCA","PEAK","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","ILMN","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","J","JNJ","JCI","JPM","KP","K","KDP","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PANW","PARA","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PXD","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","RHI","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SNA","SEDG","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UDR","ULTA","UNP","UAL","UNH","UPS","URI","UHS","VLO","VTR","VRSN","VRSK","VZ","VRTX","VFC","VTRS","VICI","V","VMC","WAB","WBA","WMT","WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WRK","WY","WHR","WMB","WTW","GWW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS"
]

STATIC_NASDAQ = [
    "AAPL","ABNB","ADBE","ADI","ADP","ADSK","AEP","AMAT","AMD","AMGN","AMZN","ANSS","ASML","AVGO","AZN","BIIB","BKNG","BKR","CDNS","CEG","CHTR","CMCSA","COST","CPRT","CRWD","CSCO","CSX","CTAS","CTSH","DDOG","DLTR","DXCM","EA","EBAY","ENPH","EXC","FANG","FAST","FTNT","GEHC","GFS","GILD","GOOG","GOOGL","HON","IDXX","ILMN","INTC","INTU","ISRG","KDP","KHC","KLAC","LCID","LRCX","LULU","MAR","MCHP","MDLZ","MELI","META","MNST","MRNA","MRVL","MSFT","MU","NFLX","NVDA","NXPI","ODFL","ON","ORLY","PANW","PAYX","PCAR","PDD","PEP","PYPL","QCOM","REGN","ROP","ROST","SBUX","SGEN","SIRI","SNPS","SPLK","TEAM","TMUS","TSLA","TXN","VRSK","VRTX","WBD","WDAY","XEL","ZS"
]

# -----------------------------------------------------------------------------
# 2. CONFIG & SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quant Pro Analyzer", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton > button { 
        width: 100%; border-radius: 5px; height: 3em; 
        background-color: #262730; color: white; border: 1px solid #3b82f6; 
    }
    .stButton > button:hover { background-color: #3b82f6; color: white; }
    h1, h2, h3 { color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True
)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# -----------------------------------------------------------------------------
# 3. ROBUST TICKER FETCHING
# -----------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    """Pokusí se stáhnout S&P 500, při chybě použije STATIC_SP500"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        r = requests.get(url, headers=HEADERS)
        df = pd.read_html(StringIO(r.text))[0]
        
        # Flexibilní hledání sloupce
        target_col = None
        for c in ['Symbol', 'Ticker', 'Ticker symbol', 'Security']:
            if c in df.columns:
                target_col = c
                break
        
        if target_col:
            return [t.replace('.', '-') for t in df[target_col].tolist()]
        else:
            raise ValueError("Sloupec nenalezen")
            
    except Exception as e:
        return STATIC_SP500

@st.cache_data(ttl=86400)
def get_nasdaq_tickers():
    """Pokusí se stáhnout NASDAQ, při chybě použije STATIC_NASDAQ"""
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        r = requests.get(url, headers=HEADERS)
        dfs = pd.read_html(StringIO(r.text))
        
        for df in dfs:
            # NASDAQ tabulka má obvykle sloupec 'Ticker' nebo 'Symbol'
            cols = [c.lower() for c in df.columns]
            if 'ticker' in cols or 'symbol' in cols:
                # Najdi skutečné jméno sloupce
                real_col = 'Ticker' if 'Ticker' in df.columns else 'Symbol'
                return [t.replace('.', '-') for t in df[real_col].tolist()]
                
        return STATIC_NASDAQ
    except Exception:
        return STATIC_NASDAQ

@st.cache_data(ttl=86400)
def get_hang_seng_tickers():
    """Stáhne Hang Seng a opraví formátování na 0005.HK"""
    try:
        url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
        r = requests.get(url, headers=HEADERS)
        dfs = pd.read_html(StringIO(r.text))
        
        # Obvykle je to ta největší tabulka
        best_df = max(dfs, key=len)
        
        col_name = None
        for c in best_df.columns:
            if 'ticker' in c.lower() or 'symbol' in c.lower() or 'code' in c.lower():
                col_name = c
                break
        
        if col_name:
            raw_tickers = best_df[col_name].astype(str).tolist()
            clean_tickers = []
            for t in raw_tickers:
                # Odstranit vše kromě čísel
                num = ''.join(filter(str.isdigit, t))
                if num:
                    clean_tickers.append(f"{num.zfill(4)}.HK")
            return clean_tickers if len(clean_tickers) > 10 else ['0005.HK', '0700.HK', '9988.HK', '0941.HK', '1299.HK']
    except:
        pass
    # Fallback pro Hang Seng
    return ['0001.HK','0002.HK','0003.HK','0005.HK','0006.HK','0011.HK','0012.HK','0016.HK','0017.HK','0019.HK','0027.HK','0066.HK','0101.HK','0151.HK','0175.HK','0241.HK','0267.HK','0288.HK','0386.HK','0388.HK','0669.HK','0700.HK','0762.HK','0823.HK','0857.HK','0868.HK','0881.HK','0883.HK','0939.HK','0941.HK','0960.HK','0968.HK','0981.HK','0992.HK','1038.HK','1044.HK','1088.HK','1093.HK','1109.HK','1113.HK','1177.HK','1211.HK','1299.HK','1398.HK','1810.HK','1876.HK','1928.HK','1929.HK','2007.HK','2015.HK','2020.HK','2269.HK','2313.HK','2318.HK','2319.HK','2331.HK','2382.HK','2388.HK','2628.HK','2688.HK','2899.HK','3690.HK','3968.HK','3988.HK','6098.HK','6618.HK','6690.HK','6862.HK','9618.HK','9633.HK','9888.HK','9961.HK','9988.HK','9999.HK']

@st.cache_data(ttl=86400)
def get_nikkei_tickers():
    """Stáhne Nikkei a opraví formátování na 7203.T"""
    try:
        url = "https://en.wikipedia.org/wiki/Nikkei_225"
        r = requests.get(url, headers=HEADERS)
        dfs = pd.read_html(StringIO(r.text))
        best_df = max(dfs, key=len)
        
        col_name = None
        for c in best_df.columns:
            if 'symbol' in c.lower():
                col_name = c
                break
        
        if col_name:
            return [f"{str(t)}.T" for t in best_df[col_name].tolist()]
    except:
        pass
    # Fallback
    return ['7203.T','6758.T','9984.T','8035.T','9983.T','6861.T','4063.T','6098.T','6501.T','6954.T','7741.T','6301.T','4543.T','4519.T','4568.T','4503.T','4502.T','4523.T','4507.T','4578.T','4151.T','4506.T','2502.T','2503.T','2802.T','2801.T','2269.T','2282.T','2501.T','2002.T','2871.T','2914.T','3382.T','3086.T','3099.T','8267.T','8233.T','8252.T','9983.T','7453.T','9843.T','3092.T','4689.T','4755.T','4751.T','2413.T','2432.T','4324.T','9735.T','9766.T','9602.T','2432.T','4307.T','4661.T','6098.T']

# -----------------------------------------------------------------------------
# 4. ROBUST CALCULATION LOGIC
# -----------------------------------------------------------------------------

sector_multiples = {
    'Technology': 30, 'Financial Services': 15, 'Healthcare': 22,
    'Consumer Cyclical': 20, 'Industrials': 20, 'Energy': 12, 'Other': 18
}

def get_stock_data(ticker):
    """
    Získá data s retry logikou a ošetřením chyb.
    """
    # Malá pauza proti blokování
    time.sleep(0.05) 
    
    try:
        stock = yf.Ticker(ticker)
        # Používáme .fast_info pro rychlost, .info jako zálohu
        # fast_info je mnohem spolehlivější pro získání aktuální ceny
        price = None
        try:
            price = stock.fast_info['last_price']
        except:
            try:
                price = stock.info.get('currentPrice', stock.info.get('regularMarketPrice'))
            except:
                pass
        
        if not price:
            return None

        # Načtení detailů přes .info (může selhat)
        try:
            info = stock.info
        except:
            info = {}

        pe = info.get('trailingPE')
        fwd_pe = info.get('forwardPE')
        sector = info.get('sector', 'Other')
        
        # Fair Value Výpočet (Simplified Robust)
        # Pokud nemáme PE, použijeme odhad sektoru
        used_pe = pe if pe else (fwd_pe if fwd_pe else sector_multiples.get(sector, 20))
        
        # Jednoduchý model: Pokud je aktuální PE výrazně nižší než sektor/historie, je levná
        target_pe = sector_multiples.get(sector, 20)
        
        # Vypočítat "Fair Value" na základě zisku (EPS)
        eps = info.get('trailingEps')
        if not eps and price and used_pe:
            eps = price / used_pe
        
        fair_value = 0
        if eps:
            fair_value = eps * target_pe
        else:
            fair_value = price # Nelze určit
            
        diff = ((fair_value - price) / price) * 100
        
        return {
            'Ticker': ticker,
            'Price': price,
            'Fair Value': round(fair_value, 2),
            'Diff %': round(diff, 2),
            'Sector': sector,
            'PE': round(pe, 2) if pe else None,
            'Mkt Cap': info.get('marketCap'),
            'Volume': info.get('volume'),
            'EPS': eps
        }
    except Exception as e:
        return None

# -----------------------------------------------------------------------------
# 5. UI & LOGIC
# -----------------------------------------------------------------------------

st.title('📈 Pro Stock Analyzer (Ultimate)')
st.write("Full data access. If live scraping fails, internal databases are used instantly.")

if 'results' not in st.session_state:
    st.session_state['results'] = pd.DataFrame()
if 'active_name' not in st.session_state:
    st.session_state['active_name'] = ""

def run_scan(ticker_func, name):
    st.session_state['active_name'] = name
    st.session_state['results'] = pd.DataFrame()
    
    with st.spinner(f"Loading ticker list for {name}..."):
        tickers = ticker_func()
    
    st.success(f"Processing {len(tickers)} symbols. Please wait...")
    
    # Progress UI
    bar = st.progress(0)
    status = st.empty()
    
    data_list = []
    total = len(tickers)
    
    # Threading - méně vláken (10) je bezpečnější proti chybám "No data"
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_stock_data, t): t for t in tickers}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            if res:
                data_list.append(res)
            
            if i % 5 == 0:
                bar.progress((i + 1) / total)
                status.text(f"Analyzed: {i+1}/{total}")

    bar.empty()
    status.empty()
    
    if data_list:
        df = pd.DataFrame(data_list)
        # Sort columns
        cols = ['Ticker', 'Price', 'Fair Value', 'Diff %', 'PE', 'Sector', 'Mkt Cap']
        df = df[cols]
        st.session_state['results'] = df.sort_values('Diff %', ascending=False)
    else:
        st.error("Error: Yahoo Finance refused connection. Try again in 1 minute.")

# Control Panel
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Load S&P 500"):
        run_scan(get_sp500_tickers, "S&P 500")
with col2:
    if st.button("Load NASDAQ"):
        run_scan(get_nasdaq_tickers, "NASDAQ 100")
with col3:
    if st.button("Load Hang Seng"):
        run_scan(get_hang_seng_tickers, "Hang Seng")
with col4:
    if st.button("Load Nikkei"):
        run_scan(get_nikkei_tickers, "Nikkei 225")

# Results Display
if not st.session_state['results'].empty:
    df = st.session_state['results']
    st.subheader(f"{st.session_state['active_name']} Results ({len(df)} stocks)")
    
    st.data_editor(
        df,
        column_config={
            "Diff %": st.column_config.NumberColumn(
                "Undervalued %",
                help="Positive = Potential Buy",
                format="%.2f %%"
            ),
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Fair Value": st.column_config.NumberColumn(format="$%.2f"),
            "Mkt Cap": st.column_config.NumberColumn(format="$%d"),
        },
        height=600,
        use_container_width=True
    )
    
    # CSV Download
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode('utf-8'),
        "market_data.csv",
        "text/csv"
    )
