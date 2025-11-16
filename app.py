import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from scipy.stats import norm
import numpy as np
from io import StringIO

# NYU Stern multiples from 2025 data
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

# Sector mapping
sector_mapping = {
    'Technology': 'Technology',
    'Financial Services': 'Financial Services',
    'Real Estate': 'Real Estate',
    'Consumer Cyclical': 'Consumer Cyclical',
    'Consumer Defensive': 'Consumer Defensive',
    'Healthcare': 'Healthcare',
    'Utilities': 'Utilities',
    'Communication Services': 'Communication Services',
    'Energy': 'Energy',
    'Industrials': 'Industrials',
    'Basic Materials': 'Basic Materials',
    'Other': 'Other'
}

@st.cache_data(ttl=86400)
def get_index_tickers():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    tickers = []

    # S&P500
    try:
        response = requests.get('https://en.wikipedia.org/wiki/List_of_S&P_500_companies', headers=headers)
        response.raise_for_status()
        sp500 = pd.read_html(StringIO(response.text))[0]['Symbol'].tolist()
        tickers.extend(sp500)
    except Exception as e:
        st.warning(f'S&P500 fetch failed: {e}')
        tickers.extend(['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'AVGO', 'GOOG', 'META', 'TSLA', 'BRK.B', 'LLY', 'JPM', 'WMT', 'V', 'ORCL', 'XOM', 'MA', 'JNJ', 'NFLX', 'PLTR', 'ABBV', 'COST', 'AMD', 'BAC', 'HD', 'PG', 'GE', 'CVX', 'CSCO', 'KO', 'UNH', 'IBM', 'MU', 'WFC', 'MS', 'CAT', 'AXP', 'PM', 'TMUS', 'GS', 'RTX', 'CRM', 'MRK', 'ABT', 'MCD', 'TMO', 'PEP', 'LIN', 'ISRG', 'UBER', 'DIS', 'APP', 'QCOM', 'LRCX', 'INTU', 'T', 'AMGN', 'AMAT', 'C', 'NOW', 'NEE', 'VZ', 'INTC', 'SCHW', 'ANET', 'BLK', 'APH', 'BKNG', 'TJX', 'GEV', 'DHR', 'GILD', 'BSX', 'ACN', 'SPGI', 'KLAC', 'BA', 'TXN', 'PFE', 'PANW', 'ADBE', 'SYK', 'ETN', 'CRWD', 'COF', 'WELL', 'UNP', 'PGR', 'DE', 'LOW', 'HON', 'MDT', 'CB', 'ADI', 'PLD', 'COP', 'VRTX', 'HOOD', 'BX', 'HCA', 'LMT', 'KKR', 'CEG', 'PH', 'MCK', 'CME', 'ADP', 'CMCSA', 'SO', 'CVS', 'MO', 'SBUX', 'NEM', 'DUK', 'BMY', 'NKE', 'GD', 'TT', 'DELL', 'MMC', 'DASH', 'MMM', 'ICE', 'AMT', 'CDNS', 'MCO', 'WM', 'ORLY', 'SHW', 'HWM', 'UPS', 'NOC', 'JCI', 'EQIX', 'BK', 'MAR', 'COIN', 'APO', 'TDG', 'AON', 'CTAS', 'WMB', 'ABNB', 'MDLZ', 'ECL', 'USB', 'REGN', 'SNPS', 'MNST', 'CSX', 'RSG', 'DDOG', 'AEP', 'AZO', 'TRV', 'PWR', 'CMI', 'ADSK', 'NSC', 'MSI', 'FDX', 'CL', 'HLT', 'WDAY', 'FTNT', 'PYPL', 'MSTR', 'WBD', 'IDXX', 'ROST', 'PCAR', 'EA', 'NXPI', 'ROP', 'BKR', 'XEL', 'ZS', 'FAST', 'EXC', 'AXON', 'TTWO', 'FANG', 'CCEP', 'PAYX', 'TEAM', 'CPRT', 'KDP', 'CTSH', 'GEHC', 'VRSK', 'KHC', 'MCHP', 'CSGP', 'ODFL', 'CHTR', 'BIIB', 'DXCM', 'TTD', 'LULU', 'DRI', 'CHD', 'TYL', 'RL', 'CTRA', 'NVR', 'IP', 'AMCR', 'CPAY', 'KEY', 'ON', 'TSN', 'CDW', 'WST', 'BG', 'PFG', 'EXPD', 'J', 'TRMB', 'CHRW', 'SW', 'CNC', 'ZBH', 'PKG', 'GPC', 'EVRG', 'GPN', 'MKC', 'GDDY', 'Q', 'INVH', 'LNT', 'PSKY', 'SNA', 'PNR', 'APTV', 'LUV', 'IFF', 'IT', 'DD', 'LII', 'HOLX', 'GEN', 'ESS', 'FTV', 'DOW', 'WY', 'BBY', 'JBHT', 'MAA', 'ERIE', 'LYB', 'TKO', 'COO', 'TXT', 'UHS', 'OMC', 'ALLE', 'DPZ', 'DPZ', 'KIM', 'FOX', 'EG', 'FOXA', 'ALB', 'FFIV', 'AVY', 'CF', 'BF.B', 'SOLV', 'NDSN', 'BALL', 'REG', 'CLX', 'MAS', 'WYNN', 'AKAM', 'HRL', 'VTRS', 'HII', 'IEX', 'ZBRA', 'HST', 'DECK', 'DOC', 'JKHY', 'SJM', 'BEN', 'UDR', 'AIZ', 'BLDR', 'BXP', 'DAY', 'CPT', 'HAS', 'PNW', 'RVTY', 'GL', 'IVZ', 'FDS', 'SWK', 'SWKS', 'EPAM', 'AES', 'ALGN', 'NWSA', 'MRNA', 'BAX', 'CPB', 'TECH', 'TAP', 'PAYC', 'ARE', 'POOL', 'AOS', 'IPG', 'MGM', 'GNRC', 'APA', 'DVA', 'HSIC', 'FRT', 'CAG', 'NCLH', 'MOS', 'CRL', 'LW', 'LKQ', 'MTCH', 'MOH', 'SOLS', 'MHK', 'NWS'])

@st.cache_data(ttl=86400)
def get_index_tickers():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    tickers = []

    # S&P500
    try:
        response = requests.get('https://en.wikipedia.org/wiki/List_of_S&P_500_companies', headers=headers)
        response.raise_for_status()
        sp500 = pd.read_html(StringIO(response.text))[0]['Symbol'].tolist()
        tickers.extend(sp500)
    except Exception as e:
        st.warning(f'S&P500 fetch failed: {e}')
        # Hardcoded fallback from browse
        sp500_fallback = 'NVDA,AAPL,MSFT,AMZN,GOOGL,AVGO,GOOG,META,TSLA,BRK.B,LLY,JPM,WMT,V,ORCL,XOM,MA,JNJ,NFLX,PLTR,ABBV,COST,AMD,BAC,HD,PG,GE,CVX,CSCO,KO,UNH,IBM,MU,WFC,MS,CAT,AXP,PM,TMUS,GS,RTX,CRM,MRK,ABT,MCD,TMO,PEP,LIN,ISRG,UBER,DIS,APP,QCOM,LRCX,INTU,T,AMGN,AMAT,C,NOW,NEE,VZ,INTC,SCHW,ANET,BLK,APH,BKNG,TJX,GEV,DHR,GILD,BSX,ACN,SPGI,KLAC,BA,TXN,PFE,PANW,ADBE,SYK,ETN,CRWD,COF,WELL,UNP,PGR,DE,LOW,HON,MDT,CB,ADI,PLD,COP,VRTX,HOOD,BX,HCA,LMT,KKR,CEG,PH,MCK,CME,ADP,CMCSA,SO,CVS,MO,SBUX,NEM,DUK,BMY,NKE,GD,TT,DELL,MMC,DASH,MMM,ICE,AMT,CDNS,MCO,WM,ORLY,SHW,HWM,UPS,NOC,JCI,EQIX,BK,MAR,COIN,APO,TDG,AON,CTAS,WMB,ABNB,MDLZ,ECL,USB,REGN,SNPS,MNST,CSX,RSG,DDOG,AEP,AZO,TRV,PWR,CMI,ADSK,NSC,MSI,FDX,CL,HLT,WDAY,FTNT,PYPL,MSTR,WBD,IDXX,ROST,PCAR,EA,NXPI,ROP,BKR,XEL,ZS,FAST,EXC,AXON,TTWO,FANG,CCEP,PAYX,TEAM,CPRT,KDP,CTSH,GEHC,VRSK,KHC,MCHP,CSGP,ODFL,CHTR,BIIB,DXCM,TTD,LULU,DRI,CHD,TYL,RL,CTRA,NVR,IP,AMCR,CPAY,KEY,ON,TSN,CDW,WST,BG,PFG,EXPD,J,TRMB,CHRW,SW,CNC,ZBH,PKG,GPC,EVRG,GPN,MKC,GDDY,Q,INVH,LNT,PSKY,SNA,PNR,APTV,LUV,IFF,IT,DD,LII,HOLX,GEN,ESS,FTV,DOW,WY,BBY,JBHT,MAA,ERIE,LYB,TKO,COO,TXT,UHS,OMC,ALLE,DPZ,KIM,FOX,EG,FOXA,ALB,FFIV,AVY,CF,BF.B,SOLV,NDSN,BALL,REG,CLX,MAS,WYNN,AKAM,HRL,VTRS,HII,IEX,ZBRA,HST,DECK,DOC,JKHY,SJM,BEN,UDR,AIZ,BLDR,BXP,DAY,CPT,HAS,PNW,RVTY,GL,IVZ,FDS,SWK,SWKS,EPAM,AES,ALGN,NWSA,MRNA,BAX,CPB,TECH,TAP,PAYC,ARE,POOL,AOS,IPG,MGM,GNRC,APA,DVA,HSIC,FRT,CAG,NCLH,MOS,CRL,LW,LKQ,MTCH,MOH,SOLS,MHK,NWS'.split(',')
        tickers.extend(sp500_fallback)

    # NASDAQ-100
    try:
        response = requests.get('https://en.wikipedia.org/wiki/Nasdaq-100', headers=headers)
        response.raise_for_status()
        nasdaq = pd.read_html(StringIO(response.text))[4]['Ticker'].tolist()
        tickers.extend(nasdaq)
    except Exception as e:
        st.warning(f'NASDAQ fetch failed: {e}')
        nasdaq_fallback = 'NVDA,AAPL,MSFT,AMZN,GOOGL,AVGO,GOOG,META,TSLA,NFLX,PLTR,COST,AMD,ASML,CSCO,MU,AZN,TMUS,PEP,LIN,ISRG,SHOP,APP,QCOM,LRCX,PDD,INTU,AMGN,AMAT,INTC,BKNG,GILD,KLAC,ARM,TXN,PANW,ADBE,CRWD,HON,ADI,VRTX,CEG,MELI,ADP,CMCSA,SBUX,DASH,CDNS,ORLY,MAR,CTAS,MRVL,ABNB,MDLZ,REGN,SNPS,MNST,CSX,DDOG,AEP,ADSK,TRI,WDAY,FTNT,PYPL,MSTR,WBD,IDXX,ROST,PCAR,EA,NXPI,ROP,BKR,XEL,ZS,FAST,EXC,AXON,TTWO,FANG,CCEP,PAYX,TEAM,CPRT,KDP,CTSH,GEHC,VRSK,KHC,MCHP,CSGP,ODFL,CHTR,BIIB,DXCM,TTD,LULU,ON,CDW,GFS'.split(',')
        tickers.extend(nasdaq_fallback)

    # Hang Seng
    try:
        response = requests.get('https://en.wikipedia.org/wiki/Hang_Seng_Index', headers=headers)
        response.raise_for_status()
        hang_df = pd.read_html(StringIO(response.text))[2]
        hang = [f"{int(code):04d}.HK" for code in hang_df['Ticker'].str.extract(r'(\d+)', expand=False).dropna()]
        tickers.extend(hang)
    except Exception as e:
        st.warning(f'Hang Seng fetch failed: {e}')
        hang_fallback = '0005.HK,0011.HK,00388.HK,00939.HK,01299.HK,01398.HK,02318.HK,02388.HK,02628.HK,02628.HK,03968.HK,03988.HK,0002.HK,0003.HK,0006.HK,0836.HK,01038.HK,02688.HK,0012.HK,0016.HK,0017.HK,0101.HK,0688.HK,0823.HK,0960.HK,01109.HK,01113.HK,01209.HK,01997.HK,06098.HK,0001.HK,0027.HK,0066.HK,0175.HK,0241.HK,0267.HK,0288.HK,0291.HK,0316.HK,0322.HK,0386.HK,0669.HK,0700.HK,0762.HK,0857.HK,0868.HK,0881.HK,0883.HK,0941.HK,0968.HK,0981.HK,0992.HK,01044.HK,01088.HK,01093.HK,01099.HK,01177.HK,01211.HK,01378.HK,01810.HK,01876.HK,01928.HK,01929.HK,02015.HK,02020.HK,02269.HK,02313.HK,02319.HK,02331.HK,02359.HK,02382.HK,02899.HK,03690.HK,03692.HK,06618.HK,06690.HK,06862.HK,09618.HK,09633.HK,09888.HK,09961.HK,09988.HK,09999.HK'.split(',')
        tickers.extend(hang_fallback)

    # Nikkei
    try:
        response = requests.get('https://en.wikipedia.org/wiki/Nikkei_225', headers=headers)
        response.raise_for_status()
        nikkei_df = pd.read_html(StringIO(response.text))[3]
        nikkei = [f"{int(code)}.T" for code in nikkei_df['Code'].dropna()]
        tickers.extend(nikkei)
    except Exception as e:
        st.warning(f'Nikkei fetch failed: {e}')
        nikkei_fallback = '7203.T,6758.T,9984.T,8306.T,6501.T,9983.T,8316.T,7974.T,8035.T,6857.T,8058.T,7011.T,8001.T,6861.T,4519.T,8411.T,9432.T,6098.T,8031.T,8766.T,9434.T,2914.T,9433.T,4063.T,6503.T,7741.T,6701.T,6702.T,4502.T,8267.T,8002.T,4568.T,7267.T,6367.T,8053.T,6902.T,6981.T,8015.T,5803.T,8725.T,3382.T,4661.T,6146.T,5802.T,8801.T,5108.T,6301.T,6762.T,6954.T,7269.T,8750.T,4578.T,7751.T,8591.T,6178.T,8630.T,9020.T,6752.T,8802.T,9022.T,4901.T,2802.T,1605.T,8308.T,4307.T,6273.T,4543.T,6723.T,4503.T,1925.T,8830.T,8604.T,5401.T,9766.T,7013.T,4452.T,8309.T,3659.T,7832.T,6971.T,4689.T,5020.T,1812.T,2502.T,9503.T,6988.T,6594.T,7270.T,6920.T,6326.T,4507.T,7733.T,9735.T,1801.T,9531.T,1928.T,9101.T,4755.T,9532.T,1802.T'.split(',')
        tickers.extend(nikkei_fallback)

    return list(set(tickers))

# Rest of the code remains the same as in the previous full code.

# ... (copy the rest: calculate_fair_price, calculate_earnings_beat_prob, black_scholes, calculate_earnings_vol, get_options_value, get_upcoming_earnings, and the app part)
