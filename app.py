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
        response = requests.get('https://www.slickcharts.com/sp500', headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        sp500 = [td.text.strip() for td in soup.select('table tbody tr td:nth-child(3) a')]
        tickers.extend(sp500)
    except Exception as e:
        st.warning(f'S&P500 fetch failed: {e}')
        sp500_fallback = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'AVGO', 'GOOG', 'META', 'TSLA', 'BRK.B', 'LLY', 'JPM', 'WMT', 'V', 'ORCL', 'XOM', 'MA', 'JNJ', 'NFLX', 'PLTR', 'ABBV', 'COST', 'AMD', 'BAC', 'HD', 'PG', 'GE', 'CVX', 'CSCO', 'KO', 'UNH', 'IBM', 'MU', 'WFC', 'MS', 'CAT', 'AXP', 'PM', 'TMUS', 'GS', 'RTX', 'CRM', 'MRK', 'ABT', 'MCD', 'TMO', 'PEP', 'LIN', 'ISRG', 'UBER', 'DIS', 'APP', 'QCOM', 'LRCX', 'INTU', 'T', 'AMGN', 'AMAT', 'C', 'NOW', 'NEE', 'VZ', 'INTC', 'SCHW', 'ANET', 'BLK', 'APH', 'BKNG', 'TJX', 'GEV', 'DHR', 'GILD', 'BSX', 'ACN', 'SPGI', 'KLAC', 'BA', 'TXN', 'PFE', 'PANW', 'ADBE', 'SYK', 'ETN', 'CRWD', 'COF', 'WELL', 'UNP', 'PGR', 'DE', 'LOW', 'HON', 'MDT', 'CB', 'ADI', 'PLD', 'COP', 'VRTX', 'HOOD', 'BX', 'HCA', 'LMT', 'KKR', 'CEG', 'PH', 'MCK', 'CME', 'ADP', 'CMCSA', 'SO', 'CVS', 'MO', 'SBUX', 'NEM', 'DUK', 'BMY', 'NKE', 'GD', 'TT', 'DELL', 'MMC', 'DASH', 'MMM', 'ICE', 'AMT', 'CDNS', 'MCO', 'WM', 'ORLY', 'SHW', 'HWM', 'UPS', 'NOC', 'JCI', 'EQIX', 'BK', 'MAR', 'COIN', 'APO', 'TDG', 'AON', 'CTAS', 'WMB', 'ABNB', 'MDLZ', 'ECL', 'USB', 'REGN', 'SNPS', 'MNST', 'CSX', 'RSG', 'DDOG', 'AEP', 'AZO', 'TRV', 'PWR', 'CMI', 'ADSK', 'NSC', 'MSI', 'FDX', 'CL', 'HLT', 'WDAY', 'FTNT', 'PYPL', 'MSTR', 'WBD', 'IDXX', 'ROST', 'PCAR', 'EA', 'NXPI', 'ROP', 'BKR', 'XEL', 'ZS', 'FAST', 'EXC', 'AXON', 'TTWO', 'FANG', 'CCEP', 'PAYX', 'TEAM', 'CPRT', 'KDP', 'CTSH', 'GEHC', 'VRSK', 'KHC', 'MCHP', 'CSGP', 'ODFL', 'CHTR', 'BIIB', 'DXCM', 'TTD', 'LULU', 'DRI', 'CHD', 'TYL', 'RL', 'CTRA', 'NVR', 'IP', 'AMCR', 'CPAY', 'KEY', 'ON', 'TSN', 'CDW', 'WST', 'BG', 'PFG', 'EXPD', 'J', 'TRMB', 'CHRW', 'SW', 'CNC', 'ZBH', 'PKG', 'GPC', 'EVRG', 'GPN', 'MKC', 'GDDY', 'Q', 'INVH', 'LNT', 'PSKY', 'SNA', 'PNR', 'APTV', 'LUV', 'IFF', 'IT', 'DD', 'LII', 'HOLX', 'GEN', 'ESS', 'FTV', 'DOW', 'WY', 'BBY', 'JBHT', 'MAA', 'ERIE', 'LYB', 'TKO', 'COO', 'TXT', 'UHS', 'OMC', 'ALLE', 'DPZ', 'KIM', 'FOX', 'EG', 'FOXA', 'ALB', 'FFIV', 'AVY', 'CF', 'BF.B', 'SOLV', 'NDSN', 'BALL', 'REG', 'CLX', 'MAS', 'WYNN', 'AKAM', 'HRL', 'VTRS', 'HII', 'IEX', 'ZBRA', 'HST', 'DECK', 'DOC', 'JKHY', 'SJM', 'BEN', 'UDR', 'AIZ', 'BLDR', 'BXP', 'DAY', 'CPT', 'HAS', 'PNW', 'RVTY', 'GL', 'IVZ', 'FDS', 'SWK', 'SWKS', 'EPAM', 'AES', 'ALGN', 'NWSA', 'MRNA', 'BAX', 'CPB', 'TECH', 'TAP', 'PAYC', 'ARE', 'POOL', 'AOS', 'IPG', 'MGM', 'GNRC', 'APA', 'DVA', 'HSIC', 'FRT', 'CAG', 'NCLH', 'MOS', 'CRL', 'LW', 'LKQ', 'MTCH', 'MOH', 'SOLS', 'MHK', 'NWS']
        tickers.extend(sp500_fallback)

    # NASDAQ-100
    try:
        response = requests.get('https://en.wikipedia.org/wiki/Nasdaq-100', headers=headers)
        response.raise_for_status()
        nasdaq = pd.read_html(StringIO(response.text))[4]['Ticker'].tolist()
        tickers.extend(nasdaq)
    except Exception as e:
        st.warning(f'NASDAQ fetch failed: {e}')
        nasdaq_fallback = ['ADBE', 'AMD', 'ABNB', 'GOOGL', 'GOOG', 'AMZN', 'AEP', 'AMGN', 'ADI', 'AAPL', 'AMAT', 'APP', 'ARM', 'ASML', 'AZN', 'TEAM', 'ADSK', 'ADP', 'AXON', 'BKR', 'BIIB', 'BKNG', 'AVGO', 'CDNS', 'CDW', 'CHTR', 'CTAS', 'CSCO', 'CCEP', 'CTSH', 'CMCSA', 'CEG', 'CPRT', 'CSGP', 'COST', 'CRWD', 'CSX', 'DDOG', 'DXCM', 'FANG', 'DASH', 'EA', 'EXC', 'FAST', 'FTNT', 'GEHC', 'GILD', 'GFS', 'HON', 'IDXX', 'INTC', 'INTU', 'ISRG', 'KDP', 'KLAC', 'KHC', 'LRCX', 'LIN', 'LULU', 'MAR', 'MRVL', 'MELI', 'META', 'MCHP', 'MU', 'MSFT', 'MSTR', 'MDLZ', 'MNST', 'NFLX', 'NVDA', 'NXPI', 'ORLY', 'ODFL', 'ON', 'PCAR', 'PLTR', 'PANW', 'PAYX', 'PYPL', 'PDD', 'PEP', 'QCOM', 'REGN', 'ROP', 'ROST', 'SHOP', 'SOLS', 'SBUX', 'SNPS', 'TMUS', 'TTWO', 'TSLA', 'TXN', 'TRI', 'TTD', 'VRSK', 'VRTX', 'WBD', 'WDAY', 'XEL', 'ZS']
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
        hang_fallback = ['0005.HK', '0011.HK', '0388.HK', '0939.HK', '1299.HK', '1398.HK', '2318.HK', '2388.HK', '2628.HK', '3968.HK', '3988.HK', '0002.HK', '0003.HK', '0006.HK', '0836.HK', '1038.HK', '2688.HK', '0012.HK', '0016.HK', '0017.HK', '0101.HK', '0688.HK', '0823.HK', '0960.HK', '1109.HK', '1113.HK', '1209.HK', '1997.HK', '6098.HK', '0001.HK', '0027.HK', '0066.HK', '0175.HK', '0241.HK', '0267.HK', '0288.HK', '0291.HK', '0316.HK', '0322.HK', '0386.HK', '0669.HK', '0700.HK', '0762.HK', '0857.HK', '0868.HK', '0881.HK', '0883.HK', '0941.HK', '0968.HK', '0981.HK', '0992.HK', '1044.HK', '1088.HK', '1093.HK', '1099.HK', '1177.HK', '1211.HK', '1378.HK', '1810.HK', '1876.HK', '1928.HK', '1929.HK', '2015.HK', '2020.HK', '2269.HK', '2313.HK', '2319.HK', '2331.HK', '2359.HK', '2382.HK', '2899.HK', '3690.HK', '3692.HK', '6618.HK', '6690.HK', '6862.HK', '9618.HK', '9633.HK', '9888.HK', '9961.HK', '9988.HK', '9999.HK']
        tickers.extend(hang_fallback)

    # Nikkei 225
    try:
        response = requests.get('https://indexes.nikkei.co.jp/en/nkave/index/component', headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        nikkei = [td.text.strip() + '.T' for td in soup.select('table tbody tr td:first-child')]
        tickers.extend(nikkei)
    except Exception as e:
        st.warning(f'Nikkei fetch failed: {e}')
        nikkei_fallback = ['4151.T', '4502.T', '4503.T', '4506.T', '4507.T', '4519.T', '4523.T', '4568.T', '4578.T', '4062.T', '6479.T', '6501.T', '6503.T', '6504.T', '6506.T', '6526.T', '6645.T', '6674.T', '6701.T', '6702.T', '6723.T', '6724.T', '6752.T', '6753.T', '6758.T', '6762.T', '6770.T', '6841.T', '6857.T', '6861.T', '6902.T', '6920.T', '6952.T', '6954.T', '6963.T', '6971.T', '6976.T', '6981.T', '7735.T', '7751.T', '7752.T', '8035.T', '7201.T', '7202.T', '7203.T', '7205.T', '7211.T', '7261.T', '7267.T', '7269.T', '7270.T', '7272.T', '4543.T', '4902.T', '6146.T', '7731.T', '7733.T', '7741.T', '9432.T', '9433.T', '9434.T', '9984.T', '5831.T', '7186.T', '8304.T', '8306.T', '8308.T', '8309.T', '8316.T', '8331.T', '8354.T', '8411.T', '8253.T', '8591.T', '8697.T', '8601.T', '8604.T', '8630.T', '8725.T', '8750.T', '8766.T', '8795.T', '1332.T', '2002.T', '2269.T', '2282.T', '2501.T', '2502.T', '2503.T', '2801.T', '2802.T', '2871.T', '2914.T', '3086.T', '3092.T', '3099.T', '3382.T', '7453.T', '8233.T', '8252.T', '8267.T', '9843.T', '9983.T', '2413.T', '2432.T', '3659.T', '3697.T', '4307.T', '4324.T', '4385.T', '4661.T', '4689.T', '4704.T', '4751.T', '4755.T', '6098.T', '6178.T', '6532.T', '7974.T', '9602.T', '9735.T', '9766.T', '1605.T', '3401.T', '3402.T', '3861.T', '3405.T', '3407.T', '4004.T', '4005.T', '4021.T', '4042.T', '4043.T', '4061.T', '4063.T', '4183.T', '4188.T', '4208.T', '4452.T', '4901.T', '4911.T', '6988.T', '5019.T', '5020.T', '5101.T', '5108.T', '5201.T', '5214.T', '5233.T', '5301.T', '5332.T', '5333.T', '5401.T', '5406.T', '5411.T', '3436.T', '5706.T', '5711.T', '5713.T', '5714.T', '5801.T', '5802.T', '5803.T', '2768.T', '8001.T', '8002.T', '8015.T', '8031.T', '8053.T', '8058.T', '1721.T', '1801.T', '1802.T', '1803.T', '1808.T', '1812.T', '1925.T', '1928.T', '1963.T', '5631.T', '6103.T', '6113.T', '6273.T', '6301.T', '6302.T', '6305.T', '6326.T', '6361.T', '6367.T', '6471.T', '6472.T', '6473.T', '7004.T', '7011.T', '7013.T', '7012.T', '7832.T', '7911.T', '7912.T', '7951.T', '3289.T', '8801.T', '8802.T', '8804.T', '8830.T', '9001.T', '9005.T', '9007.T', '9008.T', '9009.T', '9020.T', '9021.T', '9022.T', '9064.T', '9147.T', '9101.T', '9104.T', '9107.T', '9201.T', '9202.T', '9501.T', '9502.T', '9503.T', '9531.T', '9532.T']
        tickers.extend(nikkei_fallback)

    return list(set(tickers))

def calculate_fair_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials.transpose().iloc[0] if not stock.financials.empty else {}
        balance = stock.balance_sheet.transpose().iloc[0] if not stock.balance_sheet.empty else {}
        cashflow = stock.cashflow.transpose().iloc[0] if not stock.cashflow.empty else {}
        earnings_dates = stock.earnings_dates if not stock.earnings_dates.empty else pd.DataFrame()

        # Metrics...
        # (keep the same as before)

        # Fair vals...
        # (keep the same)

        result = {
            'Ticker': ticker,
            'Sector': sector,
            'Current Price': current_price,
            'Fair Price': round(fair_price, 2),
            'Diff Nominal': round(diff_nominal, 2),
            'Diff %': round(diff_percent, 2),
            'Value Bet': 'Yes' if diff_percent > 0 else 'No'
        }
        result.update(metrics)

        beat_prob, miss_prob = calculate_earnings_beat_prob(stock, earnings_dates, multiples)
        result['Beat Prob (1-5)'] = beat_prob
        result['Miss Prob (1-5)'] = miss_prob

        return result
    except Exception as e:
        st.warning(f'Error for {ticker}: {str(e)}')
        return None

def calculate_earnings_beat_prob(stock, earnings_dates, multiples):
    try:
        beat_rate = 0.5
        esp = 0
        if not earnings_dates.empty and 'Reported EPS' in earnings_dates.columns and 'EPS Estimate' in earnings_dates.columns:
            historical = earnings_dates[earnings_dates.index < datetime.now()]
            if not historical.empty:
                beat_rate = (historical['Reported EPS'] > historical['EPS Estimate']).mean()

            future = earnings_dates[earnings_dates.index > datetime.now()]
            if not future.empty:
                latest_est = future['EPS Estimate'].iloc[0]
                consensus = stock.info.get('forwardEps', latest_est)
                esp = ((latest_est - consensus) / consensus * 100) if consensus != 0 else 0

        earnings_growth = stock.info.get('earningsGrowth', 0)
        short_interest = stock.info.get('shortPercentOfFloat', 0)
        sector_growth = multiples['growth']

        score = 0
        if beat_rate > 0.7: score += 2
        elif beat_rate > 0.5: score += 1
        if esp > 5: score += 1
        if earnings_growth > 0.1: score += 1
        if short_interest < 5: score += 1
        if sector_growth > 0.1: score += 1
        beat_prob = min(max(score, 1), 5)

        miss_score = 0
        if beat_rate < 0.3: miss_score += 2
        elif beat_rate < 0.5: miss_score += 1
        if esp < -5: miss_score += 1
        if earnings_growth < 0: miss_score += 1
        if short_interest > 10: miss_score += 1
        if sector_growth < 0.05: miss_score += 1
        miss_prob = min(max(miss_score, 1), 5)

        return beat_prob, miss_prob
    except:
        return 0, 0

# ... (keep black_scholes, calculate_earnings_vol, get_options_value)

def get_upcoming_earnings(tickers):
    upcoming = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            earnings_dates = stock.earnings_dates
            if not earnings_dates.empty:
                earn_date = earnings_dates.index[0]
                if datetime.now() < earn_date < datetime.now() + timedelta(days=30):
                    upcoming.append(t)
        except:
            pass
    return upcoming

# App
st.title('Advanced Quant Stock & Options Analyzer')
st.markdown('Fair prices from NYU 2025 data, earnings beat probs from historical + factors. Live data.')

# Single ticker
ticker = st.text_input('Ticker:')
if ticker:
    result = calculate_fair_price(ticker.upper())
    if result:
        beat_prob = result['Beat Prob (1-5)']
        miss_prob = result['Miss Prob (1-5)']
        beat_icon = '🟢' * beat_prob + '⚪' * (5 - beat_prob) + ' ↑'
        miss_icon = '🔴' * miss_prob + '⚪' * (5 - miss_prob) + ' ↓'
        result['Beat Icon'] = beat_icon
        result['Miss Icon'] = miss_icon
        st.write(pd.DataFrame([result]))
    options_df = get_options_value(ticker.upper())
    if not options_df.empty:
        st.write('Options:')
        def color_diff(val):
            return 'background-color: green' if val > 0 else 'background-color: red'
        styled_options = options_df.sort_values('Value %', ascending=False).style.map(color_diff, subset=['Value %'])
        st.dataframe(styled_options, width='stretch')
    else:
        st.write('No options data available for this ticker.')

# All stocks
if st.button('Load & Sort All (may take time)'):
    with st.spinner('Processing...'):
        tickers = get_index_tickers()
        results = []
        for t in tickers:
            res = calculate_fair_price(t)
            if res:
                results.append(res)
        df = pd.DataFrame(results).sort_values('Diff %', ascending=False)

        for idx, row in df.iterrows():
            beat_prob = row['Beat Prob (1-5)']
            miss_prob = row['Miss Prob (1-5)']
            df.at[idx, 'Beat Icon'] = '🟢' * beat_prob + ' ↑'
            df.at[idx, 'Miss Icon'] = '🔴' * miss_prob + ' ↓'

        def color_diff(val):
            return 'background-color: green' if val > 0 else 'background-color: red'

        if not df.empty:
            styled_df = df.style.map(color_diff, subset=['Diff %'])
            st.dataframe(styled_df, width='stretch')
        else:
            st.write('No stock data available.')

        # Upcoming earnings
        st.subheader('Upcoming Earnings Filter')
        upcoming = get_upcoming_earnings(tickers)
        if upcoming:
            up_df = df[df['Ticker'].isin(upcoming)].sort_values('Beat Prob (1-5)', ascending=False)
            if not up_df.empty:
                styled_up = up_df.style.map(color_diff, subset=['Diff %'])
                st.dataframe(styled_up, width='stretch')
            else:
                st.write('No data for upcoming earnings.')
        else:
            st.write('No upcoming earnings found.')

        # Options all
        options_all = []
        for t in tickers[:50]:
            opt_df = get_options_value(t)
            if not opt_df.empty:
                options_all.append(opt_df)
        if options_all:
            st.subheader('Top Value Options')
            option_type = st.selectbox('Type:', ['ALL', 'CALL', 'PUT'])
            min_value = st.slider('Min Value %:', 0, 100, 10)
            options_df = pd.concat(options_all)
            filtered = options_df if option_type == 'ALL' else options_df[options_df['Type'] == option_type]
            filtered = filtered[filtered['Value %'] >= min_value].sort_values('Value %', ascending=False)
            if not filtered.empty:
                styled_filtered = filtered.style.map(color_diff, subset=['Value %'])
                st.dataframe(styled_filtered, width='stretch')
            else:
                st.write('No options meeting the criteria.')
        else:
            st.write('No options data available.')

st.markdown(f'Updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}') 
