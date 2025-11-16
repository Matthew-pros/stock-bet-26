```python
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

    # S&P500 from Slickcharts
    try:
        response = requests.get('https://www.slickcharts.com/sp500', headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        sp500 = [row.find_all('td')[2].find('a').text for row in soup.find('table', class_='table').find_all('tr')[1:]]
        tickers.extend(sp500)
    except Exception as e:
        st.warning(f'S&P500 fetch failed: {e}')
        sp500_fallback = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'AVGO', 'GOOG', 'META', 'TSLA', 'BRK.B', 'LLY', 'JPM', 'WMT', 'V', 'ORCL', 'XOM', 'MA', 'JNJ', 'NFLX', 'PLTR', 'ABBV', 'COST', 'AMD', 'BAC', 'HD', 'PG', 'GE', 'CVX', 'CSCO', 'KO', 'UNH', 'IBM', 'MU', 'WFC', 'MS', 'CAT', 'AXP', 'PM', 'TMUS', 'GS', 'RTX', 'CRM', 'MRK', 'ABT', 'MCD', 'TMO', 'PEP', 'LIN', 'ISRG', 'UBER', 'DIS', 'APP', 'QCOM', 'LRCX', 'INTU', 'T', 'AMGN', 'AMAT', 'C', 'NOW', 'NEE', 'VZ', 'INTC', 'SCHW', 'ANET', 'BLK', 'APH', 'BKNG', 'TJX', 'GEV', 'DHR', 'GILD', 'BSX', 'ACN', 'SPGI', 'KLAC', 'BA', 'TXN', 'PFE', 'PANW', 'ADBE', 'SYK', 'ETN', 'CRWD', 'COF', 'WELL', 'UNP', 'PGR', 'DE', 'LOW', 'HON', 'MDT', 'CB', 'ADI', 'PLD', 'COP', 'VRTX', 'HOOD', 'BX', 'HCA', 'LMT', 'KKR', 'CEG', 'PH', 'MCK', 'CME', 'ADP', 'CMCSA', 'SO', 'CVS', 'MO', 'SBUX', 'NEM', 'DUK', 'BMY', 'NKE', 'GD', 'TT', 'DELL', 'MMC', 'DASH', 'MMM', 'ICE', 'AMT', 'CDNS', 'MCO', 'WM', 'ORLY', 'SHW', 'HWM', 'UPS', 'NOC', 'JCI', 'EQIX', 'BK', 'MAR', 'COIN', 'APO', 'TDG', 'AON', 'CTAS', 'WMB', 'ABNB', 'MDLZ', 'ECL', 'USB', 'REGN', 'SNPS', 'MNST', 'CSX', 'RSG', 'DDOG', 'AEP', 'AZO', 'TRV', 'PWR', 'CMI', 'ADSK', 'NSC', 'MSI', 'FDX', 'CL', 'HLT', 'WDAY', 'FTNT', 'PYPL', 'MSTR', 'WBD', 'IDXX', 'ROST', 'PCAR', 'EA', 'NXPI', 'ROP', 'BKR', 'XEL', 'ZS', 'FAST', 'EXC', 'AXON', 'TTWO', 'FANG', 'CCEP', 'PAYX', 'TEAM', 'CPRT', 'KDP', 'CTSH', 'GEHC', 'VRSK', 'KHC', 'MCHP', 'CSGP', 'ODFL', 'CHTR', 'BIIB', 'DXCM', 'TTD', 'LULU', 'DRI', 'CHD', 'TYL', 'RL', 'CTRA', 'NVR', 'IP', 'AMCR', 'CPAY', 'KEY', 'ON', 'TSN', 'CDW', 'WST', 'BG', 'PFG', 'EXPD', 'J', 'TRMB', 'CHRW', 'SW', 'CNC', 'ZBH', 'PKG', 'GPC', 'EVRG', 'GPN', 'MKC', 'GDDY', 'Q', 'INVH', 'LNT', 'PSKY', 'SNA', 'PNR', 'APTV', 'LUV', 'IFF', 'IT', 'DD', 'LII', 'HOLX', 'GEN', 'ESS', 'FTV', 'DOW', 'WY', 'BBY', 'JBHT', 'MAA', 'ERIE', 'LYB', 'TKO', 'COO', 'TXT', 'UHS', 'OMC', 'ALLE', 'DPZ', 'KIM', 'FOX', 'EG', 'FOXA', 'ALB', 'FFIV', 'AVY', 'CF', 'BF.B', 'SOLV', 'NDSN', 'BALL', 'REG', 'CLX', 'MAS', 'WYNN', 'AKAM', 'HRL', 'VTRS', 'HII', 'IEX', 'ZBRA', 'HST', 'DECK', 'DOC', 'JKHY', 'SJM', 'BEN', 'UDR', 'AIZ', 'BLDR', 'BXP', 'DAY', 'CPT', 'HAS', 'PNW', 'RVTY', 'GL', 'IVZ', 'FDS', 'SWK', 'SWKS', 'EPAM', 'AES', 'ALGN', 'NWSA', 'MRNA', 'BAX', 'CPB', 'TECH', 'TAP', 'PAYC', 'ARE', 'POOL', 'AOS', 'IPG', 'MGM', 'GNRC', 'APA', 'DVA', 'HSIC', 'FRT', 'CAG', 'NCLH', 'MOS', 'CRL', 'LW', 'LKQ', 'MTCH', 'MOH', 'SOLS', 'MHK', 'NWS']
        tickers.extend(sp500_fallback)

    # NASDAQ-100
    try:
        response = requests.get('https://www.slickcharts.com/nasdaq100', headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        nasdaq = [row.find_all('td')[2].find('a').text for row in soup.find('table', class_='table').find_all('tr')[1:]]
        tickers.extend(nasdaq)
    except Exception as e:
        st.warning(f'NASDAQ fetch failed: {e}')
        nasdaq_fallback = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'AVGO', 'GOOG', 'META', 'TSLA', 'NFLX', 'PLTR', 'COST', 'AMD', 'ASML', 'CSCO', 'MU', 'AZN', 'TMUS', 'PEP', 'LIN', 'ISRG', 'SHOP', 'APP', 'QCOM', 'LRCX', 'PDD', 'INTU', 'AMGN', 'AMAT', 'INTC', 'BKNG', 'GILD', 'KLAC', 'ARM', 'TXN', 'PANW', 'ADBE', 'CRWD', 'HON', 'ADI', 'VRTX', 'CEG', 'MELI', 'ADP', 'CMCSA', 'SBUX', 'DASH', 'CDNS', 'ORLY', 'MAR', 'CTAS', 'MRVL', 'ABNB', 'MDLZ', 'REGN', 'SNPS', 'MNST', 'CSX', 'DDOG', 'AEP', 'ADSK', 'TRI', 'WDAY', 'FTNT', 'PYPL', 'MSTR', 'WBD', 'IDXX', 'ROST', 'PCAR', 'EA', 'NXPI', 'ROP', 'BKR', 'XEL', 'ZS', 'FAST', 'EXC', 'AXON', 'TTWO', 'FANG', 'CCEP', 'PAYX', 'TEAM', 'CPRT', 'KDP', 'CTSH', 'GEHC', 'VRSK', 'KHC', 'MCHP', 'CSGP', 'ODFL', 'CHTR', 'BIIB', 'DXCM', 'TTD', 'LULU', 'ON', 'CDW', 'GFS']
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
        hang_fallback = ['0005.HK', '0011.HK', '00388.HK', '00939.HK', '01299.HK', '01398.HK', '02318.HK', '02388.HK', '02628.HK', '03968.HK', '03988.HK', '0002.HK', '0003.HK', '0006.HK', '0836.HK', '01038.HK', '02688.HK', '0012.HK', '0016.HK', '0017.HK', '0101.HK', '0688.HK', '0823.HK', '0960.HK', '01109.HK', '01113.HK', '01209.HK', '01997.HK', '06098.HK', '0001.HK', '0027.HK', '0066.HK', '0175.HK', '0241.HK', '0267.HK', '0288.HK', '0291.HK', '0316.HK', '0322.HK', '0386.HK', '0669.HK', '0700.HK', '0762.HK', '0857.HK', '0868.HK', '0881.HK', '0883.HK', '0941.HK', '0968.HK', '0981.HK', '0992.HK', '01044.HK', '01088.HK', '01093.HK', '01099.HK', '01177.HK', '01211.HK', '01378.HK', '01810.HK', '01876.HK', '01928.HK', '01929.HK', '02015.HK', '02020.HK', '02269.HK', '02313.HK', '02319.HK', '02331.HK', '02359.HK', '02382.HK', '02899.HK', '03690.HK', '03692.HK', '06618.HK', '06690.HK', '06862.HK', '09618.HK', '09633.HK', '09888.HK', '09961.HK', '09988.HK', '09999.HK']
        tickers.extend(hang_fallback)

    # Nikkei
    try:
        response = requests.get('https://indexes.nikkei.co.jp/en/nkave/index/component', headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        nikkei = [a.text.strip() + '.T' for a in soup.find_all('a', class_='ticker') if a.text.strip().isdigit()]
        tickers.extend(nikkei)
    except Exception as e:
        st.warning(f'Nikkei fetch failed: {e}')
        nikkei_fallback = ['7203.T', '6758.T', '9984.T', '8306.T', '6501.T', '9983.T', '8316.T', '7974.T', '8035.T', '6857.T', '8058.T', '7011.T', '8001.T', '6861.T', '4519.T', '8411.T', '9432.T', '6098.T', '8031.T', '8766.T', '9434.T', '2914.T', '9433.T', '4063.T', '6503.T', '7741.T', '6701.T', '6702.T', '4502.T', '8267.T', '8002.T', '4568.T', '7267.T', '6367.T', '8053.T', '6902.T', '6981.T', '8015.T', '5803.T', '8725.T', '3382.T', '4661.T', '6146.T', '5802.T', '8801.T', '5108.T', '6301.T', '6762.T', '6954.T', '7269.T', '8750.T', '4578.T', '7751.T', '8591.T', '6178.T', '8630.T', '9020.T', '6752.T', '8802.T', '9022.T', '4901.T', '2802.T', '1605.T', '8308.T', '4307.T', '6273.T', '4543.T', '6723.T', '4503.T', '1925.T', '8830.T', '8604.T', '5401.T', '9766.T', '7013.T', '4452.T', '8309.T', '3659.T', '7832.T', '6971.T', '4689.T', '5020.T', '1812.T', '2502.T', '9503.T', '6988.T', '6594.T', '7270.T', '6920.T', '6326.T', '4507.T', '7733.T', '9735.T', '1801.T', '9531.T', '1928.T', '9101.T', '4755.T', '9532.T', '1802.T']
        tickers.extend(nikkei_fallback)

    return list(set(tickers))

def calculate_fair_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or 'regularMarketPrice' not in info:  # Check for valid symbol data
            raise ValueError("No symbol data found")
        financials = stock.financials.transpose().iloc[0] if not stock.financials.empty else {}
        balance = stock.balance_sheet.transpose().iloc[0] if not stock.balance_sheet.empty else {}
        cashflow = stock.cashflow.transpose().iloc[0] if not stock.cashflow.empty else {}
        earnings_dates = stock.earnings_dates if hasattr(stock, 'earnings_dates') else pd.DataFrame()

        # Metrics
        metrics = {
            'EPS': info.get('trailingEps', 0),
            'Forward EPS': info.get('forwardEps', 0),
            'ROE': info.get('returnOnEquity', 0),
            'ROA': info.get('returnOnAssets', 0),
            'Beta': info.get('beta', 1),
            'Debt/Equity': info.get('debtToEquity', 0),
            'Revenue Growth': info.get('revenueGrowth', 0),
            'PEG Ratio': info.get('pegRatio', 0),
            'P/S Trailing': info.get('priceToSalesTrailing12Months', 0),
            'EV/Revenue': info.get('enterpriseToRevenue', 0),
            'EV/EBITDA': info.get('enterpriseToEbitda', 0),
            'Profit Margins': info.get('profitMargins', 0),
            'Operating Margins': info.get('operatingMargins', 0),
            'Gross Margins': info.get('grossMargins', 0),
            'Quick Ratio': info.get('quickRatio', 0),
            'Current Ratio': info.get('currentRatio', 0),
            'Book Value': info.get('bookValue', 0),
            'EBITDA': financials.get('EBITDA', 0),
            'Free Cashflow': info.get('freeCashflow', 0),
            'Operating Cashflow': info.get('operatingCashflow', 0),
            'Total Debt': info.get('totalDebt', 0),
            'Total Revenue': info.get('totalRevenue', 0),
            'Earnings Growth': info.get('earningsGrowth', 0),
            'Payout Ratio': info.get('payoutRatio', 0),
            'Trailing Annual Div Yield': info.get('trailingAnnualDividendYield', 0),
            'Five Year Avg Div Yield': info.get('fiveYearAvgDividendYield', 0),
            'Trailing PE': info.get('trailingPE', 0),
            'Forward PE': info.get('forwardPE', 0),
            'Shares Outstanding': info.get('sharesOutstanding', 1),
            'Market Cap': info.get('marketCap', 0),
            'Enterprise Value': info.get('enterpriseValue', 0),
            'Net Income': info.get('netIncomeToCommon', 0),
            'Gross Profits': info.get('grossProfits', 0),
            'EBITDA Margins': info.get('ebitdaMargins', 0),
            'Op Cashflow/Share': info.get('operatingCashflowPerShare', 0),
            'Free Cashflow/Share': info.get('freeCashflowPerShare', 0),
            'Cash/Share': info.get('cashPerShare', 0),
            'Tangible Book Value': info.get('tangibleBookValue', 0),
            'Shareholders Equity/Share': info.get('shareholdersEquityPerShare', 0),
            'Interest Debt/Share': info.get('interestDebtPerShare', 0),
            'Price To Book': info.get('priceToBook', 0),
            'Price To Free Cashflow': info.get('priceToFreeCashFlow', 0),
            'Price Fair Value': info.get('priceFairValue', 0),
            'Short Interest': info.get('shortPercentOfFloat', 0),
        }

        bvps = metrics['Book Value']
        eps = metrics['EPS']
        forward_eps = metrics['Forward EPS']
        roe = metrics['ROE']
        payout = metrics['Payout Ratio']
        growth = (1 - payout) * roe if roe > 0 else sector_multiples.get(sector_mapping.get(info.get('sector', 'Other'), 'Other'), {}).get('growth', 0.05)
        cost_equity = 0.03 + metrics['Beta'] * 0.06
        ebitda = metrics['EBITDA']
        shares = metrics['Shares Outstanding']
        net_debt = metrics['Total Debt'] - info.get('totalCash', 0)
        revenue_ps = info.get('revenuePerShare', 0)
        current_price = info.get('currentPrice', 0)

        sector = info.get('sector', 'Other')
        multiples = sector_multiples.get(sector_mapping.get(sector, 'Other'), sector_multiples['Other'])

        # Fair vals
        fair_current_pe = eps * multiples['current_pe'] if eps > 0 else 0
        fair_forward_pe = forward_eps * multiples['forward_pe'] if forward_eps > 0 else 0
        fair_pb = bvps * multiples['pb']
        fair_ps = revenue_ps * multiples['ps']
        fair_evebitda = max(0, (ebitda * multiples['evebitda'] - net_debt) / shares if shares > 0 else 0)
        fair_ddm = (eps * (1 - payout) / (cost_equity - growth)) if (cost_equity > growth > 0) else (fair_current_pe + fair_forward_pe) / 2

        # Optimized weights
        fair_price = 0.3 * fair_current_pe + 0.25 * fair_forward_pe + 0.15 * fair_pb + 0.15 * fair_ps + 0.15 * fair_evebitda + 0.05 * fair_ddm
        fair_price /= multiples['peg'] if multiples['peg'] > 0 else 1
        fair_price *= (1 - 0.05 * (metrics['Debt/Equity'] / 100))
        fair_price *= 0.95

        diff_nominal = fair_price - current_price
        diff_percent = (diff_nominal / current_price * 100) if current_price > 0 else 0

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

        # Earnings beat
        beat_prob, miss_prob = calculate_earnings_beat_prob(stock, earnings_dates, multiples)
        result['Beat Prob (1-5)'] = beat_prob
        result['Miss Prob (1-5)'] = miss_prob

        return result
    except Exception as e:
        st.warning(f'Error for {ticker}: {str(e)}')
        return None

def calculate_earnings_beat_prob(stock, earnings_dates, multiples):
    try:
        if not earnings_dates.empty and 'Reported EPS' in earnings_dates.columns and 'EPS Estimate' in earnings_dates.columns:
            beat_rate = (earnings_dates['Reported EPS'] > earnings_dates['EPS Estimate']).mean()
        else:
            beat_rate = 0.5

        # ESP approximation
        if not earnings_dates.empty:
            latest_est = earnings_dates.iloc[0]['EPS Estimate']
            consensus = stock.info.get('forwardEps', latest_est)
            esp = ((latest_est - consensus) / consensus * 100) if consensus != 0 else 0
        else:
            esp = 0

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

def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_earnings_vol(ticker):
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.quarterly_earnings['Earnings'].pct_change().dropna()
        return earnings.std() * np.sqrt(4) if len(earnings) >= 4 else 0.2
    except:
        return 0.2

def get_options_value(ticker, r=0.04):
    try:
        stock = yf.Ticker(ticker)
        current = stock.info.get('currentPrice', 0)
        sigma = calculate_earnings_vol(ticker)
        options_data = []
        for exp in stock.options[:3]:
            chain = stock.option_chain(exp)
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            T = (exp_date - datetime.now()).days / 365
            for df, opt_type in [(chain.calls, 'call'), (chain.puts, 'put')]:
                for _, row in df.iterrows():
                    K = row['strike']
                    market = row['lastPrice']
                    theo = black_scholes(current, K, T, r, sigma, opt_type)
                    value_diff = (theo - market) / market * 100 if market > 0 else 0
                    options_data.append({
                        'Ticker': ticker,
                        'Type': opt_type.upper(),
                        'Strike': K,
                        'Expiration': exp,
                        'Market Price': market,
                        'Theo Price': round(theo, 2),
                        'Value %': round(value_diff, 2),
                        'Vol': round(sigma, 2)
                    })
        return pd.DataFrame(options_data)
    except:
        return pd.DataFrame()

def get_upcoming_earnings(tickers):
    upcoming = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            cal = stock.calendar
            if not cal.empty and 'Earnings Date' in cal.index:
                earn_date = cal['Earnings Date'][0]
                if isinstance(earn_date, pd.Series):
                    earn_date = earn_date.iloc[0]
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
        if 'Value %' in options_df.columns:
            styled_options = options_df.sort_values('Value %', ascending=False).style.map(color_diff, subset=['Value %'])
            st.dataframe(styled_options, width='stretch')
        else:
            st.dataframe(options_df, width='stretch')

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

        if 'Diff %' in df.columns:
            styled_df = df.style.map(color_diff, subset=['Diff %'])
            st.dataframe(styled_df, width='stretch')
        else:
            st.dataframe(df, width='stretch')

        # Upcoming earnings
        st.subheader('Upcoming Earnings Filter')
        upcoming = get_upcoming_earnings(tickers)
        if upcoming:
            up_df = df[df['Ticker'].isin(upcoming)].sort_values('Beat Prob (1-5)', ascending=False)
            if 'Diff %' in up_df.columns:
                styled_up = up_df.style.map(color_diff, subset=['Diff %'])
                st.dataframe(styled_up, width='stretch')
            else:
                st.dataframe(up_df, width='stretch')
        else:
            st.write('No upcoming earnings found.')

        # Options
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
            if not filtered.empty and 'Value %' in filtered.columns:
                styled_filtered = filtered.style.map(color_diff, subset=['Value %'])
                st.dataframe(styled_filtered, width='stretch')
            else:
                st.dataframe(filtered, width='stretch')

st.markdown(f'Updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}') 
```
