import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from scipy.stats import norm
import numpy as np
import lxml  # Ensure lxml is installed via requirements.txt

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
    tickers = []
    try:
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        sp500 = pd.read_html(sp500_url)[0]['Symbol'].tolist()
        tickers.extend(sp500)
    except Exception as e:
        st.warning(f'S&P500 fetch failed: {e}')
        tickers.extend(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])

    try:
        nasdaq_url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        nasdaq = pd.read_html(nasdaq_url)[4]['Ticker'].tolist()
        tickers.extend(nasdaq)
    except Exception as e:
        st.warning(f'NASDAQ fetch failed: {e}')

    try:
        hang_url = 'https://en.wikipedia.org/wiki/Hang_Seng_Index'
        hang_df = pd.read_html(hang_url)[2]
        hang = hang_df['Ticker'].tolist()
        hang = [str(code).zfill(4) + '.HK' for code in hang_df['Ticker'].str.extract(r'(\d+)', expand=False).dropna().astype(int)]
        tickers.extend(hang)
    except Exception as e:
        st.warning(f'Hang Seng fetch failed: {e}')
        tickers.extend(['0005.HK', '0011.HK', '0388.HK'])  # Fallback examples

    try:
        nikkei_url = 'https://en.wikipedia.org/wiki/Nikkei_225'
        nikkei_df = pd.read_html(nikkei_url)[3]
        nikkei = nikkei_df['Code'].tolist()
        nikkei = [str(int(code)) + '.T' for code in nikkei if not pd.isna(code)]
        tickers.extend(nikkei)
    except Exception as e:
        st.warning(f'Nikkei fetch failed: {e}')
        tickers.extend(['7203.T', '9984.T', '6758.T'])  # Fallback

    return list(set(tickers))[:500]

def calculate_fair_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
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
        if not earnings_dates.empty and 'Reported EPS' in earnings_dates and 'EPS Estimate' in earnings_dates:
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
            if 'Earnings Date' in cal and not cal['Earnings Date'].empty:
                earn_date = cal['Earnings Date'][0]
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

        styled_df = df.style.map(color_diff, subset=['Diff %'])
        st.dataframe(styled_df, width='stretch')

        # Upcoming earnings
        st.subheader('Upcoming Earnings Filter')
        upcoming = get_upcoming_earnings(tickers)
        if upcoming:
            up_df = df[df['Ticker'].isin(upcoming)].sort_values('Beat Prob (1-5)', ascending=False)
            styled_up = up_df.style.map(color_diff, subset=['Diff %'])
            st.dataframe(styled_up, width='stretch')
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
            styled_filtered = filtered.style.map(color_diff, subset=['Value %'])
            st.dataframe(styled_filtered, width='stretch')

st.markdown(f'Updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}') 
