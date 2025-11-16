import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Dictionary průměrných multiples by sector (z 2025 research: NYU Stern, Eqvista, Siblis)
sector_multiples = {
    'Technology': {'pe': 40.65, 'pb': 6.5, 'evebitda': 18.2},
    'Financial Services': {'pe': 15.2, 'pb': 1.32, 'evebitda': 9.5},
    'Real Estate': {'pe': 39.5, 'pb': 2.1, 'evebitda': 15.8},
    'Consumer Discretionary': {'pe': 29.21, 'pb': 4.2, 'evebitda': 12.5},
    'Industrials': {'pe': 24.41, 'pb': 3.5, 'evebitda': 11.0},
    'Healthcare': {'pe': 24.93, 'pb': 4.0, 'evebitda': 13.5},
    'Energy': {'pe': 12.5, 'pb': 1.8, 'evebitda': 7.8},
    'Utilities': {'pe': 18.0, 'pb': 1.9, 'evebitda': 10.2},
    'Communication Services': {'pe': 21.65, 'pb': 3.0, 'evebitda': 9.0},
    'Consumer Staples': {'pe': 20.5, 'pb': 4.5, 'evebitda': 12.0},
    'Materials': {'pe': 16.0, 'pb': 2.2, 'evebitda': 8.5},
    # Přidej další jestli potřeba, fallback na average
    'Other': {'pe': 22.0, 'pb': 3.0, 'evebitda': 11.0}
}

# Funkce pro fetch list tickers z indexů
@st.cache_data(ttl=86400)  # Cache na 24h
def get_index_tickers():
    # S&P500 z Wikipedia
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500 = pd.read_html(sp500_url)[0]['Symbol'].tolist()
    
    # NASDAQ100 z Slickcharts
    nasdaq_url = 'https://www.slickcharts.com/nasdaq100'
    response = requests.get(nasdaq_url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text, 'html.parser')
    nasdaq = [row.find_all('td')[2].text for row in soup.find_all('tr')[1:]]
    
    # Hang Seng z Investing.com (scraping, ale pro prod použij API)
    hang_url = 'https://www.investing.com/indices/hang-sen-40-components'
    response = requests.get(hang_url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text, 'html.parser')
    hang = [td.text.strip() + '.HK' for td in soup.select('td.left.noWrap a')]  # Přidej .HK pro yfinance
    
    # Nikkei z Markets Insider
    nikkei_url = 'https://markets.businessinsider.com/index/components/nikkei_225'
    response = requests.get(nikkei_url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text, 'html.parser')
    nikkei = [row.find_all('td')[0].text.strip() + '.T' for row in soup.find_all('tr')[1:]]  # .T pro Tokyo
    
    all_tickers = list(set(sp500 + nasdaq + hang + nikkei))  # Unique
    return all_tickers

# Funkce pro výpočet fair price
def calculate_fair_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
        balance = stock.balance_sheet
        cashflow = stock.cashflow
        
        # Klíčové metrics
        eps = info.get('trailingEps', 0)
        bvps = info.get('bookValue', 0)
        ebitda = financials.loc['EBITDA'].iloc[0] if 'EBITDA' in financials.index else 0
        shares = info.get('sharesOutstanding', 1)
        net_debt = balance.loc['Long Term Debt'].iloc[0] + balance.loc['Short Long Term Debt'].iloc[0] - balance.loc['Cash'].iloc[0] if all(k in balance.index for k in ['Long Term Debt', 'Short Long Term Debt', 'Cash']) else 0
        current_price = info.get('currentPrice', 0)
        
        # Sector
        sector = info.get('sector', 'Other')
        multiples = sector_multiples.get(sector, sector_multiples['Other'])
        
        # Fair valuations
        fair_pe = eps * multiples['pe'] if eps > 0 else 0
        fair_pb = bvps * multiples['pb']
        fair_evebitda = (ebitda * multiples['evebitda'] - net_debt) / shares if shares > 0 else 0
        
        # Weighted fair price
        fair_price = 0.4 * fair_pe + 0.3 * fair_pb + 0.3 * fair_evebitda
        fair_price *= 0.95  # 5% marže pro konzervativnost
        
        diff_nominal = fair_price - current_price
        diff_percent = (diff_nominal / current_price * 100) if current_price > 0 else 0
        
        return {
            'Ticker': ticker,
            'Sector': sector,
            'Current Price': current_price,
            'Fair Price': round(fair_price, 2),
            'Diff Nominal': round(diff_nominal, 2),
            'Diff %': round(diff_percent, 2),
            'Value Bet': 'Yes' if diff_percent > 0 else 'No'
        }
    except Exception as e:
        return None

# Streamlit app
st.title('Quant Fair Price Calculator & Value Bets App')
st.markdown('Vypočítává férovou cenu akcie podle fundamentálů, ignoruje tržní cenu. Sorted list pro value bets z US/CN/JP indexů.')

# Sekce 1: Single ticker
ticker = st.text_input('Zadej ticker (např. AAPL):')
if ticker:
    result = calculate_fair_price(ticker.upper())
    if result:
        st.write(pd.DataFrame([result]))
    else:
        st.error('Chyba při fetch dat. Zkus jiný ticker.')

# Sekce 2: Sorted list všech akcií
if st.button('Načti a seřaď všechny akcie (může trvat 5-10 min)'):
    with st.spinner('Fetching a calculating...'):
        tickers = get_index_tickers()
        results = []
        for t in tickers:
            res = calculate_fair_price(t)
            if res:
                results.append(res)
        df = pd.DataFrame(results)
        df = df.sort_values('Diff %', ascending=False).reset_index(drop=True)
        st.dataframe(df)
        st.download_button('Stáhni CSV', df.to_csv(), 'value_bets.csv')

st.markdown(f'Aktualizováno: {datetime.now().strftime("%Y-%m-%d %H:%M")} | Data z yfinance (live).')
