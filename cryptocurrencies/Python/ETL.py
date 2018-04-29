import re
import pandas as pd
from sqlalchemy import create_engine

cryptocurrency_names_daily_data = \
    ('bitcoin', 'bitcoin_cash', 'bitconnect', 'dash', 'ethereum', 'ethereum_classic', 'iota',
     'litecoin', 'monero', 'nem', 'neo', 'numeraire', 'omisego', 'qtum', 'ripple', 'stratis', 'waves')

# SQLAlchemy for connecting to a local MySQL instance.
# mysqlconnector
engine = create_engine('mysql+pymysql://john:Iwbicvi1994mysql@localhost:3306/cryptocurrencies', echo=False)

# Daily data
def load_daily_csvs(write_to_SQL=False, set_date_index=True):
    """
    Load daily data for 17 cryptocurrencies from April 28, 2013 to November 7, 2017.
    The data was obtained from https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory.
    """
    vars = {}
    for cryptocurrency_name in cryptocurrency_names_daily_data:
        vars[cryptocurrency_name] = pd.read_csv('../data/daily/{}_price.csv'.format(cryptocurrency_name))
    if write_to_SQL:
        # Create SQL tables for the cryptocurrencies - one for each cryptocurrency.
        for cryptocurrency_name in cryptocurrency_names_daily_data:
            cryptocurrency = vars[cryptocurrency_name]
            def volume_to_int(volume):
                return 0 if volume == '-' else int(volume.replace(',', ''))
            def market_cap_to_int(market_cap):
                return 0 if market_cap == '-' else float(market_cap.replace(',', ''))
            cryptocurrency['Date'] = cryptocurrency['Date'].apply(lambda date: pd.to_datetime(date))
            cryptocurrency['Volume'] = cryptocurrency['Volume'].apply(volume_to_int)
            cryptocurrency['Market Cap'] = cryptocurrency['Market Cap'].apply(market_cap_to_int)
            cryptocurrency.sort_values('Date', inplace=True)
            # print(cryptocurrency_name, cryptocurrency.head())
            cryptocurrency.to_sql(name=cryptocurrency_name, con=engine, if_exists='replace', index=False)

    if set_date_index:
        for cryptocurrency_name in cryptocurrency_names_daily_data:
            vars[cryptocurrency_name].set_index('Date', inplace=True)

    return vars['bitcoin'], vars['bitcoin_cash'], vars['bitconnect'], vars['dash'], vars['ethereum'], \
           vars['ethereum_classic'], vars['iota'], vars['litecoin'], vars['monero'], vars['nem'], vars['neo'], \
           vars['numeraire'], vars['omisego'], vars['qtum'], vars['ripple'], vars['stratis'], vars['waves']


def load_daily_sql():
    vars = {}
    for cryptocurrency_name in cryptocurrency_names_daily_data:
        vars[cryptocurrency_name] = pd.read_sql("SELECT * FROM {}".format(cryptocurrency_name), engine, index_col="Date")

    return vars['bitcoin'], vars['bitcoin_cash'], vars['bitconnect'], vars['dash'], vars['ethereum'], \
           vars['ethereum_classic'], vars['iota'], vars['litecoin'], vars['monero'], vars['nem'], vars['neo'], \
           vars['numeraire'], vars['omisego'], vars['qtum'], vars['ripple'], vars['stratis'], vars['waves']


# Hourly data
def load_hourly_csvs(set_date_index=True):
    """
    Load hourly data for 10 cryptocurrencies from July 1, 2017 to April 27, 2018.
    The data was obtained indirectly from http://www.cryptodatadownload.com/.
    """
    cryptocurrencies_with_hourly_data = \
        [('bitcoin', 'Gdax_BTCUSD_1h'), ('dash', 'Poloniex_DASHUSD_1h'),
         ('ethereum_classic', 'Poloniex_ETCUSD_1h'), ('ethereum', 'Gdax_ETHUSD_1h'),
         ('litecoin', 'Gdax_LTCUSD_1h'), ('monero', 'Poloniex_XMRUSD_1h'), ('nem', 'Poloniex_XEMBTC_1h'),
         ('neo', 'Bittrex_NEOUSD_1h'), ('omisego', 'Bittrex_OMGUSD_1h'), ('ripple', 'Kraken_XRPUSD_1h')]
    vars = {}
    for cryptocurrency_name, filename in cryptocurrencies_with_hourly_data:
        vars[cryptocurrency_name] = pd.read_csv('../data/hourly/{}.csv'.format(filename), header=1)

    if set_date_index:
        initial_time_format = '%Y-%m-%d %I-%p'
        for cryptocurrency_name, _ in cryptocurrencies_with_hourly_data:
            vars[cryptocurrency_name].set_index(pd.to_datetime(vars[cryptocurrency_name]['Date'],
                                                               format=initial_time_format), inplace=True)
            vars[cryptocurrency_name].drop('Date', axis=1, inplace=True)
    # Nem is stored as a fraction of Bitcoin's value rather than in USD, so we have to calculate its value in USD.
    vars['nem']['Close'] = vars['nem']['Close'] * vars['bitcoin']['Close']
    return vars['bitcoin'], vars['dash'], vars['ethereum_classic'], vars['ethereum'], vars['litecoin'], \
           vars['monero'], vars['nem'], vars['neo'], vars['omisego'], vars['ripple']


if __name__ == '__main__':
    bitcoin, bitcoin_cash, bitconnect, dash, ethereum, ethereum_classic, iota, litecoin, \
    monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, waves = load_daily_csvs(write_to_SQL=True)
    bitcoin, dash, ethereum_classic, ethereum, litecoin, monero, nem, neo, omisego, ripple = load_hourly_csvs()