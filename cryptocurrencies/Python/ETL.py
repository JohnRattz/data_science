import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# SQLAlchemy for connecting to a local MySQL instance.
# mysqlconnector
engine = create_engine('mysql+pymysql://john:Iwbicvi1994mysql@localhost:3306/cryptocurrencies', echo=False)

# Daily data
def load_data(resolution, source='csv', write_to_SQL=False):
    """
    Load daily data for 17 cryptocurrencies from April 28, 2013 to November 7, 2017.
    The data was obtained from https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory.

    Parameters
    ----------
    resolution: str
        Specifies the resolution of the data to load. Either 'daily' or 'hourly'.\n
        Daily data spans April 28, 2013 to November 7, 2017.\n
        Hourly data spans July 1, 2017 to April 27, 2018.
    source: str
        Specifies the data source. Can be either 'csv', which loads from the `data` directory, or 'sql',
        which loads from a MySQL server running locally (only works on my PC and with daily data).
    write_to_SQL: bool
        Whether to write the loaded data into the local MySQL server (only works on my PC and with daily data).

    Returns
    -------
    num_currencies: int
        The number of cryptocurrencies in the data.
    currencies_labels: list
        Some cryptocurrency labels.
    currencies_tickers: list
        Standard tickers for the cryptocurrencies in `currencies_labels`.
    currencies_labels_and_tickers: list
        Labels followed by their tickers. This and the preceding two return values are parallel containers.
    prices: pandas.core.frame.DataFrame
        A `DataFrame` containing the closing values for several cryptocurrencies. Indexed by date.
    """
    # Variable names and data associated with loading them.
    cryptocurrency_names_daily_data = \
        ('bitcoin', 'bitcoin_cash', 'bitconnect', 'dash', 'ethereum', 'ethereum_classic', 'iota',
         'litecoin', 'monero', 'nem', 'neo', 'numeraire', 'omisego', 'qtum', 'ripple', 'stratis', 'waves')
    cryptocurrencies_with_hourly_data = \
        [('bitcoin', 'Gdax_BTCUSD_1h'), ('dash', 'Poloniex_DASHUSD_1h'),
         ('ethereum_classic', 'Poloniex_ETCUSD_1h'), ('ethereum', 'Gdax_ETHUSD_1h'),
         ('litecoin', 'Gdax_LTCUSD_1h'), ('monero', 'Poloniex_XMRUSD_1h'), ('nem', 'Poloniex_XEMBTC_1h'),
         ('neo', 'Bittrex_NEOUSD_1h'), ('omisego', 'Bittrex_OMGUSD_1h'), ('ripple', 'Kraken_XRPUSD_1h')]

    vars = {}
    if resolution=='daily':
        for cryptocurrency_name in cryptocurrency_names_daily_data:
            vars[cryptocurrency_name] = pd.read_csv('../data/daily/{}_price.csv'.format(cryptocurrency_name)) \
                if source == 'csv' else pd.read_sql("SELECT * FROM {}".format(cryptocurrency_name), engine, index_col="Date")
    else:
        for cryptocurrency_name, filename in cryptocurrencies_with_hourly_data:
            vars[cryptocurrency_name] = pd.read_csv('../data/hourly/{}.csv'.format(filename), header=1)

    if resolution=='daily' and write_to_SQL:
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
            cryptocurrency.to_sql(name=cryptocurrency_name, con=engine, if_exists='replace', index=False)

    if resolution=='daily':
        for cryptocurrency_name in cryptocurrency_names_daily_data:
            vars[cryptocurrency_name].set_index('Date', inplace=True)
    else:
        initial_time_format = '%Y-%m-%d %I-%p'
        for cryptocurrency_name, _ in cryptocurrencies_with_hourly_data:
            vars[cryptocurrency_name].set_index(pd.to_datetime(vars[cryptocurrency_name]['Date'],
                                                               format=initial_time_format), inplace=True)
            vars[cryptocurrency_name].drop('Date', axis=1, inplace=True)
    num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices = \
        combine_data(vars, resolution=resolution)
    return num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices

def combine_data(vars, resolution):
    """
    Combines data into a single pandas `DataFrame`, indexed by date.

    Parameters
    ----------
    vars: dict
        Mapping of cryptocurrency names (e.g. 'bitcoin') to `DataFrame` objects.
    resolution: str
        Specifies the resolution of the data to load. Either 'daily' or 'hourly'.\n
        Daily data spans April 28, 2013 to November 7, 2017.\n
        Hourly data spans July 1, 2017 to April 27, 2018.

    Returns
    -------
    num_currencies: int
        The number of cryptocurrencies in the data.
    currencies_labels: list
        Some cryptocurrency labels.
    currencies_tickers: list
        Standard tickers for the cryptocurrencies in `currencies_labels`.
    currencies_labels_and_tickers: list
        Labels followed by their tickers. This and the preceding two return values are parallel containers.
    prices: pandas.core.frame.DataFrame
        A `DataFrame` containing the closing values for several cryptocurrencies.
    """
    currencies_labels_tickers = None
    if resolution == 'daily':
        currencies_labels_tickers = np.array(
            [['Bitcoin Cash', 'BCH'], ['Bitcoin', 'BTC'], ['BitConnect', 'BCC'], ['Dash', 'DASH'],
             ['Ethereum Classic', 'ETC'], ['Ethereum', 'ETH'], ['Iota', 'MIOTA'], ['Litecoin', 'LTC'],
             ['Monero', 'XMR'], ['Nem', 'XEM'], ['Neo', 'NEO'], ['Numeraire', 'NMR'], ['Omisego', 'OMG'],
             ['Qtum', 'QTUM'], ['Ripple', 'XRP'], ['Stratis', 'STRAT'], ['Waves', 'WAVES']])
    else:
        currencies_labels_tickers = np.array(
            [['Bitcoin', 'BTC'], ['Dash', 'DASH'], ['Ethereum Classic', 'ETC'], ['Ethereum', 'ETH'],
             ['Litecoin', 'LTC'], ['Monero', 'XMR'], ['Nem', 'XEM'], ['Neo', 'NEO'], ['Omisego', 'OMG'],
             ['Ripple', 'XRP']])

    currencies_labels = currencies_labels_tickers[:, 0]
    currencies_tickers = currencies_labels_tickers[:, 1]
    currencies_labels_and_tickers = ["{} ({})".format(currencies_label, currencies_ticker)
                                     for currencies_label, currencies_ticker in
                                     zip(currencies_labels, currencies_tickers)]
    num_currencies = len(currencies_labels_tickers)

    data_to_concat = None
    if resolution == 'daily':
        data_to_concat = [vars['bitcoin_cash'], vars['bitcoin'], vars['bitconnect'], vars['dash'],
                          vars['ethereum_classic'], vars['ethereum'], vars['iota'], vars['litecoin'], vars['monero'],
                          vars['nem'], vars['neo'], vars['numeraire'], vars['omisego'], vars['qtum'], vars['ripple'],
                          vars['stratis'], vars['waves']]
    else:
        data_to_concat = [vars['bitcoin'], vars['dash'], vars['ethereum_classic'], vars['ethereum'], vars['litecoin'],
                          vars['monero'], vars['nem'], vars['neo'], vars['omisego'], vars['ripple']]

    currencies = pd.concat(data_to_concat, axis=1, keys=currencies_labels_and_tickers)
    currencies.columns.names = ['Name', 'Currency Info']
    currencies.set_index(pd.to_datetime(currencies.index), inplace=True)
    currencies.index.name = 'Date'
    currencies.sort_index(inplace=True)
    # Retrieve the closing prices for the currencies.
    prices = currencies.xs(key='Close', axis=1, level='Currency Info')
    return num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices


# if __name__ == '__main__':
#     num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices = \
#         load_data(resolution='daily')
#     num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices = \
#         load_data(resolution='hourly')
#     print("num_currencies: ", num_currencies)
#     print("currencies_labels: ", currencies_labels)
#     print("currencies_tickers: ", currencies_tickers)
#     print("currencies_labels_and_tickers: ", currencies_labels_and_tickers)
#     print("prices.columns: ", prices.columns)
#     print("prices.head(1): ", prices.head(1))
