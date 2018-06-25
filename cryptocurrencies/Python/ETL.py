import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# SQLAlchemy for connecting to a local MySQL instance.
# mysqlconnector
engine = create_engine('mysql+pymysql://john:Iwbicvi1994mysql@localhost:3306/cryptocurrencies', echo=False)

def load_data(resolution, date_range=None, allow_mixing=True, source='csv', write_to_SQL=False):
    """
    Load data for cryptocurrencies. The cryptocurrencies and date ranges available depend on
    the resolution of the data, which can be either daily or hourly.
    The daily data was obtained from https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory.
    The hourly data was obtained from http://www.cryptodatadownload.com/.

    Parameters
    ----------
    resolution: str
        Specifies the resolution of the data to load. Either 'daily' or 'hourly'.\n
        TODO: Document which cryptocurrencies are in daily and hourly data.
        Daily data spans April 28, 2013 to November 7, 2017.\n
        Hourly data spans July 1, 2017 to April 27, 2018 and covers fewer cryptocurrencies than daily data.
    date_range: tuple
        A 2-tuple of the start and end dates of data to retrieve, specified as strings in the format:
        `YYYY-MM-DD`. Only considered if `resolution=='daily'`.
    allow_mixing: bool
        Whether or not to allow data to be retrieved from data sources other than the primary data source if the
        primary data source does not contain data for the specified `date_range`.
        The primary data sources are determined by `resolution`. See the `data` directory's structure.
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
    def set_str_date_col_to_datetime_index_hourly(hourly_data):
        """
        Sets the index of a `pandas.DataFrame` to the "Date" column, formatting it from strings to `datetime64`.

        Parameters
        ----------
        hourly_data: pandas.core.frame.DataFrame
            The `DataFrame` to set the index to the "Date" column.

        Returns
        -------
        hourly_data: pandas.core.frame.DataFrame
        """
        initial_time_format = '%Y-%m-%d %I-%p'
        hourly_data.set_index(pd.to_datetime(hourly_data['Date'], format=initial_time_format), inplace=True)
        hourly_data.drop('Date', axis=1, inplace=True)
        return hourly_data

    daily_source_start_date = '2013-04-28'
    daily_source_end_date = '2017-11-07'
    hourly_source_start_date = '2017-07-01'
    hourly_source_end_date = '2018-04-27'

    # Variable names and data associated with loading them.
    cryptocurrency_names_daily_data = \
        ('bitcoin', 'bitcoin_cash', 'bitconnect', 'dash', 'ethereum', 'ethereum_classic', 'iota',
         'litecoin', 'monero', 'nem', 'neo', 'numeraire', 'omisego', 'qtum', 'ripple', 'stratis', 'waves')
    cryptocurrencies_with_hourly_data = \
        {'bitcoin': 'Gdax_BTCUSD_1h', 'dash': 'Poloniex_DASHUSD_1h',
         'ethereum_classic': 'Poloniex_ETCUSD_1h', 'ethereum': 'Gdax_ETHUSD_1h',
         'litecoin': 'Gdax_LTCUSD_1h', 'monero': 'Poloniex_XMRUSD_1h', 'nem': 'Poloniex_XEMBTC_1h',
         'neo': 'Bittrex_NEOUSD_1h', 'omisego': 'Bittrex_OMGUSD_1h', 'ripple': 'Kraken_XRPUSD_1h'}
    vars = {}

    # Record the starting and ending dates to retrieve data for.
    start_date, end_date = [None] * 2
    merge_hourly_to_daily = False
    if date_range is not None:
        desired_start_date, desired_end_date = date_range
        if resolution=='daily':
            start_date = desired_start_date if daily_source_start_date <= desired_start_date \
                                               <= daily_source_end_date else daily_source_start_date
            end_date = desired_end_date if daily_source_start_date <= desired_start_date <= desired_end_date \
                                           <= hourly_source_end_date else daily_source_end_date
            if daily_source_end_date < end_date:
                merge_hourly_to_daily = True
        else:
            start_date = desired_start_date if hourly_source_start_date <= desired_start_date \
                                               <= hourly_source_end_date else hourly_source_start_date
            end_date = desired_end_date if hourly_source_start_date <= start_date <= desired_end_date \
                                           <= hourly_source_end_date else hourly_source_end_date

    # Load from the appropriate data source.
    if resolution=='daily':
        for cryptocurrency_name in cryptocurrency_names_daily_data:
            vars[cryptocurrency_name] = pd.read_csv('../data/daily/{}_price.csv'.format(cryptocurrency_name)) \
                if source == 'csv' else pd.read_sql("SELECT * FROM {}".format(cryptocurrency_name), engine, index_col="Date")
    else:
        for cryptocurrency_name, filename in cryptocurrencies_with_hourly_data.items():
            vars[cryptocurrency_name] = pd.read_csv('../data/hourly/{}.csv'.format(filename), header=1)

    # Format the index to `numpy.datetime64`.
    data_source = cryptocurrency_names_daily_data if resolution=='daily' else cryptocurrencies_with_hourly_data
    for cryptocurrency_name in data_source:
        if resolution=='daily':
            vars[cryptocurrency_name].set_index(pd.to_datetime(vars[cryptocurrency_name]['Date']), inplace=True)
            vars[cryptocurrency_name].drop('Date', axis=1, inplace=True)
        else:
            vars[cryptocurrency_name] = set_str_date_col_to_datetime_index_hourly(vars[cryptocurrency_name])
        # Sort dates in ascending order (from oldest to newest).
        vars[cryptocurrency_name].sort_index(inplace=True)

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
            cryptocurrency.to_sql(name=cryptocurrency_name, con=engine, if_exists='replace', index=False)

    # Acquire the date range specified in `date_range` as a combination of daily and hourly data.
    if resolution=='daily':
        for cryptocurrency_name in cryptocurrency_names_daily_data:
            daily_data = vars[cryptocurrency_name]
            if merge_hourly_to_daily:
                filename = cryptocurrencies_with_hourly_data.get(cryptocurrency_name, None)
                if filename is not None:
                    hourly_data = pd.read_csv('../data/hourly/{}.csv'.format(filename), header=1)
                    hourly_data = set_str_date_col_to_datetime_index_hourly(hourly_data)
                    hourly_data = hourly_data.loc[pd.to_datetime(end_date):
                                                  pd.to_datetime(daily_source_end_date) + pd.DateOffset(days=1), :]
                    hourly_close_data = hourly_data[['Close']]
                    hourly_close_data_agg_daily = hourly_close_data.resample('D').mean()
                    vars[cryptocurrency_name] = pd.concat([daily_data, hourly_close_data_agg_daily])
    else:
        for cryptocurrency_name in cryptocurrencies_with_hourly_data:
            vars[cryptocurrency_name] = vars[cryptocurrency_name].loc[start_date:end_date, :]

    num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices = \
        combine_data(vars, resolution=resolution)

    return num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices

def combine_data(vars, resolution):
    """
    Combines data (stored in `vars`) for several cryptocurrencies into a single
    pandas `DataFrame`, indexed by date and containing closing values for each
    cryptocurrency in columns named with the currency labels and tickers.

    Parameters
    ----------
    vars: dict
        Mapping of cryptocurrency names (e.g. 'bitcoin') to `DataFrame` objects - each
        containing at least a 'Close' column with the closing values through time.
    resolution: str
        Specifies the resolution of the data to load. Either 'daily' or 'hourly'.\n
        Daily data spans April 28, 2013 to November 7, 2017.\n
        Hourly data spans July 1, 2017 to April 27, 2018 and covers fewer cryptocurrencies than daily data.

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
    currencies.index.name = 'Date'
    # Retrieve the closing prices for the currencies.
    prices = currencies.xs(key='Close', axis=1, level='Currency Info')
    return num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices


if __name__ == '__main__':
    # num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices = \
    #     load_data(resolution='daily', date_range=('2013-04-28', '2017-12-31'))
    num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices = \
        load_data(resolution='hourly', date_range=('2017-07-01', '2017-12-31'))
#     print(prices.index[[0,-1]], type(prices.index[0]))
#     print("num_currencies: ", num_currencies)
#     print("currencies_labels: ", currencies_labels)
#     print("currencies_tickers: ", currencies_tickers)
#     print("currencies_labels_and_tickers: ", currencies_labels_and_tickers)
#     print("prices.columns: ", prices.columns)
    import pandas as pd

    index_headers = ['batch_size', 'hidden_layer_sizes', 'dropout_rate',
                     # Parameters for Adam optimizer.
                     'lr', 'beta_1', 'beta_2']

    param_sets = [{'batch_size': 4, 'hidden_layer_sizes': str((512, 128, 32)), 'dropout_rate': 0.1,
                   'lr': 1e-4, 'beta_1': 0.7, 'beta_2': 0.8},
                  {'batch_size': 4, 'hidden_layer_sizes': str((512, 128, 32)), 'dropout_rate': 0.1,
                   'lr': 1e-4, 'beta_1': 0.7, 'beta_2': 0.9},
                  {'batch_size': 4, 'hidden_layer_sizes': str((512, 128, 32)), 'dropout_rate': 0.1,
                   'lr': 1e-4, 'beta_1': 0.7, 'beta_2': 0.9}]

    keras_param_sets_occurrences = {}

    for param_set in param_sets: # Pseudo-simulating the training loop.
        param_vals_tup = tuple(param_set.values())
        num_occurrences = keras_param_sets_occurrences.setdefault(param_vals_tup, 0)
        keras_param_sets_occurrences[param_vals_tup] = num_occurrences + 1
    # print("keras_param_sets_occurrences:", keras_param_sets_occurrences)
    keras_unique_param_sets_tuples = list(keras_param_sets_occurrences.keys())
    index = pd.MultiIndex.from_tuples(keras_unique_param_sets_tuples, names=index_headers)
    # print("index:", index)
    keras_param_sets_occurrences_series = pd.Series(list(keras_param_sets_occurrences.values()), index=index)
    # print("keras_param_sets_occurrences_series:", keras_param_sets_occurrences_series)
    num_occurrences_str = 'num_occurrences'
    keras_param_sets_occurrences_frame = keras_param_sets_occurrences_series.to_frame(num_occurrences_str)
    print("keras_param_sets_occurrences_frame:", keras_param_sets_occurrences_frame)
    keras_param_sets_occurrences_frame.reset_index(inplace=True)
    # print("occ ind:", keras_param_sets_occurrences_frame.index)
    # print("keras_param_sets_occurrences_frame.columns:", keras_param_sets_occurrences_frame.columns)
    import matplotlib.pyplot as plt
    for ind_col_name in keras_param_sets_occurrences_frame.columns:
        if ind_col_name == num_occurrences_str:
            continue
        agg_data = (keras_param_sets_occurrences_frame.groupby(ind_col_name)[num_occurrences_str]).sum()
        print(agg_data, agg_data.index, type(agg_data))
        agg_data.plot(x=ind_col_name, y=num_occurrences_str, kind='bar')
        plt.xticks(rotation=15, fontsize=6)
        # Format y-axis labels as integers.
        import math
        y = np.unique(agg_data.values)
        # Credit to https://stackoverflow.com/a/12051323/5449970 for this little code snippet.
        plt.yticks(list(range(int(math.floor(min(y))), int(math.ceil(max(y)) + 1))))
        plt.tight_layout()
        plt.show()
    print(keras_param_sets_occurrences_frame.sort_values(num_occurrences_str, ascending=True).iloc[-1,:])
