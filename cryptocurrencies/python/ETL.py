import pandas as pd
from sqlalchemy import create_engine

cryptocurrency_names = \
    ('bitcoin', 'bitcoin_cash', 'bitconnect', 'dash', 'ethereum', 'ethereum_classic', 'iota', 'litecoin', \
     'monero', 'nem', 'neo', 'numeraire', 'omisego', 'qtum', 'ripple', 'stratis', 'waves')

bitcoin = None; bitcoin_cash = None; bitconnect = None; dash = None; ethereum_classic = None; ethereum = None;
iota = None; litecoin = None; monero = None; nem = None; neo = None; numeraire = None; omisego = None; qtum = None;
ripple = None; stratis = None; waves = None;

engine = create_engine('mysql+mysqlconnector://john:Iwbicvi1994mysql@localhost:3306/cryptocurrencies',
                               echo=False)


def load_csvs(write_to_SQL=False, set_date_index=True):
    for cryptocurrency_name in cryptocurrency_names:
        globals()[cryptocurrency_name] = pd.read_csv('../data/{}_price.csv'.format(cryptocurrency_name))

    if write_to_SQL:
        # Create SQL tables for the cryptocurrencies - one for each cryptocurrency.
        for cryptocurrency_name in cryptocurrency_names:
            cryptocurrency = globals()[cryptocurrency_name]
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
        for cryptocurrency_name in cryptocurrency_names:
            globals()[cryptocurrency_name].set_index('Date', inplace=True)

    return bitcoin, bitcoin_cash, bitconnect, dash, ethereum, ethereum_classic, iota, litecoin, \
           monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, waves


def load_sql():
    for i in cryptocurrency_names:
        globals()[i] = pd.read_sql("SELECT * FROM {}".format(i), engine, index_col="Date")

    return bitcoin, bitcoin_cash, bitconnect, dash, ethereum, ethereum_classic, iota, litecoin, \
           monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, waves


if __name__ == '__main__':
    load_csvs(write_to_SQL=True, set_date_index=True)