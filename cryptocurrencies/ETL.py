import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR

cryptocurrency_names = \
    ('bitcoin', 'bitcoin_cash', 'bitconnect', 'dash', 'ethereum', 'ethereum_classic', 'iota', 'litecoin', \
     'monero', 'nem', 'neo', 'numeraire', 'omisego', 'qtum', 'ripple', 'stratis', 'waves')

bitcoin = None; bitcoin_cash = None; bitconnect = None; dash = None; ethereum_classic = None; ethereum = None;
iota = None; litecoin = None; monero = None; nem = None; neo = None; numeraire = None; omisego = None; qtum = None;
ripple = None; stratis = None; waves = None;

engine = create_engine('mysql+mysqlconnector://john:Iwbicvi1994mysql@localhost:3306/cryptocurrencies',
                               echo=False)

def load_csvs(write_to_SQL=False, set_date_index=True):
    for i in cryptocurrency_names:
        globals()[i] = pd.read_csv('data/{}_price.csv'.format(i))
    # bitcoin = pd.read_csv('data/bitcoin_price.csv')
    # bitcoin_cash = pd.read_csv('data/bitcoin_cash_price.csv')
    # bitconnect = pd.read_csv('data/bitconnect_price.csv')
    # dash = pd.read_csv('data/dash_price.csv')
    # etherium_classic = pd.read_csv('data/ethereum_classic_price.csv')
    # etherium = pd.read_csv('data/ethereum_price.csv')
    # iota = pd.read_csv('data/iota_price.csv')
    # litecoin = pd.read_csv('data/litecoin_price.csv')
    # monero = pd.read_csv('data/monero_price.csv')
    # nem = pd.read_csv('data/nem_price.csv')
    # neo = pd.read_csv('data/neo_price.csv')
    # numeraire = pd.read_csv('data/numeraire_price.csv')
    # omisego = pd.read_csv('data/omisego_price.csv')
    # qtum = pd.read_csv('data/qtum_price.csv')
    # ripple = pd.read_csv('data/ripple_price.csv')
    # stratis = pd.read_csv('data/stratis_price.csv')
    # waves = pd.read_csv('data/waves_price.csv')

    if write_to_SQL:
        # Create SQL tables for the cryptocurrencies - one for each cryptocurrency.
        for i in cryptocurrency_names:
            cryptocurrency = globals()[i]
            def volume_to_int(volume):
                return 0 if volume == '-' else int(volume.replace(',', ''))
            def market_cap_to_int(market_cap):
                return 0 if market_cap == '-' else float(market_cap.replace(',', ''))
            cryptocurrency['Volume'] = cryptocurrency['Volume'].apply(volume_to_int)
            cryptocurrency['Market Cap'] = cryptocurrency['Market Cap'].apply(market_cap_to_int)
            cryptocurrency.to_sql(name=i, con=engine, if_exists='replace', index=False)#,
                                  #dtype={"Volume": })#"Date": VARCHAR(i.index.str.len().max())})

    if set_date_index:
        for i in cryptocurrency_names:
            globals()[i].set_index('Date', inplace=True)

    return bitcoin, bitcoin_cash, bitconnect, dash, ethereum, ethereum_classic, iota, litecoin, \
           monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, waves

def load_sql():
    for i in cryptocurrency_names:
        globals()[i] = pd.read_sql("SELECT * FROM {}".format(i), engine, index_col="Date")
    # bitcoin_cash = pd.read_csv('data/bitcoin_cash_price.csv')
    # bitconnect = pd.read_csv('data/bitconnect_price.csv')
    # dash = pd.read_csv('data/dash_price.csv')
    # etherium_classic = pd.read_csv('data/ethereum_classic_price.csv')
    # etherium = pd.read_csv('data/ethereum_price.csv')
    # iota = pd.read_csv('data/iota_price.csv')
    # litecoin = pd.read_csv('data/litecoin_price.csv')
    # monero = pd.read_csv('data/monero_price.csv')
    # nem = pd.read_csv('data/nem_price.csv')
    # neo = pd.read_csv('data/neo_price.csv')
    # numeraire = pd.read_csv('data/numeraire_price.csv')
    # omisego = pd.read_csv('data/omisego_price.csv')
    # qtum = pd.read_csv('data/qtum_price.csv')
    # ripple = pd.read_csv('data/ripple_price.csv')
    # stratis = pd.read_csv('data/stratis_price.csv')
    # waves = pd.read_csv('data/waves_price.csv')

    return bitcoin, bitcoin_cash, bitconnect, dash, ethereum, ethereum_classic, iota, litecoin, \
           monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, waves


if __name__ == '__main__':
    load_csvs(write_to_SQL=True, set_date_index=False)