import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from matplotlib import style
import os
from decouple import config

style.use('ggplot')

quandl_key = config('Q_KEY')
print(quandl_key)

autos = ['F', 'HMC', 'TM', 'TTM', 'PAG', 'GM', 'OSK', 'FCAU']


def get_auto_prices(autos, plot=True, save=False):
    close_list = []
    for auto in autos:
        temp_df = pdr.get_data_yahoo(auto)
        close_list.append(
            temp_df['Close']
            .fillna(0).rename(auto))
    close_df = pd.concat(close_list, axis=1)
    if save:
        close_df.to_csv('auto_prices.csv')
    if plot:
        plt.plot(close_df.index, close_df.values)

        plt.show()
    return close_df


def get_steel():
    shanghai_steel1 = \
        quandl.get("CHRIS/SHFE_RB1", authtoken="-h4jm8-epYp2YfshRaBA")
    shanghai_steel2 = \
        quandl.get("CHRIS/SHFE_WR2", authtoken="-h4jm8-epYp2YfshRaBA")

    american_steel1 = \
        quandl.get("CHRIS/CME_HR1", authtoken="-h4jm8-epYp2YfshRaBA")
    american_steel2 = \
        quandl.get("CHRIS/CME_HR2", authtoken="-h4jm8-epYp2YfshRaBA")

    steel_list = [shanghai_steel1, shanghai_steel2,
                  american_steel1, american_steel2]

    concat_list = []
    for steel in steel_list:
        close_price = steel['Settle']
        concat_list.append(close_price)
    all_steel = pd.concat(concat_list, axis=1)
    avg_price = all_steel.mean(axis=1)
    rolling = avg_price.rolling(10).mean()

    cut_off = 0.01
    steel = avg_price.copy()
    steel[steel > 1 + cut_off * rolling] = 1.01 * rolling
    steel[steel < 1 - cut_off * rolling] = 0.99 * rolling

    return steel


def get_palladium():
    palladium = quandl.get("LPPM/PALL", authtoken="-h4jm8-epYp2YfshRaBA")
    price = palladium['USD AM']
    return price


def get_rubber():
    indo = quandl.get("CHRIS/SGX_IR1", authtoken="-h4jm8-epYp2YfshRaBA")
    malay = quandl.get("CHRIS/SGX_IR1", authtoken="-h4jm8-epYp2YfshRaBA")
    thai = quandl.get("CHRIS/SGX_TR1", authtoken="-h4jm8-epYp2YfshRaBA")
    # hk = quandl.get("HKEX/06865", authtoken="-h4jm8-epYp2YfshRaBA")
    producers = [indo, malay, thai]
    df_list = []
    for prod in producers:
        close = prod['Close']
        df_list.append(close)
    all = pd.concat(df_list, axis=1)
    avg_price = all.mean(axis=1)
    return avg_price


def get_copper():
    copper = quandl.get("CHRIS/MCX_CU1", authtoken="-h4jm8-epYp2YfshRaBA")
    close = copper['Close']
    return close


def get_aluminum():
    aluminum = quandl.get("CHRIS/SHFE_AL1", authtoken="-h4jm8-epYp2YfshRaBA")
    close = aluminum['Close']
    return close


# autodf = get_auto_prices(autos, save=True)


def synthetic_car(plot=False, save=True):
    '''
    Aggregate Car:
    Steel: 60%
    Palladium at 0.06 / 2000 = 0.003%
    Rubber: (4 * 7.5 kg * 30% real rubber + 7 kg misc) / 2000 = 1.185%
    Copper 55/2.2 / 2000 = 1.375%
    Aluminum = 16.6%
    '''
    palladium = get_palladium()
    steel = get_steel()
    copper = get_copper()
    aluminum = get_aluminum()

    steel = 0.6 * steel
    palladium = 0.00003 * palladium
    copper = 0.01375 * copper
    aluminum = 0.166 * aluminum

    synthetic = steel + palladium + copper + aluminum

    print(synthetic.index)
    synthetic.interpolate(method='from_derivatives', inplace=True)
    if plot:
        plt.plot(synthetic.index, synthetic.values)
        plt.show()
    if save:
        synthetic.to_csv('synthetic_auto.csv')
    return synthetic


synthetic = pd.read_csv('synthetic_auto.csv')
synthetic.index = pd.to_datetime(synthetic.iloc[:, 0])
autos = pd.read_csv('auto_prices.csv')

autos.set_index('Date', inplace=True)
autos.index = pd.to_datetime(autos.index)
synthetic = synthetic.iloc[-1300:-1, 1]


for auto in list(autos.columns):

    syn_auto_df = pd.concat([autos[auto], synthetic], axis=1)
    syn_auto_df.dropna(axis=0, how='any', inplace=True)
    syn_auto_df.interpolate(method='from_derivatives', inplace=True)

    cor = syn_auto_df.corr().values[0, 1]

    print(f'{auto} correlation: {cor:.2f}')
