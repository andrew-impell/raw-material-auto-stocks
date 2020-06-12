import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from matplotlib import style
from decouple import config
import matplotlib

# matplotlib.rcParams['text.usetex'] = True
style.use('ggplot')
quandl_key = config('Q_KEY')


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
        quandl.get("CHRIS/SHFE_RB1", authtoken=quandl_key)
    shanghai_steel2 = \
        quandl.get("CHRIS/SHFE_WR2", authtoken=quandl_key)

    american_steel1 = \
        quandl.get("CHRIS/CME_HR1", authtoken=quandl_key)
    american_steel2 = \
        quandl.get("CHRIS/CME_HR2", authtoken=quandl_key)

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
    palladium = quandl.get("LPPM/PALL", authtoken=quandl_key)
    price = palladium['USD AM']
    return price


def get_rubber():
    indo = quandl.get("CHRIS/SGX_IR1", authtoken=quandl_key)
    malay = quandl.get("CHRIS/SGX_IR1", authtoken=quandl_key)
    thai = quandl.get("CHRIS/SGX_TR1", authtoken=quandl_key)
    # hk = quandl.get("HKEX/06865", authtoken=quandl_key)
    producers = [indo, malay, thai]
    df_list = []
    for prod in producers:
        close = prod['Close']
        df_list.append(close)
    all = pd.concat(df_list, axis=1)
    avg_price = all.mean(axis=1)
    return avg_price


def get_copper():
    copper = quandl.get("CHRIS/MCX_CU1", authtoken=quandl_key)
    close = copper['Close']
    return close


def get_aluminum():
    aluminum = quandl.get("CHRIS/SHFE_AL1", authtoken=quandl_key)
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


def get_correlation_dict(lag=1):

    cor_dict = {}

    for auto in list(autos.columns):

        syn_auto_df = pd.concat([autos[auto],
                                 synthetic.shift(lag, fill_value=0)], axis=1)
        syn_auto_df.dropna(axis=0, how='any', inplace=True)
        syn_auto_df.interpolate(method='from_derivatives', inplace=True)

        cor = syn_auto_df.corr().values[0, 1]
        cor_dict[auto] = cor

        print(f'{auto} correlation: {cor:.2f}')

    sorted_dict = \
        {k: v for k, v in sorted(cor_dict.items(), key=lambda item: item[1])}

    return sorted_dict, cor_dict


sorted_dict, cor_dict = get_correlation_dict(lag=0)

# auto_mean = synthetic.mean() / autos.mean().mean()


def opt_lag_val(max_lag=100):
    max_ss = 0
    best_i = 0
    ss_set = []
    for i in range(-max_lag, max_lag):
        sorted_dict, cor_dict = get_correlation_dict(lag=i)
        temp_ss = np.mean(np.abs(np.array(list(sorted_dict.values()))))
        ss_set.append(temp_ss)
        if temp_ss > max_ss:
            max_ss = temp_ss
            best_i = i
    return best_i, ss_set


best_i, ss_set = opt_lag_val()
print('Best lag: ', best_i)

max_lag = 100
plt.xlabel('Lag')
plt.ylabel('mean(abs(correlations))')
plt.title('Optimal Lag')
plt.plot(np.arange(-max_lag, max_lag), ss_set)
plt.show()


def graph_correlations(sorted_dict):
    plt.figure()
    plt.subplots_adjust(hspace=.5)

    plt.subplot(211)
    plt.title('Negative Correlation')
    plt.xlabel('Date')
    plt.ylabel('Percent Change')
    plt.plot(synthetic.index, synthetic.pct_change().values,
             label='Synthetic', alpha=0.5)
    for comp in list(sorted_dict.keys())[:3]:
        cor_label = str(comp) + ": " + str(cor_dict[comp])
        plt.plot(autos.index, autos[comp].pct_change().values,
                 label=cor_label, alpha=0.5)
    plt.legend()

    plt.subplot(212)
    plt.title('Positive Correlation')
    plt.xlabel('Date')
    plt.ylabel('Percent Change')
    plt.plot(synthetic.index, 50 * synthetic.pct_change().values,
             label='Synthetic', alpha=0.5)
    for comp in list(sorted_dict.keys())[-3:]:
        cor_label = str(comp) + ": " + str(cor_dict[comp])
        plt.plot(autos.index, autos[comp].pct_change().values,
                 label=cor_label, alpha=0.5)
    plt.legend()
    plt.show()
    return 0


a = graph_correlations(sorted_dict)
