# %load_ext autoreload
# %autoreload 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib as ta
import datetime, time
import pickle
from tqdm import tqdm_notebook

import pdblp as pb
import blpapi

def update(start='20100101'):
    
    con = pb.BCon(port=8194, timeout=5000)
    con.start()

    df_code = pd.read_excel('data/etf_pair_code.xlsx', dtype={'symbol':str})
    code_list = df_code.symbol.unique().tolist()
    ticker_list = (df_code.symbol + ' ' + df_code.region).unique().tolist()

    fields = ['HIGH', 'OPEN', 'LOW', 'VOLUME', 'PX_LAST']
    # cols = ['High', 'Open', 'Low', 'Volume', 'Adj Close']
    # cols = ['High', 'Low', 'Open', 'Adj Close', 'Volume']

#     start = '20100101'
    end = datetime.datetime.today().strftime('%Y%m%d')

    ticker_list.remove('HKD Curncy')
    #ticker_list.remove('ita')
    intervals = np.linspace(0, len(ticker_list), 10, dtype=int)
    tic = time.perf_counter()

    df = pd.DataFrame()

    for i in tqdm_notebook(range(len(intervals)-1)):

        time.sleep(5)

        temp_list = ticker_list[intervals[i]:intervals[i+1]]

        temp_df = con.bdh(temp_list, fields, start, end)

        temp_df = pd.concat([temp_df[x].assign(code=x.split(' ')[0]).groupby(level=0, axis=1).first() for x in temp_df.columns.levels[0]], 
                  axis=0,sort=False)#.reset_index().set_index(['date', 'code'])
    #     temp_df.columns = cols

        df = pd.concat([df, temp_df], axis=0)

    df = df[['HIGH', 'OPEN', 'LOW', 'VOLUME', 'PX_LAST', 'code']]
    df.columns = ['High', 'Open', 'Low', 'Volume', 'Adj Close', 'code']

    toc = time.perf_counter()
    df.to_csv('data/data.csv')
    con.stop()
    print('retrieve data cost time: %.2f' % (toc-tic))