import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import statsmodels.api as sm
try:
    import QUANTAXIS as QA
except:
    print("QUANTAXIS module doesn't install")

import datetime, time
try:
    from tqdm import tqdm_notebook
    import talib as ta
except:
    pass
#=============================================

def get_data(path, output_type='df'):
    data = pd.read_csv(path)
    data.columns = ['date', 'high', 'low', 'open', 'close', 'volume', 'Adj Close', 'code']
    data = data.set_index(['date', 'code'])
    if output_type == 'qa':
        data = QA.QA_DataStruct_Stock_day(data)
    return data

def get_code_list_by_sector(code_list_df, sector, region='US Equity'):
    if region == 'all':
        df = code_list_df.query('sector=="%s"' % (sector))
    else:
        df = code_list_df.query('sector=="%s" & region=="%s"' %\
                          (sector, region))
    return df

def select_code(data, code):
    def _select_code(data, code):
        return data.loc[(slice(None), code), :].xs(code, level=1)
    try:
        return _select_code(data, code)
    except:
        raise ValueError('CANNOT FIND THIS CODE {}'.format(code))
        
def position_side(ratio, avg_method='all', period=14):
    """
    avg_method -- all
               -- rolling
               -- ewm
    """
    if avg_method == 'all':
        position_side_list = []
        for i in range(ratio.shape[0]):
            if i > 1:
                if ratio[i] > np.mean(ratio[:i-1]):
                    position_side_list.append(1)
                elif ratio[i] < np.mean(ratio[:i-1]):
                    position_side_list.append(-1)
                else:
                    position_side_list.append(np.nan)
            elif i == 1:
                if ratio[1] > ratio[0]:
                    position_side_list.append(1)
                elif ratio[1] < ratio[0]:
                    position_side_list.append(-1)
                else:
                    position_side_list.append(np.nan)
            elif i == 0:
                position_side_list.append(np.nan)
        position_side_series = pd.Series(position_side_list, ratio.index)
    elif avg_method == 'rolling':
        position_side_series = pd.Series(np.where(ratio>ratio.\
            rolling(period).mean(), 1, -1).tolist(), ratio.index)
    elif avg_method == 'ewm':
        position_side_series = pd.Series(np.where(ratio>ratio.\
            ewm(span=period).mean(), 1, -1).tolist(), ratio.index)
    return position_side_series

def single_pair_trading(data, code1, code2, method='all', period=170):
    """
    data -- 这里用的是事先下载好的数据，亦可使用在线获取到的数据, 离线数据仅etf成分股的数据
    code1/code2 -- pair trading的两只股票的代码, 美股就是大写英文字母，港股则是4位数字+'.hk'的字符串
                比如 'MSFT', 'AMZN','0770.hk'(ps, 腾讯的数据并没有进行下载)
    method -- 判断买卖依据的方法
           -- all : 判断当前的ratio值是否大于之前所有ratio的平均值;
           -- rolling : 判断当前ratio是否大于最近一段时间的ratio平均值,计算方式是算数移动平均
           -- ewm : 判断当前ratio是否大于最近一段时间的ratio平均值,计算方式是指数移动平均
    """
    # retrieve adj close
    asset_1_close = select_code(data, code1)['Adj Close']
    asset_2_close = select_code(data, code2)['Adj Close']
    
    asset_1_close.index = pd.DatetimeIndex(asset_1_close.index)
    asset_2_close.index = pd.DatetimeIndex(asset_2_close.index)
    
    # calulate ratio
    ratio = asset_1_close / asset_2_close
    
    position_side_series = position_side(ratio, method, period)
    real_signal = position_side_series.shift(1)
    long_asset_series = real_signal.apply(lambda x : code1 if x==1 else
                                          (code2 if x==-1 else np.nan))
    
    asset_1_pct = asset_1_close.pct_change()
    asset_2_pct = asset_2_close.pct_change()
    
    long_return = ((1+real_signal)/2 * asset_1_pct + \
                 (1-real_signal)/2 * asset_2_pct).mean()
    long_short_return = (real_signal * (asset_1_pct - asset_2_pct)).mean()
    
    result = {
        "ratio":ratio,
        "real_signal":real_signal,
        "long_asset_series":long_asset_series,
        "long_return":long_return,
        "long_short_return":long_short_return
    }
    return result

def batch_pair_trading(code_list, data, method, period, show_bar):

    df_ratio = pd.DataFrame()
    df_long_asset = pd.DataFrame()
    lens = len(code_list)
    long_return_mat = np.zeros((lens,lens))
    long_short_return_mat = np.zeros((lens,lens))
    
    
#     for i in range(lens):
#         for j in range(i+1, lens):
    if show_bar:        
        for i in tqdm_notebook(range(lens)):
            for j in tqdm_notebook(range(i+1, lens)):           

                code1 = code_list[i]
                code2 = code_list[j]
                temp_result = single_pair_trading(data, code1, code2, method, period)
                temp_result['ratio'].name = "%s-%s" % (code1,code2)
                temp_result['long_asset_series'].name = "%s-%s" % (code1,code2)

                df_ratio = pd.concat([df_ratio, temp_result['ratio']], axis=1)
                df_long_asset = pd.concat([df_long_asset, temp_result['long_asset_series']],
                                         axis=1)
                long_return_mat[i, j] = temp_result['long_return']
                long_short_return_mat[i, j] = temp_result['long_short_return']
    else:
        for i in range(lens):
            for j in range(i+1, lens):           

                code1 = code_list[i]
                code2 = code_list[j]
                temp_result = single_pair_trading(data, code1, code2, method, period)
                temp_result['ratio'].name = "%s-%s" % (code1,code2)
                temp_result['long_asset_series'].name = "%s-%s" % (code1,code2)

                df_ratio = pd.concat([df_ratio, temp_result['ratio']], axis=1)
                df_long_asset = pd.concat([df_long_asset, temp_result['long_asset_series']],
                                         axis=1)
                long_return_mat[i, j] = temp_result['long_return']
                long_short_return_mat[i, j] = temp_result['long_short_return']

    df_long_return = pd.DataFrame(long_return_mat)
    df_long_return.index = code_list
    df_long_return.columns = code_list
    df_long_return

    df_long_short_return = pd.DataFrame(long_short_return_mat)
    df_long_short_return.index = code_list
    df_long_short_return.columns = code_list
    df_long_short_return
    
    result = {}
    
    result['ratio'] = df_ratio
    result['long_asset'] = df_long_asset
    result['long_return'] = df_long_return
    result['long_short_return'] = df_long_short_return
    
    return result

def calc_pair_trading(symbol, market_type='us', is_saving=False, method='all', period=170, show_bar=False):
    """
    symbol -- 可以是etf 代码 or sector name; 也可以是一组Equity symbol构成的list
    """
    if market_type == 'us':
        data = us_data
    elif market_type == 'hk':
        data = hk_data
        
    if symbol in etf_list:
        code_list = code_list_by_sector.query('etf_symbol == "%s"' % symbol).symbol.tolist()
    elif symbol in sector_list:
        code_list = code_list_by_sector.query('sector == "%s"' % symbol).symbol.tolist()
    elif type(symbol) == list:
        code_list = symbol
        
    result = batch_pair_trading(code_list, data, method, period, show_bar)
    
    return result

code_list_by_sector = pd.read_excel('data/etf_pair_code.xlsx', dtype={'symbol':str})
sector_list = code_list_by_sector.sector.unique().tolist()
etf_list = code_list_by_sector.etf_symbol.unique().tolist()
region_list = code_list_by_sector.region.unique().tolist()

us_data = get_data('data/us_data.csv')
hk_data = get_data('data/hk_data.csv')