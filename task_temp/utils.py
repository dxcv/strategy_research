import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import statsmodels.api as sm
try:
    import QUANTAXIS as QA
except:
    print("QUANTAXIS module doesn't install")
import talib as ta
import datetime, time

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
        
def position_side(ratio, avg_method='all', period=None):
    """
    avg_method -- all
               -- rolling
               -- emwa
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
        
    return position_side_series

def single_pair_trading(data, code1, code2):
    """
    
    """
    # retrieve adj close
    asset_1_close = select_code(data, code1)['Adj Close']
    asset_2_close = select_code(data, code2)['Adj Close']
    
    asset_1_close.index = pd.DatetimeIndex(asset_1_close.index)
    asset_2_close.index = pd.DatetimeIndex(asset_2_close.index)
    
    # calulate ratio
    ratio = asset_1_close / asset_2_close
    
    position_side_series = position_side(ratio)
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