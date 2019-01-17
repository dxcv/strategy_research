
# coding: utf-8

# ====================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from string import ascii_letters # 导入大小写英文字母
import os # 用于返回当前系统路径
import talib as ta # 用于计算指标
import pickle
# import statsmodels.api as sm

# try:
#     import QUANTAXIS as QA
# except:
#     print("QUANTAXIS module doesn't install")

import datetime, time
try:
    from tqdm import tqdm_notebook
except:
    pass


# ====================


def get_data(path, output_type='df'):
    """
    Implements:
        从csv中提取到数据
    
    Arguments:
        path -- 字符串类型, 数据文件的路径
        output_type -- 字符串类型, 数据输出的格式; 'df' or 'qa'
    
    Returns:
        data -- asset data, 数据类型: dataframe or QA_DataStruct
    """
    data = pd.read_csv(path)
    try:
        data.columns = ['date', 'high', 'low', 'open', 'close', 'volume', 'Adj Close', 'code']
    except:
        data.columns = ['date', 'high', 'open', 'low', 'volume', 'Adj Close', 'code']
    data.date = pd.DatetimeIndex(data.date)
    data = data.set_index(['date', 'code'])
    if output_type == 'qa':
        data = QA.QA_DataStruct_Stock_day(data)
    return data


# ====================
def get_code_list_by_etf(code_list_df, etf_symbol, region):
    """
    Implements:
        按行业和地区筛选股票代码
    
    Arguments:
        code_list_df -- DataFrame数据类型, 通过读取'etf_pair_code.xlsx'得到的代码列表
        etf -- 字符串类型, etf 代码
        region -- 字符串类型, 地区名称
    
    Returns:
        df -- DataFrame数据类型, 筛选之后的股票代码
    """
    
    if region == 'all':
        df = code_list_df.query('etf_symbol=="%s"' % (etf_symbol))
    else:
        region = region.upper() + ' Equity'
        df = code_list_df.query('etf_symbol=="%s" & region=="%s"' % (etf_symbol, region))
    return df

def get_code_list_by_sector(code_list_df, sector, region):
    """
    Implements:
        按行业和地区筛选股票代码
    
    Arguments:
        code_list_df -- DataFrame数据类型, 通过读取'etf_pair_code.xlsx'得到的代码列表
        sector -- 字符串类型, 行业名称
        region -- 字符串类型, 地区名称
    
    Returns:
        df -- DataFrame数据类型, 筛选之后的股票代码
    """
    
    if region == 'all':
        df = code_list_df.query('sector=="%s"' % (sector))
    else:
        region = region.upper() + ' Equity'
        df = code_list_df.query('sector=="%s" & region=="%s"' % (sector, region))
    return df


# ====================


def select_code(data, code):
    """
    Implements:
        按code筛选数据
    
    Arguments:
        data -- 由date和code字段MultiIndex构成的DataFrame数据结构;
        code -- 字符串数据类型, 用于筛选数据
    
    Returns:
        按code筛选之后的数据
    """
    def _select_code(data, code):
        return data.loc[(slice(None), code), :].xs(code, level=1)
    try:
        return _select_code(data, code)
    except:
        raise ValueError('CANNOT FIND THIS CODE {}'.format(code))


# ====================


def position_side(ratio, avg_method, period):
    """
    Implements:
        按照不同的平均方法, 计算出要做多和做空的标的;
    
    Arguments:
        ratio -- Series数据类型, 两个标的Adj Close之比算出的值;
        avg_method -- 字符串数据类型, 表示采取何种求平均值的方法
                    -- all, 表示用当前bar之前所有的ratio值求平均
                    -- rolling, 表示用最近一段时间的ratio值做算术移动平均值
                    -- ewm, 表示用最近一段时间的ratio值做指数移动平均值
        period -- integer, 当avg_method为rolling或者ewm时, 需要计算最近多少根bar的平均值
    
    Returns:
        position_side_series -- Series数据类型, 用于指示做多还是做空
    """
    if avg_method == 'all':
        position_side_list = []
        for i in range(ratio.shape[0]):
            if i > 1:
                temp_mean = np.nanmean(ratio[:i-1])
                if ratio[i] > temp_mean:
                    position_side_list.append(1)
                elif ratio[i] < temp_mean:
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
        position_side_series = pd.Series(np.where(ratio>ratio.rolling(period).apply(lambda x : np.nanmean(x), raw=True), 1, -1).tolist(), ratio.index)
    elif avg_method == 'ewm':
        position_side_series = pd.Series(np.where(ratio>ratio.ewm(span=period).apply(lambda x : np.nanmean(x), raw=True), 1, -1).tolist(), ratio.index)
    return position_side_series


# ====================

def rsi_interval(value, upper, lower, return_method='string'):
    if value <= lower:
        if return_method == 'string':
            return 'low'
        elif return_method == 'integer':
            # return -1
            return value
    elif (value > lower) & (value<=upper):
        return 0
    elif value > upper:
        if return_method == 'string':
            return 'high'
        elif return_method == 'integer':
            # return 1
            return value
# ====================

def single_pair_trading(data, code1, code2, method, period, rsi_period, upper, lower):
    """
    Implement:
        计算单对pair trading
    
    Arguments:
        
        data -- 这里用的是事先下载好的数据，亦可使用在线获取到的数据. 离线数据仅etf成分股的数据
        code1/code2 -- pair trading的两只股票的代码, 美股就是大写英文字母;
                    港股则是4位数字+'.hk'的字符串
                    比如 'MSFT', 'AMZN','0770.hk'(ps, 腾讯的数据并没有进行下载)
        method -- 字符串数据类型, 表示采取何种求平均值的方法
                    -- all, 表示用当前bar之前所有的ratio值求平均
                    -- rolling, 表示用最近一段时间的ratio值做算术移动平均值
                    -- ewm, 表示用最近一段时间的ratio值做指数移动平均值
        period -- integer, 当avg_method为rolling或者ewm时, 需要计算最近多少根bar的平均值
        
    Returns:
        result -- python中的字典数据, keys分别是以下5个:
               "ratio","real_signal","long_asset","long_return","long_short_return"
    """
    # retrieve adj close

    asset_1_close = select_code(data, code1)['Adj Close']
    asset_2_close = select_code(data, code2)['Adj Close']
    
#     asset_1_close.index = pd.DatetimeIndex(asset_1_close.index)
#     asset_2_close.index = pd.DatetimeIndex(asset_2_close.index)
    
    # normalization
#     asset_1_close = (asset_1_close - asset_1_close.mean()) / asset_1_close.std() 
#     asset_2_close = (asset_2_close - asset_2_close.mean()) / asset_2_close.std()
    
    # calulate ratio
    ratio = asset_1_close / asset_2_close
    
    # 待施工
    rsi = ta.RSI(ratio, timeperiod=rsi_period)
    
    # use rsi_mean +/- rsi_std
    # 如果用动态的rsi标准差作为上下轨，那么分别用rsi和上下轨比较，然后取并集？
    # 亦可直接把30到70之间的数设置为0即可
    
    # interval_by_str = rsi.apply(lambda x : rsi_interval(x, upper, lower, return_method='string'))
    # interval_by_int = rsi.apply(lambda x : rsi_interval(x, upper, lower, return_method='integer'))
    interval_by_int = rsi.mask((rsi>lower) & (rsi<=upper), 0)
    
    #=====
    
    position_side_series = position_side(ratio, method, period)
    real_signal = position_side_series.shift(1)
    long_asset = real_signal.apply(lambda x : code1 if x==1 else
                                          (code2 if x==-1 else np.nan))
    
    asset_1_pct = asset_1_close.pct_change()
    asset_2_pct = asset_2_close.pct_change()
    
    long_return = ((1+real_signal)/2 * asset_1_pct +  (1-real_signal)/2 * asset_2_pct).mean()
    long_short_return = (real_signal * (asset_1_pct - asset_2_pct)).mean()
    
    result = {
        "ratio":ratio,
        "real_signal":real_signal,
        "long_asset":long_asset,
        "long_return":long_return,
        "long_short_return":long_short_return,
        "rsi":rsi,
        "rsi_mask":interval_by_int
    }
    return result


# ====================


def batch_pair_trading(code_list, data, method, period, rsi_period, upper, lower, show_bar):
    """
    Implements:
        返回一组assets的两两pair trading的结果
    
    Arguments:
        code_list -- 由code组成的list
        data -- equity的行情数据
        method -- 字符串数据类型, 表示采取何种求平均值的方法
        period -- integer, 当avg_method为rolling或者ewm时, 需要计算最近多少根bar的平均值
        show_bar -- 是否显示计算过程中的进度条
    Returns:
        result -- python中的字典数据, keys分别是以下4个:
               "ratio","long_asset","long_return","long_short_return"
    """
    df_ratio = pd.DataFrame()
    df_long_asset = pd.DataFrame()
    lens = len(code_list)
    long_return_mat = np.zeros((lens,lens))
    long_short_return_mat = np.zeros((lens,lens))
    df_rsi = pd.DataFrame()
    df_rsi_mask = pd.DataFrame()
    
    
#     for i in range(lens):
#         for j in range(i+1, lens):
    if show_bar:        
        for i in tqdm_notebook(range(lens)):
            for j in range(i+1, lens):
#             for j in tqdm_notebook(range(i+1, lens)):           

                code1 = code_list[i]
                code2 = code_list[j]
                try:
                    temp_result = single_pair_trading(data, code1, code2, method, period, rsi_period,
                                                     upper, lower)
                    temp_result['ratio'].name = "%s-%s" % (code1,code2)
                    temp_result['long_asset'].name = "%s-%s" % (code1,code2)
                    temp_result['rsi'].name = "%s-%s" % (code1,code2)
                    temp_result['rsi_mask'].name = "%s-%s" % (code1,code2)
    
                    df_ratio = pd.concat([df_ratio, temp_result['ratio']], axis=1)
                    df_long_asset = pd.concat([df_long_asset, temp_result['long_asset']],
                                             axis=1)
                    long_return_mat[i, j] = temp_result['long_return']
                    long_short_return_mat[i, j] = temp_result['long_short_return']
                    df_rsi = pd.concat([df_rsi, temp_result['rsi']], axis=1)
                    df_rsi_mask = pd.concat([df_rsi_mask, temp_result['rsi_mask']], axis=1)
                except:
                    print("%s or %s don't have data, please check" % (code1, code2))
                    
                    
            
    else:
        for i in range(lens):
            for j in range(i+1, lens):           

                code1 = code_list[i]
                code2 = code_list[j]
                try:
                    temp_result = single_pair_trading(data, code1, code2, method, period, rsi_period,
                                                     upper, lower)
                    temp_result['ratio'].name = "%s-%s" % (code1,code2)
                    temp_result['long_asset'].name = "%s-%s" % (code1,code2)
                    temp_result['rsi'].name = "%s-%s" % (code1,code2)
                    temp_result['rsi_mask'].name = "%s-%s" % (code1,code2)

                    df_ratio = pd.concat([df_ratio, temp_result['ratio']], axis=1)
                    df_long_asset = pd.concat([df_long_asset, temp_result['long_asset']],
                                             axis=1)
                    long_return_mat[i, j] = temp_result['long_return']
                    long_short_return_mat[i, j] = temp_result['long_short_return']
                    df_rsi = pd.concat([df_rsi, temp_result['rsi']], axis=1)
                    df_rsi_mask = pd.concat([df_rsi_mask, temp_result['rsi_mask']], axis=1)
                except:
                    print("%s or %s don't have data, please check" % (code1, code2))

    df_long_return = pd.DataFrame(long_return_mat)
    df_long_return.index = code_list
    df_long_return.columns = code_list
    # df_long_return

    df_long_short_return = pd.DataFrame(long_short_return_mat)
    df_long_short_return.index = code_list
    df_long_short_return.columns = code_list
    # df_long_short_return
    
    result = {}
    
    result['ratio'] = df_ratio
    result['long_asset'] = df_long_asset
    result['long_return'] = df_long_return
    result['long_short_return'] = df_long_short_return
    result['rsi'] = df_rsi
    result['rsi_mask'] = df_rsi_mask
    
    return result


# ====================


def store_in_excel(result, is_saving, save_name):
    """
    Implements:
        将result保存进excel中
        
    Arguments:
        result -- python字典数据类型, 由single_pair_trading or batch_pair_trading计算得到
            keys分别是以下4个:"ratio","long_asset","long_return","long_short_return"
        is_saving -- 是否保存/保存哪些表 
                  -- True, bool type, 保存所有的表进excel
                  -- 'all', str type, 保存所有的表进excel
                  -- "ratio","long_asset","long_return","long_short_return"中任意一个进行保存
                  -- '0123', str type, 以数字代表4个key, 任意组合即为保存指定表
        save_name -- excel的文件名
                  -- 若为None, 则随机生成一个8位英文字符的文件名
                  -- str type, excel的文件名
    
    Returns:
        None
    """
    if save_name is None:
        save_name = ''.join(np.random.choice(list(ascii_letters), 8))
    
    result_path = os.getcwd()+'/result'
    if os.path.exists(result_path) is False:
        os.makedirs(result_path)
    
    writer = pd.ExcelWriter('result/'+save_name+'.xlsx', datetime_format='YYYY-MM-DD')
    keys = list(result.keys())
    
    print('saving, please hold on...', end='')
    
    if (is_saving is True) or (is_saving.lower() == 'all') or (is_saving.lower() == 'true'):
        for i in result.keys():
            result[i].to_excel(writer, sheet_name=i)
    elif is_saving.lower() == 'ratio':
        result['ratio'].to_excel(writer, sheet_name='ratio')
    elif is_saving.lower() == 'long_asset':
        result['long_asset'].to_excel(writer, sheet_name='long_asset')
    elif is_saving.lower() == 'long_return':
        result['long_return'].to_excel(writer, sheet_name='long_return')
    elif is_saving.lower() == 'long_short_return':
        result['long_short_return'].to_excel(writer, sheet_name='long_short_return')
    elif is_saving.lower() == 'rsi':
        result['rsi'].to_excel(writer, sheet_name='rsi')
    elif is_saving.lower() == 'rsi_mask':
        result['rsi_mask'].to_excel(writer, sheet_name='rsi_mask')
    else:
        for i in is_saving:
            result[keys[int(i)]].to_excel(writer, sheet_name=keys[int(i)])
            
    writer.save()
    print('\rfinish, see you          ')


# ====================


def calc_pair_trading(symbol, data_source='ol', region='all', is_saving=False, save_name=None, method='all', 
                      period=170, rsi_period=14, upper=70, lower=30, show_bar=True):
    """
    Implements:
        计算一组assets的pair trading, 并且可以选择是否存入到excel表中
    
    Arguments:
        symbol -- 字符串数据类型 or pythonlist数据类型, 
                  可以是etf 代码 或者 sector name; 
                  也可以是一组Equity symbol构成的list
        data_source -- 字符串数据类型,
                    -- 'us', 本地美股数据,
                    -- 'hk', 本地港股数据,
                    -- 'ol', 在线获取数据.
        is_saving -- 是否保存/保存哪些表 
                  -- True, bool type, 保存所有的表进excel
                  -- 'all', str type, 保存所有的表进excel
                  -- "ratio","long_asset","long_return","long_short_return"中任意一个进行保存
                  -- '0123', str type, 以数字代表4个key, 任意组合即为保存指定表
        save_name -- excel的文件名
                  -- 若为None, 则随机生成一个8位英文字符的文件名
                  -- str type, excel的文件名
        method -- 字符串数据类型, 表示采取何种求平均值的方法
                    -- all, 表示用当前bar之前所有的ratio值求平均
                    -- rolling, 表示用最近一段时间的ratio值做算术移动平均值
                    -- ewm, 表示用最近一段时间的ratio值做指数移动平均值
        period -- integer, 当avg_method为rolling或者ewm时, 需要计算最近多少根bar的平均值
        show_bar -- 是否显示计算过程中的进度条
    """    
    
    if symbol in etf_list:
        code_list = get_code_list_by_etf(df_code, symbol, region = region).symbol.tolist()
        # code_list = df_code.query('etf_symbol == "%s"' % symbol).symbol.tolist()
    elif symbol in sector_list:
        code_list = get_code_list_by_sector(df_code, symbol, region = region).symbol.tolist()
        # code_list = df_code.query('sector == "%s"' % symbol).symbol.tolist()
    elif type(symbol) == list:
        code_list = symbol
        
    if data_source == 'all':
        data = pd.concat([us_data, hk_data], axis=0)
    if data_source == 'us':
        data = us_data
    elif data_source == 'hk':
        data = hk_data
    elif data_source == 'ol':
        data = ol_data
    
        
    result = batch_pair_trading(code_list, data, method, period, rsi_period, upper, lower, show_bar)
    
    if is_saving is not False:
        store_in_excel(result, is_saving, save_name)
    
    return result


# ====================
def save_file(dataframe, name):
    with open(name, 'wb') as f:
        pickle.dump(dataframe, f)
    
def load_file(filename):
    with open(filename, 'rb') as f:
        res = pickle.load(f)
        return res
# ====================

df_code = pd.read_excel('data/etf_pair_code.xlsx', dtype={'symbol':str})
sector_list = df_code.sector.unique().tolist()
etf_list = df_code.etf_symbol.unique().tolist()
region_list = df_code.region.unique().tolist()

ol_data = get_data('data/data.csv')
# try:
# 	ol_data = get_data('data/data.csv')
# except:
# 	print('there is something wrong')
# us_data = get_data('data/us_data.csv')
# hk_data = get_data('data/hk_data.csv')

