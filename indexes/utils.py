
# coding: utf-8

# 先重整流程，然后再想办法完善数据，比如东财

# ## import libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import statsmodels.api as sm
import QUANTAXIS as QA
import talib as ta
import datetime, time


# In[2]:


plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def update_data(data, start, end):
    """
    Implements:
        update the 60mins data & save the data to the excel file
    
    Arguments:
        data -- dataframe type, 60mins data that to be updating
        start -- str, the data updated start date
        end -- str, the data updated end date
    
    Returns:
        data
    """
    # retrieve & adjust cols for concating
    data_update = QA.QA_fetch_get_index_min('tdx', '000001', start, end, '60min')
    data_update = data_update.loc[:,['open', 'close', 'high', 'low', 'vol', 'amount', 'up_count',
           'down_count', 'code', 'date', 'date_stamp', 'time_stamp', 'type']]

    # concat data
    data = pd.concat([data, data_update], axis=0)

    # save data
    writer = pd.ExcelWriter('index_60min.xlsx')
    data.to_excel(writer)
    writer.save()
    
    return data


# In[4]:


def update_and_import_data():
    """
    Implements:
        update the 60min data
        import the data
    
    Arguments:
        --
    
    Returns:
        data_min_resample_day
        index_day
    """
    # read non-update data
    data = pd.read_excel('index_60min.xlsx',index_col=0, header=0)
    
    start_update = QA.QA_util_get_next_day(data.index[-1].strftime('%Y-%m-%d'), 1)
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    end_update = QA.QA_util_get_last_day(today, 1)
    
    if start_update < today:
        data = update_data(data, start_update, end_update)
    
    # resample data
    data_min = data[(data.index.hour == 10) | (data.index.hour == 11)].loc[:, 
        ['open', 'high', 'low', 'close', 'vol', 'amount']]
    data_min_resample_day = data_min.resample('D').apply({
        'open':'first', 'high':'max', 'low':'min', 'close':'last',
        'vol':'sum','amount':'sum'
    }).dropna()
    
    start = '2016-11-16'
    end = end_update
    index_day = QA.QA_fetch_get_index_day('tdx', '000001', start, end)
    
    return (data_min_resample_day, index_day)


# In[9]:


def calc_right_nums(data_half_day, data_full_day, cur_half_day_range):
    """
    Implements:
        依据半天的涨幅，计算出有多少半天的涨幅和全天涨幅一致的数量和整体数量
    Arguments:
        
    Returns:
        
    """
    # 计算半天的涨幅和全天的涨幅
    half_day_change = (data_half_day.close / data_full_day.close.shift(1))[1:] - 1
    full_day_change = data_full_day.close.pct_change()[1:]
    
    if cur_half_day_range >= 0:
        ind = half_day_change >= cur_half_day_range
    else:
        ind = half_day_change <= cur_half_day_range
    
    half_filter = half_day_change[ind]
    full_filter = full_day_change[ind]
    
    result = (np.sign(half_filter) * np.sign(full_filter)).value_counts()
    right_nums = result.loc[1.0]
    total_nums = result.sum()
    
    return (right_nums, total_nums)


# In[10]:


def Kelly_Criterion(p, q, bet_win, bet_loss):
    """
    Implements:
        依据当前的半日涨跌幅, 赔率计算仓位
    Arguments:
        
    Returns:
        
    """
    f_star = p - q / bet_win
    f_star_reverse = q - p / bet_loss
    
    return (f_star, f_star_reverse)


# In[24]:


def calc_probability_and_position(data_half_day, data_full_day, cur_half_day_range, bet_win, bet_loss):
    """
    Implements:
        
    Arugments:
        
    Returns:
        
    """
    right_nums, total_nums = calc_right_nums(data_half_day, data_full_day, cur_half_day_range)
#     if cur_half_day_range >= 0:
    p = right_nums / total_nums
    q = 1 - p
#     else:
#         q = right_nums / total_nums
#         p = 1 - q
        
    f_star, f_star_reverse = Kelly_Criterion(p, q, bet_win, bet_loss)
    return (f_star, f_star_reverse, p, q)


# # In[5]:


# data_half_day, data_full_day = update_and_import_data()


# # In[34]:


# half_range = 0 * 1e-4
# bet_win = 0.47
# bet_loss = 1.72


# # In[35]:


# f_star, f_star_reverse = calc_probability_and_position(data_half_day, data_full_day,
                                                       # half_range, bet_win, bet_loss)
# print(f_star, f_star_reverse)


# # In[20]:


# int(f_star * 12000)

