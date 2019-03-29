
# coding: utf-8

# In[1]:

#try:
#	get_ipython().run_line_magic('load_ext', 'autoreload')
#	get_ipython().run_line_magic('autoreload', '2')
#except:
#	pass


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import talib as ta
import datetime, time
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from tqdm import tqdm_notebook
from scipy.stats import mstats


# In[3]:


plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.style.use('ggplot')


# In[22]:


def calc_exf(df):
    ret = df.ret_p1
    dret = df.dret_p1
    pre_mv = df.mv.shift(1) * df.retx_p1
    exf = df.mv - pre_mv
    exf_to_mv = exf / df.mv.shift(1)
    
    ret_roll_1 = ((ret[::-1]).rolling(1, min_periods=1).apply(np.prod, raw=True)[::-1])
    ret_roll_2 = ((ret[::-1]).rolling(2, min_periods=1).apply(np.prod, raw=True)[::-1])
    ret_roll_3 = ((ret[::-1]).rolling(3, min_periods=1).apply(np.prod, raw=True)[::-1])
    ret_roll_4 = ((ret[::-1]).rolling(4, min_periods=1).apply(np.prod, raw=True)[::-1])
    ret_roll_5 = ((ret[::-1]).rolling(5, min_periods=1).apply(np.prod, raw=True)[::-1])
    ret_roll_6 = ((ret[::-1]).rolling(6, min_periods=1).apply(np.prod, raw=True)[::-1])
    ret_roll_7 = ((ret[::-1]).rolling(7, min_periods=1).apply(np.prod, raw=True)[::-1])
    ret_roll_8 = ((ret[::-1]).rolling(8, min_periods=1).apply(np.prod, raw=True)[::-1])
    
    dret_roll_1 = ((dret[::-1]).rolling(1, min_periods=1).apply(np.prod, raw=True)[::-1])
    dret_roll_2 = ((dret[::-1]).rolling(2, min_periods=1).apply(np.prod, raw=True)[::-1])
    dret_roll_3 = ((dret[::-1]).rolling(3, min_periods=1).apply(np.prod, raw=True)[::-1])
    dret_roll_4 = ((dret[::-1]).rolling(4, min_periods=1).apply(np.prod, raw=True)[::-1])
    dret_roll_5 = ((dret[::-1]).rolling(5, min_periods=1).apply(np.prod, raw=True)[::-1])
    dret_roll_6 = ((dret[::-1]).rolling(6, min_periods=1).apply(np.prod, raw=True)[::-1])
    dret_roll_7 = ((dret[::-1]).rolling(7, min_periods=1).apply(np.prod, raw=True)[::-1])
    dret_roll_8 = ((dret[::-1]).rolling(8, min_periods=1).apply(np.prod, raw=True)[::-1])
    
    # 参考99页，引用7，这里要求的是从1974开始
    dret_roll_1_bar = dret_roll_1.shift(-1).mean()
    dret_roll_2_bar = dret_roll_2.shift(-1).mean()
    dret_roll_3_bar = dret_roll_3.shift(-1).mean()
    dret_roll_4_bar = dret_roll_4.shift(-1).mean()
    dret_roll_5_bar = dret_roll_5.shift(-1).mean()
    
    car1 = ret_roll_1 - dret_roll_1
    car2 = ret_roll_2 - dret_roll_2
    car3 = ret_roll_3 - dret_roll_3
    car4 = ret_roll_4 - dret_roll_4
    car5 = ret_roll_5 - dret_roll_5
    car6 = ret_roll_6 - dret_roll_6
    car7 = ret_roll_7 - dret_roll_7
    car8 = ret_roll_8 - dret_roll_8
    
    wr1 = ret_roll_1 / dret_roll_1
    wr2 = ret_roll_2 / dret_roll_2
    wr3 = ret_roll_3 / dret_roll_3
    wr4 = ret_roll_4 / dret_roll_4
    wr5 = ret_roll_5 / dret_roll_5
    
    wr1_bar = ret_roll_1 / dret_roll_1_bar
    wr2_bar = ret_roll_2 / dret_roll_2_bar
    wr3_bar = ret_roll_3 / dret_roll_3_bar
    wr4_bar = ret_roll_4 / dret_roll_4_bar
    wr5_bar = ret_roll_5 / dret_roll_5_bar
    
    ret_n1 = car1
    ret_n2 = car2 - car1
    ret_n3 = car3 - car2
    ret_n4 = car4 - car3
    ret_n5 = car5 - car4
    ret_n6 = car6 - car5
    ret_n7 = car7 - car6
    ret_n8 = car8 - car7
    
    res = pd.DataFrame({
        'date':df.date,
        'mv':df.mv,
        'pre_mv':pre_mv,
        'retx_p1':df.retx_p1,
        'exf':exf,
        'exf_to_mv':exf_to_mv,
        
        'car1':car1.shift(-1),
        'car3':car3.shift(-1),
        'car5':car5.shift(-1),
        
        'wr1':wr1.shift(-1),
        'wr2':wr2.shift(-1),
        'wr3':wr3.shift(-1),
        'wr4':wr4.shift(-1),
        'wr5':wr5.shift(-1),
        
        'wr1_bar':wr1_bar.shift(-1),
        'wr2_bar':wr2_bar.shift(-1),
        'wr3_bar':wr3_bar.shift(-1),
        'wr4_bar':wr4_bar.shift(-1),
        'wr5_bar':wr5_bar.shift(-1),
        
#         'ret_n1':ret_n1.shift(-1),
#         'ret_n2':ret_n2.shift(-1),
#         'ret_n3':ret_n3.shift(-1),
#         'ret_n4':ret_n4.shift(-1),
#         'ret_n5':ret_n5.shift(-1),
#         'ret_n6':ret_n6.shift(-1),
#         'ret_n7':ret_n7.shift(-1),
#         'ret_n8':ret_n8.shift(-1)
        
        'ret_n1':ret.shift(-1)-1,
        'ret_n2':ret.shift(-2)-1,
        'ret_n3':ret.shift(-3)-1,
        'ret_n4':ret.shift(-4)-1,
        'ret_n5':ret.shift(-5)-1,
        'ret_n6':ret.shift(-6)-1,
        'ret_n7':ret.shift(-7)-1,
        'ret_n8':ret.shift(-8)-1
        
    })
    return res


# load data

# In[5]:


crsp_fa = pd.read_csv('data/crsp_fa_filtered.zip',
                     parse_dates=['date'], infer_datetime_format=True)


# In[6]:


crsp_fa.head()


# In[7]:


# 运行此代码后，市值将被替换成调整后的市值
crsp_fa.mv = crsp_fa.mv_adj


# In[23]:


crsp_filter = crsp_fa.set_index('date')[:'2008']
filter_permno = crsp_filter[(crsp_filter.mv_adj.shift(1) * crsp_filter.retx_p1) >= 5e+4].permno.unique().tolist()
crsp_fa_filtered = crsp_fa.set_index('permno').loc[filter_permno].reset_index()


# In[11]:


test = crsp_fa_filtered[crsp_fa_filtered.permno == 10002]


# In[12]:


test.head(10)


# In[13]:


# 通过上面的预览，查看next return
calc_exf(test)


# In[24]:


tic = time.perf_counter()
result_exf = crsp_fa_filtered.set_index('permno').groupby('permno').apply(calc_exf)
toc = time.perf_counter()
print(toc-tic)


# In[15]:


result_exf.head()


# In[25]:


res1 = result_exf.copy()
res1 = res1.dropna(subset=['exf_to_mv']) # 只是剔除exf_to_mv中的异常值
# res1 = res1.dropna() # 只要有异常值，就全部剔除，两者差距不是很大
res1 = res1.reset_index().set_index('date')[:'2008'] # 筛选出到08年的数据
# res1 = res1.loc['1973':'2001']
# 可以自行调整limits大小
res1.exf_to_mv = mstats.winsorize(res1.exf_to_mv, limits=[0.01, 0.01]) # 这里也可以调整

per1 = np.linspace(0,1,11).tolist()
bins = res1.describe(percentiles=per1).exf_to_mv.iloc[4:-1].tolist()
bins = [bins[0]-1] + bins[1:]
# bins = bins[:-1] + [bins[-1]-1]
res1.groupby(pd.cut(res1.exf_to_mv,bins, right=True)).mean().drop(['mv', 'retx_p1', 'exf', 'permno'], axis=1)


# 可视化操作

# In[26]:


res1_grouped = res1.groupby(pd.cut(res1.exf_to_mv,bins, right=True)).mean().drop(['mv', 'retx_p1', 'exf', 'permno'], axis=1)
res1_grouped.index = list(range(1,11))


# In[27]:


res1_grouped.index.name = 'decile'


# In[19]:


res1_grouped


# In[28]:


bar_cols = ['ret_n1', 'ret_n2', 'ret_n3', 'ret_n4', 'ret_n5', 'ret_n6',
       'ret_n7', 'ret_n8']


# In[29]:


res1_grouped_transfer = res1_grouped[bar_cols].T
res1_grouped_transfer = res1_grouped_transfer.assign(decile_diff =                         res1_grouped_transfer[1]-res1_grouped_transfer[10])


# In[30]:


res1_grouped_transfer[[1, 10, 'decile_diff']].plot.bar();


# annual sorts

# 为了按照每年进行分组计算，先添加year字段

# In[23]:


# 进行复制操作，避免破坏数据
res1 = result_exf.copy()
res1 = res1.reset_index().set_index('date')[:'2008'] # 筛选出到08年的数据
res1.exf_to_mv = mstats.winsorize(res1.exf_to_mv, limits=[0.01, 0.01]) 


# In[24]:


# 如果exf_to_mv中有异常值，则drop这一行数据
res1 = res1.dropna(subset=['exf_to_mv'])
# res1 = res1.dropna()


# In[25]:


#重置index, 并添加year字段
res1 = res1.reset_index()
res1 = res1.assign(year = res1.date.dt.year)


# In[26]:


res1.head()


# In[27]:


def group_calc(g):
#     g.exf_to_mv = mstats.winsorize(g.exf_to_mv, limits=[0.05, 0.05])
    per = np.linspace(0,1,11).tolist()
    bins = g.describe(percentiles=per).exf_to_mv.iloc[4:-1].tolist()
    bins = [bins[0]-1] + bins[1:]
    res = g.groupby(pd.cut(g.exf_to_mv,bins, right=True)).mean().    drop(['mv', 'retx_p1', 'exf', 'permno', 'year'], axis=1)
    res.index = list(range(1,11))
    res.index.name = 'decile'
    return res


# In[28]:


# year_list = res1.year.sort_values().unique().tolist()


# In[29]:


res2 = res1.groupby('year').apply(group_calc)


# In[30]:


res2


# In[31]:


res2.groupby('decile').mean()


# In[32]:


test2 = res1[res1.year == 1973]
group_calc(test2)


# In[33]:


# test3 = result_exf.loc[77418]
# test3 = test3.reset_index().set_index('date')
# test3


# annual sorts下的可视化

# In[34]:


res2_grouped = res2.groupby('decile').mean()
res2_grouped.index.name = 'decile'


# In[35]:


# transfer the matrix
res2_grouped_transfer = res2_grouped[bar_cols].T


# In[36]:


res2_grouped_transfer = res2_grouped_transfer.assign(decile_diff =                         res2_grouped_transfer[1]-res2_grouped_transfer[10])


# In[37]:


res1_grouped_transfer[[1, 10, 'decile_diff']].plot.bar();


# calculate the wealth transfer

# pre_mv represent pre transaction market capitalization

# In[39]:


# result_exf = result_exf.assign(pre_mv = result_exf.mv.shift(1) * result_exf.retx_p1)
result_exf.head()


# - wt1_5
#     - 第一个数字(1)代表是WT1还是WT2; 
#     - 第二个数字(5)代表horizon, 即$WT1_{t+1,t+T}$的下标大写T

# np.where类似于excel中的if函数

# WT1

# In[40]:


result_exf = result_exf.assign(
    wt1_1 = np.where(
         result_exf.exf > 0,
         result_exf.exf * (1- result_exf.wr1),
         result_exf.exf * (1- result_exf.wr1) * result_exf.mv / result_exf.pre_mv
         ),
    wt1_2 = np.where(
         result_exf.exf > 0,
         result_exf.exf * (1- result_exf.wr2),
         result_exf.exf * (1- result_exf.wr2) * result_exf.mv / result_exf.pre_mv
         ),
    wt1_3 = np.where(
         result_exf.exf > 0,
         result_exf.exf * (1- result_exf.wr3),
         result_exf.exf * (1- result_exf.wr3) * result_exf.mv / result_exf.pre_mv
         ),
    wt1_4 = np.where(
         result_exf.exf > 0,
         result_exf.exf * (1- result_exf.wr4),
         result_exf.exf * (1- result_exf.wr4) * result_exf.mv / result_exf.pre_mv
         ),
    wt1_5 = np.where(
         result_exf.exf > 0,
         result_exf.exf * (1- result_exf.wr5),
         result_exf.exf * (1- result_exf.wr5) * result_exf.mv / result_exf.pre_mv
         ),
)


# WT2

# In[41]:


result_exf = result_exf.assign(
    wt2_1 = np.where(
         result_exf.exf > 0,
         result_exf.exf * (1- result_exf.wr1_bar),
         result_exf.exf * (1- result_exf.wr1_bar) * result_exf.mv / result_exf.pre_mv
         ),
    wt2_2 = np.where(
         result_exf.exf > 0,
         result_exf.exf * (1- result_exf.wr2_bar),
         result_exf.exf * (1- result_exf.wr2_bar) * result_exf.mv / result_exf.pre_mv
         ),
    wt2_3 = np.where(
         result_exf.exf > 0,
         result_exf.exf * (1- result_exf.wr3_bar),
         result_exf.exf * (1- result_exf.wr3_bar) * result_exf.mv / result_exf.pre_mv
         ),
    wt2_4 = np.where(
         result_exf.exf > 0,
         result_exf.exf * (1- result_exf.wr4_bar),
         result_exf.exf * (1- result_exf.wr4_bar) * result_exf.mv / result_exf.pre_mv
         ),
    wt2_5 = np.where(
         result_exf.exf > 0,
         result_exf.exf * (1- result_exf.wr5_bar),
         result_exf.exf * (1- result_exf.wr5_bar) * result_exf.mv / result_exf.pre_mv
         ),    
)


# In[42]:


result_exf.loc[77418].reset_index().set_index('date')[['wt1_5', 'wt2_5']]


# pct stand for percentage

# In[43]:


result_exf = result_exf.assign(pct_wt1_1 = result_exf.wt1_1 / result_exf.pre_mv,
                               pct_wt1_2 = result_exf.wt1_2 / result_exf.pre_mv,
                               pct_wt1_3 = result_exf.wt1_3 / result_exf.pre_mv,
                               pct_wt1_4 = result_exf.wt1_4 / result_exf.pre_mv,
                               pct_wt1_5 = result_exf.wt1_5 / result_exf.pre_mv,
                               
                               pct_wt2_1 = result_exf.wt2_1 / result_exf.pre_mv,
                               pct_wt2_2 = result_exf.wt2_2 / result_exf.pre_mv,
                               pct_wt2_3 = result_exf.wt2_3 / result_exf.pre_mv,
                               pct_wt2_4 = result_exf.wt2_4 / result_exf.pre_mv,
                               pct_wt2_5 = result_exf.wt2_5 / result_exf.pre_mv,
                              )


# In[44]:


result_exf.columns


# In[45]:


# wt columns
wt_cols = ['wt1_1', 'wt1_2', 'wt1_3',
       'wt1_4', 'wt1_5', 'wt2_1', 'wt2_2', 'wt2_3', 'wt2_4', 'wt2_5',
       'pct_wt1_1', 'pct_wt1_2', 'pct_wt1_3', 'pct_wt1_4', 'pct_wt1_5',
       'pct_wt2_1', 'pct_wt2_2', 'pct_wt2_3', 'pct_wt2_4', 'pct_wt2_5']


# In[46]:


from scipy.stats import ttest_ind


# In[47]:


res_stats = result_exf[wt_cols].describe().T[['mean', 'count']]


# In[50]:


res_stats = res_stats.assign(sums=res_stats['mean']*res_stats['count'])


# In[51]:


res_stats

