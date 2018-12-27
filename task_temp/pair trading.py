import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import statsmodels.api as sm
import datetime, time
# import pandas_datareader.data as web

# 导入计算pair trading的函数
from utils import *

try:
    import seaborn as sns
    from tqdm import tqdm_notebook
    import QUANTAXIS as QA
    import talib as ta
except:
    pass
"""
# etf symbol

code_list_by_sector.etf_symbol.value_counts().index
Index(['IBB', 'VDE', 'PNQI', 'CQQQ', 'GAMR', 'XLK', 'XLY', 'XLV', 'HACK',
       'ITA', 'XLP', 'PSJ', 'SOXX', 'SKYY', 'IAI', 'SMH', 'XLB'],
      dtype='object')

sector symbol
Biotech/ Pharma              191
Energy                       143
Internet                      95
China Internet                78
Online Gaming                 72
Hardware                      68
Consumer                      66
Medical                       63
Network/Security              53
Health Care Services          36
Defense                       32
Software/ Cloud Computing     31
ePayment                      30
Cloud Computing               29
Financials                    27
Semiconductor                 25
Others                        24
Name: sector, dtype: int64
"""

# 计算SMH etf内所有的pair trading
result = calc_pair_trading('SMH')

# plotting ratio
plt.plot(result['ratio'].index, result['ratio'].apply(lambda x : np.log(x)).values);

# 展示做多的标的, 去掉tail()显示全部
result['long_asset'].tail()

# 展示long return
result['long_return'].tail()

# 展示long short return
result['long_short_return'].tail()



