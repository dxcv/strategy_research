import numpy as np
import pandas as pd
import time
# from tqdm import tqdm_notebook

comp = pd.read_excel('demo.xlsx', sheet_name=0)
crsp = pd.read_excel('demo.xlsx', sheet_name=1, index_col=0)

def fiscal_resample(df, fys, per):
    """
    fys -- 会计年度结束月份标识
    per -- 股票唯一标识
    """
    # 收益率做累乘, 然后取最后一个值
    ret_p1 = df.ret_p1.cumprod().iloc[-1]
    # 除权除息后的收益率做累乘, 然后取最后一个值
    retx_p1 = df.retx_p1.cumprod().iloc[-1]
    # 收益率经过CPI调整之后做累乘，然后取最后一个值
    ret_p1_adj = df.ret_p1_adj.cumprod().iloc[-1]
    # 取市值的最后一个值
    mv = df.mv.resample(fys).last().iloc[-1]
    res = pd.DataFrame({'ret_p1':ret_p1, 'retx_p1':retx_p1,
                        'ret_p1_adj':ret_p1_adj, 'mv':mv}, index=[per])
    return res

tic = time.perf_counter()
# 按照每只股票和财政年度group
g = comp.groupby(['permno', 'fyear'])
df_list = []
# for name, group in tqdm_notebook(g):
for name, group in g:
    # 获取财政年度开始日期
    start = group.fysm.iloc[0].strftime('%Y-%m-%d')
    # 获取财政年度结束日期
    end = group.datadate.iloc[0].strftime('%Y-%m-%d')
    # 获取财政年度结束月份标识
    fystring = group.fystr.iloc[0]
    # 拿到对应股票时间段的数据
    sample = crsp[crsp.permno==name[0]][start:end]
    if len(sample) > 0:
        r = sample.resample(fystring)
        df_list.append(r.apply(fiscal_resample, fys=fystring, per=name[0]))

df = pd.concat(df_list, axis=0)
df.index.names = ['date', 'permno']

toc= time.perf_counter()
print(toc - tic)

