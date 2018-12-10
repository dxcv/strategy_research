
# coding: utf-8

# ```
# 通过quantopian 写algo对组合中的ETF进行回测，回测时间从2016.11.14日到最新交易日，
# 
# 初始的时候分配1250万美金，等权重分配
# 
# 开始运行后
# 
# 1. 单只标的下跌5% 需要卖出一半仓位，下跌10% 需要全部卖出
# 2. Moving average 采用 40天和60天
# 3. 单只标的RSI > 75 需要减仓一半， 指数RSI（我们参考SPX的RSI吧）>70 整个portfolio 减仓一半
# 
# 需要把整个portfolio maximum DD 控制在-5%以内，最好-3%以内(上述1,2,3不一定能达到这个风控标准，可以再叠加一些其他的技术指标比如Bolling或MACD,KDJ)
# 
# 第一个回测组合可以只配置Equity ETF 看看效果如何
# 
# 第二个回测组合可以结合Equity ETF 和Bond ETF， 其中XLU配置30% （因为它是low beta 而且相关系数最低），Tech 类的ETF 配置 30%， Bond ETF（具体选择哪些你可以自己优化选择） 配置40%
# 
# 以上说的这些组合都是long的，只是在运行的过程中跌破某个指标会卖出
# 
# 做完这些如果效果还不错(或者实在无法达到我们的风控要求) 我们再做一个纯short的组合，就是开仓的时候已经short的组合再来看情况。
# 
# 
# 你最近研究下 争取周五或周六的时候能有一个初步的成果向我汇报。
# 
# 注意，以上信息和资料都只能在你自己的电脑上操作，不可在你目前的公司电脑使用。
# ```

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import statsmodels.api as sm
# import tushare as ts
import QUANTAXIS as QA
import talib as ta
import datetime


# In[2]:


get_ipython().run_line_magic('pinfo2', 'QA.CROSS')


# In[ ]:


pd.DataFrame.groupby.


# In[ ]:


ta.


# In[ ]:


prices = data.history(security, 'price', 61, '1d')
mean_close_40 = prices.rolling(40).mean()
mean_close_60 = prices.rolling(60).mean()

if CROSS(mean_close_40, mean_close_60)[-1] == 1:
    order_target_percent(security, long_weight)

elif CROSS(mean_close_60, mean_close_40)[-1] == 0:
    order_target_percent(security, 0)

if not 
sec_rsi = ta.RSI(prices, timeperiod = 14)[-1]


# In[ ]:


"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
import numpy as np
import pandas as pd
import talib as ta

import quantopian.algorithm as algo
from quantopian.algorithm import attach_pipeline,  pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.factors import CustomFactor, AverageDollarVolume, SimpleMovingAverage
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS, StaticAssets

equity_list = symbols("IGM", "IXN", "IYW", "XLK", "PNQI", "CQQQ", "KWEB", "GAMR", "XLC", "IGV", "PSJ", "SKYY", "IGN", "HACK", "IPAY", "SOXX", "SMH", "XSD", "IHI", "XLV", "IBB", "XBI", "VHT", "IHF", "ITA", "XLP", "XLY", "XRT", "XLF", "IAI", "IYG", "KBE", "KIE", "VFH", "OIH", "VDE", "XLE", "XOP", "XHB", "XLB", "XLE", "XLI", "XLRE", "XLU", "IYR")

# universe = StaticAssets(equity_list)

long_weight = 1 / len(equity_list)


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    # Rebalance every day, 1 hour after market open.
    
    context.entry_price = {}
    
    algo.schedule_function(
        rebalance,
        algo.date_rules.every_day(),
        algo.time_rules.market_open(), # hours=1
    )

    # Record tracking variables at the end of each day.
    # algo.schedule_function(
    #     record_vars,
    #     algo.date_rules.every_day(),
    #     algo.time_rules.market_close(),
    # )

    # Create our dynamic stock selector.
    attach_pipeline(make_pipeline(), 'pipeline')


def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """

    # Base universe set to the QTradableStocksUS
    base_universe = StaticAssets(equity_list)
    
    mean_close_40 = SimpleMovingAverage(
        inputs = [USEquityPricing.close],
        window_length = 40
    )
    
    mean_close_60 = SimpleMovingAverage(
        inputs = [USEquityPricing.close],
        window_length = 60
    )

    # Factor of yesterday's close price.
    yesterday_close = USEquityPricing.close.latest

    pipe = Pipeline(
        columns={
            'close': yesterday_close,
        },
        screen=base_universe
    )
    return pipe


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = algo.pipeline_output('pipeline')

    # These are the securities that we are interested in trading each day.
    context.security_list = context.output.index

def check_positions_for_loss_or_profit(context, data):
    # Sell our positions on longs/shorts for profit or loss
    for security in context.portfolio.positions:
        if data.can_trade(security) and not get_open_orders(security):
            current_position = context.portfolio.positions[security].amount  
            cost_basis = context.portfolio.positions[security].cost_basis  
            price = data.current(security, 'price')
            # On Long & Profit
            # if price >= cost_basis * 1.10 and current_position > 0:  
            #     order_target_percent(security, 0)  
            #     log.info( str(security) + ' Sold Long for Profit')  
            #     del context.stocks_held[security]  
            # # On Short & Profit
            # if price <= cost_basis* 0.90 and current_position < 0:
            #     order_target_percent(security, 0)  
            #     log.info( str(security) + ' Sold Short for Profit')  
            #     del context.stocks_held[security]
            # On Long & Loss
            
            # 这里的 cost_basis在调仓后，成本也会跟着变吧
            if price <= cost_basis * 0.9 and current_position > 0:  
                order_target_percent(security, 0)  
                log.info( str(security) + ' Sold half Long for Loss')  

            if price <= cost_basis * 0.95 and current_position > 0:
                order_target_percent(security, long_weight / 2)
                log.info(str(security) + ' Sold Long for Loss')
                
                # del context.stocks_held[security] 
            # On Short & Loss
            # if price >= cost_basis * 1.10 and current_position < 0:  
            #     order_target_percent(security, 0)  
            #     log.info( str(security) + ' Sold Short for Loss')  
            #     del context.stocks_held[security]

    
def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    benchmark_prices = data.history(symbol('SPY'), 'prices', 15, '1d')
    spy_rsi = ta.RSI(benchmark_prices, timeperiod = 14)[-1]
    
    for security in equity_list:
        if data.can_trade(security)
            
            prices = data.history(security, 'price', 61, '1d')
            mean_close_40 = prices.rolling(40).mean()
            mean_close_60 = prices.rolling(60).mean()

            if CROSS(mean_close_40, mean_close_60)[-1] == 1:
                order_target_percent(security, long_weight)

            elif CROSS(mean_close_60, mean_close_40)[-1] == 0:
                order_target_percent(security, 0)
                
            if spy_rsi > 70:
                order_target_percent(security, long_weight/2)

            if not get_open_orders(security):
                sec_rsi = ta.RSI(prices, timeperiod = 14)[-1]
                if sec_rsi > 75:
                    order_target_percent(security, long_weight/2)
    
      for security in context.portfolio.positions:
        if data.can_trade(security) and not get_open_orders(security):
            current_position = context.portfolio.positions[security].amount  
            cost_basis = context.portfolio.positions[security].cost_basis  
            price = data.current(security, 'price')    
            
            if price <= cost_basis * 0.9 and current_position > 0:  
                order_target_percent(security, 0)  
                log.info( str(security) + ' Sold half Long for Loss')  

            if price <= cost_basis * 0.95 and current_position > 0:
                order_target_percent(security, long_weight / 2)
                log.info(str(security) + ' Sold Long for Loss')

# help function
def CROSS(A, B):
    """A<B then A>B  A上穿B B下穿A
    
    Arguments:
        A {[type]} -- [description]
        B {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    var = np.where(A<B, 1, 0)
    return (pd.Series(var, index=A.index).diff()<0).apply(int)
    

def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    pass


def handle_data(context, data):
    """
    Called every minute.
    """
    pass


# In[ ]:


for security in context.portfolio.positions:
    if data.can_trade(security): # and not get_open_orders(security)
        current_position = context.portfolio.positions[security].amount
        cost_basis = context.portfolio.positions[security].cost_basis  
        price = data.current(security, 'price')    
        
        if price <= cost_basis * 0.9 and current_position > 0:  
            order_target_percent(security, 0)  
            log.info( str(security) + ' Sold half Long for Loss')  

        if price <= cost_basis * 0.95 and current_position > 0:
            order_target_percent(security, long_weight / 2)
            log.info(str(security) + ' Sold Long for Loss')

