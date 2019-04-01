# 双均线模型 日线
# 导入函数库
from jqdata import *
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime,timedelta


# 初始化函数，设定基准等等
def initialize(context):
    # 设定参数
    set_params(context)
    # 设定回测规则
    set_backtest()

    # 开盘前运行
    run_daily(before_market_open, time='before_open', reference_security='000300.XSHG')
    # 开盘时运行
    run_daily(market_open, time='open', reference_security='000300.XSHG')
    # 收盘后运行
    run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')


def set_params(context):
    # 设置短期、长期均线
    g.short_win = 4
    # 统计样本长度
    g.long_window = 44
    # 风险参考基准、交易标的
    g.security1 = '000300.XSHG'

    
# 3 设置回测条件
def set_backtest():
    set_benchmark(g.security1)       # 设置为基准
    set_option('use_real_price', True) # 用真实价格交易
    set_option('order_volume_ratio', 1)# 放大买入限制
    log.set_level('order', 'error')    # 设置报错等级
    # 股票类每笔交易时的手续费是：买入时佣金千1，卖出时佣金千1，印花税千1, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.001, close_commission=0.001, min_commission=5), type='stock')
    # 为股票设定滑点为百分比滑点
    set_slippage(PriceRelatedSlippage(0.00246),type='stock')


## 开盘前运行函数
def before_market_open(context):
    # 输出运行时间
    log.info('函数运行时间(before_market_open)：'+str(context.current_dt.time()))


## 开盘时运行函数
def market_open(context):

    # 短期均线
    short_prices = attribute_history(g.security1, g.short_win, '1d', ['close'])['close'].mean()
    # 长期均线
    long_prices = attribute_history(g.security1, g.long_window, '1d', ['close'])['close'].mean()
    
    print('short_prices long_prices',short_prices,long_prices)
    
    # 取得当前的现金
    cash = context.portfolio.available_cash
    
    # 如果上一时间点的RSRS斜率大于买入阈值, 则全仓买入
    if short_prices > long_prices:
        # 记录这次买入
        log.info("RSRS大于买入阈值, 买入 %s" % g.security1)
        # 用所有 cash 买入
        order_value(g.security1, cash)
    # 如果上一时间点的RSRS斜率小于卖出阈值, 则空仓卖出
    elif short_prices < long_prices and context.portfolio.positions[g.security1].closeable_amount > 0:
        # 记录这次卖出
        log.info("RSRS小于卖出阈值, 卖出 %s" % g.security1)
        # 卖出所有,最终持有量为0
        order_target(g.security1, 0)


## 收盘后运行函数
def after_market_close(context):
    log.info(str('函数运行时间(after_market_close):'+str(context.current_dt.time())))
    #得到当天所有成交记录
    trades = get_trades()
    for _trade in trades.values():
        log.info('成交记录：'+str(_trade))
    log.info('一天结束')
    log.info('##############################################################')
