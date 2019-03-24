def calc_wrt(df):
    ret = df.ret_p1
    dret = df.dret_p1
    exf = df.exf
    exf_to_mv = exf / df.mv.shift(1)
    
    wr1 = ret[::-1].rolling(1, min_periods=0).apply(np.prod, raw=True)[::-1] / \
        dret[::-1].rolling(1, min_periods=0).apply(np.prod, raw=True)[::-1]
    wr3 = ret[::-1].rolling(3, min_periods=0).apply(np.prod, raw=True)[::-1] / \
        dret[::-1].rolling(3, min_periods=0).apply(np.prod, raw=True)[::-1]
    wr5 = ret[::-1].rolling(5, min_periods=0).apply(np.prod, raw=True)[::-1] / \
        dret[::-1].rolling(5, min_periods=0).apply(np.prod, raw=True)[::-1]
    
    return pd.DataFrame({
        'date':df.date,
        'exf_to_mv':exf_to_mv,
        # 这里做了时间平移处理, 数据对齐，详见后面sample说明
        'wr1':wr1.shift(-1),
        'wr3':wr3.shift(-1),
        'wr5':wr5.shift(-1)
    })

def calc_wrt_full(df):
    ret = df.ret_p1
    dret = df.dret_p1
    exf = df.exf
    exf_to_mv = exf / df.mv.shift(1)
    
    wr1 = ret[::-1].rolling(1, min_periods=0).apply(np.prod, raw=True)[::-1] / \
        dret[::-1].rolling(1, min_periods=0).apply(np.prod, raw=True)[::-1]
    
    wr2 = ret[::-1].rolling(2, min_periods=0).apply(np.prod, raw=True)[::-1] / \
        dret[::-1].rolling(2, min_periods=0).apply(np.prod, raw=True)[::-1]
    
    wr3 = ret[::-1].rolling(3, min_periods=0).apply(np.prod, raw=True)[::-1] / \
        dret[::-1].rolling(3, min_periods=0).apply(np.prod, raw=True)[::-1]
    
    wr4 = ret[::-1].rolling(4, min_periods=0).apply(np.prod, raw=True)[::-1] / \
        dret[::-1].rolling(4, min_periods=0).apply(np.prod, raw=True)[::-1]
        
    wr5 = ret[::-1].rolling(5, min_periods=0).apply(np.prod, raw=True)[::-1] / \
        dret[::-1].rolling(5, min_periods=0).apply(np.prod, raw=True)[::-1]
    
    wr6 = ret[::-1].rolling(6, min_periods=0).apply(np.prod, raw=True)[::-1] / \
        dret[::-1].rolling(6, min_periods=0).apply(np.prod, raw=True)[::-1]
    
    wr7 = ret[::-1].rolling(7, min_periods=0).apply(np.prod, raw=True)[::-1] / \
        dret[::-1].rolling(7, min_periods=0).apply(np.prod, raw=True)[::-1]
    
    wr8 = ret[::-1].rolling(8, min_periods=0).apply(np.prod, raw=True)[::-1] / \
        dret[::-1].rolling(8, min_periods=0).apply(np.prod, raw=True)[::-1]
    
    return pd.DataFrame({
        'date':df.date,
        'exf_to_mv':exf_to_mv,
        # 这里做了时间平移处理, 数据对齐，详见后面sample说明
        'wr1':wr1.shift(-1),
        'wr2':wr2.shift(-1),
        'wr3':wr3.shift(-1),
        'wr4':wr4.shift(-1),
        'wr5':wr5.shift(-1),
        'wr6':wr6.shift(-1),
        'wr7':wr7.shift(-1),
        'wr8':wr8.shift(-1)
    })

def calc_mean(df):
    return pd.DataFrame(
        {
            'N': df.permno.count(),
            'wr1':df.wr1.mean(),
            'wr2': df.wr2.mean(),
            'wr3': df.wr3.mean(),
            'wr4': df.wr4.mean(),
            'wr5': df.wr5.mean(),
            'wr6': df.wr6.mean(),
            'wr7': df.wr7.mean(),
            'wr8': df.wr8.mean()
        }
    )

    car1 = ret[::-1].rolling(1, min_periods=1).apply(np.prod, raw=True)[::-1] - \
        dret[::-1].rolling(1, min_periods=1).apply(np.prod, raw=True)[::-1]
    car3 = ret[::-1].rolling(3, min_periods=1).apply(np.prod, raw=True)[::-1] - \
        dret[::-1].rolling(3, min_periods=1).apply(np.prod, raw=True)[::-1]
    car5 = ret[::-1].rolling(5, min_periods=1).apply(np.prod, raw=True)[::-1] - \
        dret[::-1].rolling(5, min_periods=1).apply(np.prod, raw=True)[::-1]
    
    wr5 = ret[::-1].rolling(5, min_periods=1).apply(np.prod, raw=True)[::-1] / \
        dret[::-1].rolling(5, min_periods=1).apply(np.prod, raw=True)[::-1]
    
    ret_n1 = ret[::-1].rolling(1, min_periods=1).apply(np.prod, raw=True)[::-1]
    ret_n2 = ret[::-1].rolling(2, min_periods=1).apply(np.prod, raw=True)[::-1]
    ret_n3 = ret[::-1].rolling(3, min_periods=1).apply(np.prod, raw=True)[::-1]
    ret_n4 = ret[::-1].rolling(4, min_periods=1).apply(np.prod, raw=True)[::-1]
    ret_n5 = ret[::-1].rolling(5, min_periods=1).apply(np.prod, raw=True)[::-1]
    ret_n6 = ret[::-1].rolling(6, min_periods=1).apply(np.prod, raw=True)[::-1]
    ret_n7 = ret[::-1].rolling(7, min_periods=1).apply(np.prod, raw=True)[::-1]
    ret_n8 = ret[::-1].rolling(8, min_periods=1).apply(np.prod, raw=True)[::-1]
    
    res = pd.DataFrame({
        'date':df.date,
        'mv':df.mv,
#         'mv_adj':df.mv_adj,
#         'ret_p1':df.ret_p1,
        'retx_p1':df.retx_p1,
#         'dret_p1':df.dret_p1,
        'exf':exf,
        'exf_to_mv':exf_to_mv,
        'car1':car1.shift(-1),
        'car3':car3.shift(-1),
        'car5':car5.shift(-1),
        'wr5':wr5.shift(-1),
        'ret_n1':ret_n1.shift(-1),
        'ret_n2':ret_n2.shift(-1),
        'ret_n3':ret_n3.shift(-1),
        'ret_n4':ret_n4.shift(-1),
        'ret_n5':ret_n5.shift(-1),
        'ret_n6':ret_n6.shift(-1),
        'ret_n7':ret_n7.shift(-1),
        'ret_n8':ret_n8.shift(-1)
    })
    return res



def rolling_forward(df, roll_per, min_per=1):
    return ((df[::-1]).rolling(roll_per, min_per).apply(np.prod, raw=True))[::-1]

def calc_exf(df):
    ret = df.ret_p1
    dret = df.dret_p1
    exf = df.mv - df.mv.shift(1) * df.retx_p1
    exf_to_mv = exf / df.mv.shift(1)
    
    car1 = rolling_forward(ret, 1) - rolling_forward(dret, 1)
    car3 = rolling_forward(ret, 3) - rolling_forward(dret, 3)
    car5 = rolling_forward(ret, 5) - rolling_forward(dret, 5)
    
    wr5 = rolling_forward(ret, 5) - rolling_forward(dret, 5)
    
    ret_n1 = rolling_forward(ret, 1)
    ret_n2 = rolling_forward(ret, 2)
    ret_n3 = rolling_forward(ret, 3)
    ret_n4 = rolling_forward(ret, 4)
    ret_n5 = rolling_forward(ret, 5)
    ret_n6 = rolling_forward(ret, 6)
    ret_n7 = rolling_forward(ret, 7)
    ret_n8 = rolling_forward(ret, 8)
    
    res = pd.DataFrame({
        'date':df.date,
        'mv':df.mv,
#         'mv_adj':df.mv_adj,
#         'ret_p1':df.ret_p1,
        'retx_p1':df.retx_p1,
#         'dret_p1':df.dret_p1,
        'exf':exf,
        'exf_to_mv':exf_to_mv,
        'car1':car1.shift(-1),
        'car3':car3.shift(-1),
        'car5':car5.shift(-1),
        'wr5':wr5.shift(-1),
        'ret_n1':ret_n1.shift(-1),
        'ret_n2':ret_n2.shift(-1),
        'ret_n3':ret_n3.shift(-1),
        'ret_n4':ret_n4.shift(-1),
        'ret_n5':ret_n5.shift(-1),
        'ret_n6':ret_n6.shift(-1),
        'ret_n7':ret_n7.shift(-1),
        'ret_n8':ret_n8.shift(-1)
    })
    return res




def calc_exf(df):
    ret = df.ret_p1
    dret = df.dret_p1
    exf = df.mv - df.mv.shift(1) * df.retx_p1
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
    
    dret_roll_5_bar = ((dret[::-1]).rolling(5, min_periods=1).\
                       apply(np.prod, raw=True)[::-1]).mean()
    
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
    
    wr5_bar = ret_roll_5 / dret_roll_5_bar
    
#     car1 = ((ret[::-1]).rolling(1, min_periods=1).apply(np.prod, raw=True)[::-1]) - \
#         ((dret[::-1]).rolling(1, min_periods=1).apply(np.prod, raw=True)[::-1])
#     car2 = ((ret[::-1]).rolling(2, min_periods=1).apply(np.prod, raw=True)[::-1]) - \
#         ((dret[::-1]).rolling(2, min_periods=1).apply(np.prod, raw=True)[::-1])
#     car3 = ((ret[::-1]).rolling(3, min_periods=1).apply(np.prod, raw=True)[::-1]) - \
#         ((dret[::-1]).rolling(3, min_periods=1).apply(np.prod, raw=True)[::-1])
#     car4 = ((ret[::-1]).rolling(4, min_periods=1).apply(np.prod, raw=True)[::-1]) - \
#         ((dret[::-1]).rolling(4, min_periods=1).apply(np.prod, raw=True)[::-1])
#     car5 = ((ret[::-1]).rolling(5, min_periods=1).apply(np.prod, raw=True)[::-1]) - \
#         ((dret[::-1]).rolling(5, min_periods=1).apply(np.prod, raw=True)[::-1])
#     car6 = ((ret[::-1]).rolling(6, min_periods=1).apply(np.prod, raw=True)[::-1]) - \
#         ((dret[::-1]).rolling(6, min_periods=1).apply(np.prod, raw=True)[::-1])
#     car7 = ((ret[::-1]).rolling(7, min_periods=1).apply(np.prod, raw=True)[::-1]) - \
#         ((dret[::-1]).rolling(7, min_periods=1).apply(np.prod, raw=True)[::-1])
#     car8 = ((ret[::-1]).rolling(8, min_periods=1).apply(np.prod, raw=True)[::-1]) - \
#         ((dret[::-1]).rolling(8, min_periods=1).apply(np.prod, raw=True)[::-1])
    
#     wr1 = ((ret[::-1]).rolling(1, min_periods=1).apply(np.prod, raw=True)[::-1]) / \
#         ((dret[::-1]).rolling(1, min_periods=1).apply(np.prod, raw=True)[::-1])
#     wr2 = ((ret[::-1]).rolling(2, min_periods=1).apply(np.prod, raw=True)[::-1]) / \
#         ((dret[::-1]).rolling(2, min_periods=1).apply(np.prod, raw=True)[::-1])
#     wr3 = ((ret[::-1]).rolling(3, min_periods=1).apply(np.prod, raw=True)[::-1]) / \
#         ((dret[::-1]).rolling(3, min_periods=1).apply(np.prod, raw=True)[::-1])
#     wr4 = ((ret[::-1]).rolling(4, min_periods=1).apply(np.prod, raw=True)[::-1]) / \
#         ((dret[::-1]).rolling(4, min_periods=1).apply(np.prod, raw=True)[::-1])
#     wr5 = ((ret[::-1]).rolling(5, min_periods=1).apply(np.prod, raw=True)[::-1]) / \
#         ((dret[::-1]).rolling(5, min_periods=1).apply(np.prod, raw=True)[::-1])
#     wr5_bar = ((ret[::-1]).rolling(5, min_periods=1).apply(np.prod, raw=True)[::-1]) / \
#         ((dret[::-1]).rolling(5, min_periods=1).apply(np.prod, raw=True)[::-1]).mean()
    
#     ret_n1 = ((ret[::-1]).rolling(1, min_periods=1).apply(np.prod, raw=True)[::-1])


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
        'wr5_bar':wr5_bar,
        'ret_n1':ret_n1.shift(-1),
        'ret_n2':ret_n2.shift(-1),
        'ret_n3':ret_n3.shift(-1),
        'ret_n4':ret_n4.shift(-1),
        'ret_n5':ret_n5.shift(-1),
        'ret_n6':ret_n6.shift(-1),
        'ret_n7':ret_n7.shift(-1),
        'ret_n8':ret_n8.shift(-1)
        
#         'ret_n1':ret_n1,
#         'ret_n2':ret_n2,
#         'ret_n3':ret_n3,
#         'ret_n4':ret_n4,
#         'ret_n5':ret_n5,
#         'ret_n6':ret_n6,
#         'ret_n7':ret_n7,
#         'ret_n8':ret_n8
    })
    return res