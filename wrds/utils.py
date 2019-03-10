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