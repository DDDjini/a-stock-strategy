# -*- coding: utf-8 -*-
"""导出50只股票的K线数据+交易信号为JS格式，供网站个股K线Tab使用"""
import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# 加载优化版交易记录
trades_df = pd.read_csv('results/trades_optimized_50.csv')
trades_df['code'] = trades_df['code'].astype(str).str.zfill(6)
stock_codes = sorted(trades_df['code'].unique())
print(f"共 {len(stock_codes)} 只股票有交易")

# 股票名称映射（从沪深300成分股获取）
try:
    stock_list = pd.read_csv('data_cache/hs300_constituents.csv')
    stock_list['code'] = stock_list['code'].astype(str).str.zfill(6)
    name_map = dict(zip(stock_list['code'], stock_list['name']))
except:
    name_map = {}

# 构建结果
result = {}
for code in stock_codes:
    # 加载K线数据
    cache_file = f'data_cache/{code}_20180101_20260812_qfq.parquet'
    if not os.path.exists(cache_file):
        # 尝试其他命名格式
        import glob
        files = glob.glob(f'data_cache/{code}_*.parquet')
        if files:
            cache_file = files[0]
        else:
            print(f"  {code}: 无缓存数据，跳过")
            continue
    
    df = pd.read_parquet(cache_file)
    df = df.sort_values('date').reset_index(drop=True)
    
    # 只保留需要的列，减少数据量
    klines = []
    for _, row in df.iterrows():
        klines.append({
            'time': str(row['date'])[:10],
            'open': round(float(row['open']), 2),
            'high': round(float(row['high']), 2),
            'low': round(float(row['low']), 2),
            'close': round(float(row['close']), 2),
            'volume': int(row['volume'])
        })
    
    # 该股票的交易记录
    stock_trades = trades_df[trades_df['code'] == code].copy()
    trades_list = []
    for _, t in stock_trades.iterrows():
        trades_list.append({
            'entry_date': str(t['entry_date'])[:10],
            'exit_date': str(t['exit_date'])[:10],
            'entry_price': round(float(t['entry_price']), 2),
            'exit_price': round(float(t['exit_price']), 2),
            'pattern': str(t['pattern']),
            'pnl_pct': round(float(t['pnl_pct']), 4),
            'hold_days': int(t['hold_days']),
            'exit_reason': str(t['exit_reason']),
            'max_profit': round(float(t['max_profit']), 4)
        })
    
    name = name_map.get(code, code)
    result[code] = {
        'name': name,
        'klines': klines,
        'trades': trades_list
    }
    print(f"  {code} {name}: {len(klines)}根K线, {len(trades_list)}笔交易")

# 导出为JS文件
output_path = 'web/stock_kline_data.js'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('// 个股K线数据+交易信号（50只股票优化版回测）\n')
    f.write('// 自动生成，请勿手动编辑\n')
    f.write(f'const STOCK_KLINE_DATA = {json.dumps(result, ensure_ascii=False)};\n')

total_klines = sum(len(v['klines']) for v in result.values())
total_trades = sum(len(v['trades']) for v in result.values())
file_size = os.path.getsize(output_path) / 1024
print(f"\n导出完成: {len(result)}只股票, {total_klines}根K线, {total_trades}笔交易")
print(f"文件: {output_path}, 大小: {file_size:.1f}KB")
