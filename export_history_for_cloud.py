# -*- coding: utf-8 -*-
"""导出最近120天所有股票数据为单个parquet文件，用于GitHub云端运行"""
import pandas as pd
import os
import glob

START = '20200801'
END = '20260812'
data_dir = 'data_cache'

# 读取股票池
universe = pd.read_csv('data_cache/expanded_universe.csv')
universe['code'] = universe['code'].astype(str).str.zfill(6)
codes = universe['code'].tolist()
print(f"股票池: {len(codes)} 只")

# 合并所有股票最近120天数据
all_data = []
success = 0
for code in codes:
    cache_file = f'{data_dir}/{code}_{START}_{END}_qfq.parquet'
    if not os.path.exists(cache_file):
        continue
    try:
        df = pd.read_parquet(cache_file)
        if len(df) >= 60:
            # 只取最近120个交易日
            df = df.tail(120).copy()
            df['code'] = code
            all_data.append(df)
            success += 1
    except:
        continue

print(f"成功加载: {success}/{len(codes)} 只")

if all_data:
    merged = pd.concat(all_data, ignore_index=True)
    print(f"合并后: {len(merged)} 行, {merged.memory_usage(deep=True).sum()/1024/1024:.1f} MB")
    
    # 保存
    output = 'data_cache/history_120d.parquet'
    merged.to_parquet(output, index=False)
    size = os.path.getsize(output) / 1024 / 1024
    print(f"已保存: {output} ({size:.1f} MB)")
