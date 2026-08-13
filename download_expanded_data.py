# -*- coding: utf-8 -*-
"""下载扩展股票池(663只)近3年日线数据"""
import pandas as pd
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_fetcher import get_stock_daily

START = "20230801"
END = "20260812"

# 读取股票池
universe = pd.read_csv('data_cache/expanded_universe.csv')
universe['code'] = universe['code'].astype(str).str.zfill(6)
codes = universe['code'].tolist()
names = dict(zip(universe['code'], universe['name']))

print(f"共 {len(codes)} 只股票，下载 {START} 至 {END} 数据")

success = 0
failed = 0
cached = 0
failed_codes = []

for i, code in enumerate(codes):
    cache_file = f'data_cache/{code}_{START}_{END}_qfq.parquet'
    if os.path.exists(cache_file):
        cached += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(codes)}] 已缓存 {cached}, 成功 {success}, 失败 {failed}")
        continue
    
    try:
        df = get_stock_daily(code, start_date=START, end_date=END)
        if df is not None and len(df) > 0:
            success += 1
        else:
            failed += 1
            failed_codes.append(code)
    except Exception as e:
        failed += 1
        failed_codes.append(code)
        if failed <= 5:
            print(f"  {code} 失败: {e}")
    
    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(codes)}] 已缓存 {cached}, 新下载 {success}, 失败 {failed}")
        time.sleep(1)  # 避免请求过快

print(f"\n下载完成: 缓存 {cached}, 新成功 {success}, 失败 {failed}")
if failed_codes:
    print(f"失败股票: {failed_codes[:20]}")
    pd.DataFrame({'code': failed_codes}).to_csv('data_cache/failed_codes.csv', index=False)
