# -*- coding: utf-8 -*-
"""下载扩展股票池6年日线数据"""
import warnings
warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, ".")
import pandas as pd
from data_fetcher import get_stock_daily

START = "20200801"
END = "20260812"

universe = pd.read_csv('data_cache/expanded_universe.csv')
universe['code'] = universe['code'].astype(str).str.zfill(6)
codes = universe['code'].tolist()
print(f"股票池: {len(codes)}只, 时间: {START} - {END}")

success = 0
cached = 0
failed = []
for i, code in enumerate(codes):
    cache_file = f'data_cache/{code}_{START}_{END}_qfq.parquet'
    if os.path.exists(cache_file):
        cached += 1
    else:
        try:
            df = get_stock_daily(code, start_date=START, end_date=END)
            if df is not None and len(df) >= 60:
                success += 1
            else:
                failed.append(code)
        except Exception as e:
            failed.append(code)
    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(codes)}] 已缓存 {cached}, 新下载 {success}, 失败 {len(failed)}")

print(f"\n下载完成: 缓存 {cached}, 新成功 {success}, 失败 {len(failed)}")
if failed:
    pd.DataFrame({'code': failed}).to_csv('data_cache/failed_codes_6y.csv', index=False)
    print(f"失败股票: {failed[:10]}...")
