# -*- coding: utf-8 -*-
"""获取扩展股票池：沪深300 + 深证成指 + 科创50 + 创业板指，去重"""
import akshare as ak
import pandas as pd
import os

os.makedirs('data_cache', exist_ok=True)

all_stocks = set()
index_info = {}

# 1. 沪深300
try:
    df = ak.index_stock_cons_csindex(symbol="000300")
    codes = df['成分券代码'].astype(str).str.zfill(6).tolist()
    all_stocks.update(codes)
    index_info['沪深300'] = len(codes)
    print(f"沪深300: {len(codes)}只")
except Exception as e:
    print(f"沪深300获取失败: {e}")
    # 备用：用已有的hs300_constituents.csv
    try:
        df = pd.read_csv('data_cache/hs300_constituents.csv')
        codes = df['code'].astype(str).str.zfill(6).tolist()
        all_stocks.update(codes)
        index_info['沪深300(缓存)'] = len(codes)
        print(f"沪深300(缓存): {len(codes)}只")
    except:
        pass

# 2. 深证成指
try:
    df = ak.index_stock_cons(symbol="399001")
    codes = df['品种代码'].astype(str).str.zfill(6).tolist()
    all_stocks.update(codes)
    index_info['深证成指'] = len(codes)
    print(f"深证成指: {len(codes)}只")
except Exception as e:
    print(f"深证成指获取失败: {e}")

# 3. 科创50
try:
    df = ak.index_stock_cons(symbol="000688")
    codes = df['品种代码'].astype(str).str.zfill(6).tolist()
    all_stocks.update(codes)
    index_info['科创50'] = len(codes)
    print(f"科创50: {len(codes)}只")
except Exception as e:
    print(f"科创50获取失败: {e}")

# 4. 创业板指
try:
    df = ak.index_stock_cons(symbol="399006")
    codes = df['品种代码'].astype(str).str.zfill(6).tolist()
    all_stocks.update(codes)
    index_info['创业板指'] = len(codes)
    print(f"创业板指: {len(codes)}只")
except Exception as e:
    print(f"创业板指获取失败: {e}")

print(f"\n去重后总计: {len(all_stocks)}只股票")
print(f"各指数: {index_info}")

# 保存股票列表
result = pd.DataFrame({'code': sorted(all_stocks)})
# 获取股票名称
try:
    stock_info = ak.stock_info_a_code_name()
    stock_info['code'] = stock_info['code'].astype(str).str.zfill(6)
    name_map = dict(zip(stock_info['code'], stock_info['name']))
    result['name'] = result['code'].map(name_map)
except Exception as e:
    print(f"获取股票名称失败: {e}")
    result['name'] = result['code']

result.to_csv('data_cache/expanded_universe.csv', index=False)
print(f"\n已保存到 data_cache/expanded_universe.csv")
print(result.head(10))
print(f"...\n共 {len(result)} 只")
