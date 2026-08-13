# -*- coding: utf-8 -*-
"""
实时信号监控系统
- 扫描662只股票的入场/出场信号
- 飞书Webhook推送
- 状态持久化，避免重复通知
"""
import warnings
warnings.filterwarnings("ignore")
import sys, os, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import akshare as ak
import config as C
from data_fetcher import compute_technical_indicators, get_stock_daily
from pattern_recognizer import detect_all_patterns, get_best_pattern
from strategy import check_market_filter, check_false_breakout
from market_regime import MarketRegimeDetector, get_regime_params
from train_ml_from_trades import load_ml_model, score_signal
from feishu_notifier import FeishuNotifier

# ============ 配置 ============
START = "20200801"
END = "20260812"  # 历史数据截止日
STATE_FILE = "monitor_state.json"
WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK", "")  # 从环境变量读取，或直接填写
ML_THRESHOLD = 55
MIN_VOLUME_RATIO = 1.0  # 量能阈值

# ============ 状态管理 ============
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "last_scan_time": "",
        "notified_entry": [],   # ["code_date", ...]
        "notified_exit": [],
        "positions": {},        # {code: {entry_date, entry_price, pattern, highest_close}}
        "daily_stats": {}
    }

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def is_trading_time():
    """判断当前是否在交易时间"""
    now = datetime.now()
    if now.weekday() >= 5:  # 周末
        return False
    t = now.time()
    morning = (t >= datetime.strptime("09:30", "%H:%M").time() and
               t <= datetime.strptime("11:30", "%H:%M").time())
    afternoon = (t >= datetime.strptime("13:00", "%H:%M").time() and
                 t <= datetime.strptime("15:00", "%H:%M").time())
    return morning or afternoon

# ============ 数据获取 ============
def load_stock_pool():
    """加载股票池"""
    os.makedirs('data_cache', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    universe = pd.read_csv('data_cache/expanded_universe.csv')
    universe['code'] = universe['code'].astype(str).str.zfill(6)
    return universe

# 合并历史数据缓存（云端模式）
_merged_history = None

def load_merged_history():
    """加载合并的历史数据文件（云端模式用）"""
    global _merged_history
    if _merged_history is not None:
        return _merged_history
    merged_file = 'data_cache/history_120d.parquet'
    if os.path.exists(merged_file):
        try:
            _merged_history = pd.read_parquet(merged_file)
            print(f"已加载合并历史数据: {len(_merged_history)} 行, "
                  f"{_merged_history['code'].nunique()} 只股票")
        except Exception as e:
            print(f"加载合并历史数据失败: {e}")
            _merged_history = pd.DataFrame()
    else:
        _merged_history = pd.DataFrame()
    return _merged_history

def get_stock_history(code):
    """获取股票历史数据
    优先从合并文件读取（云端模式），其次读单个缓存，最后自动下载
    """
    # 1. 从合并历史数据中提取
    merged = load_merged_history()
    if not merged.empty and 'code' in merged.columns:
        df = merged[merged['code'] == code].copy()
        if len(df) >= 60:
            df = df.drop(columns=['code'], errors='ignore')
            df = df.reset_index(drop=True)
            return df
    
    # 2. 单个缓存文件
    cache_file = f'data_cache/{code}_{START}_{END}_qfq.parquet'
    if os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except:
            pass
    
    # 3. 自动下载（本地模式）
    try:
        one_year_ago = (datetime.now() - timedelta(days=400)).strftime('%Y%m%d')
        today = datetime.now().strftime('%Y%m%d')
        df = get_stock_daily(code, one_year_ago, today, adjust="qfq")
        if df is not None and len(df) >= 60:
            df.to_parquet(cache_file, index=False)
            return df
    except Exception as e:
        print(f"  [下载失败] {code}: {e}")
    return None

def get_realtime_spot(stock_codes):
    """
    获取实时行情（新浪财经API，批量请求，更稳定）
    返回DataFrame，index为股票代码，列：名称,今开,昨收,最新价,最高,最低,成交量,成交额,量比
    """
    import requests

    # 转换为新浪格式：sh600000, sz000001
    sina_codes = []
    for code in stock_codes:
        if code.startswith('6') or code.startswith('9'):
            sina_codes.append(f'sh{code}')
        else:
            sina_codes.append(f'sz{code}')

    # 新浪API每次最多约800只，662只可以一次请求
    all_data = {}
    batch_size = 700

    for i in range(0, len(sina_codes), batch_size):
        batch = sina_codes[i:i+batch_size]
        url = f"http://hq.sinajs.cn/list={','.join(batch)}"
        headers = {
            'Referer': 'https://finance.sina.com.cn',
            'User-Agent': 'Mozilla/5.0'
        }

        for retry in range(3):
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                resp.encoding = 'gbk'
                lines = resp.text.strip().split('\n')
                for line in lines:
                    if '="' not in line:
                        continue
                    # 解析: var hq_str_sh600000="浦发银行,10.5,..."
                    code_part = line.split('=')[0].replace('var hq_str_', '').strip()
                    code = code_part[2:]  # 去掉sh/sz前缀
                    data_str = line.split('="')[1].rstrip('";')
                    fields = data_str.split(',')
                    if len(fields) < 32:
                        continue
                    try:
                        all_data[code] = {
                            '名称': fields[0],
                            '今开': float(fields[1]),
                            '昨收': float(fields[2]),
                            '最新价': float(fields[3]),
                            '最高': float(fields[4]),
                            '最低': float(fields[5]),
                            '成交量': float(fields[8]) / 100,  # 股→手
                            '成交额': float(fields[9]),
                            '量比': float(fields[49]) if len(fields) > 49 else 0,
                            '涨跌幅': float(fields[32]) if len(fields) > 32 else 0,
                        }
                    except (ValueError, IndexError):
                        continue
                break
            except Exception as e:
                if retry == 2:
                    print(f"[新浪行情获取失败] 批次{i//batch_size}: {e}")
                else:
                    import time
                    time.sleep(2)

    if not all_data:
        return None

    df = pd.DataFrame.from_dict(all_data, orient='index')
    df.index.name = '代码'
    return df

def build_today_kline(hist_df, spot_row, today_str):
    """用实时快照构造今日K线，追加到历史数据"""
    if spot_row is None:
        return hist_df

    # 实时快照字段：今开、最高、最低、最新价、成交量、量比
    open_p = float(spot_row.get('今开', 0))
    high_p = float(spot_row.get('最高', 0))
    low_p = float(spot_row.get('最低', 0))
    close_p = float(spot_row.get('最新价', 0))
    volume = float(spot_row.get('成交量', 0))

    if open_p <= 0 or close_p <= 0:
        return hist_df

    today_ts = pd.Timestamp(today_str)
    # 如果今日数据已存在，替换；否则追加
    new_row = pd.DataFrame([{
        'date': today_ts, 'open': open_p, 'high': high_p,
        'low': low_p, 'close': close_p, 'volume': volume,
        'amount': float(spot_row.get('成交额', 0)),
        'amplitude': float(spot_row.get('振幅', 0)),
        'pct_chg': float(spot_row.get('涨跌幅', 0)),
        'change': float(spot_row.get('涨跌额', 0)),
        'turnover': float(spot_row.get('换手率', 0))
    }])

    # 移除历史中可能存在的今日数据
    hist_df = hist_df[hist_df['date'] != today_ts].copy()
    result = pd.concat([hist_df, new_row], ignore_index=True)
    result = result.sort_values('date').reset_index(drop=True)
    return result

# ============ 信号扫描 ============
def scan_entry_for_stock(df, code, name, index_df, regime_df, regime_detector,
                         model, ml_cols, spot_row, today_str):
    """扫描单只股票的入场信号"""
    if df is None or len(df) < 60:
        return None

    idx = len(df) - 1  # 最新一根K线（今日）
    row = df.iloc[idx]

    # 1. 量能确认：今日成交量 > 前3天平均
    if idx < 4:
        return None
    vol_ma3 = df['volume'].iloc[idx-3:idx].mean()
    if vol_ma3 <= 0:
        return None
    vol_ratio = row['volume'] / vol_ma3
    if vol_ratio < MIN_VOLUME_RATIO:
        return None

    # 2. 市场状态过滤
    today_ts = pd.Timestamp(today_str)
    regime = "range"
    regime_row = regime_df[regime_df['date'] == today_ts]
    if len(regime_row) > 0:
        regime = regime_row.iloc[0]['regime']
    else:
        # 用最近一天的状态
        regime = regime_df.iloc[-1]['regime']

    regime_params = get_regime_params(regime)
    if not check_market_filter(index_df, today_ts):
        return None

    # 3. 形态识别（最近30天窗口）
    lookback = min(30, idx)
    patterns = detect_all_patterns(df, lookback=lookback)
    if not patterns:
        return None

    # 4. 取质量分最高的形态
    best_pattern = patterns[0]
    quality = best_pattern.get('quality', 0)
    if quality < C.PATTERN_QUALITY_THRESHOLD:
        return None

    neckline = best_pattern.get('neckline', 0)
    pattern_type = best_pattern.get('pattern', '未知')

    # 5. 假突破过滤
    if not check_false_breakout(df, idx, neckline):
        return None

    # 6. 价格确认：收盘价 > 颈线
    if row['close'] <= neckline:
        return None

    # 7. ML评分
    ml_score = 50.0
    try:
        ml_score = score_signal(df, idx, neckline, pattern_type, index_df)
    except:
        pass
    if ml_score < ML_THRESHOLD:
        return None

    return {
        'code': code,
        'name': name,
        'pattern': pattern_type,
        'price': float(row['close']),
        'neckline': float(neckline),
        'volume_ratio': float(vol_ratio),
        'ml_score': float(ml_score),
        'quality_score': float(quality),
        'regime': regime,
        'signal_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def check_exit_for_position(df, code, pos_info, today_str):
    """检查持仓股票的出场信号"""
    if df is None or len(df) < 5:
        return None

    entry_date = pd.Timestamp(pos_info['entry_date'])
    entry_price = pos_info['entry_price']
    pattern = pos_info.get('pattern', '未知')

    # 找到入场后的K线索引
    entry_idx = df[df['date'] >= entry_date].index
    if len(entry_idx) == 0:
        return None
    entry_idx = entry_idx[0]

    idx = len(df) - 1  # 今日
    if idx <= entry_idx:
        return None

    row = df.iloc[idx]
    atr = row.get('atr_14', row.get('atr14', 0))

    # 1. 硬止损：收盘价 < 入场价 - 2*ATR
    if atr > 0 and row['close'] < entry_price - 2 * atr:
        return {
            'code': code, 'exit_price': float(row['close']),
            'entry_price': entry_price, 'reason': '硬止损(2ATR)',
            'hold_days': idx - entry_idx,
            'pnl_pct': (float(row['close']) - entry_price) / entry_price * 100,
            'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    # 2. 趋势线跟踪：从入场后最高点画趋势线，跌破则出场
    # 简化版：跟踪入场后的最高收盘价，当收盘价从最高点回撤超过8%时出场
    post_entry = df.iloc[entry_idx:idx+1]
    highest_close = post_entry['close'].max()
    drawdown_from_peak = (row['close'] - highest_close) / highest_close

    # 移动止盈：盈利超过10%后，回撤5%止盈；否则用趋势线
    profit_pct = (row['close'] - entry_price) / entry_price
    if profit_pct > 0.10 and drawdown_from_peak < -0.05:
        return {
            'code': code, 'exit_price': float(row['close']),
            'entry_price': entry_price, 'reason': '移动止盈(高点回撤5%)',
            'hold_days': idx - entry_idx,
            'pnl_pct': profit_pct * 100,
            'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    # 3. 跌破入场价（亏损止损）
    if row['close'] < entry_price * 0.95:
        return {
            'code': code, 'exit_price': float(row['close']),
            'entry_price': entry_price, 'reason': '亏损止损(-5%)',
            'hold_days': idx - entry_idx,
            'pnl_pct': profit_pct * 100,
            'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    # 4. 趋势线跌破：用近期低点连线
    if idx - entry_idx >= 5:
        # 找入场后的局部低点
        recent = df.iloc[entry_idx:idx+1].copy()
        recent['is_low'] = recent['low'].rolling(5, center=True).min() == recent['low']
        lows = recent[recent['is_low'] == True]
        if len(lows) >= 2:
            # 简单趋势线：最后两个低点连线
            x1, y1 = lows.iloc[-2].name, lows.iloc[-2]['low']
            x2, y2 = lows.iloc[-1].name, lows.iloc[-1]['low']
            if x2 > x1:
                slope = (y2 - y1) / (x2 - x1)
                trend_today = y2 + slope * (idx - x2)
                if row['close'] < trend_today * 0.99:
                    return {
                        'code': code, 'exit_price': float(row['close']),
                        'entry_price': entry_price, 'reason': '跌破趋势线',
                        'hold_days': idx - entry_idx,
                        'pnl_pct': profit_pct * 100,
                        'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }

    return None

# ============ 主流程 ============
def run_monitor(force_run=False, send_test=False):
    """执行一次监控扫描"""
    print("=" * 60)
    print(f"实时监控扫描 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 检查交易时间
    if not force_run and not is_trading_time():
        print("当前非交易时间，跳过扫描")
        return

    # 初始化飞书
    if not WEBHOOK_URL:
        print("[警告] 未配置飞书Webhook，信号将只打印不推送")
        print("  设置环境变量: set FEISHU_WEBHOOK=你的webhook地址")
        notifier = None
    else:
        notifier = FeishuNotifier(WEBHOOK_URL)
        if send_test:
            notifier.send_test()

    # 加载状态
    state = load_state()
    today_str = datetime.now().strftime('%Y-%m-%d')

    # 预加载合并历史数据（云端模式）
    load_merged_history()

    # 加载股票池
    universe = load_stock_pool()
    stock_codes = universe['code'].tolist()
    name_map = dict(zip(universe['code'], universe['name']))
    print(f"股票池: {len(stock_codes)}只")

    # 获取实时行情
    print("\n获取实时行情(新浪源)...")
    spot_df = get_realtime_spot(stock_codes)
    if spot_df is None:
        print("无法获取实时行情，退出")
        return
    print(f"实时行情: {len(spot_df)}只股票")

    # 加载指数数据和市场状态
    print("加载指数和市场状态...")
    from data_fetcher import get_index_daily
    index_df = get_index_daily(C.INDEX_CODE, START, END)
    index_df = compute_technical_indicators(index_df)

    # 追加指数今日数据
    if C.INDEX_CODE in spot_df.index:
        index_spot = spot_df.loc[C.INDEX_CODE] if C.INDEX_CODE in spot_df.index else None
    else:
        index_spot = None
    # 指数代码可能不同，用沪深300
    index_code_spot = "000300"
    if index_code_spot in spot_df.index:
        index_spot = spot_df.loc[index_code_spot]
        index_df = build_today_kline(index_df, index_spot, today_str)
        index_df = compute_technical_indicators(index_df)

    regime_detector = MarketRegimeDetector(n_regimes=3)
    regime_detector.fit(index_df)
    regime_df = regime_detector.predict(index_df)

    # 加载ML模型
    model, ml_cols = load_ml_model()
    print(f"ML模型已加载")

    # 初始化今日统计
    if today_str not in state['daily_stats']:
        state['daily_stats'][today_str] = {
            'entry_count': 0, 'exit_count': 0,
            'win_count': 0, 'total_pnl': 0.0
        }

    # ===== 扫描入场信号 =====
    print(f"\n扫描入场信号...")
    entry_signals = []
    scanned = 0

    for code in stock_codes:
        scanned += 1
        if scanned % 100 == 0:
            print(f"  已扫描 {scanned}/{len(stock_codes)}, 发现信号 {len(entry_signals)}")

        # 去重：今日已通知过的跳过
        signal_key = f"{code}_{today_str}_entry"
        if signal_key in state['notified_entry']:
            continue

        # 已持仓的不重复入场
        if code in state['positions']:
            continue

        # 加载历史数据（缓存不存在则自动下载）
        hist_df = get_stock_history(code)
        if hist_df is None or len(hist_df) < 60:
            continue

        # 追加今日实时数据
        spot_row = spot_df.loc[code] if code in spot_df.index else None
        if spot_row is None:
            continue
        df = build_today_kline(hist_df, spot_row, today_str)
        df = compute_technical_indicators(df)

        # 扫描信号
        name = name_map.get(code, code)
        signal = scan_entry_for_stock(
            df, code, name, index_df, regime_df, regime_detector,
            model, ml_cols, spot_row, today_str
        )
        if signal:
            entry_signals.append(signal)
            state['notified_entry'].append(signal_key)
            state['daily_stats'][today_str]['entry_count'] += 1

            # 自动记录持仓（默认用户买入，入场价=信号当天收盘价）
            state['positions'][code] = {
                'entry_date': today_str,
                'entry_price': signal['price'],
                'pattern': signal['pattern'],
                'auto_added': True
            }

            # 推送
            print(f"  🎯 入场: {name}({code}) {signal['pattern']} "
                  f"价:{signal['price']:.2f} ML:{signal['ml_score']:.0f} (已自动加入持仓监控)")
            if notifier:
                notifier.send_entry_signal(**signal)

    print(f"入场扫描完成: 发现 {len(entry_signals)} 个信号")

    # ===== 扫描出场信号 =====
    print(f"\n扫描出场信号 (持仓 {len(state['positions'])} 只)...")
    exit_signals = []
    codes_to_remove = []

    for code, pos_info in state['positions'].items():
        signal_key = f"{code}_{today_str}_exit"
        if signal_key in state['notified_exit']:
            continue

        hist_df = get_stock_history(code)
        if hist_df is None or len(hist_df) < 60:
            continue

        spot_row = spot_df.loc[code] if code in spot_df.index else None
        if spot_row is None:
            continue
        df = build_today_kline(hist_df, spot_row, today_str)
        df = compute_technical_indicators(df)

        name = name_map.get(code, code)
        exit_sig = check_exit_for_position(df, code, pos_info, today_str)
        if exit_sig:
            exit_sig['name'] = name
            exit_signals.append(exit_sig)
            state['notified_exit'].append(signal_key)
            codes_to_remove.append(code)
            state['daily_stats'][today_str]['exit_count'] += 1
            if exit_sig['pnl_pct'] >= 0:
                state['daily_stats'][today_str]['win_count'] += 1
            state['daily_stats'][today_str]['total_pnl'] += exit_sig['pnl_pct']

            print(f"  📤 出场: {name}({code}) {exit_sig['reason']} "
                  f"收益:{exit_sig['pnl_pct']:+.2f}%")
            if notifier:
                notifier.send_exit_signal(**exit_sig)

    # 移除已出场的持仓
    for code in codes_to_remove:
        del state['positions'][code]

    print(f"出场扫描完成: 发现 {len(exit_signals)} 个信号")

    # 更新状态
    state['last_scan_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_state(state)

    print(f"\n{'='*60}")
    print(f"扫描完成 | 入场:{len(entry_signals)} 出场:{len(exit_signals)} "
          f"当前持仓:{len(state['positions'])}")
    print(f"{'='*60}")

    return {
        'entry_signals': entry_signals,
        'exit_signals': exit_signals,
        'positions': len(state['positions'])
    }

def add_position(code, entry_date, entry_price, pattern="手动"):
    """手动添加持仓（用于实际买入后记录）"""
    state = load_state()
    state['positions'][code] = {
        'entry_date': entry_date,
        'entry_price': float(entry_price),
        'pattern': pattern
    }
    save_state(state)
    print(f"已添加持仓: {code} 入场价:{entry_price} 日期:{entry_date}")

def remove_position(code):
    """移除持仓（用户未实际买入时使用）"""
    state = load_state()
    if code in state['positions']:
        del state['positions'][code]
        save_state(state)
        print(f"已移除持仓: {code}")
    else:
        print(f"未找到持仓: {code}")

def list_positions():
    """列出当前持仓"""
    state = load_state()
    if not state['positions']:
        print("当前无持仓")
        return
    print(f"当前持仓 ({len(state['positions'])} 只):")
    universe = load_stock_pool()
    name_map = dict(zip(universe['code'], universe['name']))
    for code, pos in state['positions'].items():
        name = name_map.get(code, code)
        print(f"  {name}({code}) 入场:{pos['entry_date']} "
              f"价:{pos['entry_price']:.2f} 形态:{pos.get('pattern','-')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='实时信号监控')
    parser.add_argument('--force', action='store_true', help='强制运行（非交易时间也执行）')
    parser.add_argument('--test', action='store_true', help='发送测试消息')
    parser.add_argument('--add-pos', nargs=3, metavar=('CODE', 'DATE', 'PRICE'),
                        help='手动添加持仓: 代码 日期 价格')
    parser.add_argument('--remove-pos', metavar='CODE', help='移除持仓: 股票代码')
    parser.add_argument('--list-pos', action='store_true', help='列出当前持仓')
    args = parser.parse_args()

    if args.add_pos:
        add_position(args.add_pos[0], args.add_pos[1], float(args.add_pos[2]))
    elif args.remove_pos:
        remove_position(args.remove_pos)
    elif args.list_pos:
        list_positions()
    else:
        run_monitor(force_run=args.force, send_test=args.test)
