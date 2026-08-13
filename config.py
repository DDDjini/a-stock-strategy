# -*- coding: utf-8 -*-
"""
A股K线形态突破+趋势跟踪策略 - 全局配置
所有参数集中管理，便于鲁棒性测试时批量调整
"""

# ==================== 数据参数 ====================
DATA_START_DATE = "20180101"       # 回测起始日
DATA_END_DATE   = "20260801"       # 回测结束日
ADJUST_TYPE     = "qfq"            # 前复权
MIN_LIST_DAYS   = 250              # 最少上市天数（过滤次新股）
MIN_PRICE       = 2.0              # 最低股价（过滤仙股）
MAX_PRICE       = 300.0            # 最高股价（过滤高价股流动性差）
MIN_AVG_VOLUME  = 50000            # 最低日均成交量（手），过滤流动性差

# ==================== 形态识别参数 ====================
PATTERN_LOOKBACK    = 120     # 形态识别回看窗口（交易日）
MIN_PATTERN_DAYS    = 20      # 形态最短持续天数
MAX_PATTERN_DAYS    = 100     # 形态最长持续天数
TOUCH_TOLERANCE     = 0.03    # 支撑/阻力触碰容差（3%）
NECKLINE_TOUCH_MIN  = 2       # 颈线最少触碰次数

# W底参数
W_BOTTOM_DEPTH_MIN   = 0.08   # 两底相对中间峰的最小跌幅（8%）
W_BOTTOM_SYMMETRY    = 0.15   # 两底高度差最大容差（15%）
W_BOTTOM_GAP_MIN     = 10     # 两底最小间隔天数

# 头肩底参数
HSH_HEAD_DEPTH_MIN   = 0.10   # 头部相对肩部最小深度（10%）
HSH_SHOULDER_SYMM    = 0.15   # 两肩高度差最大容差（15%）

# 收敛三角参数
TRIANGLE_TOUCH_MIN   = 4      # 最少触碰次数（上下各2次）
TRIANGLE_CONVERGE    = 0.6    # 收敛度阈值（末端宽度/起始宽度 < 0.6）

# 菱形参数
DIAMOND_EXPAND_TOUCH = 3      # 扩张段最少触碰
DIAMOND_CONTRACT_TOUCH = 3    # 收缩段最少触碰

# ==================== 突破确认参数 ====================
BREAKOUT_VOL_RATIO   = 1.0    # 突破日量能 / 前3日均量 阈值（用户要求>均值，设1.0倍）
BREAKOUT_CLOSE_BUFFER = 0.01  # 收盘价突破颈线的最小幅度（1%），防假突破
BREAKOUT_CONFIRM_DAYS = 1     # 突破确认天数（0=当日，1=次日确认）

# ==================== 趋势线持仓参数 ====================
TRENDLINE_MIN_POINTS  = 3     # 趋势线最少连接点数
TRENDLINE_UPDATE_DAYS = 5     # 每N天更新一次趋势线
TRENDLINE_BREAK_BUFFER = 0.01 # 跌破趋势线缓冲（1%）
INITIAL_STOP_ATR      = 2.0   # 初始止损 = 入场价 - ATR * 系数（备用硬止损）

# ==================== 交易成本参数 ====================
COMMISSION_RATE   = 0.0003    # 佣金万三（双边）
STAMP_TAX_RATE    = 0.0005    # 印花税千一（仅卖出）
SLIPPAGE_RATE     = 0.001     # 滑点千一
MIN_COMMISSION    = 5.0       # 最低佣金5元

# ==================== 仓位管理 ====================
MAX_POSITIONS     = 10        # 最大同时持仓数
SINGLE_POSITION   = 0.10      # 单只股票最大仓位10%
INITIAL_CAPITAL   = 1000000   # 初始资金100万

# ==================== 市场过滤 ====================
INDEX_FILTER        = True    # 是否启用大盘趋势过滤
INDEX_CODE          = "sh000001"  # 上证指数
INDEX_MA_PERIOD     = 60      # 大盘60日均线过滤
SECTOR_MOMENTUM     = True    # 是否启用板块动量过滤
SECTOR_TOP_N        = 10      # 只选动量前N的板块

# ==================== 创新增强模块 ====================
USE_PATTERN_QUALITY_SCORE = True   # 形态质量评分（0-100），低于阈值不交易
PATTERN_QUALITY_THRESHOLD = 50     # 质量评分阈值（降低到50）
USE_FALSE_BREAKOUT_FILTER = True   # 假突破过滤（突破后3日内不跌回颈线2%以下）
USE_VOLUME_PROFILE       = True    # 量能分布确认
USE_MULTI_TIMEFRAME      = False   # 多周期共振（先关闭，后续作为增强选项）
USE_ATR_SIZING           = True    # ATR波动率仓位调整
