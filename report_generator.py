# -*- coding: utf-8 -*-
"""
策略报告生成器
生成完整的策略研究报告（Markdown格式）
"""
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from datetime import datetime


def generate_report(metrics, robustness_results=None, enhanced_metrics=None,
                    config_snapshot=None, output_path="results/strategy_report.md"):
    """生成策略报告"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report = []
    report.append("# A股K线形态突破+趋势跟踪策略研究报告\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    report.append("---\n")

    # ==================== 一、策略概述 ====================
    report.append("## 一、策略概述\n")
    report.append("### 1.1 策略核心理念\n")
    report.append("本策略基于经典技术分析中的**形态突破理论**，结合**趋势跟踪**思想，")
    report.append("在A股市场中识别底部反转形态（W底、头肩底、矩形底）和中继整理形态")
    report.append("（箱体、收敛三角、菱形），当价格放量突破颈线时入场，随后通过动态")
    report.append("上升趋势线跟踪持仓，跌破趋势线时出场。\n")
    report.append("策略的核心假设是：**经过充分整理后的放量突破，往往预示着趋势的启动；")
    report.append("而上升趋势线能够有效捕捉趋势的延续，同时在趋势反转时及时离场。**\n")

    report.append("### 1.2 策略流程\n")
    report.append("```\n")
    report.append("1. 数据获取：A股日线（前复权）+ 上证指数 + 技术指标\n")
    report.append("2. 形态识别：阻力位聚类 → 6种形态检测 → 质量评分\n")
    report.append("3. 突破确认：收盘价突破颈线+1%缓冲 + 量能>前3日均量\n")
    report.append("4. 多重过滤：大盘趋势(MA60) + 假突破过滤 + 形态质量阈值\n")
    report.append("5. 入场：突破确认后次日开盘买入\n")
    report.append("6. 持仓管理：动态趋势线跟踪 + ATR硬止损保护\n")
    report.append("7. 出场：跌破趋势线 / 移动止盈 / 硬止损\n")
    report.append("```\n")

    report.append("### 1.3 识别的6种形态\n")
    report.append("| 形态类型 | 类别 | 颈线定义 | 典型特征 |\n")
    report.append("|---------|------|---------|----------|\n")
    report.append("| W底 | 底部反转 | 两底中间高点 | 双底对称，右底缩量 |\n")
    report.append("| 头肩底 | 底部反转 | 两肩间高点连线 | 头部最低，右肩缩量 |\n")
    report.append("| 矩形底 | 底部反转 | 箱体上沿 | 水平支撑阻力，多次触碰 |\n")
    report.append("| 箱体 | 中继整理 | 箱体上沿 | 横盘整理，振幅收窄 |\n")
    report.append("| 收敛三角 | 中继整理 | 下降趋势线 | 高点降低+低点抬高 |\n")
    report.append("| 菱形 | 中继整理 | 收缩段上沿 | 先扩张后收缩 |\n")

    # ==================== 二、回测结果 ====================
    report.append("\n## 二、回测结果\n")
    report.append(f"**回测区间**: {config_snapshot.get('start_date', '2018-01-01')} ~ "
                  f"{config_snapshot.get('end_date', '2026-08-01')}\n")
    report.append(f"**股票池**: 沪深300成分股（过滤ST、次新股、低流动性）\n")
    report.append(f"**初始资金**: {config_snapshot.get('initial_capital', 1000000):,.0f} 元\n")
    report.append(f"**最大持仓**: {config_snapshot.get('max_positions', 10)} 只，单只≤10%\n")
    report.append(f"**交易成本**: 佣金万三 + 印花税千一(卖) + 滑点千一\n\n")

    report.append("### 2.1 核心绩效指标\n")
    report.append("| 指标 | 数值 | 评价 |\n")
    report.append("|------|------|------|\n")

    wr = metrics["win_rate"]
    wr_note = "优秀" if wr >= 0.6 else ("良好" if wr >= 0.5 else "一般")
    report.append(f"| 胜率 | {wr:.2%} | {wr_note} |\n")

    pf = metrics["profit_factor"]
    pf_note = "优秀" if pf >= 2 else ("良好" if pf >= 1.5 else "一般")
    report.append(f"| 盈亏比 | {pf:.2f} | {pf_note} |\n")

    report.append(f"| 总交易次数 | {metrics['total_trades']} | 统计显著性{'足够' if metrics['total_trades'] >= 100 else '有限'} |\n")
    report.append(f"| 平均盈利 | {metrics['avg_win_pct']:.2%} | - |\n")
    report.append(f"| 平均亏损 | {metrics['avg_loss_pct']:.2%} | - |\n")
    report.append(f"| 总收益率 | {metrics['total_return']:.2%} | - |\n")
    report.append(f"| 年化收益率 | {metrics['annual_return']:.2%} | - |\n")

    sharpe = metrics["sharpe"]
    sharpe_note = "优秀" if sharpe >= 1.5 else ("良好" if sharpe >= 1 else "一般")
    report.append(f"| 夏普比率 | {sharpe:.2f} | {sharpe_note} |\n")
    report.append(f"| Sortino | {metrics['sortino']:.2f} | - |\n")
    report.append(f"| Calmar | {metrics['calmar']:.2f} | - |\n")

    mdd = metrics["max_drawdown"]
    mdd_note = "优秀" if abs(mdd) < 0.15 else ("良好" if abs(mdd) < 0.25 else "一般")
    report.append(f"| 最大回撤 | {mdd:.2%} | {mdd_note} |\n")
    report.append(f"| 平均持仓天数 | {metrics['avg_hold_days']:.1f} | 短线波段 |\n")

    # ==================== 三、各形态表现 ====================
    report.append("\n### 2.2 各形态表现分解\n")
    if metrics.get("pattern_stats"):
        report.append("| 形态 | 交易次数 | 胜率 | 平均收益 | 平均持仓 |\n")
        report.append("|------|---------|------|---------|----------|\n")
        for pat, stat in metrics["pattern_stats"].items():
            report.append(f"| {pat} | {stat['count']} | {stat['win_rate']:.1%} | "
                          f"{stat['avg_return']:.2%} | {stat['avg_hold_days']:.0f}天 |\n")

    # 出场原因
    report.append("\n### 2.3 出场原因分布\n")
    if metrics.get("exit_stats"):
        total = sum(metrics["exit_stats"].values())
        for reason, cnt in metrics["exit_stats"].items():
            report.append(f"- **{reason}**: {cnt}次 ({cnt/total:.1%})\n")

    # ==================== 三、策略逻辑深度解析 ====================
    report.append("\n## 三、策略逻辑深度解析\n")

    report.append("### 3.1 形态识别算法\n")
    report.append("采用**摆动点检测+结构匹配**的方法：\n")
    report.append("1. 使用`scipy.signal.argrelextrema`识别局部极值点（order=5）\n")
    report.append("2. 合并相邻同类型极值，保留更极端者\n")
    report.append("3. 对每种形态定义结构化匹配条件（深度、对称性、触碰次数等）\n")
    report.append("4. 计算形态质量评分（0-100），过滤低质量形态\n\n")

    report.append("### 3.2 突破确认机制\n")
    report.append("- **价格突破**: 收盘价 > 颈线 × (1+1%)，确保有效突破而非影线触碰\n")
    report.append("- **量能确认**: 当日成交量 > 前3日均量，验证资金参与度\n")
    report.append("- **前收约束**: 前一日收盘价 ≤ 颈线，确保是从下方突破\n")
    report.append("- **假突破过滤**: 突破后3日内收盘价不低于颈线的98%\n\n")

    report.append("### 3.3 动态趋势线持仓\n")
    report.append("入场后持续跟踪价格走势：\n")
    report.append("1. 每5天重新识别入场后的摆动低点\n")
    report.append("2. 用最近2-3个有效低点线性回归，得到上升趋势线\n")
    report.append("3. 移动止盈 = max(趋势线值×0.99, 历史最高收盘价×0.92)\n")
    report.append("4. 收盘价跌破趋势线 → 出场；趋势线斜率≤0时改用高点回撤法\n")
    report.append("5. 硬止损保护: 入场价 - 2×ATR14（极端行情兜底）\n\n")

    report.append("### 3.4 创新增强模块\n")
    report.append("| 增强模块 | 作用 | 预期效果 |\n")
    report.append("|---------|------|----------|\n")
    report.append("| 形态质量评分 | 过滤模糊形态 | 提升胜率，减少假信号 |\n")
    report.append("| 大盘趋势过滤 | 只在上证指数>MA60时交易 | 规避系统性风险 |\n")
    report.append("| 假突破过滤 | 突破后3日不深跌 | 减少诱多陷阱 |\n")
    report.append("| ATR仓位管理 | 波动率调整仓位 | 控制单笔风险 |\n")
    report.append("| RSI+MACD过滤(增强版) | 动量确认 | 进一步提升胜率 |\n")
    report.append("| 分批止盈(增强版) | +5%卖半仓 | 锁定利润，提升胜率 |\n")

    # ==================== 四、鲁棒性测试 ====================
    if robustness_results:
        report.append("\n## 四、鲁棒性测试\n")

        # 参数敏感性
        if robustness_results.get("sensitivity") is not None and not robustness_results["sensitivity"].empty:
            report.append("### 4.1 参数敏感性分析\n")
            report.append("测试不同量比阈值和质量阈值组合的表现（50只股票子集）：\n\n")
            sens = robustness_results["sensitivity"]
            report.append("| 量比阈值 | 质量阈值 | 交易次数 | 胜率 | 总收益 | 夏普 | 最大回撤 |\n")
            report.append("|---------|---------|---------|------|--------|------|----------|\n")
            for _, row in sens.iterrows():
                report.append(f"| {row['vol_ratio']:.1f} | {row['quality_thresh']:.0f} | "
                              f"{row['trades']:.0f} | {row['win_rate']:.1%} | "
                              f"{row['total_return']:.1%} | {row['sharpe']:.2f} | "
                              f"{row['max_drawdown']:.1%} |\n")

        # 分周期
        if robustness_results.get("period") is not None and not robustness_results["period"].empty:
            report.append("\n### 4.2 不同市场周期表现\n")
            period_df = robustness_results["period"]
            # 动态获取可用列
            cols = period_df.columns.tolist()
            headers = ["市场周期", "交易次数", "胜率"]
            keys = ["period", "trades", "win_rate"]
            if "avg_return" in cols:
                headers.append("平均收益")
                keys.append("avg_return")
            if "total_return" in cols:
                headers.append("总收益")
                keys.append("total_return")
            if "max_drawdown" in cols:
                headers.append("最大回撤")
                keys.append("max_drawdown")
            if "sharpe" in cols:
                headers.append("夏普")
                keys.append("sharpe")
            report.append("| " + " | ".join(headers) + " |\n")
            report.append("|" + "|".join(["------"] * len(headers)) + "|\n")
            for _, row in period_df.iterrows():
                vals = []
                for k in keys:
                    v = row[k]
                    if k in ["win_rate", "avg_return", "total_return", "max_drawdown"]:
                        vals.append(f"{v:.1%}")
                    elif k == "trades":
                        vals.append(f"{v:.0f}")
                    elif k == "sharpe":
                        vals.append(f"{v:.2f}")
                    else:
                        vals.append(str(v))
                report.append("| " + " | ".join(vals) + " |\n")

        # 分年度
        if robustness_results.get("yearly") is not None and not robustness_results["yearly"].empty:
            report.append("\n### 4.3 分年度交易统计\n")
            report.append("| 年份 | 交易次数 | 胜率 | 平均收益 |\n")
            report.append("|------|---------|------|----------|\n")
            for _, row in robustness_results["yearly"].iterrows():
                report.append(f"| {int(row['year'])} | {int(row['trades'])} | "
                              f"{row['win_rate']:.1%} | {row['avg_return']:.2%} |\n")

        # 滚动窗口
        if robustness_results.get("rolling") is not None and not robustness_results["rolling"].empty:
            report.append("\n### 4.4 滚动窗口回测（2年窗口，1年步长）\n")
            report.append("| 窗口 | 交易次数 | 胜率 | 总收益 | 夏普 |\n")
            report.append("|------|---------|------|--------|------|\n")
            for _, row in robustness_results["rolling"].iterrows():
                report.append(f"| {row['window']} | {row['trades']:.0f} | "
                              f"{row['win_rate']:.1%} | {row['total_return']:.1%} | "
                              f"{row['sharpe']:.2f} |\n")

    # ==================== 五、增强版对比 ====================
    if enhanced_metrics:
        report.append("\n## 五、高胜率增强版对比\n")
        report.append("增强版在基础版上增加：RSI(45-80)过滤 + MACD柱状图>0过滤 + "
                      "收盘价>MA20过滤 + +5%分批止盈\n\n")
        report.append("| 指标 | 基础版 | 增强版 | 变化 |\n")
        report.append("|------|--------|--------|------|\n")
        for key, label in [("total_trades", "交易次数"), ("win_rate", "胜率"),
                           ("total_return", "总收益"), ("annual_return", "年化收益"),
                           ("sharpe", "夏普"), ("max_drawdown", "最大回撤"),
                           ("profit_factor", "盈亏比")]:
            base_val = metrics[key]
            enh_val = enhanced_metrics[key]
            if key in ["win_rate", "total_return", "annual_return", "max_drawdown"]:
                report.append(f"| {label} | {base_val:.2%} | {enh_val:.2%} | "
                              f"{enh_val-base_val:+.2%} |\n")
            else:
                report.append(f"| {label} | {base_val:.2f} | {enh_val:.2f} | "
                              f"{enh_val-base_val:+.2f} |\n")

    # ==================== 六、改进方向 ====================
    report.append("\n## 六、改进方向与潜在优化\n")

    report.append("### 6.1 短期可落地优化\n")
    report.append("1. **形态参数自适应**: 根据个股波动率动态调整形态识别参数（如ATR归一化）\n")
    report.append("2. **板块轮动过滤**: 只交易近20日涨幅前1/3的板块中的突破信号\n")
    report.append("3. **北向资金确认**: 突破日北向资金净流入为正（需额外数据源）\n")
    report.append("4. **龙虎榜验证**: 突破后3日内出现机构买入龙虎榜（增强信号可信度）\n")
    report.append("5. **分时级别确认**: 日线突破后，用60分钟级别二次确认入场点\n")

    report.append("\n### 6.2 中期架构升级\n")
    report.append("1. **机器学习形态评分**: 用标注好的形态数据训练XGBoost/LightGBM分类器，")
    report.append("   替代人工规则的质量评分，预期可提升胜率5-10个百分点\n")
    report.append("2. **多周期共振系统**: 日线突破 + 周线趋势 + 月线位置的三维过滤框架\n")
    report.append("3. **市场状态识别**: 用HMM或聚类算法识别牛市/熊市/震荡市，")
    report.append("   不同状态下使用不同参数组合\n")
    report.append("4. **组合优化**: 用Black-Litterman或风险平价模型分配仓位，")
    report.append("   替代等权+ATR的简单方案\n")

    report.append("\n### 6.3 长期研究方向\n")
    report.append("1. **NLP事件驱动融合**: 将财报、公告、新闻情感分析与形态突破结合，")
    report.append("   区分\"有基本面支撑的突破\"和\"纯技术面突破\"\n")
    report.append("2. **跨市场联动**: A股突破 + 港股通资金 + 美股对应板块的联动确认\n")
    report.append("3. **强化学习出场**: 用DQN/PPO优化趋势线出场时机，")
    report.append("   替代固定规则的趋势线跟踪\n")
    report.append("4. **高频数据验证**: 用Tick/分钟级数据验证突破的真实性，")
    report.append("   过滤主力对倒造成的假突破\n")

    # ==================== 七、风险提示 ====================
    report.append("\n## 七、风险提示与局限性\n")
    report.append("1. **过拟合风险**: 形态参数和过滤阈值基于历史数据优化，未来可能失效\n")
    report.append("2. **幸存者偏差**: 回测使用当前沪深300成分股，未包含已退市/调出股票\n")
    report.append("3. **流动性假设**: 假设可按开盘价成交，极端行情下可能存在滑点超预期\n")
    report.append("4. **涨跌停限制**: 涨停无法买入、跌停无法卖出，回测中已做近似处理\n")
    report.append("5. **形态主观性**: 技术形态识别本身具有主观性，不同参数可能得出不同结论\n")
    report.append("6. **市场环境变化**: 注册制改革、量化交易占比提升等可能改变形态突破的有效性\n")
    report.append("7. **80%胜率目标说明**: 纯形态突破策略的理论胜率上限约为55-65%，")
    report.append("   要达到80%+胜率需要结合分批止盈、严格过滤等手段，")
    report.append("   但会显著降低交易频率和总收益，存在胜率与收益的权衡\n")

    # ==================== 八、使用方法 ====================
    report.append("\n## 八、策略使用方法\n")
    report.append("### 8.1 实盘操作流程\n")
    report.append("1. **每日盘后**: 运行信号扫描脚本，获取次日候选买入列表\n")
    report.append("2. **盘前准备**: 对候选股设置提醒价格（颈线×1.01）和预估仓位\n")
    report.append("3. **盘中执行**: 价格触发且量能配合时，分批买入（建议分2-3笔）\n")
    report.append("4. **持仓监控**: 每日更新趋势线和移动止盈位\n")
    report.append("5. **出场执行**: 跌破趋势线时次日开盘卖出（避免盘中恐慌）\n\n")

    report.append("### 8.2 关键参数参考\n")
    report.append(f"- 量能阈值: >前3日均量（当前{config_snapshot.get('vol_ratio', 1.0)}倍）\n")
    report.append(f"- 形态质量阈值: ≥{config_snapshot.get('quality_threshold', 50)}分\n")
    report.append(f"- 突破缓冲: {config_snapshot.get('breakout_buffer', 0.01):.0%}\n")
    report.append(f"- 初始止损: 入场价 - 2×ATR14\n")
    report.append(f"- 趋势线更新: 每5个交易日\n")
    report.append(f"- 单只仓位: ≤10%（ATR调整）\n")
    report.append(f"- 最大持仓: {config_snapshot.get('max_positions', 10)}只\n")

    report.append("\n### 8.3 代码结构\n")
    report.append("```\n")
    report.append("config.py              # 全局参数配置\n")
    report.append("data_fetcher.py        # 多源数据获取（东方财富/新浪）\n")
    report.append("pattern_recognizer.py  # 6种形态识别引擎\n")
    report.append("strategy.py            # 突破确认+趋势线持仓\n")
    report.append("enhanced_strategy.py   # 高胜率增强版（多因子+分批止盈）\n")
    report.append("backtest_engine.py     # 回测引擎（含交易成本）\n")
    report.append("robustness_test.py     # 鲁棒性测试\n")
    report.append("visualization.py       # 图表生成\n")
    report.append("run_backtest.py        # 主运行脚本\n")
    report.append("```\n")

    report.append("\n---\n")
    report.append("*本报告由量化策略系统自动生成，仅供研究参考，不构成投资建议。*\n")
    report.append("*历史回测结果不代表未来表现，投资有风险，入市需谨慎。*\n")

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(report))

    print(f"报告已生成: {output_path}")
    return output_path
