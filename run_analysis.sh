#!/bin/bash
# 股票筛选分析脚本
# 用途：定时运行港股筛选分析（使用多AI提供商）

# 项目路径
PROJECT_DIR="/Users/addison/Dev/yfinance"
LOG_DIR="$PROJECT_DIR/logs"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 日志文件（带日期）
LOG_FILE="$LOG_DIR/analysis_$(date +%Y-%m-%d).log"

# 进入项目目录
cd "$PROJECT_DIR" || exit 1

# 激活虚拟环境
source venv/bin/activate

# 记录开始时间
echo "========================================" >> "$LOG_FILE"
echo "开始分析: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# 整合所有json
python3 merge_stocks.py

# 执行分析（港股 + 多AI提供商 + 多模型投票）
python3 main.py --market HK --stock-list /Users/addison/Dev/stock.json --model all --provider iflow,nvidia,opencode >> "$LOG_FILE" 2>&1

# 记录结束时间
echo "========================================" >> "$LOG_FILE"
echo "分析完成: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 退出虚拟环境
deactivate
