#!/bin/bash

# 翻译进度实时监控脚本
# 用于监控翻译进度和性能

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <输入目录> <输出目录> [刷新间隔(秒)]"
    echo "示例: $0 ./english_docs ./chinese_translations 5"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
INTERVAL="${3:-5}"  # 默认5秒刷新

# 检查目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 开始时间
START_TIME=$(date +%s)

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 监控循环
while true; do
    clear
    
    # 获取当前时间
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))
    
    # 统计文件数
    TOTAL_FILES=$(find "$INPUT_DIR" -name "*.txt" -type f | wc -l)
    COMPLETED_FILES=$(find "$OUTPUT_DIR" -name "*_中文.txt" -type f 2>/dev/null | wc -l)
    REMAINING=$((TOTAL_FILES - COMPLETED_FILES))
    
    # 计算进度百分比
    if [ $TOTAL_FILES -gt 0 ]; then
        PROGRESS=$((COMPLETED_FILES * 100 / TOTAL_FILES))
    else
        PROGRESS=0
    fi
    
    # 显示标题
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║              📊 DeepSeek 翻译进度监控                      ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # 显示统计信息
    echo -e "📁 输入目录: $INPUT_DIR"
    echo -e "📂 输出目录: $OUTPUT_DIR"
    echo -e "⏱️  运行时间: ${ELAPSED_MIN}分${ELAPSED_SEC}秒"
    echo ""
    
    # 进度条
    BAR_LENGTH=50
    FILLED_LENGTH=$((PROGRESS * BAR_LENGTH / 100))
    EMPTY_LENGTH=$((BAR_LENGTH - FILLED_LENGTH))
    
    echo -n "进度: ["
    printf "%${FILLED_LENGTH}s" | tr ' ' '█'
    printf "%${EMPTY_LENGTH}s" | tr ' ' '░'
    echo "] $PROGRESS%"
    echo ""
    
    # 文件统计
    echo -e "📊 文件统计:"
    echo -e "  总文件数: ${YELLOW}$TOTAL_FILES${NC}"
    echo -e "  已完成:   ${GREEN}$COMPLETED_FILES${NC}"
    echo -e "  剩余:     ${RED}$REMAINING${NC}"
    echo ""
    
    # 速度估算
    if [ $COMPLETED_FILES -gt 0 ] && [ $ELAPSED -gt 0 ]; then
        SPEED=$(echo "scale=2; $COMPLETED_FILES * 60 / $ELAPSED" | bc)
        REMAINING_TIME=$((REMAINING * ELAPSED / COMPLETED_FILES))
        REMAINING_MIN=$((REMAINING_TIME / 60))
        REMAINING_SEC=$((REMAINING_TIME % 60))
        
        echo -e "⚡ 性能指标:"
        echo -e "  平均速度: ${GREEN}$SPEED${NC} 文件/分钟"
        echo -e "  预计剩余: ${YELLOW}${REMAINING_MIN}分${REMAINING_SEC}秒${NC}"
        echo ""
    fi
    
    # 最近完成的文件
    echo -e "📝 最近完成的文件:"
    if [ -d "$OUTPUT_DIR" ]; then
        find "$OUTPUT_DIR" -name "*_中文.txt" -type f -printf "%T@ %p\n" 2>/dev/null | \
        sort -nr | head -5 | while read timestamp filepath; do
            filename=$(basename "$filepath")
            echo -e "  ${GREEN}✓${NC} $filename"
        done
    else
        echo -e "  ${YELLOW}等待开始...${NC}"
    fi
    
    echo ""
    
    # 系统资源
    echo -e "💻 系统资源:"
    
    # CPU使用率
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    echo -e "  CPU: ${CPU_USAGE}%"
    
    # 内存使用
    MEM_INFO=$(free -h | grep "^Mem:")
    MEM_USED=$(echo $MEM_INFO | awk '{print $3}')
    MEM_TOTAL=$(echo $MEM_INFO | awk '{print $2}')
    echo -e "  内存: $MEM_USED / $MEM_TOTAL"
    
    # GPU使用（如果有）
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1)
        if [ ! -z "$GPU_INFO" ]; then
            GPU_UTIL=$(echo $GPU_INFO | cut -d',' -f1)
            GPU_MEM_USED=$(echo $GPU_INFO | cut -d',' -f2)
            GPU_MEM_TOTAL=$(echo $GPU_INFO | cut -d',' -f3)
            echo -e "  GPU: ${GPU_UTIL}% | 显存: ${GPU_MEM_USED}MB / ${GPU_MEM_TOTAL}MB"
        fi
    fi
    
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "按 Ctrl+C 退出监控 | 刷新间隔: ${INTERVAL}秒"
    
    # 完成提示
    if [ $REMAINING -eq 0 ] && [ $TOTAL_FILES -gt 0 ]; then
        echo ""
        echo -e "${GREEN}🎉 翻译已完成！${NC}"
        echo -e "总用时: ${ELAPSED_MIN}分${ELAPSED_SEC}秒"
        break
    fi
    
    sleep $INTERVAL
done