#!/bin/bash

# 基于文档的问答对评测示例脚本

echo "======================================"
echo "基于文档的问答对评测示例"
echo "======================================"

# 设置路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# 检查是否存在示例文档
DOCS_DIR="${WORKSPACE_DIR}/websailor_domain/input_texts"
if [ ! -d "$DOCS_DIR" ]; then
    echo "错误: 文档目录不存在: $DOCS_DIR"
    echo "请确保 websailor_domain/input_texts 目录存在并包含文档文件"
    exit 1
fi

# 检查是否存在问答对文件
QA_FILE="${WORKSPACE_DIR}/websailor_domain_dataset/qa_pairs.json"
if [ ! -f "$QA_FILE" ]; then
    # 如果不存在，使用示例文件
    QA_FILE="${SCRIPT_DIR}/sample_qa_pairs.json"
    if [ ! -f "$QA_FILE" ]; then
        echo "错误: 找不到问答对文件"
        exit 1
    fi
fi

# 创建输出目录
OUTPUT_DIR="${SCRIPT_DIR}/evaluation_results"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "使用的文档目录: $DOCS_DIR"
echo "输入问答对文件: $QA_FILE"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 方法1: 使用独立的文档评测器
echo "1. 运行独立的文档评测器..."
python "${SCRIPT_DIR}/run_document_evaluation.py" \
    --documents "$DOCS_DIR" \
    --input "$QA_FILE" \
    --output "${OUTPUT_DIR}/doc_evaluated_qa_pairs.json" \
    --top-k 50 \
    --verbose

echo ""
echo "======================================"
echo ""

# 方法2: 使用集成的评测器（结合传统评测和文档评测）
echo "2. 运行集成评测器（传统 + 文档）..."
cd "$SCRIPT_DIR"
python evaluate_qa_with_docs.py \
    --input "$QA_FILE" \
    --output "${OUTPUT_DIR}/integrated_evaluated_qa_pairs.json" \
    --documents "$DOCS_DIR" \
    --doc-weight 0.6 \
    --top-k 50 \
    --report-dir "${OUTPUT_DIR}/reports"

echo ""
echo "======================================"
echo "评测完成！"
echo "======================================"
echo ""
echo "结果文件:"
echo "- 文档评测结果: ${OUTPUT_DIR}/doc_evaluated_qa_pairs.json"
echo "- 文档评测报告: ${OUTPUT_DIR}/doc_evaluated_qa_pairs_report.md"
echo "- 集成评测结果: ${OUTPUT_DIR}/integrated_evaluated_qa_pairs.json"
echo "- 集成评测报告: ${OUTPUT_DIR}/reports/"
echo ""
echo "你可以查看这些文件了解评测结果的详细信息"