#!/bin/bash

# 文档感知问答对评测示例脚本

echo "=== 文档感知问答对评测示例 ==="
echo

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 基本的文档感知评测
echo -e "${GREEN}1. 运行基本的文档感知评测${NC}"
echo "评测问答对并检查其从文档中提取的合理性..."
echo

python evaluate_qa_with_documents.py \
    --input sample_qa_pairs_with_docs.json \
    --documents sample_documents.json \
    --output results/document_aware_results.json \
    --min-reasonableness 0.6 \
    --export-issues

echo
echo -e "${YELLOW}结果已保存到: results/document_aware_results.json${NC}"
echo

# 2. 结合传统评测和文档感知评测
echo -e "${GREEN}2. 运行综合评测（传统质量 + 文档合理性）${NC}"
echo "同时评估问答对的质量和提取合理性..."
echo

python evaluate_qa_with_documents.py \
    --input sample_qa_pairs_with_docs.json \
    --documents sample_documents.json \
    --output results/combined_evaluation_results.json \
    --combine-evaluation \
    --report-dir reports/combined

echo
echo -e "${YELLOW}综合评测结果已保存${NC}"
echo

# 3. 多文档评测
echo -e "${GREEN}3. 使用多个文档源进行评测${NC}"
echo "可以同时指定多个文档文件..."
echo

# 创建额外的文档文件示例
cat > sample_documents_extra.json << EOF
[
  {
    "id": "doc_006",
    "content": "云计算是一种通过互联网提供计算资源的服务模式。它包括三种主要服务模型：基础设施即服务(IaaS)、平台即服务(PaaS)和软件即服务(SaaS)。云计算的优势包括弹性扩展、按需付费、高可用性和全球覆盖。",
    "metadata": {
      "source": "云计算指南",
      "category": "云计算"
    }
  }
]
EOF

python evaluate_qa_with_documents.py \
    --input sample_qa_pairs_with_docs.json \
    --documents sample_documents.json sample_documents_extra.json \
    --output results/multi_doc_results.json \
    --min-reasonableness 0.7

echo
echo -e "${YELLOW}多文档评测完成${NC}"
echo

# 4. 查看评测报告
echo -e "${GREEN}4. 查看生成的评测报告${NC}"
echo

if [ -f "reports/extraction_evaluation_report.md" ]; then
    echo "评测报告预览："
    head -n 30 reports/extraction_evaluation_report.md
    echo "..."
    echo
    echo -e "${YELLOW}完整报告: reports/extraction_evaluation_report.md${NC}"
else
    echo "报告文件未找到"
fi

echo
echo "=== 评测示例完成 ==="
echo
echo "提示："
echo "- 使用 --min-reasonableness 参数调整合理性阈值"
echo "- 使用 --export-issues 导出有问题的问答对"
echo "- 使用 --combine-evaluation 同时进行质量和合理性评测"
echo "- 查看 results/ 目录获取详细结果"
echo "- 查看 reports/ 目录获取评测报告"