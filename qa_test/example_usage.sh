#!/bin/bash
# 问答对质量评测系统使用示例

echo "问答对质量评测系统使用示例"
echo "=========================="
echo ""

# 1. 安装依赖
echo "1. 安装依赖包..."
echo "pip install -r requirements.txt"
echo ""

# 2. 基本使用 - 评测示例数据
echo "2. 基本使用 - 评测示例数据并选择Top 5"
echo "python evaluate_qa.py --input sample_qa_pairs.json --output best_qa_pairs.json --top-k 5"
echo ""

# 3. 快速测试模式（跳过LLM评测）
echo "3. 快速测试模式（跳过LLM评测）"
echo "python evaluate_qa.py --input sample_qa_pairs.json --output test_output.json --skip-llm --sample 5"
echo ""

# 4. 使用自定义配置
echo "4. 使用自定义配置文件"
echo "python evaluate_qa.py --input your_data.json --output filtered_data.json --config custom_config.yaml"
echo ""

# 5. 设置最低分数阈值
echo "5. 只保留分数高于0.8的问答对"
echo "python evaluate_qa.py --input sample_qa_pairs.json --output high_quality.json --min-score 0.8"
echo ""

# 6. 详细模式
echo "6. 运行详细模式查看更多信息"
echo "python evaluate_qa.py --input sample_qa_pairs.json --output output.json --verbose"
echo ""

# 7. 不生成报告
echo "7. 只评测不生成报告"
echo "python evaluate_qa.py --input sample_qa_pairs.json --output output.json --no-report"
echo ""

echo "提示："
echo "- 首次运行需要下载语义模型（约500MB）"
echo "- 使用LLM评测需要配置API密钥（在config.yaml中或设置环境变量）"
echo "- 可以通过 --skip-llm 跳过LLM评测进行快速测试"