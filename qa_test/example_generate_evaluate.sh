#!/bin/bash

# 问答对生成与评测集成系统使用示例
# 支持从多个txt文件生成问答对并进行质量评测

echo "=========================================="
echo "问答对生成与评测集成系统使用示例"
echo "=========================================="

# 基本用法：从目录中的txt文件生成问答对并评测
echo -e "\n1. 基本用法："
echo "python qa_test/generate_and_evaluate.py \\"
echo "    --input-dir ./texts \\"
echo "    --output ./outputs/best_qa_pairs.json \\"
echo "    --config qa_test/config.yaml"

# 高级用法：指定更多参数
echo -e "\n2. 高级用法（指定更多参数）："
echo "python qa_test/generate_and_evaluate.py \\"
echo "    --input-dir ./texts \\"
echo "    --output ./outputs/best_qa_pairs.json \\"
echo "    --config qa_test/config.yaml \\"
echo "    --top-k 200 \\"
echo "    --min-score 0.7 \\"
echo "    --report-dir ./reports \\"
echo "    --save-intermediate \\"
echo "    --verbose"

# 限制处理文件数量
echo -e "\n3. 限制处理文件数量（用于测试）："
echo "python qa_test/generate_and_evaluate.py \\"
echo "    --input-dir ./texts \\"
echo "    --output ./outputs/test_qa_pairs.json \\"
echo "    --config qa_test/config.yaml \\"
echo "    --max-files 3 \\"
echo "    --save-intermediate"

# 使用自定义配置
echo -e "\n4. 使用自定义知识图谱和问题生成配置："
echo "python qa_test/generate_and_evaluate.py \\"
echo "    --input-dir ./texts \\"
echo "    --output ./outputs/custom_qa_pairs.json \\"
echo "    --config qa_test/config.yaml \\"
echo "    --kg-config ./configs/kg_config.yaml \\"
echo "    --qg-config ./configs/qg_config.yaml"

# 跳过LLM评测（仅使用规则评测）
echo -e "\n5. 跳过LLM评测（加快处理速度）："
echo "python qa_test/generate_and_evaluate.py \\"
echo "    --input-dir ./texts \\"
echo "    --output ./outputs/quick_qa_pairs.json \\"
echo "    --config qa_test/config.yaml \\"
echo "    --skip-llm \\"
echo "    --no-report"

# 实际运行示例（创建测试目录和文件）
echo -e "\n=========================================="
echo "实际运行示例"
echo "=========================================="

# 创建测试目录
mkdir -p ./test_texts
mkdir -p ./test_outputs

# 创建示例txt文件
echo "创建示例txt文件..."
cat > ./test_texts/sample1.txt << 'EOF'
TCL华星光电的Mini-LED背光技术采用了先进的局部调光控制系统。该技术通过将背光源划分为数千个独立控制的区域，
每个区域都可以根据显示内容独立调整亮度。这种精细的控制方式不仅能够实现超高对比度，还能有效降低功耗。
在实际应用中，Mini-LED背光技术已经被广泛应用于高端电视、显示器和笔记本电脑等产品中。
EOF

cat > ./test_texts/sample2.txt << 'EOF'
印刷OLED技术是TCL华星光电的另一项重要创新。与传统的蒸镀工艺相比，印刷工艺具有材料利用率高、
生产成本低、适合大尺寸面板制造等优势。TCL华星光电已经建立了完整的印刷OLED产线，
并在材料开发、设备优化和工艺控制等方面取得了重要突破。
EOF

cat > ./test_texts/sample3.txt << 'EOF'
柔性显示技术是未来显示产业的重要发展方向。TCL华星光电在柔性AMOLED技术方面投入了大量研发资源，
开发出了可弯曲、可折叠的显示面板。这些柔性面板采用了特殊的衬底材料和封装技术，
能够在保证显示质量的同时实现多种形态变化，为智能手机、可穿戴设备等产品提供了新的设计可能。
EOF

echo -e "\n运行生成与评测系统..."
python qa_test/generate_and_evaluate.py \
    --input-dir ./test_texts \
    --output ./test_outputs/evaluated_qa_pairs.json \
    --config qa_test/config.yaml \
    --top-k 50 \
    --save-intermediate \
    --verbose

echo -e "\n处理完成！"
echo "生成的问答对保存在: ./test_outputs/evaluated_qa_pairs.json"
echo "所有生成的问答对保存在: ./test_outputs/evaluated_qa_pairs_all_generated.json"
echo "评测报告保存在: ./reports/"