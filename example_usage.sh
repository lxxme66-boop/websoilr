#!/bin/bash

# DeepSeek 翻译工具使用示例
# 专门处理长文档，确保完整翻译

echo "📚 DeepSeek 长文档翻译示例"
echo "=========================="
echo ""
echo "此示例演示如何确保长文档被完整翻译"
echo ""

# 创建示例目录
EXAMPLE_DIR="example_translation"
INPUT_DIR="$EXAMPLE_DIR/input"
OUTPUT_DIR="$EXAMPLE_DIR/output"

echo "1. 准备示例环境..."
mkdir -p "$INPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# 创建一个示例长文档
cat > "$INPUT_DIR/long_document.txt" << 'EOF'
This is a long document example for translation testing.

Chapter 1: Introduction
Machine learning has revolutionized many fields in recent years. From computer vision to natural language processing, the applications are endless. This document explores various aspects of machine learning and its practical applications.

The field of artificial intelligence has seen tremendous growth, particularly in deep learning. Neural networks have become increasingly sophisticated, enabling breakthroughs in areas previously thought impossible.

Chapter 2: Technical Details
Deep learning models consist of multiple layers of artificial neurons. These layers process information in a hierarchical manner, extracting features at different levels of abstraction. The training process involves adjusting weights through backpropagation.

Modern architectures like transformers have revolutionized NLP tasks. The attention mechanism allows models to focus on relevant parts of the input, leading to better understanding of context and relationships.

Chapter 3: Applications
The applications of machine learning are vast and varied:
1. Healthcare: Disease diagnosis, drug discovery, personalized medicine
2. Finance: Fraud detection, algorithmic trading, risk assessment
3. Transportation: Autonomous vehicles, traffic optimization
4. Entertainment: Recommendation systems, content generation

Chapter 4: Future Directions
As we look to the future, several trends emerge:
- Increased model efficiency and smaller footprints
- Better interpretability and explainability
- Ethical AI and bias mitigation
- Edge computing and distributed learning

The journey of AI is just beginning, and the possibilities are limitless.
EOF

echo "✅ 示例文档已创建"
echo ""

echo "2. 预处理检查..."
python batch_translate_helper.py preprocess -i "$INPUT_DIR"
echo ""

echo "3. 运行翻译（使用小片段确保完整）"
echo "提示：对于长文档，建议："
echo "  - 选择模型：2 (7B) 或 3 (14B)"
echo "  - 选择分片：1 (小片段 800字符)"
echo ""
echo "运行命令："
echo "python translate_deepseek_improved.py"
echo ""
echo "输入参数示例："
echo "  模型选择: 2"
echo "  输入目录: $INPUT_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  分片大小: 1"
echo ""

# 创建自动输入文件
cat > "auto_input.txt" << EOF
2
$INPUT_DIR
$OUTPUT_DIR
1
EOF

echo "4. 可以使用以下命令自动运行："
echo "python translate_deepseek_improved.py < auto_input.txt"
echo ""

echo "5. 翻译完成后，检查质量："
echo "python check_translation.py $INPUT_DIR $OUTPUT_DIR"
echo ""

echo "6. 监控翻译进度（在另一个终端）："
echo "./monitor_progress.sh $INPUT_DIR $OUTPUT_DIR"
echo ""

echo "💡 关键要点："
echo "- 使用小片段(800字符)处理长文档"
echo "- 检查翻译覆盖率，确保>80%"
echo "- 如果覆盖率低，减小分片大小重试"
echo "- 使用监控工具实时查看进度"
echo ""

echo "📝 完整命令序列："
echo "# 步骤1: 预处理"
echo "python batch_translate_helper.py preprocess -i your_input_folder"
echo ""
echo "# 步骤2: 翻译（选择小片段）"
echo "python translate_deepseek_improved.py"
echo ""
echo "# 步骤3: 检查质量"
echo "python check_translation.py your_input_folder your_output_folder"
echo ""
echo "# 步骤4: 如果需要重试低覆盖率文件"
echo "# retry_files.txt 会自动生成"
echo ""

echo "示例准备完成！现在可以开始翻译了。"