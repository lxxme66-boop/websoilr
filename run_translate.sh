#!/bin/bash

# DeepSeek翻译工具运行脚本

echo "🚀 启动DeepSeek翻译工具..."
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  未安装transformers，正在安装..."
    pip3 install transformers torch accelerate
fi

# 运行翻译程序
echo ""
echo "启动翻译程序..."
python3 translate_deepseek_improved.py

echo ""
echo "✅ 程序结束"