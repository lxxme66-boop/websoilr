#!/bin/bash
# 快速启动本地大模型评测

echo "=================================="
echo "QA测试系统 - 本地大模型快速启动"
echo "=================================="
echo

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

# 检查是否在qa_test目录
if [ ! -f "config.yaml" ]; then
    echo "错误: 请在qa_test目录下运行此脚本"
    exit 1
fi

# 安装依赖
echo "1. 检查并安装依赖..."
pip install -r requirements.txt -q
echo "✓ 依赖安装完成"
echo

# 检查Ollama
echo "2. 检查Ollama服务..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama已安装"
    
    # 检查Ollama服务
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama服务正在运行"
        
        # 检查模型
        if ollama list | grep -q "qwen2.5:7b"; then
            echo "✓ Qwen2.5:7b模型已安装"
        else
            echo "⚠ Qwen2.5:7b模型未安装"
            read -p "是否安装Qwen2.5:7b模型？(y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "正在下载模型（约4GB）..."
                ollama pull qwen2.5:7b
            fi
        fi
    else
        echo "⚠ Ollama服务未运行"
        echo "正在启动Ollama服务..."
        ollama serve &
        sleep 3
    fi
else
    echo "⚠ Ollama未安装"
    echo "请访问 https://ollama.com 下载并安装Ollama"
    echo "或运行: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi
echo

# 备份原配置
echo "3. 配置本地模型..."
if [ ! -f "config.yaml.bak" ]; then
    cp config.yaml config.yaml.bak
    echo "✓ 已备份原配置到 config.yaml.bak"
fi

# 更新配置为本地模型
cat > config.yaml.tmp << EOF
# 问答对质量评测系统配置文件

# 大语言模型配置
llm:
  provider: local
  api_type: ollama
  model: qwen2.5:7b
  endpoint_url: http://localhost:11434/api/chat
  temperature: 0.3
  max_tokens: 500
  timeout: 60
  cache_enabled: true

EOF

# 保留其他配置
tail -n +25 config.yaml >> config.yaml.tmp
mv config.yaml.tmp config.yaml
echo "✓ 已更新配置为使用本地Ollama模型"
echo

# 运行测试
echo "4. 运行测试..."
echo "=================================="
python3 test_local_llm.py

echo
echo "=================================="
echo "设置完成！"
echo
echo "现在您可以使用以下命令运行评测："
echo "  python evaluate_qa.py --input sample_qa_pairs.json --output results.json"
echo
echo "查看更多配置选项："
echo "  cat LOCAL_LLM_SETUP.md"
echo "=================================="