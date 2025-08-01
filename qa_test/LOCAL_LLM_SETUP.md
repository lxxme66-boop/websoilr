# 本地大模型设置指南

本指南将帮助您配置qa_test系统以使用本地大模型，而不依赖OpenAI等商业API。

## 目录

1. [支持的本地LLM服务](#支持的本地llm服务)
2. [Ollama设置](#ollama设置)
3. [vLLM设置](#vllm设置)
4. [FastChat设置](#fastchat设置)
5. [配置说明](#配置说明)
6. [故障排除](#故障排除)

## 支持的本地LLM服务

qa_test系统支持以下本地LLM服务：

- **Ollama** - 最简单的本地部署方案，支持多种开源模型
- **vLLM** - 高性能推理引擎，适合生产环境
- **FastChat** - 支持多种模型的统一接口
- **LM Studio** - 图形界面的本地LLM运行工具
- **任何OpenAI兼容的API** - 包括LocalAI、llama.cpp server等

## Ollama设置

### 1. 安装Ollama

```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# 从 https://ollama.com/download 下载安装程序
```

### 2. 下载模型

推荐使用以下中文优化模型：

```bash
# Qwen2.5系列（推荐）
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b

# DeepSeek系列
ollama pull deepseek-coder-v2:16b

# 其他选择
ollama pull llama3.2:3b-instruct-fp16
ollama pull mistral:7b-instruct
```

### 3. 启动Ollama服务

```bash
# Ollama通常会自动启动，如果没有：
ollama serve
```

### 4. 配置qa_test

编辑 `qa_test/config.yaml`：

```yaml
llm:
  provider: local
  api_type: ollama
  model: qwen2.5:7b  # 使用您下载的模型
  endpoint_url: http://localhost:11434/api/chat
  temperature: 0.3
  max_tokens: 500
  timeout: 60
```

## vLLM设置

### 1. 安装vLLM

```bash
# 需要CUDA支持
pip install vllm

# 或使用Docker
docker pull vllm/vllm-openai:latest
```

### 2. 启动vLLM服务

```bash
# 使用Qwen2.5模型
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --max-model-len 4096

# 或使用Docker
docker run --gpus all -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct
```

### 3. 配置qa_test

```yaml
llm:
  provider: local
  api_type: vllm
  model: Qwen/Qwen2.5-7B-Instruct
  endpoint_url: http://localhost:8000/v1/chat/completions
  temperature: 0.3
  max_tokens: 500
  timeout: 60
```

## FastChat设置

### 1. 安装FastChat

```bash
pip install fschat
```

### 2. 启动FastChat服务

```bash
# 启动controller
python -m fastchat.serve.controller

# 启动model worker（新终端）
python -m fastchat.serve.model_worker \
    --model-path lmsys/vicuna-7b-v1.5 \
    --controller http://localhost:21001

# 启动OpenAI API server（新终端）
python -m fastchat.serve.openai_api_server \
    --controller http://localhost:21001 \
    --port 8000
```

### 3. 配置qa_test

```yaml
llm:
  provider: local
  api_type: openai-compatible
  model: vicuna-7b-v1.5
  endpoint_url: http://localhost:8000/v1/chat/completions
  temperature: 0.3
  max_tokens: 500
  timeout: 60
```

## 配置说明

### 基本配置项

- `provider`: 设置为 `local` 以使用本地模型
- `api_type`: 选择API类型
  - `ollama`: Ollama服务
  - `vllm`: vLLM服务
  - `openai-compatible`: OpenAI兼容的API
- `model`: 模型名称（根据服务类型而定）
- `endpoint_url`: API端点URL
- `temperature`: 生成温度（0-1，越低越确定）
- `max_tokens`: 最大生成令牌数
- `timeout`: 请求超时时间（秒）

### 高级配置项

```yaml
llm:
  # ... 基本配置 ...
  
  # 可选：API密钥（某些服务需要）
  api_key: your-api-key
  
  # 可选：自定义请求头
  headers:
    X-Custom-Header: value
```

## 使用示例

### 1. 安装依赖

```bash
cd qa_test
pip install -r requirements.txt
```

### 2. 运行评测

```bash
# 评测单个问答对
python evaluate_qa.py \
    --question "什么是机器学习？" \
    --answer "机器学习是人工智能的一个分支..."

# 批量评测
python evaluate_qa.py \
    --input sample_qa_pairs.json \
    --output results.json
```

## 故障排除

### 1. 连接错误

如果出现 "Cannot connect to local LLM service" 错误：

- 检查服务是否正在运行
- 确认端口号是否正确
- 检查防火墙设置

```bash
# 测试连接
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8000/v1/models  # vLLM/OpenAI兼容
```

### 2. 超时错误

如果请求超时：

- 增加 `timeout` 配置值
- 检查模型大小是否适合您的硬件
- 考虑使用更小的模型

### 3. 内存不足

对于大模型：

- 使用量化版本（如GGUF格式）
- 减少 `max_model_len` 参数
- 使用更小的模型变体

### 4. GPU相关问题

```bash
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 查看GPU使用情况
nvidia-smi
```

## 性能优化建议

1. **模型选择**
   - 7B参数模型通常足够用于QA评测
   - 中文任务推荐Qwen2.5系列
   - 代码相关推荐DeepSeek-Coder

2. **批处理**
   - 调整 `batch_size` 配置以优化吞吐量
   - 使用 `max_workers` 控制并发请求

3. **缓存**
   - 启用 `cache_enabled` 避免重复评测
   - 定期清理缓存以释放内存

## 推荐配置

对于大多数用户，我们推荐使用Ollama + Qwen2.5：

```yaml
llm:
  provider: local
  api_type: ollama
  model: qwen2.5:7b
  temperature: 0.3
  max_tokens: 500
  cache_enabled: true
```

这个配置提供了良好的中文理解能力和合理的性能平衡。

## 更多资源

- [Ollama文档](https://github.com/ollama/ollama)
- [vLLM文档](https://docs.vllm.ai/)
- [FastChat文档](https://github.com/lm-sys/FastChat)
- [Qwen模型](https://github.com/QwenLM/Qwen2.5)