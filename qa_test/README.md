# QA Test - 问答对质量评测系统

本系统用于评测生成的问答对质量，帮助筛选高质量的训练数据。

## 主要功能

1. **问答对生成与评测集成** (新功能)
   - 支持从多个txt文件批量生成问答对
   - 自动构建知识图谱并生成多样化问题
   - 集成质量评测，一步完成生成和筛选

2. **多维度质量评测**
   - 相关性评分
   - 完整性评分
   - 清晰度评分
   - 准确性评分
   - 深度评分
   - LLM综合评分

3. **智能筛选机制**
   - 基于总分排序
   - 可配置的阈值过滤
   - Top-K选择

4. **详细的评测报告**
   - JSON格式详细报告
   - 文本摘要报告
   - 质量分布统计

## 项目简介

本项目实现了一个高质量的问答对（QA Pairs）自动评测系统，结合大语言模型（LLM）和传统NLP方法，从多个维度评估问答对的质量，帮助您筛选出最优质的问答数据。

## 功能特点

### 1. 多维度质量评估
- **相关性评分**：评估问题和答案的相关程度
- **答案完整性**：检查答案是否完整回答了问题
- **语言流畅度**：评估文本的语言质量和可读性
- **信息准确性**：通过多种方法验证答案的准确性
- **答案深度**：评估答案的详细程度和信息量

### 2. 混合评测方法
- **大语言模型评测**：
  - 支持本地大模型（Ollama、vLLM、FastChat等）
  - 支持商业API（OpenAI GPT、Anthropic Claude）
  - 灵活的API配置，易于扩展
- **传统NLP指标**：
  - BLEU分数（评估文本相似度）
  - ROUGE分数（评估文本摘要质量）
  - 困惑度（Perplexity）
  - 文本长度和复杂度分析
- **语义相似度**：使用Sentence Transformers计算语义相似度
- **关键词覆盖率**：检查答案对问题关键词的覆盖程度

### 3. 智能筛选
- 综合多维度评分的加权平均
- 可配置的评分阈值
- 自动筛选Top-K最佳问答对

## 安装与配置

### 1. 环境要求
- Python 3.8+
- CUDA（可选，用于GPU加速）

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置文件
编辑 `config.yaml` 文件，配置以下参数：
- LLM提供商（local/openai/anthropic）
- 本地模型设置（参见 [LOCAL_LLM_SETUP.md](LOCAL_LLM_SETUP.md)）
- 评测参数（权重、阈值等）
- 模型选择
- 输出设置

## 使用方法

### 1. 从多个txt文件生成并评测问答对（推荐）

使用新的集成脚本，可以直接从包含多个txt文件的目录生成问答对并进行评测：

```bash
# 基本用法
python qa_test/generate_and_evaluate.py \
    --input-dir ./texts \
    --output ./outputs/best_qa_pairs.json \
    --config qa_test/config.yaml

# 高级用法（指定更多参数）
python qa_test/generate_and_evaluate.py \
    --input-dir ./texts \
    --output ./outputs/best_qa_pairs.json \
    --config qa_test/config.yaml \
    --top-k 200 \
    --min-score 0.7 \
    --report-dir ./reports \
    --save-intermediate \
    --verbose

# 限制处理文件数量（用于测试）
python qa_test/generate_and_evaluate.py \
    --input-dir ./texts \
    --output ./outputs/test_qa_pairs.json \
    --config qa_test/config.yaml \
    --max-files 3 \
    --save-intermediate
```

#### 参数说明：
- `--input-dir`: 包含txt文件的输入目录路径
- `--output`: 输出最佳问答对文件路径
- `--config`: 配置文件路径（默认: config.yaml）
- `--top-k`: 选择Top-K个最佳问答对（默认: 100）
- `--min-score`: 最低分数阈值
- `--report-dir`: 报告输出目录（默认: reports）
- `--no-report`: 不生成评测报告
- `--skip-llm`: 跳过LLM评测（仅使用基于规则的评测）
- `--verbose`: 显示详细日志
- `--kg-config`: 知识图谱构建配置文件路径
- `--qg-config`: 问题生成配置文件路径
- `--save-intermediate`: 保存中间结果
- `--max-files`: 最多处理的txt文件数量

### 2. 仅评测已有的问答对（原有功能）

如果你已经有生成好的问答对JSON文件，可以直接进行评测：

```bash
python evaluate_qa.py --input your_qa_pairs.json --output best_qa_pairs.json --top_k 100
```

### 3. 查看结果
评测完成后，系统会生成：
- `best_qa_pairs.json`：筛选出的最佳问答对
- `evaluation_report.json`：详细的评测报告
- `evaluation_summary.txt`：评测摘要

## 技术架构

### 核心模块
1. **evaluator.py**：主评测引擎
2. **llm_evaluator.py**：大语言模型评测模块
3. **nlp_metrics.py**：传统NLP指标计算
4. **semantic_analyzer.py**：语义分析模块
5. **data_processor.py**：数据处理和预处理
6. **scorer.py**：综合评分和排序模块

### 评测流程
1. 数据加载和预处理
2. 并行执行多维度评测
3. 评分标准化和加权
4. 综合排序和筛选
5. 结果输出和报告生成

## 评测指标详解

### 1. LLM评测（权重：40%）
- 使用GPT-4评估问答对的整体质量
- 考虑因素：相关性、准确性、完整性、清晰度

### 2. 语义相似度（权重：20%）
- 使用Sentence-BERT计算问题和答案的语义相关性
- 确保答案真正回答了问题

### 3. 答案质量（权重：20%）
- 答案长度适中性
- 信息密度
- 结构清晰度

### 4. 语言流畅度（权重：10%）
- 语法正确性
- 表达自然度
- 可读性评分

### 5. 关键信息覆盖（权重：10%）
- 问题关键词在答案中的覆盖率
- 专业术语的使用准确性

## 自定义配置

您可以通过修改 `config.yaml` 来自定义评测行为：

```yaml
# 评分权重配置
weights:
  llm_score: 0.4
  semantic_similarity: 0.2
  answer_quality: 0.2
  fluency: 0.1
  keyword_coverage: 0.1

# 筛选阈值
thresholds:
  min_total_score: 0.7
  min_llm_score: 0.6
  min_answer_length: 20
  max_answer_length: 500
```

## 性能优化

- 支持批量处理，提高评测效率
- 可选的GPU加速（用于语义模型）
- 结果缓存机制，避免重复计算
- 并行处理多个评测维度

## 注意事项

1. 大语言模型API调用可能产生费用，请合理设置批次大小
2. 首次运行需要下载预训练模型（约500MB）
3. 建议在评测大量数据前先用小样本测试配置

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进本项目。

## 运行示例脚本

提供了完整的使用示例脚本：

```bash
# 查看使用示例
cat qa_test/example_generate_evaluate.sh

# 运行示例（会创建测试文件并执行）
bash qa_test/example_generate_evaluate.sh
```