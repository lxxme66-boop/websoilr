# QA Test 使用指南

## 概述

QA Test 系统现在支持两种主要工作模式：

1. **集成模式**：从多个txt文件自动生成问答对并进行质量评测
2. **评测模式**：对已有的问答对进行质量评测

## 快速开始

### 1. 从多个txt文件生成并评测问答对

这是推荐的使用方式，特别适合需要从原始文档生成高质量问答对的场景。

```bash
# 准备txt文件目录
mkdir -p ./my_texts
# 将您的txt文件放入 ./my_texts 目录

# 运行生成与评测
python qa_test/generate_and_evaluate.py \
    --input-dir ./my_texts \
    --output ./outputs/best_qa_pairs.json \
    --config qa_test/config.yaml \
    --top-k 100
```

### 2. 仅评测已有问答对

如果您已经有JSON格式的问答对，可以直接评测：

```bash
python qa_test/evaluate_qa.py \
    --input ./existing_qa_pairs.json \
    --output ./best_qa_pairs.json \
    --top-k 100
```

## 详细功能说明

### 多文件批量处理

系统会自动：
1. 扫描指定目录中的所有txt文件
2. 为每个文件构建知识图谱
3. 从知识图谱中提取子图
4. 基于子图生成多样化的问答对
5. 对所有生成的问答对进行质量评测
6. 筛选出最高质量的问答对

### 问题类型

系统支持生成四种类型的问题：

1. **事实型问题** (Factual)
   - 直接询问事实信息
   - 例如："Mini-LED背光技术的主要优势是什么？"

2. **比较型问题** (Comparison)
   - 比较两个或多个概念
   - 例如："OLED和LCD在功耗方面有什么区别？"

3. **推理型问题** (Reasoning)
   - 需要逻辑推理的问题
   - 例如："为什么印刷OLED技术更适合大尺寸面板制造？"

4. **多跳问题** (Multi-hop)
   - 需要多步推理的复杂问题
   - 例如："TCL如何通过Mini-LED技术实现高对比度显示效果？"

### 评测维度

每个问答对会从以下维度进行评分：

- **相关性** (Relevance): 问题和答案的相关程度
- **完整性** (Completeness): 答案是否完整回答了问题
- **清晰度** (Clarity): 表达是否清晰易懂
- **准确性** (Accuracy): 信息是否准确无误
- **深度** (Depth): 答案的深度和详细程度
- **LLM综合评分**: 大语言模型的整体评价

## 高级配置

### 自定义知识图谱构建

创建自定义的知识图谱配置文件：

```yaml
# kg_custom.yaml
extraction:
  entity_types:
    - "技术"
    - "产品"
    - "应用"
  relation_types:
    - "使用"
    - "包含"
    - "改进"
  chunk_size: 600
  chunk_overlap: 100
```

使用自定义配置：

```bash
python qa_test/generate_and_evaluate.py \
    --input-dir ./texts \
    --output ./outputs/qa_pairs.json \
    --kg-config ./kg_custom.yaml
```

### 自定义问题生成

创建自定义的问题生成配置：

```yaml
# qg_custom.yaml
question_generation:
  questions_per_subgraph: 5
  question_types:
    factual:
      weight: 0.5
    reasoning:
      weight: 0.3
    comparison:
      weight: 0.2
```

### 调整评测权重

在 `config.yaml` 中调整各项评分的权重：

```yaml
weights:
  relevance: 0.20
  completeness: 0.20
  clarity: 0.15
  accuracy: 0.20
  depth: 0.15
  llm_score: 0.10
```

## 性能优化建议

### 1. 处理大量文件

当需要处理大量txt文件时：

```bash
# 分批处理
python qa_test/generate_and_evaluate.py \
    --input-dir ./texts \
    --output ./outputs/batch1.json \
    --max-files 50 \
    --save-intermediate
```

### 2. 跳过LLM评测

如果需要快速处理，可以跳过LLM评测：

```bash
python qa_test/generate_and_evaluate.py \
    --input-dir ./texts \
    --output ./outputs/quick_results.json \
    --skip-llm
```

### 3. 并行处理

系统会自动并行处理多个文件，但您可以通过环境变量控制并行度：

```bash
export OMP_NUM_THREADS=8
python qa_test/generate_and_evaluate.py ...
```

## 输出文件说明

运行后会生成以下文件：

1. **主输出文件** (`best_qa_pairs.json`)
   - 包含筛选后的高质量问答对
   - 每个问答对包含得分和元数据

2. **中间结果** (使用 `--save-intermediate` 时)
   - `*_all_generated.json`: 所有生成的问答对
   - `*_invalid.json`: 无效的问答对

3. **评测报告** (在 `reports/` 目录)
   - `evaluation_report_*.json`: 详细的JSON格式报告
   - `evaluation_summary_*.txt`: 人类可读的摘要报告

## 常见问题

### Q: 如何处理非UTF-8编码的txt文件？

A: 系统默认使用UTF-8编码。如果您的文件使用其他编码，请先转换：

```bash
iconv -f GBK -t UTF-8 input.txt > output.txt
```

### Q: 生成的问题质量不高怎么办？

A: 可以尝试：
1. 调整问题生成配置，增加 `questions_per_subgraph`
2. 优化知识图谱提取，调整 `chunk_size` 和实体类型
3. 提高评测阈值，使用 `--min-score 0.8`

### Q: 处理速度太慢？

A: 建议：
1. 使用 `--skip-llm` 跳过LLM评测
2. 减少每个子图生成的问题数量
3. 使用 `--max-files` 限制处理文件数量

## 示例工作流

### 完整的生产流程

```bash
# 1. 准备数据
mkdir -p ./production_texts
cp /path/to/your/texts/*.txt ./production_texts/

# 2. 测试运行（处理少量文件）
python qa_test/generate_and_evaluate.py \
    --input-dir ./production_texts \
    --output ./test_output.json \
    --max-files 5 \
    --save-intermediate \
    --verbose

# 3. 检查结果
cat ./test_output.json | jq '.[0:5]'

# 4. 完整运行
python qa_test/generate_and_evaluate.py \
    --input-dir ./production_texts \
    --output ./final_qa_pairs.json \
    --top-k 1000 \
    --min-score 0.75 \
    --report-dir ./production_reports

# 5. 查看报告
cat ./production_reports/evaluation_summary_*.txt
```

## 扩展和定制

系统设计为模块化架构，您可以：

1. **自定义评测指标**：修改 `evaluator.py` 添加新的评分维度
2. **自定义问题模板**：在配置文件中添加领域特定的问题模板
3. **集成其他LLM**：修改 `llm_evaluator.py` 支持其他模型
4. **批量后处理**：使用 `data_processor.py` 的功能进行数据清洗

## 技术支持

如遇到问题，请：
1. 查看日志文件 `generate_evaluate_*.log`
2. 检查配置文件格式是否正确
3. 确保所有依赖已正确安装
4. 提交Issue时附上错误日志和配置文件