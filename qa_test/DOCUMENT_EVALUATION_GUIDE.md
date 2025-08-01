# 基于文档的问答对评测指南

## 概述

本指南介绍如何使用基于文档的评测功能来评估问答对的质量。该功能会将文档作为知识库，评测生成的问答对是否与文档内容一致、准确。

## 主要特性

1. **文档加载**: 自动加载指定目录下的所有 txt 和 md 文件作为知识库
2. **多维度评测**: 评测答案的相关性、准确性、覆盖度和一致性
3. **支持证据**: 为每个问答对提供来自文档的支持句子
4. **灵活集成**: 可以独立使用，也可以与传统评测方法结合

## 使用方法

### 1. 独立的文档评测器

使用 `document_based_evaluator.py` 进行纯文档评测：

```bash
python qa_test/run_document_evaluation.py \
    --documents websailor_domain/input_texts \
    --input websailor_domain_dataset/qa_pairs.json \
    --output evaluated_qa_pairs.json \
    --top-k 100
```

参数说明：
- `--documents`: 文档目录路径（包含 txt/md 文件）
- `--input`: 输入的问答对 JSON 文件
- `--output`: 输出的评测结果文件
- `--top-k`: 选择 Top-K 个最佳问答对
- `--model`: 句子嵌入模型（默认: paraphrase-multilingual-mpnet-base-v2）

### 2. 集成评测器

使用 `evaluate_qa_with_docs.py` 结合传统评测和文档评测：

```bash
python qa_test/evaluate_qa_with_docs.py \
    --input qa_pairs.json \
    --output best_qa_pairs.json \
    --documents websailor_domain/input_texts \
    --doc-weight 0.6 \
    --top-k 100
```

额外参数：
- `--doc-weight`: 文档评测的权重（0-1之间，默认0.5）
- `--config`: 配置文件路径
- `--min-score`: 最低分数阈值

### 3. 使用示例脚本

运行提供的示例脚本：

```bash
cd qa_test
./example_document_evaluation.sh
```

## 评测维度

### 1. 相关性（Relevance）
- 评估答案与文档中相关句子的语义相似度
- 使用句子嵌入模型计算余弦相似度

### 2. 准确性（Accuracy）
- 评估答案中的关键词是否与文档内容匹配
- 计算关键词重叠度

### 3. 覆盖度（Coverage）
- 评估答案覆盖了多少文档中的关键信息
- 考虑答案长度的合理性

### 4. 一致性（Consistency）
- 评估答案是否与文档内容一致，没有矛盾
- 检测否定词和语义冲突

## 输出格式

### 评测结果 JSON

```json
{
  "total_evaluated": 200,
  "selected_count": 100,
  "evaluation_summary": {
    "avg_relevance": 0.825,
    "avg_accuracy": 0.756,
    "avg_coverage": 0.689,
    "avg_consistency": 0.912,
    "avg_total": 0.796
  },
  "qa_pairs": [
    {
      "id": "qa_0",
      "question": "TCL华星的主要产品是什么？",
      "answer": "TCL华星主要生产显示面板...",
      "scores": {
        "relevance": 0.892,
        "accuracy": 0.834,
        "coverage": 0.756,
        "consistency": 0.950,
        "total": 0.858
      },
      "supporting_evidence": [
        {
          "sentence": "TCL华星光电技术有限公司专注于显示面板的研发和生产",
          "similarity": 0.923
        }
      ]
    }
  ]
}
```

### 评测报告

自动生成的 Markdown 报告包含：
- 评测概览
- 分数分布统计
- 各维度平均分
- Top 10 最佳问答对示例
- Bottom 5 最差问答对示例

## 最佳实践

1. **文档准备**
   - 确保文档内容准确、完整
   - 文档应该覆盖问答对涉及的主题
   - 使用 UTF-8 编码

2. **参数调优**
   - 根据文档质量调整文档评测权重
   - 对于高质量文档，可以增加文档评测权重到 0.7-0.8
   - 对于不确定的文档，保持默认的 0.5

3. **结果分析**
   - 查看支持证据，验证评测的合理性
   - 关注低分问答对，分析原因
   - 使用生成的报告进行深入分析

## 故障排除

### 常见问题

1. **内存不足**
   - 减少批处理大小
   - 使用更小的句子嵌入模型

2. **评测速度慢**
   - 首次运行需要下载模型
   - 可以使用 GPU 加速（如果可用）

3. **文档未加载**
   - 检查文档路径是否正确
   - 确保文档格式为 .txt 或 .md

## 扩展功能

### 自定义评测权重

在配置文件中调整各维度的权重：

```yaml
document_evaluation:
  weight: 0.6
  dimension_weights:
    relevance: 0.3
    accuracy: 0.3
    coverage: 0.2
    consistency: 0.2
```

### 添加新的文档格式

修改 `_load_documents` 方法支持新格式：

```python
supported_extensions = ['.txt', '.md', '.pdf', '.docx']
```

## 总结

基于文档的评测功能提供了一种客观、可验证的方式来评估问答对的质量。通过将生成的答案与源文档进行比对，可以确保问答对的准确性和可靠性。