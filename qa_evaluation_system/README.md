# 问答对质量评测系统

## 概述

本系统提供了一个综合的问答对（QA Pairs）质量评测框架，结合大语言模型（LLM）和传统NLP方法，从多个维度评估问答对的质量。

## 主要特性

### 1. 多维度评测
- **问题质量评测**
  - 问题清晰度和具体性
  - 问题的合理性和逻辑性
  - 问题的信息价值
  - 语法和表达规范性

- **答案质量评测**
  - 答案的准确性和相关性
  - 答案的完整性和深度
  - 答案的逻辑性和条理性
  - 语言表达的专业性

- **问答匹配度评测**
  - 问题与答案的相关性
  - 答案对问题的覆盖程度
  - 信息的一致性

### 2. 评测方法

#### 2.1 大模型评测
- 使用先进的LLM（如GPT-4、Claude、DeepSeek等）进行语义理解
- 多角度提示工程，确保评测的全面性
- 支持多个LLM的集成评测，提高可靠性

#### 2.2 传统NLP指标
- **语义相似度**：使用BERT、Sentence-BERT等计算语义相似度
- **文本质量指标**：BLEU、ROUGE、METEOR等
- **可读性分析**：Flesch Reading Ease、Gunning Fog Index等
- **语法检查**：基于规则和模型的语法错误检测

#### 2.3 规则检查
- 长度合理性检查
- 特殊字符和格式检查
- 重复内容检测
- 敏感信息过滤

### 3. 综合评分系统
- 加权组合多个评测维度
- 可配置的权重系统
- 详细的评分报告
- 质量等级划分（优秀、良好、中等、差）

## 安装

```bash
# 克隆项目
git clone <repository_url>
cd qa_evaluation_system

# 安装依赖
pip install -r requirements.txt

# 下载必要的模型
python download_models.py
```

## 使用方法

### 1. 基础使用

```python
from qa_evaluator import QAEvaluator

# 初始化评测器
evaluator = QAEvaluator(config_path="config.yaml")

# 评测单个问答对
question = "什么是机器学习？"
answer = "机器学习是人工智能的一个分支，它使计算机能够从数据中学习..."
score = evaluator.evaluate_single(question, answer)

# 批量评测
qa_pairs = [
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."}
]
results = evaluator.evaluate_batch(qa_pairs)
```

### 2. 命令行使用

```bash
# 评测JSON文件中的问答对
python evaluate_qa.py --input data/qa_pairs.json --output results/evaluation_report.json

# 使用特定配置
python evaluate_qa.py --input data/qa_pairs.json --config custom_config.yaml

# 只评测前N个样本
python evaluate_qa.py --input data/qa_pairs.json --sample 100

# 导出高质量问答对
python evaluate_qa.py --input data/qa_pairs.json --export-top 1000 --threshold 0.8
```

### 3. 高级配置

编辑 `config.yaml` 文件来自定义评测参数：

```yaml
# LLM配置
llm:
  model: "gpt-4"
  temperature: 0.1
  max_tokens: 1000

# 评测权重
weights:
  question_quality: 0.3
  answer_quality: 0.4
  relevance: 0.3

# 阈值设置
thresholds:
  excellent: 0.9
  good: 0.7
  medium: 0.5
```

## 输出格式

评测结果包含：
- 总体评分（0-1）
- 各维度详细评分
- 具体问题和改进建议
- 质量等级

示例输出：
```json
{
  "overall_score": 0.85,
  "quality_level": "良好",
  "detailed_scores": {
    "question_quality": 0.88,
    "answer_quality": 0.82,
    "relevance": 0.86
  },
  "issues": ["答案可以更加详细", "缺少具体示例"],
  "suggestions": ["增加实际应用案例", "提供更多技术细节"]
}
```

## 项目结构

```
qa_evaluation_system/
├── README.md              # 本文件
├── requirements.txt       # 依赖列表
├── config.yaml           # 默认配置文件
├── evaluate_qa.py        # 主程序入口
├── qa_evaluator.py       # 核心评测器
├── llm_evaluator.py      # LLM评测模块
├── nlp_metrics.py        # NLP指标计算
├── rule_checker.py       # 规则检查模块
├── utils.py              # 工具函数
├── download_models.py    # 模型下载脚本
└── examples/             # 示例文件
    ├── sample_qa.json
    └── custom_config.yaml
```

## 性能优化

- 支持批量处理和并行计算
- LLM调用缓存机制
- 可选的轻量级评测模式
- GPU加速支持（用于BERT等模型）

## 注意事项

1. **API密钥**：使用LLM评测需要配置相应的API密钥
2. **计算资源**：完整评测需要较多计算资源，建议使用GPU
3. **成本控制**：大规模评测时注意LLM API的调用成本
4. **数据隐私**：确保敏感数据不会发送到外部API

## 贡献指南

欢迎提交Issue和Pull Request来改进本系统。

## 许可证

MIT License