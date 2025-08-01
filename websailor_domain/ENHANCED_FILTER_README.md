# 增强版 QA 过滤功能说明

## 功能概述

增强版的 `_filter_qa_pairs` 方法在原有基础上新增了使用大语言模型（LLM）来检测和删除答案明显错误或回答无关的问答对的功能。

## 主要改进

1. **答案相关性检查**：使用大模型评估答案是否直接回答了问题
2. **答案正确性验证**：检测答案内容是否准确合理，是否包含明显错误
3. **更精确的去重算法**：基于关键词指纹的去重方式
4. **详细的过滤统计**：记录各种原因的过滤数量
5. **质量分数计算**：综合多个维度计算问题质量

## 配置说明

### 启用 LLM 验证

在配置文件中添加以下配置：

```json
{
  "use_llm_validation": true,  // 启用LLM验证
  "validity_threshold": 0.7,    // 合理性分数阈值
  "models": {
    "qa_generator_model": {
      "path": "/mnt/storage/models/Qwen/Qwen2.5-14B-Instruct",
      "description": "QA生成模型",
      "max_length": 4096,
      "temperature": 0.8
    }
  },
  "dataset_synthesis": {
    "quality_checks": {
      "min_question_length": 20,
      "max_question_length": 600,
      "min_answer_length": 20,
      "answer_validation": true,
      "check_answer_relevance": true
    }
  }
}
```

### 配置参数说明

- `use_llm_validation`: 是否启用大模型验证（默认为 true）
- `validity_threshold`: 合理性分数阈值，低于此分数的问答对会被过滤（默认 0.7）
- `min_question_length`: 问题最小长度（默认 20）
- `max_question_length`: 问题最大长度（默认 600）
- `min_answer_length`: 答案最小长度（默认 20）

## 过滤流程

1. **长度检查**：过滤掉长度不符合要求的问题
2. **答案验证**：检查答案是否存在且满足最小长度要求
3. **合理性分数检查**：基于预设的合理性分数阈值过滤
4. **去重检查**：使用关键词指纹算法去除重复问题
5. **LLM相关性检查**：使用大模型评估答案的相关性和正确性
6. **质量分数计算**：综合评估问题质量，只保留高质量问题

## 质量分数计算维度

1. **问题长度**：20-200字符最佳（2分）
2. **答案长度**：50-500字符最佳（2分）
3. **问题类型**：推理型和多跳型问题得分更高
4. **实体数量**：2-5个实体最佳（1.5分）
5. **合理性分数**：基于LLM评估的加成

## 过滤统计信息

运行后会输出详细的过滤统计：

```
质量过滤统计:
  总数: 112
  长度过滤: 10
  答案过滤: 5
  合理性过滤: 8
  重复过滤: 12
  相关性过滤: 15
  最终保留: 62
```

## 性能考虑

- LLM验证会显著增加处理时间，建议在小批量数据上使用
- 可以通过设置 `use_llm_validation: false` 来禁用LLM验证
- 质量阈值可以根据需求调整，默认为 5.0

## 使用示例

```python
# 在 question_generator_optimized.py 中已经集成
generator = QuestionGenerator(config)
qa_pairs = generator.generate_questions(subgraphs)
# _filter_qa_pairs 会在内部自动调用
```

## 注意事项

1. 确保模型路径正确且可访问
2. LLM验证需要GPU支持，建议使用支持CUDA的环境
3. 首次运行时会下载模型，需要足够的磁盘空间
4. 可以根据实际需求调整各项阈值参数