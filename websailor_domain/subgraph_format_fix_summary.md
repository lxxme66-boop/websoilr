# 子图格式修复总结

## 问题描述
在运行 WebSailor 数据集生成时，生成了 0 个高质量 QA 对。根据用户提供的信息，原因是"子图格式不统一"。

## 问题分析
经过代码分析，发现以下问题：

1. **格式不匹配**：`subgraph_sampler.py` 返回的是 NetworkX DiGraph 对象，但用户提供的代码片段（来自 `question_generator_optimized.py`）期望接收字典格式的子图。

2. **属性名称不一致**：知识图谱构建器（`knowledge_graph_builder.py`）在添加边时使用 `type` 属性，但 `question_generator.py` 中某些地方查找的是 `relation` 属性。

## 修复内容

### 1. 统一边属性访问
修改了 `question_generator.py` 中所有访问边关系的地方，使其优先查找 `type` 属性，如果不存在则回退到 `relation` 属性：

```python
# 修改前
relation = data.get('relation', '')

# 修改后  
relation = data.get('type', data.get('relation', ''))
```

### 2. 修复的具体位置
- `_analyze_subgraph` 方法：修复了关系类型的收集
- `_generate_factual_questions` 方法：修复了边数据的提取
- `_identify_path_patterns` 方法：修复了路径关系的提取
- `_generate_reasoning_questions` 方法：修复了相关边的关系提取
- `_format_subgraph_for_prompt` 方法：修复了格式化输出中的关系显示
- `_can_compare_entities` 方法：修复了实体关系的比较

### 3. 保持 NetworkX 格式
代码继续使用 NetworkX DiGraph 格式，而不是转换为字典格式，这样可以：
- 保持与 `subgraph_sampler.py` 的兼容性
- 利用 NetworkX 提供的图算法功能
- 减少代码改动

## 验证方法
创建了测试脚本 `test_subgraph_format.py` 来验证修复是否正确处理了：
- NetworkX 节点和边的访问
- `type` 和 `relation` 属性的兼容性处理
- 子图拓扑属性的访问

## 建议
1. 在整个项目中统一使用 `type` 作为边的关系属性名称
2. 如果需要支持多种格式，可以在 `QuestionGenerator` 类中添加格式转换方法
3. 添加单元测试以确保格式兼容性