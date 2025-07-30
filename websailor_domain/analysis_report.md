# WebSailor 问答对生成问题分析报告

## 问题概述

运行 `python main.py --mode generate --output_dir output_dataset` 后，系统生成了 0 个问答对。从日志输出可以看到：

```
生成问题: 100%|████████| 25/25 [06:21<00:00, 15.26s/it]
2025-07-30 18:03:02,081 - core.question_generator - INFO - 质量过滤: 0 -> 0
2025-07-30 18:03:02,081 - core.question_generator - INFO - 共生成 0 个高质量QA对
```

## 根本原因分析

### 1. 代码版本不匹配

从用户提供的问答对生成代码来看，这是一个优化版本的 `QuestionGenerator`，但实际运行的是 `/workspace/websailor_domain/core/question_generator.py` 中的原始版本。

**关键差异：**
- 用户提供的代码：使用字典格式的子图（`List[Dict]`）
- 实际运行的代码：使用 NetworkX 图对象（`List[nx.DiGraph]`）

### 2. 数据结构不兼容

实际代码中的 `generate_questions` 方法期望接收 NetworkX DiGraph 对象：
```python
def generate_questions(self, subgraphs: List[nx.DiGraph], 
                      questions_per_subgraph: int = 5) -> List[Dict]:
```

但从 `subgraph_sampler.py` 加载的子图可能是字典格式：
```python
2025-07-30 17:56:40,502 - core.subgraph_sampler - INFO - 加载了 25 个子图
```

### 3. 子图分析失败

在 `_analyze_subgraph` 方法中，代码尝试访问 NetworkX 图的属性：
```python
features = {
    'topology': subgraph.graph.get('topology', 'unknown'),
    'complexity': subgraph.graph.get('complexity', 0.5),
    'num_nodes': subgraph.number_of_nodes(),
    'num_edges': subgraph.number_of_edges(),
    ...
}
```

如果传入的不是正确的 NetworkX 图对象，这些方法调用会失败。

### 4. 异常被静默处理

在 `_generate_question` 方法中，所有异常都被捕获并静默处理：
```python
except Exception as e:
    logger.warning(f"问题生成失败: {e}")
    return None
```

这导致问题生成失败时只返回 None，而不会抛出错误。

## 解决方案

### 方案1：替换为优化版本的代码

将用户提供的优化版 `QuestionGenerator` 替换现有的版本。这个版本：
- 支持字典格式的子图
- 有更完善的问题验证和优化机制
- 包含更丰富的问题模板

### 方案2：修复现有代码的兼容性问题

修改现有的 `question_generator.py`，使其能够处理从 `subgraph_sampler` 加载的子图格式。

### 方案3：调试模式运行

添加更详细的日志输出，找出具体的失败原因。

## 推荐解决步骤

1. **备份现有代码**
   ```bash
   cp /workspace/websailor_domain/core/question_generator.py /workspace/websailor_domain/core/question_generator_original.py
   ```

2. **替换为优化版本**
   使用用户提供的优化版 `QuestionGenerator` 替换现有版本

3. **检查子图格式**
   确认 `subgraph_sampler` 输出的格式与 `question_generator` 期望的格式一致

4. **添加调试日志**
   在关键位置添加日志，追踪问题生成的具体失败点

5. **验证模型加载**
   确保 QA 生成模型正确加载并可用

## 代码修改建议

### 1. 子图格式转换

如果需要保持现有架构，可以添加格式转换函数：

```python
def convert_dict_to_networkx(subgraph_dict: Dict) -> nx.DiGraph:
    """将字典格式的子图转换为NetworkX图"""
    G = nx.DiGraph()
    
    # 添加节点
    for node in subgraph_dict.get('nodes', []):
        G.add_node(node['id'], **node)
    
    # 添加边
    for edge in subgraph_dict.get('edges', []):
        G.add_edge(edge['source'], edge['target'], **edge)
    
    # 添加图属性
    G.graph['topology'] = subgraph_dict.get('topology', 'unknown')
    G.graph['complexity'] = subgraph_dict.get('complexity', 0.5)
    
    return G
```

### 2. 错误处理改进

改进错误处理，使问题更容易诊断：

```python
def _generate_question(self, subgraph: nx.DiGraph, 
                      question_type: str, 
                      features: Dict) -> Optional[Dict]:
    """生成单个问题"""
    try:
        # 生成逻辑...
    except Exception as e:
        logger.error(f"问题生成失败 - 类型: {question_type}, 错误: {str(e)}", exc_info=True)
        # 记录更多调试信息
        logger.debug(f"子图节点数: {subgraph.number_of_nodes() if hasattr(subgraph, 'number_of_nodes') else 'N/A'}")
        logger.debug(f"子图特征: {features}")
        return None
```

### 3. 验证检查

在生成问题前添加验证：

```python
def generate_questions(self, subgraphs: List[Any], 
                      questions_per_subgraph: int = 5) -> List[Dict]:
    """为子图生成问题"""
    logger.info(f"开始为{len(subgraphs)}个子图生成问题...")
    
    # 验证输入
    if not subgraphs:
        logger.warning("没有提供子图")
        return []
    
    # 检查子图类型
    first_subgraph = subgraphs[0]
    if isinstance(first_subgraph, dict):
        logger.info("检测到字典格式的子图，进行格式转换...")
        subgraphs = [convert_dict_to_networkx(sg) for sg in subgraphs]
    elif not isinstance(first_subgraph, nx.DiGraph):
        logger.error(f"不支持的子图类型: {type(first_subgraph)}")
        return []
    
    # 继续原有逻辑...
```

## 总结

问题的根本原因是代码版本不一致和数据格式不兼容。建议使用用户提供的优化版代码，它有更好的兼容性和错误处理机制。如果必须使用现有代码，需要添加格式转换和更详细的错误处理。