# 知识图谱构建器优化指南

## 优化概述

针对您遇到的JSON解析错误和实体提取问题，我已经对 `kg_builder.py` 进行了全面优化。主要改进包括：

### 1. 改进的JSON解析策略

原始代码只能处理完整的JSON数组，导致大量解析失败。优化后的解析器支持：

- **多种JSON格式**：支持JSON数组、单行JSON对象、多行JSON对象
- **容错解析**：即使部分JSON格式错误，也能提取有效部分
- **智能修复**：自动补充缺失字段，规范化数据格式

### 2. 增强的实体验证

```python
def _validate_entity(self, entity: Dict) -> bool:
    """验证实体的完整性"""
    # 必须有id
    if 'id' not in entity or not entity['id']:
        return False
    
    # 自动补充缺失字段
    if 'name' not in entity:
        entity['name'] = entity.get('description', entity['id'])
    
    if 'type' not in entity:
        entity['type'] = '未分类'
```

### 3. 优化的文本分块

改进的分块算法确保：
- 句子完整性
- 段落边界保持
- 避免截断重要信息

### 4. 提示词优化器

新增 `PromptOptimizer` 类，提供：
- 领域特定的示例
- 结构化的输出格式
- 关键信息提取
- 后处理规范化

## 使用方法

### 基本使用

```python
from core.kg_builder import IndustrialKGBuilder

# 初始化
kg_builder = IndustrialKGBuilder(model_manager, config)

# 构建知识图谱
documents = {
    "doc1.txt": "文档内容...",
    "doc2.txt": "文档内容..."
}
graph = kg_builder.build_from_documents(documents)

# 保存图谱
kg_builder.save_graph("knowledge_graph.gexf")
```

### 配置参数

```python
class Config:
    DOMAIN = "材料科学"  # 领域名称
    CHUNK_SIZE = 1000    # 文本分块大小
    KG_EXTRACTOR_MODEL_PATH = "path/to/model"
```

## 错误处理改进

### 1. JSON解析错误
- **问题**：`Extra data: line 1 column 381`
- **解决**：使用正则表达式逐个提取JSON对象，而不是期望完整数组

### 2. 实体字段缺失
- **问题**：`[实体缺字段] {'id': '350°C', 'type': '温度'}`
- **解决**：自动补充缺失字段，使用默认值或推断值

### 3. 关系验证失败
- **问题**：关系引用不存在的实体
- **解决**：验证实体ID有效性，过滤无效关系

## 性能优化建议

### 1. 批处理优化
```python
# 限制每次处理的实体数量
entities_for_relation = entities[:10]
```

### 2. 缓存机制
可以添加实体缓存，避免重复提取：
```python
self.entity_cache = {}  # 在__init__中初始化

def _extract_knowledge(self, text_chunk):
    chunk_hash = hashlib.md5(text_chunk.encode()).hexdigest()
    if chunk_hash in self.entity_cache:
        return self.entity_cache[chunk_hash]
```

### 3. 并行处理
对于大量文档，可以使用多进程：
```python
from concurrent.futures import ProcessPoolExecutor

def process_documents_parallel(documents, max_workers=4):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for doc_name, content in documents.items():
            future = executor.submit(process_single_doc, doc_name, content)
            futures.append(future)
```

## 模型调优建议

### 1. 温度参数
- 降低temperature（0.3）以获得更一致的输出
- 使用top_p（0.9）限制输出多样性

### 2. 提示词工程
- 提供具体示例
- 明确输出格式
- 使用领域特定术语

### 3. 后处理
- 规范化ID格式
- 统一实体类型
- 验证关系有效性

## 监控和调试

### 1. 日志级别
```python
# 详细调试
logging.getLogger('core.kg_builder').setLevel(logging.DEBUG)
```

### 2. 统计信息
优化后的代码提供详细统计：
- 总实体数
- 总关系数
- 每个文档的处理情况

### 3. 可视化
使用NetworkX导出的GEXF文件可以在Gephi等工具中可视化。

## 常见问题解决

1. **模型输出不稳定**
   - 使用更低的temperature
   - 提供更多示例
   - 增加重试机制

2. **内存占用过大**
   - 减小CHUNK_SIZE
   - 分批处理文档
   - 定期清理缓存

3. **处理速度慢**
   - 使用批量推理
   - 启用GPU加速
   - 实现缓存机制

## 扩展功能

### 1. 实体消歧
```python
def disambiguate_entities(self, entities):
    """基于上下文消除实体歧义"""
    # 使用embedding相似度
    # 或规则匹配
```

### 2. 关系推理
```python
def infer_implicit_relations(self, graph):
    """推断隐含关系"""
    # 基于图结构
    # 传递性关系
```

### 3. 质量评估
```python
def evaluate_graph_quality(self, graph):
    """评估知识图谱质量"""
    metrics = {
        'density': nx.density(graph),
        'connectivity': nx.is_connected(graph.to_undirected()),
        'avg_degree': sum(dict(graph.degree()).values()) / len(graph)
    }
    return metrics
```

## 总结

优化后的知识图谱构建器具有：
- ✅ 更强的容错能力
- ✅ 更好的解析成功率
- ✅ 更规范的数据格式
- ✅ 更详细的日志信息
- ✅ 更灵活的扩展性

这些改进应该能显著减少您遇到的解析错误，提高知识图谱构建的成功率和质量。