# 知识图谱构建器改进说明

## 主要改进内容

### 1. 核心改进：从基于规则的实体抽取改为基于大模型的实体抽取

#### 原版本的问题：
- 依赖jieba分词和词性标注
- 需要大量预定义的规则和词典
- 对复杂语境理解能力有限
- 难以识别新的实体类型和关系

#### 改进后的优势：
- 使用大语言模型（LLM）直接理解文本语义
- 通过精心设计的prompt引导模型抽取实体和关系
- 能够理解复杂的技术文档和专业术语
- 灵活适应新的实体类型，无需修改代码

### 2. 具体改进点

#### 2.1 模型加载方式
```python
# 原版本：加载BERT模型用于计算置信度
self.model = AutoModel.from_pretrained('bert-base-chinese')

# 改进版：加载大语言模型用于实体关系抽取
self.model = AutoModelForCausalLM.from_pretrained(
    self.model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

#### 2.2 实体抽取方法
- **原版本**：基于词性标注和规则匹配
- **改进版**：使用LLM通过prompt一次性抽取实体和关系，返回结构化JSON

#### 2.3 文本处理优化
- 添加了更智能的文本分块方法，保持语义完整性
- 改进了句子分割规则，保留句子结束符
- 支持处理更长的文档

#### 2.4 数据验证和清理
- 增强了实体ID生成逻辑，支持中文
- 添加了更完善的数据验证机制
- 自动处理重复实体和关系

#### 2.5 输出格式增强
- 添加了更多统计信息（平均度数等）
- 保留了实体和关系的来源信息
- 支持多文件来源追踪

## 运行方式

### 1. 环境准备

```bash
# 安装依赖
pip install torch transformers networkx tqdm

# 如果使用GPU
pip install accelerate
```

### 2. 准备配置文件

```python
config = {
    'knowledge_graph': {
        'entity_types': ['技术', '产品', '材料', '工艺', '公司', '人员'],
        'relation_types': ['使用', '包含', '生产', '研发', '依赖', '改进', '替代', '认证', '合作', '应用于'],
        'chunk_size': 1000,  # 文本分块大小
        'max_chunk_overlap': 200,  # 块之间的重叠
        'extraction_rules': {
            'confidence_threshold': 0.7  # 置信度阈值
        }
    },
    'tcl_specific': {
        'technical_terms': {
            'display': ['OLED', 'LCD', 'Mini-LED', 'QLED'],
            'semiconductor': ['芯片', '半导体', '集成电路'],
            # ... 更多TCL专业术语
        }
    },
    'models': {
        'llm_model': {
            'path': 'THUDM/chatglm3-6b'  # 可以替换为其他中文LLM
        }
    }
}
```

### 3. 准备输入数据

创建输入目录并放入文本文件：
```bash
mkdir -p data/texts
# 将TCL相关的技术文档（.txt格式）放入该目录
```

### 4. 运行代码

```python
from knowledge_graph_builder_improved import KnowledgeGraphBuilder
from pathlib import Path

# 创建构建器实例
builder = KnowledgeGraphBuilder(config)

# 构建知识图谱
kg = builder.build_from_texts('./data/texts')

# 保存结果
builder.save_graph(kg, Path('./output/tcl_knowledge_graph.json'))
```

### 5. 使用不同的大模型

可以通过修改配置使用不同的中文大模型：

```python
# 使用ChatGLM3
config['models']['llm_model']['path'] = 'THUDM/chatglm3-6b'

# 使用Qwen
config['models']['llm_model']['path'] = 'Qwen/Qwen-7B-Chat'

# 使用Baichuan
config['models']['llm_model']['path'] = 'baichuan-inc/Baichuan2-7B-Chat'
```

### 6. 输出格式

生成的知识图谱JSON文件包含：
- **nodes**: 实体节点列表，包含id、name、type、confidence等信息
- **edges**: 关系边列表，包含source、target、relation、confidence等信息
- **statistics**: 统计信息，包含节点数、边数、类型分布等

### 7. 性能优化建议

1. **GPU加速**：确保CUDA可用，模型会自动使用GPU
2. **批处理**：可以调整chunk_size来平衡内存使用和处理速度
3. **模型选择**：
   - 资源充足：使用更大的模型（如13B、70B）
   - 资源有限：使用量化版本或更小的模型

### 8. 后续使用

生成的知识图谱可用于：
- 训练TCL PPO垂域模型的数据集
- 知识问答系统
- 实体链接和关系推理
- 技术文档的语义搜索

## 注意事项

1. **内存需求**：大语言模型需要较多GPU内存（建议16GB以上）
2. **处理时间**：相比基于规则的方法，处理时间会更长
3. **模型选择**：建议使用针对中文优化的模型
4. **数据质量**：输入文本的质量直接影响抽取效果

## 效果对比

| 方面 | 原版本（基于规则） | 改进版（基于LLM） |
|------|-------------------|-------------------|
| 准确率 | 中等 | 高 |
| 召回率 | 低 | 高 |
| 灵活性 | 低（需要预定义规则） | 高（通过prompt调整） |
| 处理速度 | 快 | 慢 |
| 资源需求 | 低 | 高 |
| 新实体识别 | 困难 | 容易 |
| 上下文理解 | 弱 | 强 |