# WebSailor Domain Dataset Generator

基于WebSailor核心思想构造的TCL工业垂域数据集生成器。

## 项目概述

本项目实现了WebSailor的三大核心思想，用于生成高质量的问答数据集：

### WebSailor核心思想

1. **子图采样** - 从整个知识图中抽取不同拓扑的子图作为问题候选基础
   - 星形拓扑：一个中心节点连接多个周边节点
   - 路径拓扑：线性连接的节点链
   - 簇拓扑：密集连接的节点群
   - 树形拓扑：层次结构的节点关系

2. **问题生成** - 基于子图中节点与关系，设计QA问题，覆盖多种问题类型
   - 单跳事实查询：直接查询某个实体的属性
   - 多跳事实查询：需要通过多个节点推理
   - 比较查询：比较多个实体的异同
   - 聚合查询：统计或汇总信息
   - 推理查询：需要逻辑推理的复杂问题

3. **模糊化处理** - 模糊描述中间实体或关系，添加冗余或干扰信息
   - 实体匿名化：用模糊描述替代具体实体名
   - 关系模糊化：模糊描述实体间的具体关系
   - 噪声注入：添加干扰信息和冗余描述
   - 信息隐藏：隐藏部分关键信息，增加推理难度

## 项目结构

```
websailor_domain_dataset/
├── main.py                          # 主入口文件
├── config.json                      # 配置文件
├── requirements.txt                 # 依赖包列表
├── README.md                        # 项目说明文档
├── 
├── core/                           # 核心模块
│   ├── __init__.py
│   ├── knowledge_graph_builder.py   # 知识图谱构建器
│   ├── subgraph_sampler.py         # 子图采样器（核心思想1）
│   ├── question_generator.py       # 问题生成器（核心思想2）
│   ├── obfuscation_processor.py    # 模糊化处理器（核心思想3）
│   ├── trajectory_generator.py     # 推理轨迹生成器
│   └── data_synthesizer.py         # 数据综合器
├── 
├── utils/                          # 工具模块
│   ├── __init__.py
│   ├── nlp_utils.py                # NLP工具函数
│   ├── graph_utils.py              # 图处理工具
│   └── text_utils.py               # 文本处理工具
├── 
├── input_texts/                    # 输入文本文件夹
│   ├── tcl_sample_1.txt
│   └── ...
├── 
├── output_dataset/                 # 输出数据集文件夹
│   ├── qa_pairs.json
│   ├── trajectories.json
│   ├── knowledge_graphs.json
│   └── statistics.json
├── 
├── templates/                      # 模板文件
│   ├── question_templates.json     # 问题模板
│   ├── obfuscation_patterns.json  # 模糊化模式
│   └── trajectory_templates.json  # 轨迹模板
└── 
└── examples/                       # 示例文件
    ├── sample_input/
    ├── sample_output/
    └── README_examples.md
```

## 安装和使用

### 环境要求

- Python 3.8+
- 推荐使用虚拟环境

### 安装依赖

```bash
pip install -r requirements.txt
```

### 可选：安装中文NLP模型

```bash
# 安装spacy中文模型（可选，用于更好的实体识别）
python -m spacy download zh_core_web_sm
```

### 使用方法

1. **准备输入文本**
   将TCL工业领域的文本文件放入 `input_texts/` 目录

2. **配置参数**
   根据需要修改 `config.json` 中的参数

3. **运行生成器**
   ```bash
   python main.py
   ```

4. **查看结果**
   生成的数据集将保存在 `output_dataset/` 目录中

## 配置说明

### 主要配置项

- `num_subgraphs`: 要生成的子图数量
- `question_types`: 问题类型及其权重
- `obfuscation_strategies`: 模糊化策略及其参数
- `quality_filters`: 质量过滤标准

### 子图采样配置

```json
"subgraph_sampling": {
  "sampling_strategies": [
    {
      "name": "star_topology",
      "weight": 0.3,
      "parameters": {
        "center_node_types": ["产品", "技术", "公司"],
        "min_neighbors": 3,
        "max_neighbors": 8
      }
    }
  ]
}
```

### 问题生成配置

```json
"question_generation": {
  "question_types": [
    {
      "name": "factual_single",
      "weight": 0.2,
      "templates": [
        "{entity}的{attribute}是什么？"
      ]
    }
  ]
}
```

## 输出格式

### QA对格式

```json
{
  "entry_id": "entry_1234",
  "question": "这家公司的显示技术有什么特点？",
  "answer": "该企业在量子点技术方面具有显著优势...",
  "question_type": "factual_single",
  "obfuscation_type": "entity_anonymization",
  "quality_score": 0.85,
  "trajectory": {...},
  "features": {...},
  "metadata": {...}
}
```

### 推理轨迹格式

```json
{
  "question": "...",
  "answer": "...",
  "trajectory_type": "step_by_step_reasoning",
  "steps": [
    {
      "step_id": 1,
      "step_type": "question_understanding",
      "description": "理解问题：...",
      "entities": [...],
      "relations": [...],
      "confidence": 0.9,
      "evidence": "..."
    }
  ],
  "features": {...}
}
```

## 核心特性

### 1. 多样化的子图拓扑

- **星形拓扑**：适合生成聚合类问题
- **路径拓扑**：适合生成推理链问题
- **簇拓扑**：适合生成复杂推理问题
- **树形拓扑**：适合生成层次化问题

### 2. 丰富的问题类型

- **事实查询**：单跳和多跳事实问题
- **比较分析**：实体间的对比问题
- **聚合统计**：数量和分类统计问题
- **逻辑推理**：需要推理的复杂问题

### 3. 智能模糊化

- **实体匿名化**：用"这家公司"代替具体公司名
- **关系模糊化**：用"相关"代替具体关系
- **噪声注入**：添加干扰信息
- **信息隐藏**：隐藏关键信息

### 4. 详细推理轨迹

- **逐步推理**：展示推理过程
- **图遍历路径**：基于知识图谱的推理路径
- **证据收集**：收集支持答案的证据
- **假设验证**：验证推理假设

## 扩展和定制

### 添加新的采样策略

1. 在 `core/subgraph_sampler.py` 中实现新的采样函数
2. 在配置文件中添加相应配置
3. 更新采样器的策略映射

### 添加新的问题类型

1. 在 `core/question_generator.py` 中实现生成函数
2. 在配置文件中添加问题模板
3. 更新生成器的类型映射

### 添加新的模糊化策略

1. 在 `core/obfuscation_processor.py` 中实现处理函数
2. 在配置文件中添加策略参数
3. 更新处理器的策略映射

## 技术特点

- **模块化设计**：各组件独立，易于扩展
- **配置驱动**：通过配置文件控制生成行为
- **质量保证**：多层次的质量评估和过滤
- **可扩展性**：支持新的领域和数据类型

## 注意事项

1. **内存使用**：大规模生成时注意内存占用
2. **生成时间**：复杂配置可能需要较长时间
3. **质量调优**：根据具体需求调整质量阈值
4. **领域适配**：针对不同领域需要调整实体和关系类型

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。

## 联系方式

如有问题或建议，请通过 Issue 联系我们。