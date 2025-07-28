# WebSailor Domain Dataset Construction Tool

基于WebSailor核心思想的垂域（TCL工业）数据集构建工具

## 项目概述

本项目实现了WebSailor论文中的核心数据构造思想，专门用于构建TCL工业领域的高质量训练数据集。通过知识图谱构建、子图采样、问题生成、模糊化处理和推理轨迹生成等步骤，生成适合训练大语言模型的领域特定数据。

### WebSailor核心思想实现

1. **子图采样（Subgraph Sampling）**
   - 从整个知识图谱中抽取不同拓扑结构的子图
   - 每个子图代表一种"任务场景"，包含多个目标、干扰信息和隐含路径
   - 支持星型、链式、树型、环型等拓扑结构

2. **问题生成（Question Generation）**
   - 基于子图中的节点与关系设计QA问题
   - 覆盖事实型、推理型、多跳型、比较型等多种问题类型
   - 确保问题的可回答性并生成干扰项

3. **模糊化处理（Obfuscation Processing）**
   - 将具体实体抽象化（如"这个设备"代替具体名称）
   - 信息分散，在问题中插入相关但非必要的信息
   - 添加冗余或干扰信息，增加问题密度但减少精确信息
   - 引入代词、时间、空间等歧义

## 安装与配置

### 环境要求

- Python 3.8+
- 建议使用虚拟环境

### 安装步骤

1. 克隆项目
```bash
git clone <repository_url>
cd websailor_domain_dataset
```

2. 创建虚拟环境（推荐）
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 下载必要的NLP模型（如需要）
```bash
python -m spacy download zh_core_web_sm
```

## 使用指南

### 1. 准备输入数据

在`input_texts/`目录下放置领域文本文件（.txt格式），例如：
- `domain_text_1.txt`: TCL显示技术相关文档
- `domain_text_2.txt`: 制造工艺相关文档
- `domain_text_3.txt`: 质量控制相关文档

### 2. 配置参数

编辑`config.json`文件，调整以下关键参数：

```json
{
  "subgraph_sampling": {
    "total_subgraphs": 1000,  // 生成的子图数量
    "sampling_strategies": [...] // 采样策略配置
  },
  "question_generation": {
    "question_types": {...}  // 问题类型权重
  },
  "obfuscation": {
    "strategies": {...}  // 模糊化策略
  }
}
```

### 3. 运行数据集构建

```bash
python main.py --config config.json --input-dir input_texts --output-dir output_dataset
```

可选参数：
- `--config`: 配置文件路径（默认：config.json）
- `--input-dir`: 输入文本目录（默认：input_texts）
- `--output-dir`: 输出数据集目录（默认：output_dataset）
- `--log-level`: 日志级别（默认：INFO）

### 4. 查看输出结果

生成的数据集将保存在`output_dataset/`目录下：
- `qa_pairs.json`: 问答对数据
- `trajectories.json`: 推理轨迹数据
- `knowledge_graphs.json`: 知识图谱数据
- `statistics.json`: 统计信息

## 项目结构

```
websailor_domain_dataset/
├── main.py                          # 主入口文件
├── config.json                      # 配置文件
├── requirements.txt                 # 依赖包列表
├── README.md                        # 本文档
│
├── core/                           # 核心模块
│   ├── __init__.py
│   ├── knowledge_graph_builder.py   # 知识图谱构建器
│   ├── subgraph_sampler.py         # 子图采样器（WebSailor核心）
│   ├── question_generator.py        # 问题生成器（WebSailor核心）
│   ├── obfuscation_processor.py    # 模糊化处理器（WebSailor核心）
│   ├── trajectory_generator.py     # 推理轨迹生成器
│   └── data_synthesizer.py         # 数据综合器
│
├── utils/                          # 工具模块
│   ├── __init__.py
│   ├── nlp_utils.py                # NLP工具函数
│   ├── graph_utils.py              # 图处理工具
│   └── text_utils.py               # 文本处理工具
│
├── templates/                      # 模板文件
│   ├── question_templates.json     # 问题模板
│   ├── obfuscation_patterns.json  # 模糊化模式
│   └── trajectory_templates.json   # 轨迹模板
│
├── input_texts/                    # 输入文本文件夹
│   └── (放置领域文本文件)
│
└── output_dataset/                 # 输出数据集文件夹
    └── (生成的数据集文件)
```

## 核心模块说明

### 1. KnowledgeGraphBuilder
- 从原始文本构建知识图谱
- 支持混合实体提取（NER、依存句法、规则）
- 关系抽取和图谱构建

### 2. SubgraphSampler（WebSailor核心）
- 拓扑采样：星型、链式、树型、环型子图
- 语义采样：基于语义相关性的子图
- 任务导向采样：故障诊断、工艺优化等特定任务子图

### 3. QuestionGenerator（WebSailor核心）
- 多种问题类型：事实型、推理型、多跳型、比较型
- 复杂度控制：简单、中等、复杂
- 可回答性保证和干扰项生成

### 4. ObfuscationProcessor（WebSailor核心）
- 实体抽象化：将具体实体替换为抽象描述
- 信息分散：在文本中分散插入相关信息
- 冗余注入：添加相似事实、相关背景等
- 歧义引入：代词、时间、空间、数量歧义

### 5. TrajectoryGenerator
- 多种推理模式：演绎、归纳、溯因、类比
- 包含正确路径和死胡同
- 置信度评分

### 6. DataSynthesizer
- 数据质量控制：长度过滤、去重、一致性检查
- 数据增强：问题改写、负例生成
- 格式化输出

## 与WebSailor的主要区别

1. **领域特化**：专门针对TCL工业领域进行优化，包括特定的实体类型、关系类型和问题模板

2. **实现细节**：
   - 使用中文NLP工具（jieba）进行文本处理
   - 针对工业领域的特殊模糊化模式
   - 包含领域特定的推理轨迹模板

3. **扩展功能**：
   - 支持多种图拓扑结构的采样
   - 更丰富的模糊化策略
   - 可配置的质量控制和数据增强

## 配置详解

### 知识图谱构建配置
```json
"knowledge_graph": {
  "entity_extraction": {
    "method": "hybrid",  // 混合方法
    "domain_entities": ["产品", "设备", "工艺", ...]
  }
}
```

### 子图采样配置
```json
"subgraph_sampling": {
  "sampling_strategies": [
    {
      "name": "topology_based",
      "weight": 0.4,
      "params": {
        "include_patterns": ["star", "chain", "tree", "cycle"]
      }
    }
  ]
}
```

### 模糊化配置
```json
"obfuscation": {
  "strategies": {
    "entity_abstraction": {
      "enabled": true,
      "level": "moderate"
    }
  }
}
```

## 最佳实践

1. **输入文本准备**
   - 确保文本质量高，包含丰富的领域知识
   - 文本应该涵盖多个方面（产品、工艺、质量等）
   - 建议总文本量在10MB以上

2. **参数调优**
   - 根据需求调整子图数量和复杂度
   - 平衡不同问题类型的比例
   - 适度使用模糊化，保持问题可解性

3. **质量检查**
   - 检查生成的问答对是否合理
   - 验证推理轨迹的逻辑性
   - 确保知识图谱的连通性

## 常见问题

### Q: 生成的数据量不够？
A: 可以通过以下方式增加：
- 增加`total_subgraphs`参数
- 启用数据增强功能
- 添加更多输入文本

### Q: 问题太简单或太复杂？
A: 调整`complexity_distribution`参数：
```json
"complexity_distribution": {
  "simple": 0.3,
  "medium": 0.5,
  "complex": 0.2
}
```

### Q: 模糊化后问题难以理解？
A: 降低模糊化级别或减少模糊化策略：
```json
"obfuscation": {
  "strategies": {
    "entity_abstraction": {
      "level": "low"  // 改为low
    }
  }
}
```

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

[添加许可证信息]

## 引用

如果使用本项目，请引用WebSailor原论文：
```
[WebSailor论文引用信息]
```