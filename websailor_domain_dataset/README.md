# TCL工业垂域知识推理数据集

基于WebSailor核心思想构造的TCL工业领域知识推理数据集，专注于复杂推理和模糊化处理。

## 🎯 项目概述

本项目实现了WebSailor的三大核心思想：
1. **子图采样** - 从整个知识图中抽取不同拓扑的子图作为问题候选基础
2. **问题生成** - 基于子图中节点与关系设计多样化QA问题，覆盖多种问题类型
3. **模糊化处理** - 模糊描述中间实体或关系，添加冗余或干扰信息，使问题信息密度高但精确信息少

## 🔧 WebSailor核心思想实现

### 1. 子图采样 (SubgraphSampler)
- **核心理念**: 每个子图代表一种"任务场景"，包含多个目标、干扰信息、隐含路径
- **采样策略**: 
  - 随机游走采样 - 获得连续路径结构
  - 广度优先采样 - 获得扇形扩展结构  
  - 深度优先采样 - 获得深层路径结构
  - 社区采样 - 获得密集连接子结构
  - 中心性采样 - 获得包含重要节点的子图
- **场景特征提取**: 识别潜在目标、干扰节点、隐含路径、多跳关系、模糊实体

### 2. 问题生成 (QuestionGenerator)  
- **问题类型覆盖**:
  - 事实查询 - 直接从图中获取答案
  - 关系查询 - 查询实体间的直接关系
  - 多跳推理 - 需要通过多个关系进行推理
  - 比较分析 - 涉及多个实体的比较
  - 推理判断 - 需要逻辑推理和因果分析
  - 路径查找 - 寻找实体间的关系路径
- **TCL领域定制**: 针对工业制造场景的专业问题模板

### 3. 模糊化处理 (ObfuscationProcessor)
- **实体模糊化**: 使用指代词、上位词、描述性短语替换具体实体
- **关系模糊化**: 将具体关系替换为模糊描述
- **干扰信息添加**: 插入相关但非必要的信息，增加推理难度
- **指代歧义**: 使用指代词创造多候选答案的不确定性
- **数值时间模糊化**: 模糊化精确数值和时间表达

## 📁 项目结构

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
│   ├── subgraph_sampler.py         # 子图采样器（WebSailor核心）
│   ├── question_generator.py       # 问题生成器（WebSailor核心）
│   ├── obfuscation_processor.py    # 模糊化处理器（WebSailor核心）
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
│   ├── tcl_sample_text.txt         # TCL示例文本
│   └── ...
├──
├── output_dataset/                 # 输出数据集文件夹
│   ├── qa_pairs.json               # QA对数据
│   ├── trajectories.json           # 推理轨迹数据
│   ├── knowledge_graphs.json       # 知识图谱数据
│   └── statistics.json             # 统计信息
├──
├── templates/                      # 模板文件
│   ├── question_templates.json     # 问题模板
│   ├── obfuscation_patterns.json  # 模糊化模式
│   └── trajectory_templates.json  # 轨迹模板
└──
└── examples/                       # 示例文件
    ├── sample_input/               # 示例输入
    ├── sample_output/              # 示例输出
    └── README_examples.md          # 示例说明
```

## 🚀 运行流程

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 如果使用GPU加速（可选）
pip install torch-geometric dgl
```

### 2. 配置参数
编辑 `config.json` 文件，调整以下关键参数：
- `num_subgraphs`: 子图采样数量
- `target_qa_pairs`: 目标QA对数量  
- `obfuscation_ratio`: 模糊化比例
- `question_types`: 问题类型列表

### 3. 准备输入数据
将TCL工业领域文本放入 `input_texts/` 目录，支持格式：
- `.txt` - 纯文本文件
- `.json` - 结构化文本数据
- `.csv` - 表格化数据

### 4. 运行数据集构造
```bash
# 完整流程运行
python main.py --config config.json

# 分步运行示例
python main.py --step knowledge_graph    # 仅构建知识图谱
python main.py --step subgraph_sampling  # 仅进行子图采样
python main.py --step question_generation # 仅生成问题
python main.py --step obfuscation        # 仅进行模糊化处理
```

### 5. 输出结果
运行完成后，在 `output_dataset/` 目录下生成：
- `tcl_dataset_complete.json` - 完整数据集
- `qa_pairs.json` - QA对格式
- `trajectories.json` - 推理轨迹
- `statistics.json` - 统计信息
- `dataset.csv` - CSV格式（简化版）

## 📊 数据集特色

### WebSailor特征覆盖
- ✅ **子图采样覆盖率**: 100% (所有数据来自子图采样)
- ✅ **问题生成覆盖率**: 100% (所有问题都是生成的)
- ✅ **模糊化覆盖率**: 可配置 (默认60%)
- ✅ **推理轨迹覆盖率**: 可配置 (默认80%)

### 数据质量指标
- **问题类型多样性**: 6种主要类型，均衡分布
- **难度梯度**: 1-5级难度，符合认知负荷理论
- **推理复杂度**: 支持1-5跳多步推理
- **模糊化程度**: 可量化的不确定性水平

### TCL工业领域特色
- **专业术语覆盖**: 产品、技术、工艺、材料、设备等
- **场景真实性**: 基于真实制造流程和问题
- **知识深度**: 涵盖技术原理、工艺流程、质量控制等

## 🔍 核心算法详解

### 子图采样算法
```python
# 伪代码示例
def sample_subgraph(knowledge_graph, strategy):
    """WebSailor子图采样核心算法"""
    # 1. 选择采样策略
    if strategy == "random_walk":
        nodes = random_walk_sampling(graph)
    elif strategy == "community_based":
        nodes = community_based_sampling(graph)
    # ... 其他策略
    
    # 2. 构建子图
    subgraph = graph.subgraph(nodes)
    
    # 3. 提取场景特征 (WebSailor核心)
    scenario_features = extract_scenario_features(subgraph)
    # - 潜在目标实体
    # - 干扰节点
    # - 隐含路径
    # - 多跳关系
    # - 模糊实体候选
    
    return subgraph, scenario_features
```

### 模糊化处理算法
```python
# 伪代码示例  
def obfuscate_question(question, subgraph, scenario_features):
    """WebSailor模糊化处理核心算法"""
    # 1. 实体模糊化
    for entity in question_entities:
        if entity in ambiguous_entities:
            # 使用更强的模糊化
            obfuscated_form = select_strong_obfuscation(entity)
        else:
            obfuscated_form = select_regular_obfuscation(entity)
    
    # 2. 添加干扰信息
    noise_entities = scenario_features['interference_nodes']
    add_noise_information(question, noise_entities)
    
    # 3. 增加指代歧义
    add_pronoun_ambiguity(question)
    
    return obfuscated_question
```

## 📈 评估指标

### WebSailor完整性评估
- **子图多样性**: 不同采样策略的分布均匀性
- **问题复杂度**: 平均推理跳数、涉及实体数
- **模糊化效果**: 文本变化程度、不确定性水平
- **推理一致性**: 轨迹逻辑完整性、证据充分性

### 数据集质量评估
- **覆盖度**: 知识点覆盖率、问题类型分布
- **难度分布**: 认知负荷梯度、挑战性平衡
- **真实性**: 领域专业性、场景合理性
- **可用性**: 格式规范性、标注完整性

## 🤖 与WebSailor的差异对比

| 特性 | 原始WebSailor | 本项目实现 |
|------|---------------|------------|
| **应用领域** | 通用网络文本 | TCL工业垂域 |
| **知识图谱** | 大规模通用KG | 领域专业KG |
| **子图采样** | 基础随机采样 | 5种策略+质量评估 |
| **问题生成** | 模板化生成 | 6类型+领域定制 |
| **模糊化处理** | 简单实体替换 | 多层次模糊化系统 |
| **推理轨迹** | 基础步骤记录 | 详细证据链+置信度 |
| **输出格式** | 单一JSON | 多格式+统计分析 |
| **质量控制** | 基础过滤 | 多维度质量评估 |

## 🔧 自定义扩展

### 添加新的采样策略
```python
# 在 SubgraphSampler 中添加新方法
def _custom_sampling_strategy(self, graph):
    """自定义采样策略"""
    # 实现你的采样逻辑
    return sampled_nodes
```

### 定制问题模板
编辑 `templates/question_templates.json`，添加新的问题模板：
```json
{
  "custom_templates": [
    "在{context}环境下，{entity}如何{action}？",
    "考虑到{constraint}，{entity}的{property}是多少？"
  ]
}
```

### 扩展模糊化模式
在 `ObfuscationProcessor` 中添加新的模糊化规则：
```python
def _custom_obfuscation(self, text):
    """自定义模糊化处理"""
    # 实现特定的模糊化逻辑
    return obfuscated_text
```

## 📚 引用

如果使用本数据集，请引用：
```bibtex
@dataset{tcl_websailor_dataset,
  title={TCL工业垂域知识推理数据集},
  author={基于WebSailor核心思想实现},
  year={2024},
  description={专注于复杂推理和模糊化处理的工业领域数据集}
}
```

## 🤝 贡献指南

欢迎贡献代码和改进建议：
1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件

## 📞 联系方式

- 项目维护者: [您的姓名]
- 邮箱: [您的邮箱]
- 项目地址: [GitHub链接]

---

**注意**: 本项目基于WebSailor的核心思想进行实现，专注于TCL工业垂域的知识推理数据集构造。所有核心算法都包含详细的中文注释，便于理解和扩展。