# WebSailor Domain-Specific Dataset Construction System

基于WebSailor核心思想的垂域数据集构建系统，专门用于TCL工业领域。

## 项目概述

本项目实现了WebSailor的三大核心思想：
1. **子图采样** - 从整个知识图中抽取不同拓扑的子图作为问题候选基础
2. **问题生成** - 基于子图中节点与关系，设计多样化的QA问题
3. **模糊化处理** - 添加冗余或干扰信息，使问题信息密度高但精确信息少

## 与原版WebSailor的区别

### 1. 领域特化
- **WebSailor**: 通用的网络信息搜索和推理数据集
- **本项目**: 专注于TCL工业垂直领域（显示技术、半导体制造、智能家居等）

### 2. 实体和关系类型
- **WebSailor**: 通用实体（人物、地点、事件等）
- **本项目**: 工业领域特定实体
  - Product（产品）
  - Technology（技术）
  - Component（组件）
  - Material（材料）
  - Process（工艺）
  - Standard（标准）
  - Company（公司）

### 3. 模糊化策略
- **WebSailor**: 通用的模糊化方法
- **本项目**: 领域特定的模糊化规则
  - "这款显示产品"代替具体产品名
  - "某知名电子制造商"代替公司名
  - 添加行业背景和技术演进信息作为干扰

### 4. 问题模板
- **WebSailor**: 通用问答模板
- **本项目**: 工业领域专用模板
  - 技术查询："{产品}使用什么技术？"
  - 供应链查询："{组件}的制造商是谁？"
  - 标准合规："{产品}符合哪些标准？"

## 项目结构

```
websailor_domain/
├── main.py                          # 主入口文件
├── config.json                      # 配置文件
├── requirements.txt                 # 依赖包列表
├── README.md                        # 项目说明
│
├── core/                           # 核心模块
│   ├── __init__.py
│   ├── knowledge_graph_builder.py   # 知识图谱构建器
│   ├── subgraph_sampler.py         # 子图采样器（WebSailor核心）
│   ├── question_generator.py       # 问题生成器（WebSailor核心）
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
├── input_texts/                    # 输入文本文件夹
│   ├── domain_text_1.txt
│   ├── domain_text_2.txt
│   └── ...
│
├── output_dataset/                 # 输出数据集文件夹
│   ├── qa_pairs.json
│   ├── trajectories.json
│   ├── knowledge_graphs.json
│   └── statistics.json
│
├── templates/                      # 模板文件
│   ├── question_templates.json     # 问题模板
│   ├── obfuscation_patterns.json  # 模糊化模式
│   └── trajectory_templates.json  # 轨迹模板
│
└── examples/                       # 示例文件
    ├── sample_input/
    ├── sample_output/
    └── README_examples.md
```

## 运行流程

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 下载必要的NLP模型（如需要）
python -m spacy download zh_core_web_sm
```

### 2. 准备输入数据

在 `input_texts/` 目录下放置TCL工业领域的文本文件，例如：
- 产品说明书
- 技术文档
- 行业报告
- 新闻资讯

文本示例：
```
TCL推出了新一代Mini-LED显示技术，该技术采用量子点材料，
能够实现更高的色彩准确度。这款显示器包含了先进的背光模组，
符合RoHS环保标准，由TCL华星光电制造。
```

### 3. 配置参数

编辑 `config.json` 文件，调整以下关键参数：
- `subgraph_sampling`: 子图采样策略和数量
- `question_generation`: 问题类型和模板
- `obfuscation`: 模糊化策略和强度
- `tcl_specific`: TCL领域特定配置

### 4. 运行数据集构建

```bash
# 基本运行
python main.py

# 指定参数运行
python main.py \
    --input-dir input_texts \
    --output-dir output_dataset \
    --domain TCL_Industry \
    --config config.json
```

### 5. 查看输出

生成的数据集将保存在 `output_dataset/` 目录：
- `TCL_Industry_dataset_YYYYMMDD.json`: 完整数据集
- `TCL_Industry_dataset_YYYYMMDD.jsonl`: JSONL格式
- `statistics.json`: 数据集统计信息

## 数据集格式

每个数据样本包含：
```json
{
  "id": "TCL_Industry_000001",
  "question": "某知名电子制造商推出的新型显示产品采用了什么技术？",
  "answer": "Mini-LED显示技术",
  "original_question": "TCL推出的显示器采用了什么技术？",
  "question_type": "single_hop",
  "difficulty": 0.3,
  "ambiguity_score": 0.4,
  "evidence_path": [
    {
      "source": "TCL显示器",
      "relation": "uses_technology",
      "target": "Mini-LED显示技术"
    }
  ],
  "subgraph": {
    "topology_type": "star",
    "nodes": ["TCL显示器", "Mini-LED显示技术", "量子点材料", ...],
    "edges": [...]
  },
  "trajectories": [
    {
      "reasoning_pattern": "deductive",
      "steps": [...],
      "is_successful": true
    }
  ]
}
```

## 核心算法说明

### 1. 子图采样（Subgraph Sampling）
- **星型拓扑**: 中心节点向外辐射，适合单实体查询
- **链式拓扑**: 线性关系链，适合生产流程推理
- **树形拓扑**: 层级结构，适合组件关系
- **网状拓扑**: 复杂交叉关系，适合多约束推理

### 2. 问题生成（Question Generation）
- **单跳问题**: 直接关系查询
- **多跳问题**: 需要多步推理
- **比较问题**: 对比多个实体
- **聚合问题**: 统计类查询
- **约束问题**: 多条件筛选

### 3. 模糊化处理（Obfuscation）
- **实体替换**: 用模糊描述代替具体名称
- **信息注入**: 添加相关但不必要的干扰信息
- **关系模糊化**: 使关系描述更加间接
- **上下文扩展**: 增加背景信息提高复杂度

## 评估指标

生成的数据集质量可通过以下指标评估：
- 问题多样性：不同问题类型的分布
- 推理复杂度：平均推理步数
- 模糊度分布：模糊化程度的合理性
- 答案准确性：答案与证据路径的一致性

## 扩展开发

### 添加新的实体类型
1. 在 `config.json` 中添加实体类型
2. 在 `knowledge_graph_builder.py` 中添加识别模式
3. 更新问题模板以支持新实体

### 自定义模糊化策略
1. 在 `obfuscation_processor.py` 中实现新策略
2. 在配置文件中注册策略
3. 设置策略的应用概率

### 集成外部知识库
1. 实现知识库接口
2. 在知识图谱构建时融合外部数据
3. 更新实体链接逻辑

## 注意事项

1. **数据质量**: 输入文本的质量直接影响生成数据集的质量
2. **计算资源**: 大规模数据集生成需要较多内存和计算时间
3. **领域适配**: 可根据具体领域调整实体类型和关系模式
4. **隐私保护**: 处理企业数据时注意脱敏处理

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议：
1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或合作意向，请联系项目维护者。

## 致谢

本项目基于WebSailor的核心思想开发，感谢原作者的开创性工作。