# WebSailor TCL工业垂域数据集构建系统

基于WebSailor方法论的TCL工业垂域高质量数据集自动构建系统。

## 项目概述

本项目实现了WebSailor数据构建方法论在TCL工业垂直领域的应用，通过知识图谱构建、子图采样、问题生成、信息模糊化和推理轨迹生成等核心技术，自动构建高质量的领域特定数据集，用于训练和评估大语言模型在复杂信息搜索和推理任务上的能力。

## WebSailor方法论核心思想

WebSailor是一种先进的后训练方法，旨在让大语言模型具备处理复杂信息搜索任务的能力。其核心思想包括：

### 1. 子图采样（Subgraph Sampling）
- **目的**：从整个知识图谱中抽取不同拓扑结构的子图作为问题候选基础
- **特点**：每个子图代表一种"任务场景"，可能包含：
  - 多个潜在目标
  - 干扰信息
  - 隐含的推理路径
- **采样策略**：
  - 随机游走（Random Walk）：模拟信息搜索路径
  - 广度优先搜索（BFS）：构建局部知识结构
  - 社区检测（Community-based）：基于语义聚类采样

### 2. 问题生成（Question Generation）
- **基于子图结构**：根据子图中的节点与关系设计QA问题
- **问题类型覆盖**：
  - 事实性问题（Factual）
  - 推理性问题（Reasoning）
  - 多跳问题（Multi-hop）
  - 比较性问题（Comparative）
  - 因果性问题（Causal）

### 3. 信息模糊化（Information Obfuscation）
- **实体模糊**：将具体实体描述替换为模糊指代（如"这位领导人"可能指向多个节点）
- **关系模糊**：将明确的关系描述改为含糊表达
- **噪声注入**：添加冗余或干扰信息，使问题信息密度高但精确信息少
- **目的**：增加推理难度，训练模型处理不确定性

### 4. 推理轨迹生成（Trajectory Generation）
- **多种推理类型**：
  - 演绎推理（Deductive）
  - 归纳推理（Inductive）
  - 溯因推理（Abductive）
  - 类比推理（Analogical）
- **步骤化展示**：生成从问题到答案的详细推理步骤

## TCL工业垂域特点

与通用WebSailor方法相比，TCL工业垂域数据集具有以下特点：

### 1. 领域特定知识
- **产品知识**：TCL电视、空调、冰箱、洗衣机等产品线
- **技术术语**：量子点技术、Mini LED、AIoT、画质引擎等
- **行业标准**：能效等级、环保认证、质量标准等

### 2. 多语言支持
- 同时支持中文和英文文本处理
- 跨语言实体对齐和关系抽取
- 双语问答对生成

### 3. 工业场景优化
- **产品参数推理**：基于技术参数的比较和推理
- **供应链关系**：零部件、供应商、制造工艺之间的复杂关系
- **技术演进路径**：产品迭代和技术升级的时序推理

### 4. 实体类型扩展
- 产品型号（如TCL-65Q10G）
- 技术参数（如4K分辨率、120Hz刷新率）
- 工业流程（如生产工艺、质检标准）

## 系统架构

```
websailor_domain/
├── core/                          # 核心模块
│   ├── knowledge_graph_builder.py # 知识图谱构建
│   ├── subgraph_sampler.py       # 子图采样（WebSailor核心）
│   ├── question_generator.py      # 问题生成（WebSailor核心）
│   ├── obfuscation_processor.py   # 模糊化处理（WebSailor核心）
│   ├── trajectory_generator.py    # 推理轨迹生成
│   └── data_synthesizer.py        # 数据综合器
├── utils/                         # 工具模块
│   ├── nlp_utils.py              # NLP处理工具
│   ├── graph_utils.py            # 图处理工具
│   └── text_utils.py             # 文本处理工具
├── config.json                    # 配置文件
├── main.py                        # 主程序入口
└── requirements.txt               # 依赖包列表
```

## 安装与配置

### 1. 环境要求
- Python 3.8+
- CUDA 11.7+（用于GPU加速）
- 至少32GB内存（推荐64GB）
- 至少100GB磁盘空间

### 2. 安装步骤

```bash
# 克隆项目
git clone https://github.com/your-repo/websailor_domain.git
cd websailor_domain

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 下载必要的NLP模型
python -m spacy download zh_core_web_sm
python -m nltk.downloader punkt stopwords
```

### 3. 模型配置

本项目使用多个专门的大语言模型，请确保以下模型已下载到指定路径：

- **专家模型**（轨迹生成）：`/mnt/storage/models/Qwen/Qwen3-32B-Instruct`
- **QA生成模型**：`/mnt/storage/models/Qwen/Qwen2.5-14B-Instruct`
- **推理重建模型**：`/mnt/storage/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- **知识图谱提取模型**：`/mnt/data/LLM/lhy/models/fintuned_embedding/fv6`

如需修改模型路径，请编辑 `config.json` 文件。

## 使用方法

### 1. 准备输入数据

在 `input_texts/` 目录下放置TCL工业领域的文本文件（支持.txt格式）：

```
input_texts/
├── tcl_tv_products.txt          # 电视产品介绍
├── tcl_technology_report.txt    # 技术报告
├── tcl_industry_news_en.txt     # 英文行业新闻
└── ...
```

### 2. 运行数据集生成

#### 基本用法

```bash
python main.py --input_dir input_texts --output_dir output_dataset
```

#### 高级参数

```bash
python main.py \
    --input_dir input_texts \
    --output_dir output_dataset \
    --num_subgraphs 500 \          # 采样子图数量
    --questions_per_subgraph 10 \  # 每个子图生成的问题数
    --config custom_config.json \  # 自定义配置文件
    --log_level DEBUG              # 日志级别
```

### 3. 输出结果

生成的数据集将保存在 `output_dataset/` 目录：

```
output_dataset/
├── knowledge_graphs.json    # 构建的知识图谱
├── train.jsonl             # 训练集
├── val.jsonl               # 验证集
├── test.jsonl              # 测试集
├── statistics.json         # 数据集统计信息
└── run_config.json         # 运行配置记录
```

#### 数据格式示例

```json
{
  "id": "tcl_qa_001",
  "question": "这家公司最新推出的量子点电视采用了什么显示技术，相比传统LED有哪些优势？",
  "answer": "TCL最新的量子点电视采用了QD-Mini LED技术...",
  "difficulty": 0.7,
  "question_type": "comparative",
  "language": "zh",
  "obfuscation_level": 0.6,
  "trajectory": {
    "reasoning_type": "deductive",
    "steps": [
      {
        "step": 1,
        "action": "识别关键实体",
        "thought": "问题中提到'这家公司'和'量子点电视'...",
        "result": "TCL, QD-Mini LED技术"
      },
      ...
    ]
  },
  "source_subgraph": {
    "nodes": [...],
    "edges": [...],
    "topology": "star"
  }
}
```

## 配置说明

### config.json 主要配置项

```json
{
  "models": {
    "expert_model": {
      "path": "模型路径",
      "max_length": 8192,
      "temperature": 0.7
    },
    ...
  },
  "data_settings": {
    "languages": ["zh", "en"],
    "max_subgraph_size": 20,
    "min_subgraph_size": 3,
    "subgraph_sampling_strategies": ["random_walk", "bfs", "community"],
    "obfuscation_levels": [0.3, 0.5, 0.7, 0.9],
    "question_types": ["factual", "reasoning", "multi_hop", "comparative", "causal"]
  },
  "websailor_params": {
    "uncertainty_threshold": 0.6,
    "information_density": 0.8,
    "semantic_complexity": 0.7
  }
}
```

## 评估与分析

### 数据集质量指标

运行完成后，`statistics.json` 包含详细的数据集统计信息：

- **知识图谱规模**：节点数、边数、密度
- **子图分布**：拓扑类型、复杂度分布
- **问题分布**：类型、难度、语言分布
- **模糊化效果**：应用的模糊化技术统计
- **推理轨迹质量**：平均步数、连贯性评分

### 可视化分析

```python
# 可视化知识图谱
from utils.graph_utils import visualize_graph, load_graph_from_json

kg = load_graph_from_json("output_dataset/knowledge_graphs.json")
visualize_graph(kg, output_path="kg_visualization.png")
```

## 常见问题

### Q1: 如何处理内存不足的问题？
A: 可以通过减少 `num_subgraphs` 或 `batch_size` 参数来降低内存使用。

### Q2: 如何添加新的领域特定术语？
A: 编辑 `utils/nlp_utils.py` 中的 `TCL_DOMAIN_TERMS` 字典。

### Q3: 如何调整问题难度？
A: 修改 `config.json` 中的 `obfuscation_levels` 和 `semantic_complexity` 参数。

### Q4: 支持哪些文本格式？
A: 目前支持纯文本(.txt)格式，系统会自动检测编码（UTF-8, GBK等）。

## 扩展开发

### 添加新的问题类型

在 `core/question_generator.py` 中：

```python
def _generate_custom_question(self, subgraph: nx.DiGraph, features: Dict) -> Dict:
    # 实现自定义问题生成逻辑
    pass
```

### 添加新的模糊化策略

在 `core/obfuscation_processor.py` 中：

```python
def _custom_obfuscation(self, text: str, level: float) -> str:
    # 实现自定义模糊化逻辑
    pass
```

## 引用

如果您使用本项目，请引用：

```bibtex
@software{websailor_tcl,
  title = {WebSailor TCL Industrial Domain Dataset Construction System},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-repo/websailor_domain}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

- 项目维护者：[Your Name]
- Email: your.email@example.com
- Issues: https://github.com/your-repo/websailor_domain/issues