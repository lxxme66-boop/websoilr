# WebSailor LDD（垂域数据集构建）模块功能与调用流程分析

## 一、系统概述

WebSailor是一个基于知识图谱的垂域（领域特定）数据集自动构建系统，专门用于生成高质量的训练数据，以提升大语言模型在复杂信息搜索和推理任务上的能力。系统实现了WebSailor论文中的核心数据构造思想，并针对TCL工业领域进行了优化。

## 二、核心模块功能详解

### 1. DataSynthesizer（数据综合器）
**位置**: `core/data_synthesizer.py`

**功能**：
- 作为整个系统的协调器，整合所有组件
- 管理完整的数据集生成流程
- 控制数据集的分割（训练集/验证集/测试集）
- 统计和保存生成结果

**主要方法**：
- `synthesize_dataset()`: 执行完整的数据集合成流程
- `_load_input_texts()`: 加载输入文本
- `_split_dataset()`: 数据集分割
- `_save_dataset()`: 保存最终数据集

### 2. KnowledgeGraphBuilder（知识图谱构建器）
**位置**: `core/knowledge_graph_builder.py`

**功能**：
- 从原始文本中提取实体和关系
- 构建领域知识图谱
- 支持中英文文本处理
- 实体类型识别和关系抽取

**TCL工业垂域特定实体类型**：
- 产品、技术、工艺、材料、设备
- 标准、专利、公司、人员、项目
- 参数、指标、方法、流程、组件

**TCL工业垂域特定关系类型**：
- 使用、包含、生产、研发、依赖
- 改进、替代、认证、合作、应用于
- 基于、优化、集成、测试、制造

**主要方法**：
- `build_from_texts()`: 从文本构建知识图谱
- `_extract_entities()`: 实体抽取
- `_extract_relations()`: 关系抽取
- `_merge_entities()`: 实体消歧和合并

### 3. SubgraphSampler（子图采样器）
**位置**: `core/subgraph_sampler.py`

**功能**：
- WebSailor核心组件之一
- 从完整知识图谱中采样不同拓扑结构的子图
- 每个子图代表一种"任务场景"
- 支持多种采样策略

**采样策略**：
1. **随机游走（Random Walk）**：
   - 模拟信息搜索路径
   - 创建包含隐含路径的子图
   
2. **广度优先搜索（BFS）**：
   - 构建以某节点为中心的局部知识结构
   - 适合生成局部推理问题
   
3. **社区检测（Community-based）**：
   - 基于图的社区结构采样
   - 创建语义相关的子图

**拓扑类型**：
- chain（链式）：A→B→C→D
- star（星型）：中心节点连接多个周边节点
- tree（树形）：层次结构
- cycle（环形）：包含循环
- mixed（混合）：复杂拓扑
- dense（密集）：高连接度
- sparse（稀疏）：低连接度

### 4. QuestionGenerator（问题生成器）
**位置**: `core/question_generator.py`

**功能**：
- WebSailor核心组件之一
- 基于子图结构生成多样化的问题
- 覆盖不同难度和类型的问题
- 包含问题合理性验证机制

**问题类型**：
1. **事实型（Factual）**：
   - 直接查询实体属性或关系
   - 例："TCL Q10G使用了什么显示技术？"

2. **比较型（Comparison）**：
   - 比较多个实体的属性
   - 例："Mini LED和OLED技术的主要区别是什么？"

3. **推理型（Reasoning）**：
   - 需要逻辑推理才能回答
   - 例："如果要提高产品能效等级，应该优化哪些技术参数？"

4. **多跳型（Multi-hop）**：
   - 需要多步推理
   - 例："生产使用量子点技术的65寸电视需要哪些关键材料？"

5. **因果型（Causal）**：
   - 探索因果关系
   - 例："采用新型背光技术对产品功耗有什么影响？"

**生成流程**：
1. 分析子图结构和复杂度
2. 选择合适的问题类型
3. 使用LLM生成初始问题
4. 验证问题合理性
5. 优化和改进问题质量

### 5. ObfuscationProcessor（模糊化处理器）
**位置**: `core/obfuscation_processor.py`

**功能**：
- WebSailor核心组件之一
- 增加问题的不确定性和推理难度
- 模糊化实体和关系描述
- 注入干扰信息

**模糊化策略**：
1. **实体模糊化**：
   - 具体实体 → 模糊指代
   - 例："TCL Q10G" → "这款高端产品"
   
2. **关系模糊化**：
   - 明确关系 → 含糊表达
   - 例："使用了" → "与...有关"
   
3. **噪声注入**：
   - 添加相关但非必要的信息
   - 增加信息密度但减少精确性
   
4. **上下文扩展**：
   - 添加背景信息增加复杂度
   - 例：加入行业趋势、技术发展等信息

### 6. TrajectoryGenerator（推理轨迹生成器）
**位置**: `core/trajectory_generator.py`

**功能**：
- 生成从问题到答案的推理轨迹
- 展示详细的推理步骤
- 支持多种推理模式
- 用于训练模型的推理能力

**推理模式**：
1. **演绎推理（Deductive）**：
   - 从一般到特殊
   - 基于规则和逻辑推导

2. **归纳推理（Inductive）**：
   - 从特殊到一般
   - 基于模式识别和总结

3. **溯因推理（Abductive）**：
   - 寻找最佳解释
   - 基于假设和验证

4. **类比推理（Analogical）**：
   - 基于相似性
   - 跨领域知识迁移

## 三、系统调用流程

### 主流程（main.py）

```
1. 初始化阶段
   ├── 加载配置文件（config.json）
   ├── 设置日志系统
   └── 创建输出目录

2. 组件初始化
   └── DataSynthesizer初始化
       ├── KnowledgeGraphBuilder初始化
       ├── SubgraphSampler初始化
       ├── QuestionGenerator初始化
       ├── ObfuscationProcessor初始化
       └── TrajectoryGenerator初始化

3. 数据生成流程
   ├── 步骤1：文本加载
   │   └── 读取input_texts目录下的所有文本文件
   │
   ├── 步骤2：知识图谱构建
   │   ├── 文本预处理（分词、NER）
   │   ├── 实体抽取（Entity Extraction）
   │   ├── 关系抽取（Relation Extraction）
   │   └── 图谱构建（NetworkX Graph）
   │
   ├── 步骤3：子图采样
   │   ├── 随机游走采样（1/3）
   │   ├── BFS采样（1/3）
   │   ├── 社区检测采样（1/3）
   │   └── 子图验证和筛选
   │
   ├── 步骤4：问题生成
   │   ├── 遍历每个子图
   │   ├── 分析子图结构
   │   ├── 选择问题类型
   │   ├── 使用LLM生成问题
   │   └── 问题质量验证
   │
   ├── 步骤5：模糊化处理
   │   ├── 实体模糊化
   │   ├── 关系模糊化
   │   ├── 噪声注入
   │   └── 上下文扩展
   │
   ├── 步骤6：推理轨迹生成
   │   ├── 选择推理类型
   │   ├── 生成推理步骤
   │   ├── 构建推理链
   │   └── 轨迹验证
   │
   └── 步骤7：数据集整合
       ├── 合并所有生成的数据
       ├── 数据集分割（8:1:1）
       └── 格式化输出

4. 结果保存
   ├── knowledge_graph.json（知识图谱）
   ├── subgraphs.json（子图集合）
   ├── train.json（训练集）
   ├── val.json（验证集）
   ├── test.json（测试集）
   ├── statistics.json（统计信息）
   └── run_config.json（运行配置）
```

### 详细调用链

```
main.py
└── DataSynthesizer.synthesize_dataset()
    ├── _load_input_texts()
    │   └── 读取文本文件
    │
    ├── KnowledgeGraphBuilder.build_from_texts()
    │   ├── _preprocess_text()
    │   ├── _extract_entities()
    │   ├── _extract_relations()
    │   └── _build_graph()
    │
    ├── SubgraphSampler.sample_subgraphs()
    │   ├── _random_walk_sampling()
    │   ├── _bfs_sampling()
    │   ├── _community_based_sampling()
    │   └── _filter_valid_subgraphs()
    │
    ├── QuestionGenerator.generate_questions()
    │   ├── _analyze_subgraph()
    │   ├── _select_question_type()
    │   ├── _generate_with_llm()
    │   └── _validate_question()
    │
    ├── ObfuscationProcessor.obfuscate_questions()
    │   ├── _obfuscate_entities()
    │   ├── _obfuscate_relations()
    │   ├── _add_noise()
    │   └── _expand_context()
    │
    ├── TrajectoryGenerator.generate_trajectories()
    │   ├── _select_reasoning_type()
    │   ├── _generate_trajectory()
    │   └── _validate_trajectory()
    │
    └── _save_dataset()
        ├── _split_dataset()
        └── 保存各类文件
```

## 四、数据流转

### 1. 输入数据
- 原始文本文件（TCL工业领域文档）
- 配置文件（config.json）

### 2. 中间数据
- **知识图谱**：NetworkX图结构，包含实体和关系
- **子图集合**：从大图中采样的小图，每个5-50个节点
- **初始问答对**：基于子图生成的原始问题和答案
- **模糊化问答对**：经过模糊处理的问题

### 3. 输出数据
- **训练集**：包含问题、答案、推理轨迹
- **验证集**：用于模型调优
- **测试集**：用于最终评估
- **元数据**：统计信息、配置记录

## 五、关键特性

### 1. WebSailor核心思想实现
- **子图采样**：创建多样化的任务场景
- **信息模糊化**：增加推理难度
- **推理轨迹**：提供详细的思考过程

### 2. TCL工业垂域优化
- **领域词典**：包含TCL特定术语
- **实体类型**：针对工业场景定制
- **关系类型**：反映工业领域特点

### 3. 质量保证机制
- **问题验证**：确保生成问题的合理性
- **子图筛选**：保证子图的有效性
- **轨迹验证**：检查推理步骤的逻辑性

### 4. 可扩展性
- **模块化设计**：各组件独立，易于替换
- **配置驱动**：通过配置文件控制行为
- **多语言支持**：同时处理中英文

## 六、使用示例

```bash
# 基本使用
python main.py --input_dir input_texts --output_dir output_dataset

# 自定义配置
python main.py --config my_config.json --input_dir data/raw --output_dir data/processed

# 调整生成规模
python main.py --input_dir input_texts --output_dir output_dataset \
               --num_subgraphs 500 --questions_per_subgraph 10
```

## 七、配置说明

主要配置项：
- `knowledge_graph`: 知识图谱构建参数
- `subgraph_sampling`: 子图采样策略
- `question_generation`: 问题生成配置
- `obfuscation`: 模糊化级别设置
- `trajectory_generation`: 推理轨迹参数
- `models`: 使用的AI模型配置

## 八、总结

WebSailor LDD系统通过六个核心模块的协同工作，实现了从原始文本到高质量训练数据集的自动化构建。系统遵循WebSailor方法论，通过子图采样、问题生成、信息模糊化和推理轨迹生成等步骤，创建了适合训练大语言模型进行复杂推理的数据集。针对TCL工业垂域的特定优化使得生成的数据更贴近实际应用场景。