# WebSailor Domain-Specific Dataset Construction System

基于WebSailor核心思想构建的TCL工业垂域数据集生成系统

## 项目概述

本项目实现了WebSailor的三大核心思想，专门针对TCL工业领域进行了优化：

1. **子图采样** - 从知识图中抽取不同拓扑的子图作为问题候选基础
2. **问题生成** - 基于子图中节点与关系，设计多样化QA问题
3. **模糊化处理** - 模糊描述中间实体或关系，添加干扰信息增加推理难度

## 与原版WebSailor的区别

### 1. 领域专门化
- **原版WebSailor**: 通用网络信息检索
- **TCL版本**: 专注于TCL工业领域（显示技术、半导体、智能家居等）

### 2. 实体和关系类型
- **原版**: 通用实体（人物、地点、事件等）
- **TCL版**: 工业特定实体（Product、Technology、Component、Material、Process、Standard等）

### 3. 模糊化策略
- **原版**: 通用模糊化（"这个人"、"那个地方"）
- **TCL版**: 工业特定模糊化（"这款显示产品"、"某半导体组件"、"相关技术标准"）

### 4. 问题模板
- **原版**: 通用信息查询
- **TCL版**: 工业领域问题（技术参数、生产流程、标准认证、供应链关系等）

## 并行优化版本特性

### 1. 多模型并行加载
- 同时加载NER、关系抽取、问题生成等多个模型
- 自动分配GPU资源，支持多GPU并行
- 减少启动时间50%以上

### 2. 批处理优化
- 文本批量处理，提高GPU利用率
- 动态批次大小调整
- 内存高效模式支持

### 3. 异步处理流水线
- 不同处理阶段并行执行
- 异步IO减少等待时间
- 支持流式处理大规模数据

### 4. 性能监控
- 实时性能指标
- 进度条显示
- 检查点保存和恢复

## 项目结构

```
websailor_domain/
├── main.py                      # 主入口文件（基础版本）
├── main_parallel.py             # 并行优化版本入口
├── config.json                  # 基础配置文件
├── config_parallel.json         # 并行优化配置
├── requirements.txt             # 依赖包列表
│
├── core/                        # 核心模块
│   ├── __init__.py
│   ├── knowledge_graph_builder.py       # 知识图谱构建器
│   ├── enhanced_knowledge_graph_builder.py  # 增强版KG构建器
│   ├── subgraph_sampler.py             # 子图采样器（核心）
│   ├── question_generator.py           # 问题生成器（核心）
│   ├── obfuscation_processor.py        # 模糊化处理器（核心）
│   ├── trajectory_generator.py         # 推理轨迹生成器
│   ├── data_synthesizer.py             # 数据综合器
│   └── model_manager.py                # 模型管理器（并行加载）
│
├── utils/                       # 工具模块
│   ├── __init__.py
│   ├── nlp_utils.py            # NLP工具函数
│   ├── graph_utils.py          # 图处理工具
│   └── text_utils.py           # 文本处理工具
│
├── input_texts/                 # 输入文本文件夹
│   ├── sample_tcl_text.txt
│   └── ...
│
├── output_dataset/              # 输出数据集文件夹
│   ├── knowledge_graph.json
│   ├── dataset.json
│   ├── train.json
│   ├── validation.json
│   └── test.json
│
├── templates/                   # 模板文件
│   ├── question_templates.json
│   ├── obfuscation_patterns.json
│   └── trajectory_templates.json
│
└── examples/                    # 示例文件
    ├── sample_input/
    ├── sample_output/
    │   └── sample_dataset.json
    └── README_examples.md
```

## 运行流程

### 环境准备

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **下载预训练模型**（如果使用LLM）
```bash
# 下载中文BERT模型
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('bert-base-chinese'); AutoModel.from_pretrained('bert-base-chinese')"
```

### 数据准备

1. 将TCL领域相关文本放入 `input_texts/` 目录
2. 文本格式：每个txt文件包含一段或多段TCL相关描述

### 运行系统

#### 基础版本
```bash
# 使用默认配置运行
python main.py

# 指定输入输出目录
python main.py --input-dir ./my_texts --output-dir ./my_dataset

# 使用自定义配置
python main.py --config my_config.json --domain TCL_Industry
```

#### 并行优化版本
```bash
# 使用GPU并行处理
python main_parallel.py --use-gpu --async-mode

# 指定配置文件
python main_parallel.py --config config_parallel.json

# 调试模式
python main_parallel.py --debug
```

### 配置说明

主要配置项：
- `domain`: 领域名称
- `parallel`: 并行处理配置
- `models`: 模型配置
- `knowledge_graph`: 知识图谱构建参数
- `subgraph_sampling`: 子图采样策略
- `question_generation`: 问题生成配置
- `obfuscation`: 模糊化处理策略
- `trajectory_generation`: 推理轨迹生成
- `data_synthesis`: 数据综合输出

## 输出数据集格式

```json
{
  "domain": "TCL_Industry",
  "version": "1.0",
  "created_at": "2024-01-01T10:00:00",
  "samples": [
    {
      "id": "TCL_Industry_000001",
      "question": "某知名电子制造商推出的新型显示产品采用了什么技术？",
      "answer": "Mini-LED显示技术",
      "question_type": "single_hop",
      "difficulty": 0.3,
      "evidence_path": [...],
      "subgraph": {...},
      "trajectories": [...],
      "obfuscation_metadata": {...}
    }
  ],
  "knowledge_graph": {...},
  "statistics": {...}
}
```

## 核心算法说明

### 1. 子图采样算法
- **星型拓扑**: 以中心节点为核心的辐射结构
- **链式拓扑**: 适合生产流程的线性结构
- **树形拓扑**: 组件层级关系的树状结构
- **网状拓扑**: 复杂交叉关系的网络结构

### 2. 问题生成策略
- **单跳问题**: 直接关系查询
- **多跳问题**: 需要多步推理
- **比较问题**: 实体对比分析
- **聚合问题**: 统计类查询
- **约束问题**: 条件筛选查询

### 3. 模糊化技术
- **实体替换**: 具体名称→模糊描述
- **信息注入**: 添加相关但无用信息
- **关系模糊**: 明确关系→间接表达
- **上下文扩展**: 增加背景描述

## 性能优化建议

1. **GPU使用**
   - 使用 `--use-gpu` 启用GPU加速
   - 多GPU环境自动分配负载

2. **批处理大小**
   - 根据GPU内存调整 `batch_size`
   - 使用 `memory_efficient_mode` 处理大规模数据

3. **并行处理**
   - 调整 `max_workers` 参数
   - 启用 `async-mode` 提高吞吐量

4. **缓存优化**
   - 启用 `cache_enabled` 减少重复计算
   - 设置合适的 `cache_size_mb`

## 常见问题

1. **内存不足**
   - 减小批处理大小
   - 启用内存高效模式
   - 使用检查点恢复

2. **GPU利用率低**
   - 增加批处理大小
   - 启用混合精度训练
   - 使用异步处理模式

3. **处理速度慢**
   - 使用并行版本
   - 增加工作线程数
   - 优化数据预处理

## 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License