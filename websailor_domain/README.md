# WebSailor TCL工业垂域数据集构建系统

基于WebSailor方法论的TCL工业领域专用数据集构建系统，用于生成高质量的复杂推理训练数据。

## 项目概述

本项目将WebSailor的先进数据构建方法应用于TCL工业垂直领域，通过知识图谱构建、子图采样、问题生成、模糊化处理和推理轨迹生成等步骤，创建适合训练大语言模型进行复杂工业领域推理的数据集。

### 核心特性

1. **子图采样（Subgraph Sampling）**
   - 从知识图谱中提取不同拓扑结构的子图（链式、星型、树型、环型、混合型）
   - 每个子图代表一个"任务场景"，包含多个目标、干扰信息和隐含路径
   - 确保数据集覆盖多样化的推理模式

2. **问题生成（Question Generation）**
   - 基于子图中的节点和关系设计QA问题
   - 覆盖事实型、比较型、推理型、多跳型、反事实型、时序型、因果型等问题类型
   - 使用大语言模型生成自然、流畅的问题表述

3. **模糊化处理（Obfuscation Processing）**
   - 将具体实体替换为模糊描述（如"这种技术"、"某个产品"）
   - 添加冗余或干扰信息，增加信息密度但降低精确度
   - 引入歧义和多重指代，提升推理难度

4. **推理轨迹生成（Trajectory Generation）**
   - 使用专家模型生成详细的推理步骤
   - 支持链式思维、树式思维、图式思维等多种推理格式
   - 提供透明的推理过程，便于模型学习

## 与原始WebSailor的区别

### 1. 领域适配
- **原始WebSailor**: 面向通用Web信息检索和推理任务
- **TCL版本**: 专注于TCL工业领域，包括显示技术、智能制造、半导体、新材料等

### 2. 知识图谱构建
- **原始WebSailor**: 从Web文档提取通用知识
- **TCL版本**: 
  - 定制化的实体类型：产品、技术、工艺、材料、设备、标准、专利等
  - 工业特定的关系类型：使用、生产、研发、依赖、改进等
  - 集成TCL专业术语词典

### 3. 问题复杂度
- **原始WebSailor**: 偏重信息检索和简单推理
- **TCL版本**: 
  - 强化技术参数比较和工艺流程推理
  - 增加产业链分析和技术演进问题
  - 包含标准合规性和专利相关问题

### 4. 语言特性
- **原始WebSailor**: 主要英文
- **TCL版本**: 中英文混合，保留技术术语的原始表达

### 5. 模型配置
- 使用多个专门的大语言模型：
  - 专家模型（轨迹生成）：Qwen3-32B-Instruct
  - QA生成模型：Qwen2.5-14B-Instruct  
  - 推理重建模型：DeepSeek-R1-Distill-Qwen-32B
  - 知识图谱提取模型：fintuned_embedding/fv6

## 安装要求

### 系统要求
- Python 3.8+
- CUDA 11.0+（用于GPU加速）
- 至少32GB内存
- 100GB可用磁盘空间

### 安装步骤

1. 克隆项目
```bash
git clone <repository_url>
cd websailor_domain
```

2. 创建虚拟环境
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

4. 下载spaCy中文模型（可选）
```bash
python -m spacy download zh_core_web_sm
```

## 使用方法

### 1. 准备输入数据

将TCL工业领域的文本文件放入 `input_texts/` 目录：
```bash
input_texts/
├── tcl_display_tech.txt      # 显示技术相关文本
├── tcl_smart_manufacturing.txt # 智能制造相关文本
└── ...                        # 其他领域文本
```

### 2. 配置参数

编辑 `config.json` 文件，调整以下关键参数：
- `models`: 各个LLM的路径配置
- `num_subgraph_samples`: 子图采样数量
- `obfuscation_rate`: 模糊化处理比例
- `train_ratio/val_ratio/test_ratio`: 数据集划分比例

### 3. 运行系统

#### 完整流程运行
```bash
python main.py --mode full --input_dir input_texts --output_dir output_dataset
```

#### 分步骤运行

1. 构建知识图谱
```bash
python main.py --mode build_kg --input_dir input_texts --output_dir output_dataset
```

2. 子图采样
```bash
python main.py --mode sample --output_dir output_dataset
```

3. 生成问题和轨迹
```bash
python main.py --mode generate --output_dir output_dataset
```

4. 综合数据集
```bash
python main.py --mode synthesize --output_dir output_dataset
```

### 4. 查看输出

生成的数据集将保存在 `output_dataset/` 目录：
```
output_dataset/
├── knowledge_graphs.json    # 构建的知识图谱
├── sampled_subgraphs.json  # 采样的子图
├── qa_pairs.json           # 生成的QA对（含轨迹）
├── train.json              # 训练集
├── val.json                # 验证集
├── test.json               # 测试集
└── statistics.json         # 数据集统计信息
```

## 数据格式

### QA对格式示例
```json
{
  "id": "qa_001",
  "question": "这种显示技术与传统工艺相比有什么优势？",
  "answer": "相比传统蒸镀工艺，该技术采用喷墨打印方式，大幅降低了生产成本...",
  "original_question": "印刷OLED技术与传统蒸镀工艺相比有什么优势？",
  "question_type": "comparison",
  "difficulty": 0.75,
  "subgraph": {
    "nodes": [...],
    "edges": [...]
  },
  "trajectory": {
    "reasoning_type": "deductive",
    "format": "chain_of_thought",
    "steps": [
      {
        "step": 1,
        "thought": "首先需要识别'这种显示技术'指的是什么...",
        "action": "identify_entity",
        "result": "根据上下文，这种显示技术指的是印刷OLED技术"
      },
      ...
    ]
  }
}
```

## 高级功能

### 自定义实体和关系类型

在 `config.json` 中修改：
```json
"knowledge_graph": {
  "entity_types": ["自定义类型1", "自定义类型2", ...],
  "relation_types": ["自定义关系1", "自定义关系2", ...]
}
```

### 添加领域特定模板

创建自定义问题模板文件：
```bash
templates/custom_questions.json
```

### 集成新的LLM

在相应的模块中修改模型加载逻辑，例如在 `question_generator.py` 中：
```python
def _load_qa_generator(self):
    # 添加新模型的加载代码
    pass
```

## 常见问题

### Q: 如何处理内存不足的问题？
A: 可以减少批处理大小或使用模型量化技术。在配置中调整 `batch_size` 参数。

### Q: 如何提高数据质量？
A: 
- 增加输入文本的质量和多样性
- 调整 `quality_checks` 中的阈值
- 使用更大的LLM模型

### Q: 如何加速处理过程？
A: 
- 使用GPU加速
- 启用多进程处理（在代码中设置 `num_workers`）
- 减少子图采样数量进行快速原型验证

## 贡献指南

欢迎提交Issue和Pull Request。提交代码前请确保：
1. 代码符合PEP 8规范
2. 添加适当的注释和文档
3. 通过所有测试用例

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 致谢

- WebSailor原始论文作者
- TCL研究院提供的领域知识支持
- 开源社区的各种工具和库