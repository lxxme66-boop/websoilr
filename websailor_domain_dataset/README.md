# WebSailor Domain Dataset

本项目实现了WebSailor核心思想的数据构造流程，适用于TCL工业等垂域：

- **子图采样**：从知识图中采样不同拓扑子图，模拟多样任务场景。
- **问题生成**：基于子图节点与关系，自动生成多类型QA问题。
- **模糊化处理**：对问题进行模糊描述，添加干扰信息，提升数据复杂度。

## 快速开始

```bash
pip install -r requirements.txt
python main.py
```

输出数据集位于 `output_dataset/qa_pairs.json`。