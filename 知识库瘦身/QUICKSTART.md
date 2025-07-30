# 快速开始指南

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 准备数据

将您的文档数据整理成JSON格式，保存为 `data.json`：

```json
[
  {
    "doc_id": "文档唯一ID",
    "论文标题": "文档标题",
    "Abstract": "文档内容或摘要"
  }
]
```

## 3. 运行去重

```bash
python optimized_deduplication.py
```

## 4. 查看结果

程序运行完成后，查看生成的文件：
- `unique_documents.json` - 去重后的文档
- `duplicate_documents.csv` - 重复文档详情
- `*.png` - 可视化分析图表

## 5. 运行示例

如果想先看看效果，可以运行示例程序：

```bash
python example_usage.py
```

## 常见问题

### Q: 如何调整去重的严格程度？
A: 修改创建 `DocumentDeduplicator` 时的阈值参数：
- `tfidf_threshold`: 降低此值会识别更多相似文档（默认0.7）
- `minhash_threshold`: 降低此值会增加候选文档对（默认0.5）

### Q: 处理大量文档时速度慢怎么办？
A: 确保 `n_jobs=-1` 以使用所有CPU核心，或者适当降低 `num_perm` 值（如64）

### Q: 中文显示有问题怎么办？
A: 程序会自动尝试多种中文字体，如果仍有问题，请安装中文字体包