# DeepSeek翻译工具 - 改进版

## 🌟 主要改进

1. **修复长文档翻译不完整的问题**
   - 优化了文本分割算法，确保长文档能够被完整翻译
   - 提供三种分割策略：自动分割、段落分割、智能分割

2. **更好的分割控制**
   - 自动分割：按固定长度分割，但尽量在句子边界
   - 段落分割：保持段落完整性，适合结构化文档
   - 智能分割：优先保持段落和句子的完整性

3. **增强的进度显示**
   - 显示文件大小和分割片段数
   - 实时显示每个片段的翻译进度
   - 统计成功率和总耗时

## 📋 系统要求

- Python 3.8+
- CUDA支持的GPU（推荐）
- 至少16GB内存（运行7B模型）
- 足够的磁盘空间存储模型

## 🚀 快速开始

### 方法1：使用运行脚本（推荐）

```bash
# 给脚本添加执行权限
chmod +x run_translate.sh

# 运行脚本
./run_translate.sh
```

### 方法2：直接运行Python

```bash
# 安装依赖（如果还没安装）
pip install transformers torch accelerate

# 运行翻译程序
python3 translate_deepseek_improved.py
```

## 📖 使用说明

1. **选择模型**
   ```
   1. DeepSeek-R1-Distill-Qwen-1.5B (最快，适合快速翻译)
   2. DeepSeek-R1-Distill-Qwen-7B (推荐，平衡速度和质量)
   3. DeepSeek-R1-Distill-Qwen-14B (高质量，速度较慢)
   4. gte_Qwen2-7B-instruct (替代选项)
   ```

2. **输入文件夹路径**
   - 包含要翻译的英文txt文件的文件夹
   - 支持相对路径和绝对路径

3. **输出文件夹路径**
   - 默认为 `./chinese_translations`
   - 会自动创建不存在的文件夹

4. **选择分割策略**
   - **自动分割**（推荐）：每1000字符分割一次，在句子边界处断开
   - **段落分割**：保持段落完整，适合有明确段落结构的文档
   - **智能分割**：综合考虑段落和句子边界，最大化保持文本结构

## 🔧 高级配置

### 修改分割长度

编辑 `translate_deepseek_improved.py`，找到分割函数中的 `1000` 参数：

```python
# 修改这些行中的 1000 为你想要的长度
chunks = split_content_auto(content, 1000)  # 改为如 1500
chunks = split_content_by_paragraph(content, 1000)
chunks = split_content_smart(content, 1000)
```

### 调整生成参数

找到 `model.generate` 部分：

```python
outputs = model.generate(
    inputs.to(model.device),
    max_new_tokens=2048,      # 最大生成长度
    temperature=0.3,          # 温度（0-1，越低越保守）
    do_sample=True,          # 是否采样
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
```

## 📊 性能参考

基于测试结果（仅供参考）：

| 模型 | 速度 | 质量 | 内存需求 |
|------|------|------|----------|
| 1.5B | 最快 | 一般 | ~4GB |
| 7B | 中等 | 良好 | ~16GB |
| 14B | 较慢 | 优秀 | ~32GB |

## 🐛 常见问题

### 1. 内存不足错误

**解决方案**：
- 选择更小的模型
- 减小分割长度
- 使用CPU模式（在代码中设置 `device_map="cpu"`）

### 2. 翻译不完整

**解决方案**：
- 确保使用改进版脚本 `translate_deepseek_improved.py`
- 选择"自动分割"或"智能分割"策略
- 检查输出文件，查看分割片段数是否合理

### 3. CUDA错误

**解决方案**：
```bash
# 检查CUDA版本
nvidia-smi

# 重新安装对应版本的PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 4. 模型加载失败

**解决方案**：
- 检查模型路径是否正确
- 确保有足够的磁盘空间
- 尝试使用 `trust_remote_code=True` 参数

## 📝 输出格式

翻译后的文件将保存为：
- 文件名：`原文件名_中文.txt`
- 位置：指定的输出文件夹
- 格式：保持原文的段落结构

## 💡 使用建议

1. **首次使用**：先用小文件测试，确认效果后再批量翻译
2. **长文档**：使用"自动分割"或"智能分割"策略
3. **技术文档**：建议使用7B或14B模型以获得更准确的术语翻译
4. **批量处理**：将所有待翻译文件放在同一文件夹中

## 🔄 更新日志

### v2.0 (当前版本)
- ✅ 修复长文档只翻译一部分的问题
- ✅ 新增三种文本分割策略
- ✅ 改进进度显示和错误处理
- ✅ 优化内存使用

### v1.0
- 基础翻译功能
- 支持多个DeepSeek模型

## 📧 问题反馈

如遇到问题，请检查：
1. Python和依赖是否正确安装
2. 模型路径是否正确
3. 输入文件是否为UTF-8编码
4. 是否有足够的内存和磁盘空间

---

**提示**：翻译质量很大程度上取决于选择的模型和分割策略。建议根据实际需求进行调整。