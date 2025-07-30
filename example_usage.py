#!/usr/bin/env python3
"""
文档去重系统使用示例
"""

import json
from optimized_document_deduplication import DocumentDeduplicator

# 创建示例数据
sample_data = [
    {
        "doc_id": "001",
        "论文标题": "深度学习在自然语言处理中的应用",
        "Abstract": "本文探讨了深度学习技术在自然语言处理领域的最新应用，包括文本分类、情感分析和机器翻译等方面。通过实验验证了深度学习模型的有效性。"
    },
    {
        "doc_id": "002",
        "论文标题": "深度学习在NLP中的应用研究",
        "Abstract": "本文探讨了深度学习技术在自然语言处理领域的最新应用，包括文本分类、情感分析和机器翻译等方面。通过实验验证了深度学习模型的有效性。"
    },
    {
        "doc_id": "003",
        "论文标题": "机器学习算法综述",
        "Abstract": "本文全面介绍了各种机器学习算法，包括监督学习、无监督学习和强化学习的基本原理和应用场景。"
    },
    {
        "doc_id": "004",
        "论文标题": "机器学习算法研究综述",
        "Abstract": "本文全面介绍了各种机器学习算法，包括监督学习、无监督学习和强化学习的基本原理和应用场景。"
    },
    {
        "doc_id": "005",
        "论文标题": "计算机视觉技术发展",
        "Abstract": "计算机视觉是人工智能的重要分支，本文介绍了计算机视觉的发展历程、关键技术和最新进展。"
    },
    {
        "doc_id": "006",
        "论文标题": "自然语言处理技术综述",
        "Abstract": "本文系统地介绍了自然语言处理的基本概念、主要任务和技术方法，包括词法分析、句法分析、语义理解等。"
    }
]

# 保存示例数据
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(sample_data, f, ensure_ascii=False, indent=2)

print("示例数据已创建！")
print(f"文档数量：{len(sample_data)}")
print("\n开始去重处理...\n")

# 创建去重器实例（使用自定义参数）
deduplicator = DocumentDeduplicator(
    fingerprint_threshold=0.9,    # 指纹匹配阈值
    minhash_threshold=0.5,        # MinHash阈值  
    tfidf_threshold=0.7,          # TF-IDF相似度阈值
    num_perm=128,                 # MinHash排列数
    n_jobs=2,                     # 使用2个CPU核心
    batch_size=1000               # 批处理大小
)

# 加载文档
documents = deduplicator.load_documents('data.json')

# 执行去重
duplicates = deduplicator.deduplicate(documents)

# 分析和导出结果
deduplicator.analyze_and_export(documents, duplicates)

print("\n去重完成！")
print("\n生成的文件：")
print("- precision_deduplication.log: 详细日志")
print("- duplicate_documents.csv: 重复文档详情")
print("- duplicate_ids.csv: 重复文档ID列表")  
print("- unique_documents.json: 去重后的文档")
print("- *.png: 可视化分析图表")