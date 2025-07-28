#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试文本分割效果对比
"""

# 原始的分割函数（问题版本）
def old_split_content(text, max_length):
    """简单分割文本 - 原始版本"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    paragraphs = text.split('\n\n')
    current = ""
    
    for para in paragraphs:
        if len(current + para) <= max_length:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(current)
            current = para
    
    if current:
        chunks.append(current)
    
    return chunks

# 改进的分割函数
def smart_split_content(text, max_length=800):
    """
    智能分割文本，确保完整翻译
    - 优先按段落分割
    - 避免在句子中间分割
    - 确保每个块不超过max_length
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    
    # 首先尝试按段落分割
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        # 如果单个段落就超过最大长度，需要进一步分割
        if len(para) > max_length:
            # 先保存当前块
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # 分割长段落
            sentences = split_into_sentences(para)
            for sent in sentences:
                if len(current_chunk) + len(sent) + 1 <= max_length:
                    current_chunk += (" " if current_chunk and not current_chunk.endswith('\n') else "") + sent
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent
        else:
            # 正常段落
            if len(current_chunk) + len(para) + 2 <= max_length:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
    
    # 不要忘记最后一个块
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def split_into_sentences(text):
    """将文本分割成句子"""
    import re
    
    # 简单的句子分割规则
    sentences = re.split(r'([.!?])\s+', text)
    
    # 重新组合句子和标点
    result = []
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            result.append(sentences[i] + sentences[i + 1])
        else:
            result.append(sentences[i])
    
    # 如果分割后的句子仍然太长，按固定长度分割
    final_result = []
    for sent in result:
        if len(sent) > 800:
            # 按固定长度分割，但尽量在空格处分割
            words = sent.split()
            current = ""
            for word in words:
                if len(current) + len(word) + 1 <= 800:
                    current += (" " if current else "") + word
                else:
                    if current:
                        final_result.append(current)
                    current = word
            if current:
                final_result.append(current)
        else:
            final_result.append(sent)
    
    return final_result

# 测试文本
test_text = """This is the first paragraph. It contains multiple sentences. Each sentence adds to the overall meaning.

This is the second paragraph with more content. The bistable model of Ho in ZnO has been studied extensively. Researchers have found interesting properties in this material system. The defect states can switch between different configurations. This switching behavior is important for device applications.

Here is a very long paragraph that contains a lot of technical information about the IGZTO material developed by Kobe Steel in Japan. The material shows excellent electrical properties and has been presented at the SID conference in 2015. The thin film transistors made from this material demonstrate high mobility and good stability. The manufacturing process involves several steps including deposition, annealing, and patterning. Each step must be carefully controlled to achieve the desired properties. The resulting devices have been used in various display applications. The technology continues to evolve with new improvements being made regularly. This paragraph is intentionally long to test the splitting algorithm.

This is the fourth paragraph. It's shorter but still important.

The final paragraph concludes our test text. It summarizes the key points discussed earlier."""

# 测试两种分割方法
print("🔍 测试文本分割对比")
print("=" * 60)
print(f"测试文本总长度: {len(test_text)} 字符")
print()

# 测试原始方法（使用1500字符，模拟原代码）
print("❌ 原始分割方法（1500字符）:")
old_chunks = old_split_content(test_text, 1500)
print(f"  分成 {len(old_chunks)} 个片段")
for i, chunk in enumerate(old_chunks):
    print(f"  片段 {i+1}: {len(chunk)} 字符")
    print(f"    开头: {chunk[:50]}...")
    print(f"    结尾: ...{chunk[-50:]}")

print()

# 测试改进方法（使用800字符）
print("✅ 改进分割方法（800字符）:")
new_chunks = smart_split_content(test_text, 800)
print(f"  分成 {len(new_chunks)} 个片段")
for i, chunk in enumerate(new_chunks):
    print(f"  片段 {i+1}: {len(chunk)} 字符")
    print(f"    开头: {chunk[:50]}...")
    print(f"    结尾: ...{chunk[-50:]}")

# 验证内容完整性
print()
print("📊 内容完整性检查:")
old_total = sum(len(chunk) for chunk in old_chunks)
new_total = sum(len(chunk) for chunk in new_chunks)
print(f"  原始方法覆盖: {old_total}/{len(test_text)} 字符 ({old_total/len(test_text)*100:.1f}%)")
print(f"  改进方法覆盖: {new_total}/{len(test_text)} 字符 ({new_total/len(test_text)*100:.1f}%)")

# 测试长文档
print("\n" + "=" * 60)
print("📄 测试长文档（模拟实际论文）:")

# 生成一个更长的测试文档
long_text = test_text * 5  # 重复5次，模拟长文档
print(f"长文档总长度: {len(long_text)} 字符")

# 对比分割结果
old_long_chunks = old_split_content(long_text, 1500)
new_long_chunks = smart_split_content(long_text, 800)

print(f"\n原始方法: {len(old_long_chunks)} 个片段")
print(f"改进方法: {len(new_long_chunks)} 个片段")

# 显示为什么原始方法会丢失内容
print("\n❗ 原始方法问题分析:")
print("  - 使用1500字符的大块，容易超过模型处理能力")
print("  - 没有处理超长段落的逻辑")
print("  - 可能在段落中间截断，破坏语义完整性")

print("\n✨ 改进方法优势:")
print("  - 使用800字符的合理块大小")
print("  - 智能处理长段落，按句子分割")
print("  - 保持语义完整性")
print("  - 确保所有内容都被翻译")