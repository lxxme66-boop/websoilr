#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版本：优化长文档分割，确保完整翻译
"""

import os
import time
from pathlib import Path

def simple_translate():
    """简化的翻译流程"""
    
    # 直接指定模型路径
    model_paths = {
        "1": "/mnt/storage/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "2": "/mnt/storage/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
        "3": "/mnt/storage/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "4": "/mnt/storage/models/gte_Qwen2-7B-instruct"
    }
    
    print("🚀 改进版DeepSeek翻译工具")
    print("=" * 50)
    
    print("可用模型:")
    print("1. DeepSeek-R1-Distill-Qwen-1.5B (最快)")
    print("2. DeepSeek-R1-Distill-Qwen-7B (推荐)")
    print("3. DeepSeek-R1-Distill-Qwen-14B (高质量)")
    print("4. DeepSeek-R1-Distill-Qwen-32B (最高质量)")
    
    # 选择模型
    while True:
        choice = input("请选择模型 (1-4): ").strip()
        if choice in model_paths:
            model_path = model_paths[choice]
            # 检查路径是否存在
            if os.path.exists(model_path):
                print(f"✅ 选择模型: {Path(model_path).name}")
                break
            else:
                print(f"❌ 模型路径不存在: {model_path}")
                continue
        else:
            print("❌ 无效选择")
    
    # 输入输出路径
    input_dir = input("📁 输入英文txt文件夹路径: ").strip()
    if not input_dir or not Path(input_dir).exists():
        print("❌ 输入路径无效")
        return
    
    output_dir = input("📂 输出文件夹路径 [默认: ./chinese_translations]: ").strip()
    if not output_dir:
        output_dir = "./chinese_translations"
    
    # 询问分块大小
    chunk_size = input("📏 每个分块的字符数 [默认: 800，推荐600-1000]: ").strip()
    if not chunk_size:
        chunk_size = 800
    else:
        try:
            chunk_size = int(chunk_size)
        except:
            chunk_size = 800
    
    print(f"\n开始加载模型: {Path(model_path).name}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # 加载模型
        print("🔄 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        print("🔄 加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ 模型加载成功")
        
        # 开始翻译
        translate_files(input_dir, output_dir, model, tokenizer, chunk_size)
        
    except Exception as e:
        print(f"❌ 错误: {e}")

def translate_files(input_dir, output_dir, model, tokenizer, chunk_size=800):
    """翻译文件"""
    import torch
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    txt_files = list(input_path.glob("*.txt"))
    
    if not txt_files:
        print("❌ 没有找到txt文件")
        return
    
    print(f"📁 找到 {len(txt_files)} 个文件")
    print(f"📏 分块大小: {chunk_size} 字符")
    
    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n[{i}/{len(txt_files)}] 翻译: {txt_file.name}")
        
        try:
            # 读取文件
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print("⚠ 文件为空，跳过")
                continue
            
            # 显示文件信息
            total_chars = len(content)
            print(f"  文件大小: {total_chars} 字符")
            
            # 智能分割内容
            chunks = smart_split_content(content, chunk_size)
            print(f"  分成 {len(chunks)} 个片段")
            
            translated_chunks = []
            
            for j, chunk in enumerate(chunks):
                print(f"  翻译片段 {j+1}/{len(chunks)} ({len(chunk)} 字符)", end="", flush=True)
                
                # 构建prompt
                prompt = f"请将以下英文翻译成中文，保持原文格式：\n\n{chunk}\n\n中文翻译："
                
                # 生成翻译
                inputs = tokenizer.encode(prompt, return_tensors="pt")
                
                # 限制输入长度
                if inputs.shape[1] > 2048:
                    inputs = inputs[:, :2048]
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=2048,
                        temperature=0.3,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 提取翻译
                if "中文翻译：" in response:
                    translation = response.split("中文翻译：")[-1].strip()
                else:
                    translation = response[len(prompt):].strip()
                
                # 如果翻译为空或太短，可能是出错了
                if len(translation) < len(chunk) * 0.3:  # 中文通常比英文短，但不应该太短
                    print(" ⚠️ 翻译可能不完整")
                else:
                    print(" ✓")
                
                translated_chunks.append(translation)
                
                # 避免请求过快
                time.sleep(0.1)
            
            # 保存结果
            result = "\n\n".join(translated_chunks)
            output_file = output_path / f"{txt_file.stem}_中文.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            
            # 显示翻译统计
            translated_chars = len(result)
            print(f"✅ 完成: {output_file.name}")
            print(f"  原文: {total_chars} 字符, 译文: {translated_chars} 字符")
            
        except Exception as e:
            print(f"❌ 翻译失败: {e}")
            import traceback
            traceback.print_exc()

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
    
    # 验证是否有内容丢失
    total_original = len(text)
    total_chunks = sum(len(chunk) for chunk in chunks)
    if total_chunks < total_original * 0.9:  # 允许10%的空白字符差异
        print(f"  ⚠️ 警告：可能有内容丢失 (原文:{total_original} 分块后:{total_chunks})")
    
    return chunks

def split_into_sentences(text):
    """
    将文本分割成句子
    """
    import re
    
    # 简单的句子分割规则
    # 匹配句号、问号、感叹号后跟空格或换行
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

if __name__ == "__main__":
    simple_translate()