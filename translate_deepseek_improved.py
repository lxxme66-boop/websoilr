#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进版本：优化文本分割算法，确保长文档能够被正确分割翻译
"""

import os
import time
from pathlib import Path
import re

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
    print("4. gte_Qwen2-7B-instruct (替代选项)")
    
    # 选择模型
    while True:
        choice = input("\n请选择模型 (1-4): ").strip()
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
    input_dir = input("\n📁 输入英文txt文件夹路径: ").strip()
    if not input_dir or not Path(input_dir).exists():
        print("❌ 输入路径无效")
        return
    
    output_dir = input("📂 输出文件夹路径 [默认: ./chinese_translations]: ").strip()
    if not output_dir:
        output_dir = "./chinese_translations"
    
    # 询问分割策略
    print("\n文本分割选项:")
    print("1. 自动分割 (推荐，每1000字符)")
    print("2. 按段落分割 (适合结构化文档)")
    print("3. 智能分割 (根据句子边界)")
    
    split_choice = input("选择分割策略 (1-3) [默认: 1]: ").strip()
    if not split_choice:
        split_choice = "1"
    
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
        translate_files(input_dir, output_dir, model, tokenizer, split_choice)
        
    except Exception as e:
        print(f"❌ 错误: {e}")

def translate_files(input_dir, output_dir, model, tokenizer, split_choice):
    """翻译文件"""
    import torch
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    txt_files = list(input_path.glob("*.txt"))
    
    if not txt_files:
        print("❌ 没有找到txt文件")
        return
    
    print(f"\n📁 找到 {len(txt_files)} 个文件")
    
    # 记录统计信息
    total_start = time.time()
    success_count = 0
    
    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n[{i}/{len(txt_files)}] 翻译: {txt_file.name}")
        file_start = time.time()
        
        try:
            # 读取文件
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print("⚠ 文件为空，跳过")
                continue
            
            print(f"  文件大小: {len(content)} 字符")
            
            # 根据选择的策略分割内容
            if split_choice == "1":
                chunks = split_content_auto(content, 1000)
            elif split_choice == "2":
                chunks = split_content_by_paragraph(content, 1000)
            else:
                chunks = split_content_smart(content, 1000)
            
            print(f"  分割成 {len(chunks)} 个片段")
            translated_chunks = []
            
            for j, chunk in enumerate(chunks):
                print(f"  翻译片段 {j+1}/{len(chunks)} ({len(chunk)} 字符)", end="", flush=True)
                
                # 构建prompt
                prompt = f"请将以下英文翻译成中文，保持原文格式：\n\n{chunk}\n\n中文翻译："
                
                # 生成翻译
                inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=2048, truncation=True)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.to(model.device),
                        max_new_tokens=2048,
                        temperature=0.3,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 提取翻译
                if "中文翻译：" in response:
                    translation = response.split("中文翻译：")[-1].strip()
                else:
                    translation = response[len(prompt):].strip()
                
                translated_chunks.append(translation)
                print(" ✓")
            
            # 保存结果
            result = "\n\n".join(translated_chunks)
            output_file = output_path / f"{txt_file.stem}_中文.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            
            file_time = time.time() - file_start
            print(f"✅ 完成: {output_file.name} (耗时: {file_time:.1f}秒)")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 翻译失败: {e}")
    
    # 总结
    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"翻译完成！")
    print(f"成功: {success_count}/{len(txt_files)} 个文件")
    print(f"总耗时: {total_time:.1f}秒")
    print(f"输出目录: {output_path.absolute()}")

def split_content_auto(text, max_length):
    """自动分割文本 - 按固定长度分割，尽量在句子边界"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        # 计算这一块的结束位置
        end_pos = min(current_pos + max_length, len(text))
        
        # 如果不是最后一块，尝试在句子边界分割
        if end_pos < len(text):
            # 查找最近的句子结束符
            chunk = text[current_pos:end_pos]
            
            # 优先级：句号 > 问号 > 感叹号 > 换行 > 逗号 > 空格
            for delimiter in ['. ', '? ', '! ', '\n', ', ', ' ']:
                last_delimiter = chunk.rfind(delimiter)
                if last_delimiter > max_length * 0.5:  # 至少保留一半长度
                    end_pos = current_pos + last_delimiter + len(delimiter)
                    break
        
        chunks.append(text[current_pos:end_pos].strip())
        current_pos = end_pos
    
    return chunks

def split_content_by_paragraph(text, max_length):
    """按段落分割，确保每个片段不超过最大长度"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # 如果段落本身就太长，需要进一步分割
        if len(para) > max_length:
            # 先保存当前块
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # 分割长段落
            sub_chunks = split_content_auto(para, max_length)
            chunks.extend(sub_chunks)
        
        # 如果加上这个段落不会超过限制
        elif len(current_chunk) + len(para) + 2 <= max_length:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        
        # 如果会超过限制，保存当前块并开始新块
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
    
    # 保存最后一块
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def split_content_smart(text, max_length):
    """智能分割 - 优先保持段落完整性，其次是句子完整性"""
    if len(text) <= max_length:
        return [text]
    
    # 首先按段落分割
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # 如果段落太长，按句子分割
        if len(para) > max_length:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # 按句子分割段落
            sentences = re.split(r'(?<=[.!?])\s+', para)
            
            for sentence in sentences:
                if len(sentence) > max_length:
                    # 句子还是太长，强制分割
                    sub_chunks = split_content_auto(sentence, max_length)
                    chunks.extend(sub_chunks)
                elif len(current_chunk) + len(sentence) + 1 <= max_length:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
        
        # 段落长度合适
        elif len(current_chunk) + len(para) + 4 <= max_length:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
    
    # 保存最后一块
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

if __name__ == "__main__":
    simple_translate()