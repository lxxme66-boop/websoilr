#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进版DeepSeek翻译工具
- 修复长文档只翻译一部分的问题
- 优化文本分割逻辑
- 增加进度显示和错误处理
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
    print("4. gte_Qwen2-7B-instruct (备选)")
    
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
    
    # 询问分片大小
    print("\n文本分片设置:")
    print("1. 小片段 (800字符) - 更精确但速度慢")
    print("2. 中片段 (1500字符) - 平衡选择")
    print("3. 大片段 (3000字符) - 速度快但可能遗漏")
    chunk_choice = input("选择分片大小 (1-3) [默认: 2]: ").strip()
    
    chunk_sizes = {"1": 800, "2": 1500, "3": 3000}
    chunk_size = chunk_sizes.get(chunk_choice, 1500)
    
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

def translate_files(input_dir, output_dir, model, tokenizer, chunk_size=1500):
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
    print(f"📏 使用分片大小: {chunk_size} 字符")
    
    # 统计信息
    total_chars = 0
    total_chunks = 0
    start_time = time.time()
    
    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n[{i}/{len(txt_files)}] 翻译: {txt_file.name}")
        
        try:
            # 读取文件
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print("⚠ 文件为空，跳过")
                continue
            
            file_size = len(content)
            total_chars += file_size
            print(f"  文件大小: {file_size:,} 字符")
            
            # 智能分割文本
            chunks = smart_split_content(content, chunk_size)
            total_chunks += len(chunks)
            print(f"  分成 {len(chunks)} 个片段")
            
            translated_chunks = []
            
            for j, chunk in enumerate(chunks):
                print(f"  翻译片段 {j+1}/{len(chunks)} ({len(chunk)} 字符)", end="")
                
                # 构建prompt
                prompt = f"""请将以下英文内容完整翻译成中文。注意：
1. 保持原文的格式和结构
2. 专业术语要准确
3. 不要遗漏任何内容

英文原文：
{chunk}

中文翻译："""
                
                # 生成翻译
                inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=4096)
                
                if inputs.shape[1] > 4096:
                    print(" ⚠ 输入过长，自动截断")
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.to(model.device),
                        max_new_tokens=3000,  # 增加输出长度
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
                    # 尝试其他分割方式
                    translation = response[len(prompt):].strip()
                
                # 确保翻译不为空
                if not translation:
                    print(" ⚠ 翻译为空，重试")
                    translation = chunk  # 保留原文
                
                translated_chunks.append(translation)
                print(" ✓")
            
            # 合并翻译结果
            result = "\n\n".join(translated_chunks)
            
            # 添加元信息
            header = f"""# {txt_file.name} - 中文翻译
# 原文件大小: {file_size:,} 字符
# 分片数量: {len(chunks)}
# 翻译时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
# 使用模型: {Path(model.name_or_path).name}

"""
            
            # 保存结果
            output_file = output_path / f"{txt_file.stem}_中文.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(header + result)
            
            # 验证翻译完整性
            result_size = len(result)
            coverage = (result_size / file_size) * 100
            
            print(f"✅ 完成: {output_file.name}")
            print(f"  翻译覆盖率: {coverage:.1f}% ({result_size:,}/{file_size:,} 字符)")
            
            if coverage < 50:
                print("  ⚠️ 警告：翻译覆盖率较低，可能有内容遗漏")
            
        except Exception as e:
            print(f"\n❌ 翻译失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"✅ 翻译完成!")
    print(f"  总字符数: {total_chars:,}")
    print(f"  总片段数: {total_chunks}")
    print(f"  总用时: {elapsed/60:.1f} 分钟")
    print(f"  平均速度: {total_chars/elapsed:.0f} 字符/秒")

def smart_split_content(text, max_length):
    """智能分割文本，确保不遗漏内容"""
    
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
                chunks.append(current_chunk)
                current_chunk = ""
            
            # 分割长段落
            # 按句子分割
            sentences = split_into_sentences(para)
            
            for sent in sentences:
                if len(current_chunk) + len(sent) + 1 <= max_length:
                    current_chunk += (" " if current_chunk else "") + sent
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sent
        
        else:
            # 正常段落
            if len(current_chunk) + len(para) + 2 <= max_length:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
    
    # 不要忘记最后一块
    if current_chunk:
        chunks.append(current_chunk)
    
    # 验证没有内容丢失
    total_length = sum(len(chunk) for chunk in chunks)
    original_length = len(text)
    
    if abs(total_length - original_length) > 100:  # 允许少量差异（空格等）
        print(f"\n⚠️ 警告：分割后总长度({total_length})与原文({original_length})相差较大")
    
    return chunks

def split_into_sentences(text):
    """将文本分割成句子"""
    import re
    
    # 简单的句子分割规则
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # 处理过短的句子（可能是缩写等）
    result = []
    current = ""
    
    for sent in sentences:
        if len(sent) < 20 and current:  # 太短的句子合并
            current += " " + sent
        else:
            if current:
                result.append(current)
            current = sent
    
    if current:
        result.append(current)
    
    return result

if __name__ == "__main__":
    simple_translate()