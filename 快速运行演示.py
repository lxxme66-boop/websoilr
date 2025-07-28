#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSailor Domain Dataset 快速运行演示

这个脚本演示了如何使用WebSailor Domain Dataset工具的核心功能
"""

import os
import json
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_demo_input():
    """创建演示用的输入文件"""
    demo_text = """
人工智能技术发展概述

人工智能（AI）是计算机科学的一个重要分支，致力于研究和开发能够模拟人类智能的机器系统。

核心技术包括：
1. 机器学习算法
2. 深度神经网络
3. 自然语言处理
4. 计算机视觉
5. 知识图谱构建

应用领域：
- 智能推荐系统
- 语音识别技术
- 图像识别算法
- 自动驾驶汽车
- 智能医疗诊断

技术挑战：
人工智能面临的主要挑战包括数据质量问题、算法可解释性、计算资源需求、隐私保护等。

发展趋势：
未来人工智能将向通用人工智能（AGI）方向发展，同时在各个垂直领域实现更深入的应用。
    """
    
    input_dir = Path("demo_input")
    input_dir.mkdir(exist_ok=True)
    
    input_file = input_dir / "ai_tech.txt"
    with open(input_file, 'w', encoding='utf-8') as f:
        f.write(demo_text.strip())
    
    logger.info(f"创建演示输入文件: {input_file}")
    return input_dir

def run_simple_demo():
    """运行简化的演示"""
    logger.info("=== WebSailor Domain Dataset 快速演示 ===")
    
    # 1. 创建演示输入
    input_dir = create_demo_input()
    
    # 2. 简单的文本处理演示
    logger.info("演示文本处理功能...")
    
    try:
        import jieba
        import jieba.posseg as pseg
        
        # 读取演示文件
        demo_file = input_dir / "ai_tech.txt"
        with open(demo_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 分词演示
        logger.info("进行中文分词...")
        words = list(jieba.cut(text))
        logger.info(f"分词结果示例: {words[:10]}")
        
        # 词性标注演示
        logger.info("进行词性标注...")
        pos_words = list(pseg.cut(text[:100]))  # 只处理前100个字符
        logger.info(f"词性标注示例: {[(w, p) for w, p in pos_words[:5]]}")
        
        # 提取关键词
        logger.info("提取关键词...")
        import jieba.analyse
        keywords = jieba.analyse.extract_tags(text, topK=10)
        logger.info(f"关键词: {keywords}")
        
        # 创建输出目录和结果文件
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        
        # 保存结果
        result = {
            "input_file": str(demo_file),
            "word_count": len(words),
            "keywords": keywords,
            "pos_sample": [(w, p) for w, p in pos_words[:10]],
            "status": "success"
        }
        
        result_file = output_dir / "demo_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"演示结果已保存到: {result_file}")
        logger.info("✅ 演示运行成功！")
        
        return True
        
    except Exception as e:
        logger.error(f"演示运行出错: {e}")
        return False

def show_usage_guide():
    """显示使用指南"""
    print("""
🚀 WebSailor Domain Dataset 使用指南

## 运行环境
- Python 3.8+
- 已安装所需依赖包

## 基本使用方法

### 1. 快速演示（推荐）
python3 快速运行演示.py

### 2. 完整功能运行
python3 main.py --input-dir input_texts --output-dir output_dataset

### 3. 自定义配置运行
python3 main.py --config custom_config.json --input-dir your_input --output-dir your_output

## 输入要求
- 将您的领域文本文件（.txt格式，UTF-8编码）放入input_texts目录
- 支持中文和英文文本
- 建议文件大小在1MB以内

## 输出结果
程序会在指定的输出目录生成：
- 知识图谱文件
- 子图数据集
- 问答对数据
- 可视化结果

## 配置参数
可以通过修改config.json文件调整：
- 子图采样数量
- 实体提取参数
- 生成策略配置

## 常见问题
1. 如果遇到依赖包问题，请运行: pip install -r requirements.txt
2. 如果内存不足，可以减少配置文件中的采样数量
3. 确保输入文本为UTF-8编码

## 技术支持
如有问题，请检查日志输出或查看README.md文档
    """)

if __name__ == "__main__":
    print("选择运行模式：")
    print("1. 运行快速演示")
    print("2. 显示使用指南")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            run_simple_demo()
        elif choice == "2":
            show_usage_guide()
        else:
            print("运行快速演示...")
            run_simple_demo()
            
    except KeyboardInterrupt:
        print("\n演示已取消")
    except Exception as e:
        print(f"运行出错: {e}")
        show_usage_guide()