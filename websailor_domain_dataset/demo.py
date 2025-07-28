#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSailor Domain Dataset 快速运行演示
"""

import os
import json
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_simple_demo():
    """运行简化的演示"""
    logger.info("=== WebSailor Domain Dataset 快速演示 ===")
    
    try:
        import jieba
        import jieba.posseg as pseg
        
        # 演示文本
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
        """
        
        # 分词演示
        logger.info("进行中文分词...")
        words = list(jieba.cut(demo_text))
        logger.info(f"分词结果示例: {words[:10]}")
        
        # 词性标注演示
        logger.info("进行词性标注...")
        pos_words = list(pseg.cut(demo_text[:100]))
        logger.info(f"词性标注示例: {[(w, p) for w, p in pos_words[:5]]}")
        
        # 提取关键词
        logger.info("提取关键词...")
        import jieba.analyse
        keywords = jieba.analyse.extract_tags(demo_text, topK=10)
        logger.info(f"关键词: {keywords}")
        
        # 创建输出目录和结果文件
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        
        # 保存结果
        result = {
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
        
        print("\n🎉 演示完成！主要功能展示：")
        print(f"- 成功分词 {len(words)} 个词汇")
        print(f"- 提取关键词: {', '.join(keywords[:5])}...")
        print(f"- 结果保存在: {result_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"演示运行出错: {e}")
        return False

def show_usage_guide():
    """显示使用指南"""
    print("""
🚀 WebSailor Domain Dataset 使用指南

## 运行环境
当前环境: Python 3.13, Linux 6.12.8+
依赖已安装: ✅

## 基本使用方法

### 1. 快速演示（当前脚本）
python3 demo.py

### 2. 完整功能运行
python3 main.py --input-dir input_texts --output-dir output_dataset

### 3. 查看帮助
python3 main.py --help

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

## 项目文件结构
websailor_domain_dataset/
├── main.py              # 主程序入口
├── config.json          # 配置文件
├── requirements.txt     # 依赖包列表
├── core/               # 核心功能模块
├── utils/              # 工具函数
├── templates/          # 模板文件
└── input_texts/        # 输入文本目录

## 下载代码
项目代码已打包为: websailor_domain_dataset_20250728_023039.tar.gz
位置: /workspace/websailor_domain_dataset_20250728_023039.tar.gz
    """)

if __name__ == "__main__":
    print("WebSailor Domain Dataset 演示程序")
    print("1. 运行快速演示")
    print("2. 显示使用指南")
    print("3. 退出")
    
    while True:
        try:
            choice = input("\n请选择 (1-3): ").strip()
            
            if choice == "1":
                run_simple_demo()
                break
            elif choice == "2":
                show_usage_guide()
                break
            elif choice == "3":
                print("再见！")
                break
            else:
                print("请输入 1、2 或 3")
                
        except KeyboardInterrupt:
            print("\n\n程序已退出")
            break
        except Exception as e:
            print(f"运行出错: {e}")
            break