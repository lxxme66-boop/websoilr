#!/usr/bin/env python3
"""
测试修复后的问题生成器
"""

import json
import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.question_generator import QuestionGenerator
import networkx as nx

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_test_subgraph():
    """创建一个测试子图"""
    # 创建NetworkX图
    G = nx.DiGraph()
    
    # 添加节点
    G.add_node("TCL电视", type="产品")
    G.add_node("OLED技术", type="技术")
    G.add_node("量子点技术", type="技术")
    G.add_node("8K分辨率", type="技术")
    
    # 添加边
    G.add_edge("TCL电视", "OLED技术", type="使用")
    G.add_edge("TCL电视", "量子点技术", type="使用")
    G.add_edge("TCL电视", "8K分辨率", type="支持")
    G.add_edge("OLED技术", "量子点技术", type="配合")
    
    return G

def test_question_generation():
    """测试问题生成"""
    # 创建配置
    config = {
        "models": {
            "qa_generator_model": {
                "path": "THUDM/chatglm-6b",  # 使用较小的模型进行测试
                "max_length": 2048,
                "temperature": 0.7
            }
        },
        "question_generation": {
            "question_types": ["factual", "reasoning", "multi_hop", "comparative"],
            "complexity_levels": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
            "language_patterns": {"zh_cn": 0.7, "en": 0.3}
        },
        "simple_mode": True  # 使用简单模式进行测试
    }
    
    # 初始化问题生成器
    print("初始化问题生成器...")
    qg = QuestionGenerator(config)
    
    # 创建测试子图
    print("创建测试子图...")
    subgraphs = [create_test_subgraph()]
    
    # 测试子图转换
    print("\n测试子图格式转换...")
    converted = qg._convert_digraph_to_dict(subgraphs[0])
    print(f"转换后的子图: {converted['num_nodes']} 个节点, {converted['num_edges']} 条边")
    print(f"节点类型: {converted['node_types']}")
    print(f"关系类型: {converted['relation_types']}")
    
    # 生成问题
    print("\n开始生成问题...")
    qa_pairs = qg.generate_questions(subgraphs, questions_per_subgraph=5)
    
    # 显示结果
    print(f"\n成功生成 {len(qa_pairs)} 个问答对:")
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\n--- 问答对 {i} ---")
        print(f"类型: {qa['type']}")
        print(f"语言: {qa['language']}")
        print(f"问题: {qa['question']}")
        print(f"答案: {qa['answer']}")
        print(f"合理性分数: {qa.get('validity_score', 'N/A')}")
    
    # 保存结果
    output_file = "test_qa_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    test_question_generation()