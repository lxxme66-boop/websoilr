#!/usr/bin/env python3
"""
测试优化后的问题生成器
"""

import sys
import networkx as nx
sys.path.append('/workspace/websailor_domain')

from core.question_generator import QuestionGenerator

# 创建测试配置
test_config = {
    'models': {
        'qa_generator_model': {
            'path': 'THUDM/chatglm-6b',
            'max_length': 2048,
            'temperature': 0.8
        }
    },
    'question_generation': {
        'question_types': ['factual', 'reasoning', 'multi_hop', 'comparison'],
        'complexity_levels': {
            'easy': 0.3,
            'medium': 0.5,
            'hard': 0.2
        },
        'language_patterns': {
            'zh_cn': 0.7,
            'en': 0.3
        }
    },
    'dataset_synthesis': {
        'quality_checks': {
            'min_question_length': 10,
            'max_question_length': 200,
            'answer_validation': True,
            'diversity_threshold': 0.8
        }
    },
    'tcl_specific': {}
}

# 创建测试子图
def create_test_subgraph():
    G = nx.DiGraph()
    
    # 添加节点
    G.add_node('TCL电视', type='产品')
    G.add_node('QLED技术', type='技术')
    G.add_node('量子点材料', type='材料')
    G.add_node('SMT工艺', type='工艺')
    
    # 添加边
    G.add_edge('TCL电视', 'QLED技术', relation='使用')
    G.add_edge('QLED技术', '量子点材料', relation='依赖')
    G.add_edge('TCL电视', 'SMT工艺', relation='应用于')
    
    # 添加图属性
    G.graph['topology'] = 'star'
    
    return G

def main():
    print("测试优化后的问题生成器...")
    
    # 初始化问题生成器
    try:
        qg = QuestionGenerator(test_config)
        print("✓ 问题生成器初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return
    
    # 创建测试子图
    subgraph = create_test_subgraph()
    print(f"✓ 创建测试子图: {subgraph.number_of_nodes()}个节点, {subgraph.number_of_edges()}条边")
    
    # 测试 _analyze_subgraph 方法
    try:
        features = qg._analyze_subgraph(subgraph)
        print(f"✓ 子图分析成功: {features}")
    except Exception as e:
        print(f"✗ 子图分析失败: {e}")
        return
    
    # 测试问题生成
    try:
        qa_pairs = qg.generate_questions([subgraph])
        print(f"✓ 生成了 {len(qa_pairs)} 个问题")
        
        # 打印生成的问题
        for i, qa in enumerate(qa_pairs[:3]):  # 只打印前3个
            print(f"\n问题{i+1}: {qa.get('question', 'N/A')}")
            print(f"类型: {qa.get('type', 'N/A')}")
            print(f"语言: {qa.get('language', 'N/A')}")
    except Exception as e:
        print(f"✗ 问题生成失败: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == '__main__':
    main()