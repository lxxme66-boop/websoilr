"""
测试问题生成器修复
验证问答对能否正常生成
"""

import json
import logging
from core.question_generator import QuestionGenerator

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 测试配置
test_config = {
    'models': {
        'qa_generator_model': {
            'path': '/mnt/storage/models/Qwen/Qwen2.5-14B-Instruct',
            'max_length': 2048,
            'temperature': 0.7,
            'top_p': 0.9
        }
    },
    'data_settings': {
        'question_types': ['factual', 'reasoning', 'multi_hop', 'comparative', 'causal']
    }
}

# 创建测试子图
test_subgraphs = [
    {
        'id': 0,
        'nodes': [
            {'id': 'TCL电视', 'type': '产品'},
            {'id': 'OLED显示屏', 'type': '技术'},
            {'id': '量子点技术', 'type': '技术'},
            {'id': '画质处理芯片', 'type': '组件'}
        ],
        'edges': [
            {'source': 'TCL电视', 'target': 'OLED显示屏', 'relation': '使用'},
            {'source': 'TCL电视', 'target': '量子点技术', 'relation': '集成'},
            {'source': 'OLED显示屏', 'target': '画质处理芯片', 'relation': '依赖'}
        ],
        'topology': 'star',
        'complexity': 0.6
    },
    {
        'id': 1,
        'nodes': [
            {'id': '生产线', 'type': '设备'},
            {'id': 'SMT工艺', 'type': '工艺'},
            {'id': '质量检测', 'type': '流程'}
        ],
        'edges': [
            {'source': '生产线', 'target': 'SMT工艺', 'relation': '应用'},
            {'source': 'SMT工艺', 'target': '质量检测', 'relation': '需要'}
        ],
        'topology': 'chain',
        'complexity': 0.5
    }
]

def test_question_generation():
    """测试问题生成"""
    print("开始测试问题生成器...")
    
    # 初始化问题生成器
    generator = QuestionGenerator(test_config)
    
    # 生成问题
    questions = generator.generate_questions(test_subgraphs, questions_per_subgraph=3)
    
    # 打印结果
    print(f"\n生成了 {len(questions)} 个问题")
    
    for i, q in enumerate(questions):
        print(f"\n问题 {i+1}:")
        print(f"类型: {q['type']}")
        print(f"语言: {q['language']}")
        print(f"问题: {q['question']}")
        print(f"答案: {q['answer'][:100]}...")  # 只显示前100字符
        print(f"涉及实体: {q['entities']}")
    
    # 保存结果
    with open('test_questions.json', 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    
    print("\n测试结果已保存到 test_questions.json")

if __name__ == '__main__':
    test_question_generation()