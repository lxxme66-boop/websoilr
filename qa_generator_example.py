#!/usr/bin/env python3
"""
结构化数据问答对生成器
用于从JSON Lines格式的术语数据生成多样化的问答对
"""

import json
import random
from typing import List, Dict, Tuple
from datetime import datetime

class QAGenerator:
    """问答对生成器类"""
    
    def __init__(self):
        # 问题模板库
        self.question_templates = {
            'abbreviation': [
                "{abbreviation}是什么的缩写？",
                "{abbreviation}代表什么？",
                "{abbreviation}的全称是什么？",
                "请问{abbreviation}是什么意思？",
                "{abbreviation}展开后是什么？"
            ],
            'translation': [
                "{english_full_name}的中文是什么？",
                "{english_full_name}翻译成中文是？",
                "请将{english_full_name}翻译为中文",
                "{chinese_full_name}的英文是什么？",
                "{chinese_full_name}用英语怎么说？"
            ],
            'definition': [
                "什么是{chinese_full_name}？",
                "请解释一下{chinese_full_name}",
                "{chinese_full_name}是什么意思？",
                "能说明一下{english_full_name}吗？",
                "简述{chinese_full_name}的含义"
            ],
            'comprehensive': [
                "{abbreviation}的全称和中文分别是什么？",
                "请解释{abbreviation}（包括英文全称和中文）",
                "{abbreviation}是什么？请给出英文和中文",
                "说明{abbreviation}的完整含义"
            ]
        }
        
        # 答案模板库
        self.answer_templates = {
            'abbreviation': [
                "{english_full_name}",
                "{abbreviation}是{english_full_name}的缩写",
                "{abbreviation}代表{english_full_name}"
            ],
            'translation': [
                "{chinese_full_name}",
                "中文是：{chinese_full_name}",
                "{english_full_name}"
            ],
            'definition': [
                "{chinese_full_name}（{english_full_name}）是{abbreviation}的含义",
                "{chinese_full_name}，英文为{english_full_name}，缩写为{abbreviation}"
            ],
            'comprehensive': [
                "{abbreviation}是{english_full_name}的缩写，中文意思是{chinese_full_name}",
                "英文全称：{english_full_name}，中文：{chinese_full_name}",
                "{english_full_name}（{chinese_full_name}）"
            ]
        }
    
    def load_data(self, filepath: str) -> List[Dict]:
        """加载JSON Lines格式的数据"""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def generate_single_qa(self, item: Dict, qa_type: str) -> Tuple[str, str]:
        """生成单个问答对"""
        # 随机选择问题模板
        q_template = random.choice(self.question_templates[qa_type])
        question = q_template.format(**item)
        
        # 选择对应的答案模板
        if qa_type == 'abbreviation':
            answer = item['english_full_name']
        elif qa_type == 'translation':
            if 'english_full_name' in q_template:
                answer = item['chinese_full_name']
            else:
                answer = item['english_full_name']
        elif qa_type in ['definition', 'comprehensive']:
            a_template = random.choice(self.answer_templates[qa_type])
            answer = a_template.format(**item)
        
        return question, answer
    
    def generate_qa_pairs(self, data: List[Dict], 
                         num_per_type: int = 2) -> List[Dict]:
        """批量生成问答对"""
        qa_pairs = []
        
        for item in data:
            # 为每种类型生成指定数量的问答对
            for qa_type in self.question_templates.keys():
                for _ in range(num_per_type):
                    try:
                        question, answer = self.generate_single_qa(item, qa_type)
                        qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'type': qa_type,
                            'source': item['abbreviation'],
                            'timestamp': datetime.now().isoformat()
                        })
                    except Exception as e:
                        print(f"生成问答对时出错: {e}")
        
        # 打乱顺序
        random.shuffle(qa_pairs)
        return qa_pairs
    
    def export_qa_pairs(self, qa_pairs: List[Dict], 
                       output_format: str = 'jsonl') -> None:
        """导出问答对到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_format == 'jsonl':
            filename = f'qa_pairs_{timestamp}.jsonl'
            with open(filename, 'w', encoding='utf-8') as f:
                for qa in qa_pairs:
                    f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        
        elif output_format == 'json':
            filename = f'qa_pairs_{timestamp}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        
        elif output_format == 'txt':
            filename = f'qa_pairs_{timestamp}.txt'
            with open(filename, 'w', encoding='utf-8') as f:
                for i, qa in enumerate(qa_pairs, 1):
                    f.write(f"问题{i}: {qa['question']}\n")
                    f.write(f"答案{i}: {qa['answer']}\n")
                    f.write(f"类型: {qa['type']}\n")
                    f.write("-" * 50 + "\n")
        
        print(f"已导出 {len(qa_pairs)} 个问答对到文件: {filename}")
        return filename
    
    def generate_statistics(self, qa_pairs: List[Dict]) -> Dict:
        """生成统计信息"""
        stats = {
            'total': len(qa_pairs),
            'by_type': {},
            'by_source': {}
        }
        
        for qa in qa_pairs:
            # 按类型统计
            qa_type = qa['type']
            stats['by_type'][qa_type] = stats['by_type'].get(qa_type, 0) + 1
            
            # 按来源统计
            source = qa['source']
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
        
        return stats


def main():
    """主函数：演示如何使用QAGenerator"""
    
    # 创建示例数据文件
    sample_data = [
        {"abbreviation": "VOD", "english_full_name": "Video On Demand", "chinese_full_name": "视频点播"},
        {"abbreviation": "Viscosity (η)", "english_full_name": "Viscosity (η)", "chinese_full_name": "粘滞系数/黏度"},
        {"abbreviation": "Via Hole", "english_full_name": "Via Hole", "chinese_full_name": "过孔"},
        {"abbreviation": "API", "english_full_name": "Application Programming Interface", "chinese_full_name": "应用程序接口"},
        {"abbreviation": "CPU", "english_full_name": "Central Processing Unit", "chinese_full_name": "中央处理器"}
    ]
    
    # 保存示例数据
    with open('sample_terms.jsonl', 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 初始化生成器
    generator = QAGenerator()
    
    # 加载数据
    data = generator.load_data('sample_terms.jsonl')
    print(f"已加载 {len(data)} 条术语数据")
    
    # 生成问答对
    qa_pairs = generator.generate_qa_pairs(data, num_per_type=1)
    print(f"\n已生成 {len(qa_pairs)} 个问答对")
    
    # 显示部分示例
    print("\n问答对示例：")
    print("=" * 60)
    for qa in qa_pairs[:5]:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print(f"类型: {qa['type']}")
        print("-" * 60)
    
    # 导出到文件
    generator.export_qa_pairs(qa_pairs, 'jsonl')
    generator.export_qa_pairs(qa_pairs, 'txt')
    
    # 生成统计信息
    stats = generator.generate_statistics(qa_pairs)
    print(f"\n统计信息：")
    print(f"总问答对数: {stats['total']}")
    print(f"按类型分布: {stats['by_type']}")
    print(f"按来源分布: {stats['by_source']}")


if __name__ == "__main__":
    main()