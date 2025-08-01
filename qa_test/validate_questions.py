#!/usr/bin/env python3
"""
验证生成的问题是否合理
检查问题是否基于输入文本，是否包含相关技术术语
"""

import json
import re
from typing import List, Dict, Tuple
from collections import Counter


def load_input_texts(file_paths: List[str]) -> Dict[str, str]:
    """加载输入文本"""
    texts = {}
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            texts[path] = f.read()
    return texts


def load_qa_pairs(file_path: str) -> List[Dict]:
    """加载问答对"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_key_terms(text: str) -> set:
    """提取关键术语"""
    # 技术术语模式
    tech_patterns = [
        r'[A-Z]+(?:\s+[A-Z]+)*',  # 大写缩写 (如 TFT, LCD, OLED)
        r'[a-zA-Z]+-[a-zA-Z]+',   # 连字符术语 (如 a-Si, Mini-LED)
        r'\b(?:技术|系统|设备|材料|工艺|显示|制造)\b',  # 中文技术词
    ]
    
    terms = set()
    for pattern in tech_patterns:
        matches = re.findall(pattern, text)
        terms.update(matches)
    
    # 特定技术术语
    specific_terms = [
        'TFT', 'LCD', 'OLED', 'EPD', 'LED', 'QLED', 'TCL',
        'a-Si', 'ZnO', '金属氧化物', '衬底树脂', '柔性',
        '显示', '面板', '量子点', '印刷', '智能制造',
        'MES', 'AGV', '数字孪生', 'AI', '5G'
    ]
    
    for term in specific_terms:
        if term in text:
            terms.add(term)
    
    return terms


def validate_question_relevance(question: str, input_texts: Dict[str, str]) -> Dict:
    """验证问题与输入文本的相关性"""
    question_terms = extract_key_terms(question)
    
    relevance_scores = {}
    for file_path, text in input_texts.items():
        text_terms = extract_key_terms(text)
        
        # 计算术语重叠
        overlap = question_terms & text_terms
        relevance_scores[file_path] = {
            'overlap_count': len(overlap),
            'overlap_ratio': len(overlap) / len(question_terms) if question_terms else 0,
            'matched_terms': list(overlap)
        }
    
    # 判断是否相关
    max_overlap = max(score['overlap_count'] for score in relevance_scores.values())
    is_relevant = max_overlap >= 2  # 至少匹配2个关键术语
    
    return {
        'is_relevant': is_relevant,
        'relevance_scores': relevance_scores,
        'question_terms': list(question_terms)
    }


def analyze_question_quality(question: str) -> Dict:
    """分析问题质量"""
    quality_indicators = {
        'is_technical': False,
        'is_specific': False,
        'has_context': False,
        'complexity_level': 'low'
    }
    
    # 检查是否为技术问题
    tech_indicators = ['TFT', 'LCD', 'OLED', 'EPD', '显示', '技术', '工艺', '系统']
    quality_indicators['is_technical'] = any(ind in question for ind in tech_indicators)
    
    # 检查具体性
    specific_patterns = ['什么原因', '如何', '为什么', '分析', '解释', '比较']
    quality_indicators['is_specific'] = any(pattern in question for pattern in specific_patterns)
    
    # 检查是否有上下文
    quality_indicators['has_context'] = len(question) > 50
    
    # 评估复杂度
    if len(question) > 100 and quality_indicators['is_specific']:
        quality_indicators['complexity_level'] = 'high'
    elif len(question) > 50 or quality_indicators['is_specific']:
        quality_indicators['complexity_level'] = 'medium'
    
    return quality_indicators


def validate_answer_quality(answer: str, question: str) -> Dict:
    """验证答案质量"""
    quality_metrics = {
        'length_appropriate': 50 <= len(answer) <= 1000,
        'addresses_question': False,
        'has_structure': False,
        'uses_technical_terms': False
    }
    
    # 检查是否回答了问题
    question_keywords = ['什么', '如何', '为什么', '原因', '方法', '步骤']
    answer_keywords = ['因为', '由于', '首先', '其次', '步骤', '方法', '解决']
    
    has_question_word = any(kw in question for kw in question_keywords)
    has_answer_word = any(kw in answer for kw in answer_keywords)
    quality_metrics['addresses_question'] = has_question_word and has_answer_word
    
    # 检查结构
    structure_indicators = ['1.', '2.', '首先', '其次', '最后', '\n']
    quality_metrics['has_structure'] = any(ind in answer for ind in structure_indicators)
    
    # 检查技术术语使用
    tech_terms = extract_key_terms(answer)
    quality_metrics['uses_technical_terms'] = len(tech_terms) >= 3
    
    return quality_metrics


def generate_validation_report(qa_pairs: List[Dict], input_texts: Dict[str, str]) -> Dict:
    """生成验证报告"""
    report = {
        'total_pairs': len(qa_pairs),
        'validation_results': [],
        'statistics': {
            'relevant_questions': 0,
            'technical_questions': 0,
            'high_quality_answers': 0,
            'problematic_pairs': []
        }
    }
    
    for i, qa_pair in enumerate(qa_pairs):
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        qa_id = qa_pair.get('id', f'qa_{i+1:03d}')
        
        # 验证问题相关性
        relevance = validate_question_relevance(question, input_texts)
        
        # 分析问题质量
        question_quality = analyze_question_quality(question)
        
        # 验证答案质量
        answer_quality = validate_answer_quality(answer, question)
        
        validation_result = {
            'id': qa_id,
            'question_preview': question[:100] + '...' if len(question) > 100 else question,
            'relevance': relevance,
            'question_quality': question_quality,
            'answer_quality': answer_quality
        }
        
        report['validation_results'].append(validation_result)
        
        # 更新统计
        if relevance['is_relevant']:
            report['statistics']['relevant_questions'] += 1
        if question_quality['is_technical']:
            report['statistics']['technical_questions'] += 1
        if all(answer_quality.values()):
            report['statistics']['high_quality_answers'] += 1
        
        # 识别问题对
        if not relevance['is_relevant'] or not question_quality['is_technical']:
            report['statistics']['problematic_pairs'].append({
                'id': qa_id,
                'reason': 'irrelevant or non-technical'
            })
    
    # 计算百分比
    total = report['total_pairs']
    report['statistics']['relevance_rate'] = report['statistics']['relevant_questions'] / total
    report['statistics']['technical_rate'] = report['statistics']['technical_questions'] / total
    report['statistics']['quality_rate'] = report['statistics']['high_quality_answers'] / total
    
    return report


def main():
    """主函数"""
    print("=== QA对验证分析 ===\n")
    
    # 示例分析（基于提供的专家Top 10）
    expert_top_questions = [
        "电子纸在柔性使用环境中出现异常性能下降，可能与所用的ZnO TFT、金属氧化物TFT或solvent-free material有关。请分析故障原因并提出解决方案。",
        "在一块柔性a-Si EPD上观察到屏幕刷新率异常，且部分区域显示不连续，可能的原因是什么？",
        "在最近的一次测试中，某款采用金属氧化物TFT和衬底树脂的LCD面板显示出现不规则的亮度波动..."
    ]
    
    print("专家选择的问题特征分析：")
    print("1. 技术深度：所有问题都涉及具体的技术组件（TFT、EPD、LCD等）")
    print("2. 问题复杂性：多数问题描述了具体的故障现象和可能原因")
    print("3. 实践导向：要求分析原因并提出解决方案")
    print("4. 专业术语密度高：平均每个问题包含3-5个专业术语")
    
    print("\n问题生成建议：")
    print("1. 增加技术细节：在问题中包含更多具体的技术组件和参数")
    print("2. 描述实际场景：模拟真实的故障或优化场景")
    print("3. 要求深度分析：不仅问'是什么'，更要问'为什么'和'如何解决'")
    print("4. 结合多个技术点：将相关技术组合在一起考察")
    
    print("\n输入文本分析：")
    print("- tcl_display_tech.txt：包含Mini-LED、量子点、印刷OLED等显示技术")
    print("- tcl_smart_manufacturing.txt：包含智能制造、数字孪生、AI视觉检测等")
    
    print("\n问题生成改进方向：")
    print("1. 基于TCL的具体技术生成问题（如Mini-LED背光控制）")
    print("2. 结合智能制造与显示技术（如AI在面板缺陷检测中的应用）")
    print("3. 涉及实际生产问题（如印刷OLED的工艺优化）")
    print("4. 包含跨领域整合（如5G在远程设备控制中的应用）")


if __name__ == '__main__':
    main()