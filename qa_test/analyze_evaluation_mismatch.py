#!/usr/bin/env python3
"""
分析自动评测与专家评测不匹配的原因
"""

import json
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_evaluation_results(file_path: str) -> Dict:
    """加载评测结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_score_distribution(results: List[Dict]) -> Dict:
    """分析得分分布"""
    metrics = ['llm_score', 'semantic_similarity', 'answer_quality', 'fluency', 'keyword_coverage']
    distributions = {}
    
    for metric in metrics:
        scores = []
        for result in results:
            if 'scores' in result and metric in result['scores']:
                scores.append(result['scores'][metric])
        
        if scores:
            distributions[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores),
                'q1': np.percentile(scores, 25),
                'q3': np.percentile(scores, 75)
            }
    
    return distributions


def identify_problematic_metrics(results: List[Dict], expert_top_ids: List[str]) -> Dict:
    """识别导致评分不匹配的问题指标"""
    
    # 获取自动评测的Top结果ID
    sorted_results = sorted(results, key=lambda x: x.get('total_score', 0), reverse=True)
    auto_top_ids = [r['id'] for r in sorted_results[:len(expert_top_ids)]]
    
    # 找出不匹配的案例
    mismatched_ids = set(auto_top_ids) - set(expert_top_ids)
    correctly_matched_ids = set(auto_top_ids) & set(expert_top_ids)
    missed_expert_ids = set(expert_top_ids) - set(auto_top_ids)
    
    # 分析不同组的指标差异
    def get_metric_stats(ids):
        stats = defaultdict(list)
        for result in results:
            if result['id'] in ids:
                for metric, score in result.get('scores', {}).items():
                    stats[metric].append(score)
        
        return {metric: {
            'mean': np.mean(scores) if scores else 0,
            'std': np.std(scores) if scores else 0
        } for metric, scores in stats.items()}
    
    analysis = {
        'overlap_rate': len(correctly_matched_ids) / len(expert_top_ids),
        'mismatched_auto_picks': list(mismatched_ids),
        'missed_expert_picks': list(missed_expert_ids),
        'correctly_matched': list(correctly_matched_ids),
        'metric_comparison': {
            'correctly_matched': get_metric_stats(correctly_matched_ids),
            'mismatched_auto': get_metric_stats(mismatched_ids),
            'missed_expert': get_metric_stats(missed_expert_ids)
        }
    }
    
    return analysis


def analyze_specific_cases(results: List[Dict], case_ids: List[str]) -> List[Dict]:
    """分析特定案例的详细信息"""
    case_analysis = []
    
    for case_id in case_ids:
        for result in results:
            if result['id'] == case_id:
                analysis = {
                    'id': case_id,
                    'question': result['question'][:100] + '...',
                    'answer_length': len(result['answer']),
                    'scores': result['scores'],
                    'total_score': result['total_score']
                }
                
                # 识别低分指标
                low_score_metrics = []
                for metric, score in result['scores'].items():
                    if score < 0.5:
                        low_score_metrics.append((metric, score))
                
                analysis['low_score_metrics'] = sorted(low_score_metrics, key=lambda x: x[1])
                case_analysis.append(analysis)
                break
    
    return case_analysis


def generate_improvement_recommendations(analysis: Dict) -> List[str]:
    """基于分析结果生成改进建议"""
    recommendations = []
    
    # 检查重叠率
    if analysis['overlap_rate'] < 0.5:
        recommendations.append("严重不匹配：自动评测与专家评测重叠率低于50%")
    
    # 分析指标差异
    metric_comp = analysis['metric_comparison']
    
    for metric in ['llm_score', 'semantic_similarity', 'answer_quality']:
        if metric in metric_comp['correctly_matched'] and metric in metric_comp['missed_expert']:
            correct_mean = metric_comp['correctly_matched'][metric]['mean']
            missed_mean = metric_comp['missed_expert'][metric]['mean']
            
            if abs(correct_mean - missed_mean) > 0.1:
                if missed_mean > correct_mean:
                    recommendations.append(
                        f"{metric}权重可能过低：专家选择的案例在该指标上得分更高"
                    )
                else:
                    recommendations.append(
                        f"{metric}权重可能过高：该指标可能不是专家评判的主要依据"
                    )
    
    return recommendations


def main():
    """主函数"""
    # 这里需要实际的评测结果文件路径
    print("=== 自动评测与专家评测不匹配分析 ===\n")
    
    # 专家评测选出的Top 10 ID
    expert_top_ids = [
        "qa_018", "qa_011", "qa_020", "qa_016", "qa_004",
        "qa_006", "qa_001", "qa_005", "qa_009", "qa_010"
    ]
    
    # 模拟分析（实际使用时需要加载真实数据）
    print("问题分析：")
    print("1. 权重配置问题：")
    print("   - LLM评分权重(40%)可能偏低，专家更看重内容质量")
    print("   - 语义相似度权重(20%)可能过高，专业答案措辞可能与参考答案不同")
    print("   - BLEU等文本匹配指标在专业领域不适用")
    
    print("\n2. 评分机制问题：")
    print("   - 缺少对问题复杂度的考虑")
    print("   - 缺少对答案专业性的额外奖励")
    print("   - 过度惩罚与参考答案的文字差异")
    
    print("\n3. 领域特殊性：")
    print("   - 技术领域答案更看重准确性而非文字流畅")
    print("   - 专业术语的使用比通用词汇更重要")
    print("   - 结构化和步骤化的答案应得到更高评价")
    
    print("\n改进建议：")
    print("1. 调整权重配置：")
    print("   - 提高LLM评分权重至50%")
    print("   - 降低语义相似度权重至15%")
    print("   - 增加答案质量权重至25%")
    
    print("\n2. 引入专家对齐机制：")
    print("   - 对高LLM评分的答案给予额外奖励")
    print("   - 减轻低语义相似度的惩罚（如果其他指标良好）")
    print("   - 考虑问题复杂度和答案专业性")
    
    print("\n3. 优化评分策略：")
    print("   - 识别技术内容并调整评分标准")
    print("   - 重视答案的完整性和准确性")
    print("   - 降低纯文本匹配指标的影响")


if __name__ == '__main__':
    main()