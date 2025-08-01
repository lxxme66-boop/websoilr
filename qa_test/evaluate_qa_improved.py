#!/usr/bin/env python3
"""
改进的问答对质量评测主程序
使用专家对齐的评分策略
"""

import os
import sys
import json
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator import QAEvaluator, QAPair, EvaluationResult
from data_processor import DataProcessor
from scorer_improved import ImprovedScorer


def setup_logging(config: Dict) -> logging.Logger:
    """设置日志系统"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # 创建logger
    logger = logging.getLogger('ImprovedQAEvaluation')
    logger.setLevel(log_level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    if log_config.get('console', True):
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    # 文件处理器
    log_file = log_config.get('file', 'qa_evaluation_improved.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_with_improved_scorer(evaluator: QAEvaluator, qa_pairs: List[QAPair], 
                                 improved_scorer: ImprovedScorer, logger: logging.Logger) -> List[Dict]:
    """
    使用改进的评分器进行评测
    
    Args:
        evaluator: 原始评测器
        qa_pairs: 问答对列表
        improved_scorer: 改进的评分器
        logger: 日志记录器
        
    Returns:
        评测结果列表
    """
    logger.info(f"Starting evaluation of {len(qa_pairs)} QA pairs with improved scorer...")
    
    # 使用原始评测器获取各维度得分
    evaluation_results = evaluator.evaluate_batch(qa_pairs)
    
    # 转换为字典格式并应用改进的评分
    improved_results = []
    
    for result in evaluation_results:
        # 提取数据
        qa_data = {
            'id': result.qa_pair.id,
            'question': result.qa_pair.question,
            'answer': result.qa_pair.answer,
            'scores': result.scores,
            'details': result.details
        }
        
        # 使用改进的评分器计算专家对齐得分
        expert_score, adjustments = improved_scorer.compute_expert_aligned_score(
            result.scores, 
            result.qa_pair.question, 
            result.qa_pair.answer
        )
        
        qa_data['total_score'] = result.total_score  # 原始总分
        qa_data['expert_aligned_score'] = expert_score  # 专家对齐得分
        qa_data['score_adjustments'] = adjustments  # 调整详情
        
        improved_results.append(qa_data)
    
    logger.info(f"Completed evaluation with improved scoring")
    
    return improved_results


def analyze_scoring_differences(original_results: List[Dict], logger: logging.Logger):
    """分析原始评分和专家对齐评分的差异"""
    
    differences = []
    
    for result in original_results:
        diff = {
            'id': result['id'],
            'original_score': result['total_score'],
            'expert_aligned_score': result['expert_aligned_score'],
            'difference': result['expert_aligned_score'] - result['total_score'],
            'adjustments': result['score_adjustments']['adjustments']
        }
        differences.append(diff)
    
    # 排序找出差异最大的案例
    differences.sort(key=lambda x: abs(x['difference']), reverse=True)
    
    logger.info("\n=== Top 10 Scoring Differences ===")
    for i, diff in enumerate(differences[:10]):
        logger.info(f"\n{i+1}. ID: {diff['id']}")
        logger.info(f"   Original Score: {diff['original_score']:.3f}")
        logger.info(f"   Expert Aligned Score: {diff['expert_aligned_score']:.3f}")
        logger.info(f"   Difference: {diff['difference']:+.3f}")
        if diff['adjustments']:
            logger.info("   Adjustments:")
            for adj in diff['adjustments']:
                logger.info(f"     - {adj['reason']}: {adj['value']:+.3f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='改进的问答对质量评测（专家对齐版）'
    )
    
    # 必需参数
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='输入问答对文件路径（JSON格式）'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出最佳问答对文件路径'
    )
    
    # 可选参数
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='配置文件路径（默认: config.yaml）'
    )
    
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=10,
        help='选择Top-K个最佳问答对（默认: 10）'
    )
    
    parser.add_argument(
        '--report-dir',
        default='reports_improved',
        help='报告输出目录（默认: reports_improved）'
    )
    
    parser.add_argument(
        '--compare-original',
        action='store_true',
        help='是否与原始评分进行对比'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    logger = setup_logging(config)
    
    # 创建评测器和改进的评分器
    evaluator = QAEvaluator(config)
    improved_scorer = ImprovedScorer(expert_alignment=True)
    
    # 加载问答对
    logger.info(f"Loading QA pairs from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    # 转换为QAPair对象
    qa_pairs = []
    for item in qa_data:
        qa_pair = QAPair(
            question=item['question'],
            answer=item['answer'],
            id=item.get('id'),
            metadata=item.get('metadata', {})
        )
        qa_pairs.append(qa_pair)
    
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")
    
    # 执行评测
    results = evaluate_with_improved_scorer(evaluator, qa_pairs, improved_scorer, logger)
    
    # 如果需要对比原始评分
    if args.compare_original:
        analyze_scoring_differences(results, logger)
    
    # 获取Top-K结果
    top_k_results = improved_scorer.get_top_k_expert_aligned(results, k=args.top_k)
    
    # 生成统计报告
    logger.info("\n=== Evaluation Statistics ===")
    
    # 计算统计信息
    all_scores = [r['expert_aligned_score'] for r in results]
    logger.info(f"Total evaluated: {len(results)}")
    logger.info(f"Expert-aligned score statistics:")
    logger.info(f"  Mean: {np.mean(all_scores):.3f}")
    logger.info(f"  Std: {np.std(all_scores):.3f}")
    logger.info(f"  Min: {np.min(all_scores):.3f}")
    logger.info(f"  Max: {np.max(all_scores):.3f}")
    logger.info(f"  Median: {np.median(all_scores):.3f}")
    
    # 显示Top-K结果
    logger.info(f"\n=== Top {args.top_k} QA Pairs (Expert Aligned) ===")
    for i, result in enumerate(top_k_results):
        logger.info(f"\n{i+1}. ID: {result['id']}")
        logger.info(f"   Expert Aligned Score: {result['expert_aligned_score']:.3f}")
        logger.info(f"   Original Score: {result['total_score']:.3f}")
        logger.info(f"   LLM Score: {result['scores']['llm_score']:.3f}")
        logger.info(f"   Question: {result['question'][:100]}...")
    
    # 保存结果
    output_data = []
    for result in top_k_results:
        output_item = {
            'id': result['id'],
            'question': result['question'],
            'answer': result['answer'],
            'expert_aligned_score': result['expert_aligned_score'],
            'original_score': result['total_score'],
            'scores': result['scores']
        }
        output_data.append(output_item)
    
    # 写入输出文件
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nTop {args.top_k} QA pairs saved to {args.output}")
    
    # 生成详细报告
    if config.get('output', {}).get('generate_report', True):
        report_dir = Path(args.report_dir)
        report_dir.mkdir(exist_ok=True)
        
        # 生成完整报告
        report_path = report_dir / f"evaluation_report_improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_evaluated': len(results),
                'top_k': args.top_k,
                'config': config
            },
            'statistics': {
                'expert_aligned_scores': {
                    'mean': float(np.mean(all_scores)),
                    'std': float(np.std(all_scores)),
                    'min': float(np.min(all_scores)),
                    'max': float(np.max(all_scores)),
                    'median': float(np.median(all_scores))
                }
            },
            'top_k_results': output_data,
            'all_results': results if config.get('output', {}).get('save_details', False) else None
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Detailed report saved to {report_path}")


if __name__ == '__main__':
    main()