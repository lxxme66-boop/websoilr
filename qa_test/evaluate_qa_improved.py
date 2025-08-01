#!/usr/bin/env python3
"""
改进的问答对质量评测主程序
使用更符合专家偏好的评分系统
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

from evaluator_improved import ImprovedQAEvaluator, QAPair, EvaluationResult
from data_processor import DataProcessor


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
    
    # 使用改进的权重配置
    if 'weights' not in config:
        config['weights'] = {}
    
    # 覆盖默认权重以更好地对齐专家偏好
    config['weights'].update({
        'llm_score': 0.25,          # 降低LLM评测权重
        'semantic_similarity': 0.10,  # 降低语义相似度权重
        'answer_quality': 0.20,      # 保持答案质量权重
        'fluency': 0.05,            # 降低流畅度权重
        'keyword_coverage': 0.05,    # 降低关键词权重
        'conciseness': 0.15,        # 新增：简洁性
        'structure_clarity': 0.10,   # 新增：结构清晰度
        'actionability': 0.10        # 新增：可操作性
    })
    
    return config


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='改进的问答对质量评测，更好地对齐专家偏好'
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
        default=100,
        help='选择Top-K个最佳问答对（默认: 100）'
    )
    
    parser.add_argument(
        '--min-score',
        type=float,
        help='最低分数阈值（覆盖配置文件中的设置）'
    )
    
    parser.add_argument(
        '--report-dir',
        default='reports',
        help='报告输出目录（默认: reports）'
    )
    
    parser.add_argument(
        '--expert-mode',
        action='store_true',
        help='启用专家模式，使用更严格的评分标准'
    )
    
    return parser.parse_args()


def load_qa_pairs(file_path: str, data_processor: DataProcessor) -> List[QAPair]:
    """加载问答对数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs = []
    for idx, item in enumerate(data):
        # 数据预处理
        processed = data_processor.preprocess_qa_pair(
            item.get('question', ''),
            item.get('answer', '')
        )
        
        qa_pair = QAPair(
            question=processed['question'],
            answer=processed['answer'],
            id=item.get('id', f'qa_{idx:03d}'),
            metadata=item.get('metadata', {})
        )
        qa_pairs.append(qa_pair)
    
    return qa_pairs


def save_results(results: List[Dict], output_path: str, top_k: int = None):
    """保存评测结果"""
    # 如果指定了top_k，只保存前k个
    if top_k:
        results = results[:top_k]
    
    # 准备输出数据
    output_data = []
    for result in results:
        output_item = {
            'question': result['question'],
            'answer': result['answer'],
            'id': result['id'],
            'total_score': result['total_score'],
            'scores': result['scores']
        }
        
        # 添加额外的评分信息
        if 'base_score' in result:
            output_item['base_score'] = result['base_score']
        if 'penalty' in result:
            output_item['penalty'] = result['penalty']
        
        # 只保留关键的详情
        if 'details' in result:
            output_item['details'] = {
                'llm_details': result['details'].get('llm_details', {}),
                'quality_details': result['details'].get('quality_details', {})
            }
        
        output_data.append(output_item)
    
    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def save_report(report: Dict, report_dir: str):
    """保存评测报告"""
    # 创建报告目录
    os.makedirs(report_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存完整报告
    report_path = os.path.join(report_dir, f'evaluation_report_{timestamp}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 保存摘要报告
    summary = {
        'timestamp': timestamp,
        'total_evaluated': report['total_evaluated'],
        'statistics': report['statistics']['total_score'],
        'score_distribution': report['score_distribution'],
        'expert_alignment': report.get('expert_alignment', {})
    }
    
    summary_path = os.path.join(report_dir, f'summary_{timestamp}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return report_path, summary_path


def print_summary(report: Dict, top_results: List[Dict]):
    """打印评测摘要"""
    print("\n" + "="*60)
    print("改进的问答对质量评测报告")
    print("="*60)
    
    # 基础统计
    stats = report['statistics']['total_score']
    print(f"\n总评测数量: {report['total_evaluated']}")
    print(f"平均得分: {stats['mean']:.3f}")
    print(f"标准差: {stats['std']:.3f}")
    print(f"最高分: {stats['max']:.3f}")
    print(f"最低分: {stats['min']:.3f}")
    
    # 分数分布
    print("\n分数分布:")
    dist = report['score_distribution']
    for i, (bin_range, count, percentage) in enumerate(
        zip(dist['bins'], dist['counts'], dist['percentages'])
    ):
        bar = '█' * int(percentage / 2)
        print(f"  {bin_range}: {bar} {count} ({percentage:.1f}%)")
    
    # 专家对齐信息
    if 'expert_alignment' in report:
        alignment = report['expert_alignment']
        if 'recommendations' in alignment:
            print("\n改进建议:")
            for rec in alignment['recommendations']:
                print(f"  - {rec}")
    
    # Top 5 结果
    print(f"\nTop 5 最佳问答对:")
    for i, result in enumerate(top_results[:5], 1):
        print(f"\n{i}. ID: {result['id']} (得分: {result['total_score']:.3f})")
        print(f"   问题: {result['question'][:80]}...")
        if 'penalty' in result and result['penalty'] > 0:
            print(f"   惩罚: -{result['penalty']:.3f} (基础分: {result['base_score']:.3f})")


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    logger = setup_logging(config)
    logger.info("Starting improved QA evaluation...")
    
    # 更新配置中的阈值
    if args.min_score is not None:
        config['thresholds']['min_total_score'] = args.min_score
    
    # 专家模式：使用更严格的标准
    if args.expert_mode:
        config['thresholds']['min_total_score'] = 0.75
        config['thresholds']['min_answer_quality'] = 0.7
        logger.info("Expert mode enabled - using stricter criteria")
    
    try:
        # 初始化组件
        data_processor = DataProcessor()
        evaluator = ImprovedQAEvaluator(config)
        
        # 加载问答对
        logger.info(f"Loading QA pairs from {args.input}")
        qa_pairs = load_qa_pairs(args.input, data_processor)
        logger.info(f"Loaded {len(qa_pairs)} QA pairs")
        
        # 批量评测
        logger.info("Evaluating QA pairs...")
        results = evaluator.evaluate_batch(qa_pairs, show_progress=True)
        
        # 排序结果
        logger.info("Ranking results...")
        ranked_results = evaluator.rank_results(results)
        
        # 应用过滤条件
        filters = {
            'min_total_score': config['thresholds']['min_total_score']
        }
        
        # 添加其他过滤条件
        for key, value in config['thresholds'].items():
            if key.startswith('min_') and key != 'min_total_score':
                filters[key] = value
        
        filtered_results = evaluator.scorer.apply_filters(ranked_results, filters)
        logger.info(f"Filtered to {len(filtered_results)} results")
        
        # 保存结果
        logger.info(f"Saving top {args.top_k} results to {args.output}")
        save_results(filtered_results, args.output, args.top_k)
        
        # 生成报告
        if config['output']['generate_report']:
            logger.info("Generating evaluation report...")
            report = evaluator.generate_report(results)
            report_path, summary_path = save_report(report, args.report_dir)
            logger.info(f"Report saved to {report_path}")
            logger.info(f"Summary saved to {summary_path}")
            
            # 打印摘要
            print_summary(report, filtered_results)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


if __name__ == '__main__':
    main()