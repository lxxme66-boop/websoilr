#!/usr/bin/env python3
"""
问答对质量评测主程序
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

from evaluator import QAEvaluator, QAPair, EvaluationResult
from data_processor import DataProcessor


def setup_logging(config: Dict) -> logging.Logger:
    """设置日志系统"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # 创建logger
    logger = logging.getLogger('QAEvaluation')
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
    log_file = log_config.get('file', 'qa_evaluation.log')
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


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='评测问答对质量并筛选最佳结果'
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
        '--no-report',
        action='store_true',
        help='不生成评测报告'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        help='仅评测前N个样本（用于测试）'
    )
    
    parser.add_argument(
        '--skip-llm',
        action='store_true',
        help='跳过LLM评测（用于快速测试）'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出模式'
    )
    
    return parser.parse_args()


def convert_to_qa_pairs(qa_data: List[Dict]) -> List[QAPair]:
    """将字典数据转换为QAPair对象"""
    qa_pairs = []
    for i, item in enumerate(qa_data):
        qa_pair = QAPair(
            question=item['question'],
            answer=item['answer'],
            id=item.get('id', f'qa_{i}'),
            metadata=item.get('metadata', {})
        )
        qa_pairs.append(qa_pair)
    return qa_pairs


def save_evaluation_results(results: List[EvaluationResult], output_path: str,
                          save_details: bool = True):
    """保存评测结果"""
    output_data = []
    
    for result in results:
        qa_data = {
            'question': result.qa_pair.question,
            'answer': result.qa_pair.answer,
            'id': result.qa_pair.id,
            'total_score': result.total_score,
            'scores': result.scores
        }
        
        if save_details:
            qa_data['details'] = result.details
        
        if result.qa_pair.metadata:
            qa_data['metadata'] = result.qa_pair.metadata
        
        output_data.append(qa_data)
    
    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def generate_evaluation_report(results: List[EvaluationResult], evaluator: QAEvaluator,
                             report_dir: str, config: Dict):
    """生成评测报告"""
    os.makedirs(report_dir, exist_ok=True)
    
    # 生成基础报告
    report = evaluator.generate_report(results)
    
    # 添加额外统计信息
    scorer = evaluator.scorer
    statistics = scorer.calculate_statistics([{
        'scores': r.scores,
        'total_score': r.total_score
    } for r in results])
    
    report['detailed_statistics'] = statistics
    
    # 分数分布
    distribution = scorer.get_score_distribution([{
        'total_score': r.total_score
    } for r in results])
    report['score_distribution_detailed'] = distribution
    
    # 异常值分析
    outliers = scorer.identify_outliers([{
        'scores': r.scores,
        'total_score': r.total_score
    } for r in results])
    report['outliers'] = outliers
    
    # 保存JSON报告
    report_path = os.path.join(report_dir, f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 生成文本摘要
    summary_path = os.path.join(report_dir, f'evaluation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("问答对质量评测报告摘要\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"评测总数: {report['total_evaluated']}\n")
        f.write(f"平均得分: {report['statistics']['mean_score']:.3f}\n")
        f.write(f"标准差: {report['statistics']['std_score']:.3f}\n")
        f.write(f"最高分: {report['statistics']['max_score']:.3f}\n")
        f.write(f"最低分: {report['statistics']['min_score']:.3f}\n\n")
        
        f.write("分数分布:\n")
        for range_str, count in report['score_distribution'].items():
            f.write(f"  {range_str}: {count}\n")
        
        f.write("\n质量洞察:\n")
        for insight in report['quality_insights']:
            f.write(f"\n{insight['metric']}:\n")
            f.write(f"  最佳样本 (分数: {insight['best_example']['score']:.3f}):\n")
            f.write(f"    {insight['best_example']['question']}\n")
            f.write(f"  最差样本 (分数: {insight['worst_example']['score']:.3f}):\n")
            f.write(f"    {insight['worst_example']['question']}\n")
    
    return report_path, summary_path


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
    logger = setup_logging(config)
    
    logger.info("问答对质量评测系统启动")
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出文件: {args.output}")
    logger.info(f"配置文件: {args.config}")
    
    # 覆盖配置中的某些参数
    if args.min_score is not None:
        config['thresholds']['min_total_score'] = args.min_score
    
    if args.skip_llm:
        config['weights']['llm_score'] = 0.0
        logger.info("已跳过LLM评测")
    
    try:
        # 初始化数据处理器
        data_processor = DataProcessor(config.get('preprocessing', {}))
        
        # 加载问答对数据
        logger.info("正在加载问答对数据...")
        qa_data = data_processor.load_qa_pairs(args.input)
        
        # 数据验证和清洗
        valid_qa_data, invalid_qa_data = data_processor.batch_validate(qa_data)
        if invalid_qa_data:
            logger.warning(f"发现 {len(invalid_qa_data)} 个无效问答对")
            # 保存无效数据供检查
            invalid_path = args.output.replace('.json', '_invalid.json')
            data_processor.save_qa_pairs(invalid_qa_data, invalid_path)
        
        # 去重
        unique_qa_data = data_processor.deduplicate_qa_pairs(valid_qa_data)
        logger.info(f"去重后剩余 {len(unique_qa_data)} 个问答对")
        
        # 如果指定了样本数量，只取前N个
        if args.sample:
            unique_qa_data = unique_qa_data[:args.sample]
            logger.info(f"仅评测前 {args.sample} 个样本")
        
        # 转换为QAPair对象
        qa_pairs = convert_to_qa_pairs(unique_qa_data)
        
        # 初始化评测器
        logger.info("正在初始化评测器...")
        evaluator = QAEvaluator(config)
        
        # 执行评测
        logger.info(f"开始评测 {len(qa_pairs)} 个问答对...")
        results = evaluator.evaluate_batch(qa_pairs)
        
        # 选择最佳结果
        logger.info(f"正在选择Top-{args.top_k}个最佳问答对...")
        top_results = evaluator.select_top_k(results, args.top_k)
        
        # 保存结果
        logger.info(f"正在保存结果到 {args.output}...")
        save_evaluation_results(
            top_results, 
            args.output,
            save_details=config.get('output', {}).get('save_details', True)
        )
        
        # 生成报告
        if not args.no_report and config.get('output', {}).get('generate_report', True):
            logger.info("正在生成评测报告...")
            report_path, summary_path = generate_evaluation_report(
                results, evaluator, args.report_dir, config
            )
            logger.info(f"报告已保存到: {report_path}")
            logger.info(f"摘要已保存到: {summary_path}")
        
        # 输出统计信息
        logger.info(f"评测完成！")
        logger.info(f"总评测数: {len(results)}")
        logger.info(f"筛选后数量: {len(top_results)}")
        if top_results:
            avg_score = sum(r.total_score for r in top_results) / len(top_results)
            logger.info(f"筛选后平均分: {avg_score:.3f}")
        
    except Exception as e:
        logger.error(f"评测过程中发生错误: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("评测系统正常退出")


if __name__ == '__main__':
    main()