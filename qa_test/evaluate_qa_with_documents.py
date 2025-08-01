#!/usr/bin/env python3
"""
文档感知的问答对质量评测主程序
评估问题从文档中提取的合理性
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

from document_aware_evaluator import (
    DocumentAwareEvaluator, 
    DocumentAwareQAPair, 
    ExtractionEvaluationResult
)
from evaluator import QAEvaluator, QAPair


def setup_logging(config: Dict) -> logging.Logger:
    """设置日志系统"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # 创建logger
    logger = logging.getLogger('DocumentAwareQAEvaluation')
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
    log_file = log_config.get('file', 'document_aware_qa_evaluation.log')
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
        description='评测问答对质量并检查文档提取合理性'
    )
    
    # 必需参数
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='输入问答对文件路径（JSON格式）'
    )
    
    parser.add_argument(
        '--documents', '-d',
        required=True,
        nargs='+',
        help='源文档文件路径（可以是多个文件）'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出评测结果文件路径'
    )
    
    # 可选参数
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='配置文件路径（默认: config.yaml）'
    )
    
    parser.add_argument(
        '--min-reasonableness',
        type=float,
        default=0.6,
        help='最低合理性分数阈值（默认: 0.6）'
    )
    
    parser.add_argument(
        '--report-dir',
        default='reports',
        help='报告输出目录（默认: reports）'
    )
    
    parser.add_argument(
        '--combine-evaluation',
        action='store_true',
        help='是否同时进行传统质量评估和文档合理性评估'
    )
    
    parser.add_argument(
        '--export-issues',
        action='store_true',
        help='是否导出有问题的问答对'
    )
    
    return parser.parse_args()


def load_qa_pairs(file_path: str) -> List[DocumentAwareQAPair]:
    """加载问答对数据"""
    logger = logging.getLogger('DocumentAwareQAEvaluation')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"加载问答对文件失败: {e}")
        return []
    
    qa_pairs = []
    
    # 处理不同的数据格式
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                qa_pair = DocumentAwareQAPair(
                    question=item.get('question', ''),
                    answer=item.get('answer', ''),
                    source_document=item.get('source_document'),
                    document_id=item.get('document_id'),
                    metadata=item.get('metadata', {})
                )
                # 添加ID
                if 'id' in item:
                    qa_pair.metadata['id'] = item['id']
                qa_pairs.append(qa_pair)
    elif isinstance(data, dict):
        # 处理字典格式
        for key, value in data.items():
            if isinstance(value, dict):
                qa_pair = DocumentAwareQAPair(
                    question=value.get('question', ''),
                    answer=value.get('answer', ''),
                    source_document=value.get('source_document'),
                    document_id=value.get('document_id'),
                    metadata={'id': key}
                )
                qa_pairs.append(qa_pair)
    
    logger.info(f"成功加载 {len(qa_pairs)} 个问答对")
    return qa_pairs


def evaluate_with_documents(
    qa_pairs: List[DocumentAwareQAPair],
    document_paths: List[str],
    config: Dict,
    min_reasonableness: float = 0.6
) -> Dict:
    """使用文档进行评估"""
    logger = logging.getLogger('DocumentAwareQAEvaluation')
    
    # 创建文档感知评测器
    doc_evaluator = DocumentAwareEvaluator(config)
    
    # 加载文档
    logger.info("正在加载源文档...")
    doc_evaluator.load_documents(document_paths)
    
    # 批量评估
    logger.info("正在评估问答对的提取合理性...")
    results = doc_evaluator.batch_evaluate(
        qa_pairs,
        progress_callback=lambda current, total: logger.info(
            f"评估进度: {current}/{total} ({current/total*100:.1f}%)"
        )
    )
    
    # 生成报告
    report = doc_evaluator.generate_extraction_report(results)
    
    # 筛选合理的问答对
    reasonable_results = [
        r for r in results 
        if r.reasonableness_score >= min_reasonableness
    ]
    
    # 统计信息
    stats = {
        'total_evaluated': len(results),
        'reasonable_count': len(reasonable_results),
        'unreasonable_count': len(results) - len(reasonable_results),
        'average_score': report['average_reasonableness_score'],
        'score_distribution': report['score_distribution'],
        'dimension_analysis': report['dimension_analysis'],
        'document_coverage': report['document_coverage']
    }
    
    return {
        'results': results,
        'reasonable_results': reasonable_results,
        'report': report,
        'stats': stats
    }


def combine_evaluations(
    qa_pairs: List[DocumentAwareQAPair],
    document_paths: List[str],
    config: Dict
) -> Dict:
    """结合传统评估和文档合理性评估"""
    logger = logging.getLogger('DocumentAwareQAEvaluation')
    
    # 文档合理性评估
    doc_eval_results = evaluate_with_documents(
        qa_pairs, document_paths, config
    )
    
    # 传统质量评估
    logger.info("正在进行传统质量评估...")
    traditional_evaluator = QAEvaluator(config)
    
    # 转换为传统QAPair格式
    traditional_pairs = [
        QAPair(
            question=qa.question,
            answer=qa.answer,
            id=qa.metadata.get('id'),
            metadata=qa.metadata
        )
        for qa in qa_pairs
    ]
    
    traditional_results = traditional_evaluator.batch_evaluate(traditional_pairs)
    
    # 合并结果
    combined_results = []
    for i, (doc_result, trad_result) in enumerate(
        zip(doc_eval_results['results'], traditional_results)
    ):
        combined = {
            'qa_pair': doc_result.qa_pair,
            'document_evaluation': {
                'reasonableness_score': doc_result.reasonableness_score,
                'extraction_scores': doc_result.extraction_scores,
                'document_found': doc_result.document_context is not None
            },
            'quality_evaluation': {
                'total_score': trad_result.total_score,
                'scores': trad_result.scores
            },
            'combined_score': (
                0.6 * doc_result.reasonableness_score + 
                0.4 * trad_result.total_score
            )
        }
        combined_results.append(combined)
    
    return {
        'combined_results': combined_results,
        'document_evaluation': doc_eval_results,
        'quality_evaluation': traditional_results
    }


def save_results(results: Dict, output_path: str, args):
    """保存评测结果"""
    logger = logging.getLogger('DocumentAwareQAEvaluation')
    
    # 准备输出数据
    output_data = {
        'metadata': {
            'evaluation_time': datetime.now().isoformat(),
            'input_file': args.input,
            'document_files': args.documents,
            'total_qa_pairs': results['stats']['total_evaluated'],
            'reasonable_count': results['stats']['reasonable_count'],
            'min_reasonableness_threshold': args.min_reasonableness
        },
        'statistics': results['stats'],
        'report': results['report']
    }
    
    # 添加合理的问答对
    if results.get('reasonable_results'):
        output_data['reasonable_qa_pairs'] = [
            {
                'question': r.qa_pair.question,
                'answer': r.qa_pair.answer,
                'reasonableness_score': r.reasonableness_score,
                'extraction_scores': r.extraction_scores,
                'document_id': r.document_context.document_id if r.document_context else None,
                'metadata': r.qa_pair.metadata
            }
            for r in results['reasonable_results']
        ]
    
    # 保存结果
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"评测结果已保存到: {output_path}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")
        
    # 导出有问题的问答对
    if args.export_issues and results['report'].get('issues'):
        issues_path = output_path.replace('.json', '_issues.json')
        try:
            with open(issues_path, 'w', encoding='utf-8') as f:
                json.dump(results['report']['issues'], f, ensure_ascii=False, indent=2)
            logger.info(f"问题问答对已导出到: {issues_path}")
        except Exception as e:
            logger.error(f"导出问题失败: {e}")


def generate_detailed_report(results: Dict, report_dir: str):
    """生成详细报告"""
    logger = logging.getLogger('DocumentAwareQAEvaluation')
    
    # 创建报告目录
    os.makedirs(report_dir, exist_ok=True)
    
    # 生成Markdown报告
    report_path = os.path.join(report_dir, 'extraction_evaluation_report.md')
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 文档感知问答对评测报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 总体统计
            stats = results['stats']
            f.write("## 总体统计\n\n")
            f.write(f"- 总评测数: {stats['total_evaluated']}\n")
            f.write(f"- 合理问答对数: {stats['reasonable_count']}\n")
            f.write(f"- 不合理问答对数: {stats['unreasonable_count']}\n")
            f.write(f"- 平均合理性分数: {stats['average_score']:.3f}\n\n")
            
            # 分数分布
            f.write("## 分数分布\n\n")
            f.write("| 分数范围 | 数量 | 占比 |\n")
            f.write("|---------|------|------|\n")
            for range_key, count in stats['score_distribution'].items():
                percentage = count / stats['total_evaluated'] * 100
                f.write(f"| {range_key} | {count} | {percentage:.1f}% |\n")
            f.write("\n")
            
            # 维度分析
            f.write("## 维度分析\n\n")
            f.write("| 维度 | 平均分 | 标准差 | 最低分 | 最高分 |\n")
            f.write("|------|--------|--------|--------|--------|\n")
            for dim, scores in stats['dimension_analysis'].items():
                f.write(f"| {dim} | {scores['mean']:.3f} | "
                       f"{scores['std']:.3f} | {scores['min']:.3f} | "
                       f"{scores['max']:.3f} |\n")
            f.write("\n")
            
            # 文档覆盖率
            if stats.get('document_coverage'):
                f.write("## 文档覆盖率\n\n")
                f.write("| 文档ID | 问答对数量 |\n")
                f.write("|--------|------------|\n")
                for doc_id, count in stats['document_coverage'].items():
                    f.write(f"| {doc_id} | {count} |\n")
                f.write("\n")
            
            # 主要问题
            if results['report'].get('issues'):
                f.write("## 主要问题\n\n")
                for issue in results['report']['issues'][:10]:  # 只显示前10个
                    f.write(f"### 问答对 {issue['qa_id']}\n")
                    f.write(f"- 问题: {issue['question']}\n")
                    f.write(f"- 分数: {issue['score']:.3f}\n")
                    f.write(f"- 主要问题: {', '.join(issue['main_issues'])}\n\n")
                    
        logger.info(f"详细报告已生成: {report_path}")
        
    except Exception as e:
        logger.error(f"生成报告失败: {e}")


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    logger = setup_logging(config)
    logger.info("开始文档感知的问答对评测")
    
    # 加载问答对
    qa_pairs = load_qa_pairs(args.input)
    if not qa_pairs:
        logger.error("没有加载到问答对数据")
        return 1
    
    # 执行评估
    if args.combine_evaluation:
        # 结合传统评估和文档合理性评估
        results = combine_evaluations(qa_pairs, args.documents, config)
        # 使用文档评估结果
        eval_results = results['document_evaluation']
    else:
        # 仅进行文档合理性评估
        eval_results = evaluate_with_documents(
            qa_pairs, 
            args.documents, 
            config,
            args.min_reasonableness
        )
    
    # 保存结果
    save_results(eval_results, args.output, args)
    
    # 生成详细报告
    generate_detailed_report(eval_results, args.report_dir)
    
    # 打印总结
    logger.info("\n" + "="*50)
    logger.info("评测完成！")
    logger.info(f"总评测数: {eval_results['stats']['total_evaluated']}")
    logger.info(f"合理问答对: {eval_results['stats']['reasonable_count']}")
    logger.info(f"平均合理性分数: {eval_results['stats']['average_score']:.3f}")
    logger.info("="*50)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())