#!/usr/bin/env python3
"""
带文档验证的问答对质量评测主程序
确保生成的问答对基于原始文档内容
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
from document_relevance_checker import DocumentRelevanceChecker
from data_processor import DataProcessor


def setup_logging(config: Dict) -> logging.Logger:
    """设置日志系统"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # 创建logger
    logger = logging.getLogger('QAEvaluationWithDocs')
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
    log_file = log_config.get('file', 'qa_evaluation_with_docs.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 使用改进的权重配置（包含文档有效性）
    if 'weights' not in config:
        config['weights'] = {}
    
    config['weights'].update({
        'llm_score': 0.20,          # LLM评测权重
        'semantic_similarity': 0.08,  # 语义相似度权重
        'answer_quality': 0.15,      # 答案质量权重
        'fluency': 0.05,            # 流畅度权重
        'keyword_coverage': 0.05,    # 关键词权重
        'conciseness': 0.12,        # 简洁性
        'structure_clarity': 0.08,   # 结构清晰度
        'actionability': 0.08,       # 可操作性
        'document_validity': 0.19    # 文档有效性（最重要）
    })
    
    return config


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='带文档验证的问答对质量评测'
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
    
    parser.add_argument(
        '--docs', '-d',
        required=True,
        help='原始文档文件路径（用于验证）'
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
        help='最低分数阈值'
    )
    
    parser.add_argument(
        '--min-validity',
        type=float,
        default=0.5,
        help='最低文档有效性阈值（默认: 0.5）'
    )
    
    parser.add_argument(
        '--report-dir',
        default='reports',
        help='报告输出目录（默认: reports）'
    )
    
    parser.add_argument(
        '--strict-mode',
        action='store_true',
        help='严格模式：只保留高文档相关性的问答对'
    )
    
    return parser.parse_args()


def load_documents(file_path: str) -> List[str]:
    """加载原始文档"""
    documents = []
    
    # 支持多种格式
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                documents = data
            elif isinstance(data, dict):
                # 尝试常见的键名
                for key in ['documents', 'docs', 'texts', 'content']:
                    if key in data:
                        documents = data[key]
                        break
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            # 按段落分割
            content = f.read()
            documents = [p.strip() for p in content.split('\n\n') if p.strip()]
    else:
        # 尝试作为纯文本读取
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = [f.read()]
    
    return documents


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


def evaluate_with_document_check(
    qa_pairs: List[QAPair],
    evaluator: ImprovedQAEvaluator,
    doc_checker: DocumentRelevanceChecker,
    logger: logging.Logger
) -> List[Dict]:
    """
    评测问答对并检查文档相关性
    """
    logger.info("Starting evaluation with document relevance check...")
    
    # 批量评测
    results = evaluator.evaluate_batch(qa_pairs, show_progress=True)
    
    # 转换为字典格式
    dict_results = []
    for r in results:
        result_dict = evaluator._result_to_dict(r)
        dict_results.append(result_dict)
    
    # 检查文档相关性
    logger.info("Checking document relevance...")
    for result in dict_results:
        question = result['question']
        answer = result['answer']
        
        # 计算文档有效性得分
        validity_scores = doc_checker.compute_qa_validity_score(question, answer)
        
        # 添加到评分中
        result['scores']['document_validity'] = validity_scores['validity_score']
        result['validity_details'] = validity_scores
        
        # 重新计算总分（包含文档有效性）
        qa_dict = {'question': question, 'answer': answer}
        result['total_score'] = evaluator.scorer.compute_total_score(
            result['scores'], 
            qa_dict
        )
    
    return dict_results


def filter_by_validity(results: List[Dict], min_validity: float) -> List[Dict]:
    """根据文档有效性过滤结果"""
    filtered = []
    for result in results:
        validity_score = result['scores'].get('document_validity', 0)
        if validity_score >= min_validity:
            filtered.append(result)
    return filtered


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
            'scores': result['scores'],
            'validity_details': result.get('validity_details', {})
        }
        
        # 添加关键的详情
        if 'details' in result:
            output_item['details'] = {
                'llm_details': result['details'].get('llm_details', {}),
                'quality_details': result['details'].get('quality_details', {})
            }
        
        output_data.append(output_item)
    
    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def generate_validity_report(results: List[Dict]) -> Dict:
    """生成文档有效性报告"""
    validity_scores = [r['scores'].get('document_validity', 0) for r in results]
    question_relevance = [r['validity_details']['question_relevance'] for r in results if 'validity_details' in r]
    answer_consistency = [r['validity_details']['answer_consistency'] for r in results if 'validity_details' in r]
    
    report = {
        'document_validity': {
            'mean': sum(validity_scores) / len(validity_scores) if validity_scores else 0,
            'min': min(validity_scores) if validity_scores else 0,
            'max': max(validity_scores) if validity_scores else 0
        },
        'question_relevance': {
            'mean': sum(question_relevance) / len(question_relevance) if question_relevance else 0,
            'min': min(question_relevance) if question_relevance else 0,
            'max': max(question_relevance) if question_relevance else 0
        },
        'answer_consistency': {
            'mean': sum(answer_consistency) / len(answer_consistency) if answer_consistency else 0,
            'min': min(answer_consistency) if answer_consistency else 0,
            'max': max(answer_consistency) if answer_consistency else 0
        }
    }
    
    # 找出文档相关性问题
    low_validity = [r for r in results if r['scores'].get('document_validity', 0) < 0.5]
    if low_validity:
        report['low_validity_examples'] = [
            {
                'id': r['id'],
                'question': r['question'][:100] + '...',
                'validity_score': r['scores'].get('document_validity', 0),
                'issues': []
            }
            for r in low_validity[:5]
        ]
        
        # 分析问题
        for example in report['low_validity_examples']:
            result = next(r for r in low_validity if r['id'] == example['id'])
            details = result.get('validity_details', {})
            
            if details.get('question_details', {}).get('factual_grounding', 1) < 0.5:
                example['issues'].append('问题中的实体在文档中未找到')
            if details.get('answer_details', {}).get('hallucination_risk', 0) > 0.5:
                example['issues'].append('答案可能包含文档中没有的信息')
            if details.get('answer_details', {}).get('factual_accuracy', 1) < 0.5:
                example['issues'].append('答案缺乏文档支持')
    
    return report


def print_summary(report: Dict, validity_report: Dict, top_results: List[Dict]):
    """打印评测摘要"""
    print("\n" + "="*60)
    print("带文档验证的问答对质量评测报告")
    print("="*60)
    
    # 基础统计
    stats = report['statistics']['total_score']
    print(f"\n总评测数量: {report['total_evaluated']}")
    print(f"平均得分: {stats['mean']:.3f}")
    print(f"标准差: {stats['std']:.3f}")
    print(f"最高分: {stats['max']:.3f}")
    print(f"最低分: {stats['min']:.3f}")
    
    # 文档有效性统计
    print("\n文档有效性统计:")
    doc_validity = validity_report['document_validity']
    print(f"  平均有效性: {doc_validity['mean']:.3f}")
    print(f"  最低有效性: {doc_validity['min']:.3f}")
    print(f"  最高有效性: {doc_validity['max']:.3f}")
    
    print(f"\n问题相关性: {validity_report['question_relevance']['mean']:.3f}")
    print(f"答案一致性: {validity_report['answer_consistency']['mean']:.3f}")
    
    # 低有效性示例
    if 'low_validity_examples' in validity_report:
        print("\n文档相关性较低的示例:")
        for example in validity_report['low_validity_examples']:
            print(f"\n  ID: {example['id']} (有效性: {example['validity_score']:.3f})")
            print(f"  问题: {example['question']}")
            if example['issues']:
                print(f"  问题: {', '.join(example['issues'])}")
    
    # Top 5 结果
    print(f"\nTop 5 最佳问答对:")
    for i, result in enumerate(top_results[:5], 1):
        print(f"\n{i}. ID: {result['id']} (得分: {result['total_score']:.3f})")
        print(f"   文档有效性: {result['scores'].get('document_validity', 0):.3f}")
        print(f"   问题: {result['question'][:80]}...")


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    logger = setup_logging(config)
    logger.info("Starting QA evaluation with document validation...")
    
    # 更新配置中的阈值
    if args.min_score is not None:
        config['thresholds']['min_total_score'] = args.min_score
    
    # 严格模式：提高文档有效性要求
    if args.strict_mode:
        args.min_validity = max(args.min_validity, 0.7)
        logger.info(f"Strict mode enabled - minimum validity: {args.min_validity}")
    
    try:
        # 加载原始文档
        logger.info(f"Loading documents from {args.docs}")
        documents = load_documents(args.docs)
        logger.info(f"Loaded {len(documents)} documents")
        
        # 初始化组件
        data_processor = DataProcessor()
        evaluator = ImprovedQAEvaluator(config)
        doc_checker = DocumentRelevanceChecker(documents)
        
        # 加载问答对
        logger.info(f"Loading QA pairs from {args.input}")
        qa_pairs = load_qa_pairs(args.input, data_processor)
        logger.info(f"Loaded {len(qa_pairs)} QA pairs")
        
        # 评测并检查文档相关性
        results = evaluate_with_document_check(
            qa_pairs, evaluator, doc_checker, logger
        )
        
        # 根据文档有效性过滤
        logger.info(f"Filtering by document validity (min: {args.min_validity})")
        valid_results = filter_by_validity(results, args.min_validity)
        logger.info(f"Kept {len(valid_results)} valid results")
        
        # 排序结果
        logger.info("Ranking results...")
        ranked_results = evaluator.scorer.rank_qa_pairs(valid_results)
        
        # 应用其他过滤条件
        filters = {
            'min_total_score': config['thresholds']['min_total_score']
        }
        filtered_results = evaluator.scorer.apply_filters(ranked_results, filters)
        logger.info(f"Final filtered to {len(filtered_results)} results")
        
        # 保存结果
        logger.info(f"Saving top {args.top_k} results to {args.output}")
        save_results(filtered_results, args.output, args.top_k)
        
        # 生成报告
        if config['output']['generate_report']:
            logger.info("Generating evaluation report...")
            
            # 基础评测报告
            report = evaluator.generate_report(
                [evaluator._result_to_dict(r) for r in evaluator.evaluate_batch(qa_pairs, show_progress=False)]
            )
            
            # 文档有效性报告
            validity_report = generate_validity_report(results)
            
            # 保存报告
            os.makedirs(args.report_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            report_path = os.path.join(args.report_dir, f'evaluation_report_with_docs_{timestamp}.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'evaluation_report': report,
                    'validity_report': validity_report
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Report saved to {report_path}")
            
            # 打印摘要
            print_summary(report, validity_report, filtered_results)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


if __name__ == '__main__':
    main()