#!/usr/bin/env python3
"""
集成文档评测的问答对质量评测主程序
支持基于文档内容的准确性评测
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluator import QAEvaluator, QAPair, EvaluationResult
from document_based_evaluator import DocumentBasedEvaluator, DocumentEvaluationResult
from data_processor import DataProcessor


class IntegratedQAEvaluator:
    """集成了文档评测的问答对评测器"""
    
    def __init__(self, config: Dict, document_dir: Optional[str] = None):
        """
        初始化评测器
        
        Args:
            config: 配置字典
            document_dir: 文档目录路径（可选）
        """
        self.config = config
        self.logger = self._setup_logger()
        
        # 初始化原有评测器
        self.qa_evaluator = QAEvaluator(config)
        
        # 如果提供了文档目录，初始化文档评测器
        self.doc_evaluator = None
        if document_dir and os.path.exists(document_dir):
            self.logger.info(f"Initializing document-based evaluator with documents from: {document_dir}")
            self.doc_evaluator = DocumentBasedEvaluator(
                document_dir=document_dir,
                model_name=config.get('document_evaluation', {}).get('model', 'paraphrase-multilingual-mpnet-base-v2')
            )
        
        # 权重配置
        doc_eval_config = config.get('document_evaluation', {})
        self.use_document_evaluation = self.doc_evaluator is not None
        self.doc_weight = doc_eval_config.get('weight', 0.5) if self.use_document_evaluation else 0.0
        self.qa_weight = 1.0 - self.doc_weight
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('IntegratedQAEvaluator')
        logger.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
    def evaluate_single(self, qa_pair: QAPair) -> Dict:
        """
        评测单个问答对
        
        Args:
            qa_pair: 问答对
            
        Returns:
            综合评测结果
        """
        result = {
            'qa_pair': qa_pair,
            'scores': {},
            'details': {}
        }
        
        # 原有评测
        qa_result = self.qa_evaluator.evaluate_single(qa_pair)
        result['scores']['qa_evaluation'] = qa_result.scores
        result['scores']['qa_total'] = qa_result.total_score
        result['details']['qa_evaluation'] = qa_result.details
        
        # 文档评测（如果启用）
        if self.use_document_evaluation:
            doc_result = self.doc_evaluator.evaluate_qa_pair(qa_pair)
            result['scores']['document_evaluation'] = {
                'relevance': doc_result.relevance_score,
                'accuracy': doc_result.accuracy_score,
                'coverage': doc_result.coverage_score,
                'consistency': doc_result.consistency_score,
                'total': doc_result.total_score
            }
            result['details']['document_evaluation'] = {
                'supporting_sentences': doc_result.supporting_sentences,
                'key_terms_matched': doc_result.details.get('key_terms_matched', {})
            }
            
            # 计算综合得分
            result['scores']['total'] = (
                qa_result.total_score * self.qa_weight +
                doc_result.total_score * self.doc_weight
            )
        else:
            result['scores']['total'] = qa_result.total_score
        
        return result
    
    def evaluate_batch(self, qa_pairs: List[QAPair]) -> List[Dict]:
        """批量评测问答对"""
        results = []
        
        for qa_pair in qa_pairs:
            self.logger.info(f"Evaluating: {qa_pair.question[:50]}...")
            result = self.evaluate_single(qa_pair)
            results.append(result)
            
            # 记录得分
            if self.use_document_evaluation:
                self.logger.info(
                    f"  QA Score: {result['scores']['qa_total']:.3f}, "
                    f"Doc Score: {result['scores']['document_evaluation']['total']:.3f}, "
                    f"Total: {result['scores']['total']:.3f}"
                )
            else:
                self.logger.info(f"  Score: {result['scores']['total']:.3f}")
        
        return results


def setup_logging(config: Dict) -> logging.Logger:
    """设置日志系统"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    logger = logging.getLogger('QAEvaluation')
    logger.setLevel(log_level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if log_config.get('console', True):
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
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
        description='评测问答对质量（支持基于文档的评测）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 仅使用传统评测
  python evaluate_qa_with_docs.py \\
    --input qa_pairs.json \\
    --output best_qa_pairs.json

  # 使用文档评测
  python evaluate_qa_with_docs.py \\
    --input qa_pairs.json \\
    --output best_qa_pairs.json \\
    --documents ../websailor_domain/input_texts

  # 调整文档评测权重
  python evaluate_qa_with_docs.py \\
    --input qa_pairs.json \\
    --output best_qa_pairs.json \\
    --documents ../websailor_domain/input_texts \\
    --doc-weight 0.7
        """
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
        '--documents', '-d',
        help='文档目录路径（用于基于文档的评测）'
    )
    
    parser.add_argument(
        '--doc-weight',
        type=float,
        default=0.5,
        help='文档评测权重（0-1之间，默认: 0.5）'
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
        '--report-dir',
        default='reports',
        help='报告输出目录（默认: reports）'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细输出'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 加载配置
    config = load_config(args.config)
    
    # 更新文档评测配置
    if args.documents:
        config.setdefault('document_evaluation', {})
        config['document_evaluation']['weight'] = args.doc_weight
    
    # 设置日志
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("问答对质量评测系统")
    logger.info("=" * 60)
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出文件: {args.output}")
    logger.info(f"配置文件: {args.config}")
    if args.documents:
        logger.info(f"文档目录: {args.documents}")
        logger.info(f"文档评测权重: {args.doc_weight}")
    logger.info(f"Top-K: {args.top_k}")
    
    try:
        # 加载问答对
        logger.info("\n加载问答对...")
        with open(args.input, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        # 处理不同的JSON格式
        if isinstance(qa_data, dict) and 'qa_pairs' in qa_data:
            qa_pairs_data = qa_data['qa_pairs']
        else:
            qa_pairs_data = qa_data
        
        # 转换为QAPair对象
        qa_pairs = []
        for i, item in enumerate(qa_pairs_data):
            qa_pair = QAPair(
                question=item['question'],
                answer=item['answer'],
                id=item.get('id', f'qa_{i}'),
                metadata=item.get('metadata', {})
            )
            qa_pairs.append(qa_pair)
        
        logger.info(f"加载了 {len(qa_pairs)} 个问答对")
        
        # 创建评测器
        logger.info("\n初始化评测器...")
        evaluator = IntegratedQAEvaluator(config, args.documents)
        
        # 评测
        logger.info("\n开始评测...")
        results = evaluator.evaluate_batch(qa_pairs)
        
        # 按总分排序
        results.sort(key=lambda x: x['scores']['total'], reverse=True)
        
        # 应用最低分数阈值
        if args.min_score:
            results = [r for r in results if r['scores']['total'] >= args.min_score]
            logger.info(f"应用最低分数阈值 {args.min_score} 后剩余 {len(results)} 个问答对")
        
        # 选择Top-K
        selected_results = results[:args.top_k]
        
        # 准备输出数据
        output_data = {
            'metadata': {
                'total_evaluated': len(qa_pairs),
                'selected_count': len(selected_results),
                'evaluation_time': datetime.now().isoformat(),
                'config': {
                    'top_k': args.top_k,
                    'min_score': args.min_score,
                    'use_document_evaluation': evaluator.use_document_evaluation,
                    'document_weight': evaluator.doc_weight if evaluator.use_document_evaluation else 0
                }
            },
            'summary': {
                'avg_total_score': sum(r['scores']['total'] for r in selected_results) / len(selected_results) if selected_results else 0
            },
            'qa_pairs': []
        }
        
        # 如果使用了文档评测，添加相关摘要
        if evaluator.use_document_evaluation:
            doc_scores = [r['scores']['document_evaluation'] for r in selected_results]
            output_data['summary']['document_evaluation'] = {
                'avg_relevance': sum(s['relevance'] for s in doc_scores) / len(doc_scores) if doc_scores else 0,
                'avg_accuracy': sum(s['accuracy'] for s in doc_scores) / len(doc_scores) if doc_scores else 0,
                'avg_coverage': sum(s['coverage'] for s in doc_scores) / len(doc_scores) if doc_scores else 0,
                'avg_consistency': sum(s['consistency'] for s in doc_scores) / len(doc_scores) if doc_scores else 0
            }
        
        # 构建输出的问答对
        for result in selected_results:
            qa_item = {
                'id': result['qa_pair'].id,
                'question': result['qa_pair'].question,
                'answer': result['qa_pair'].answer,
                'scores': result['scores'],
                'metadata': result['qa_pair'].metadata
            }
            
            # 如果使用了文档评测，添加支持证据
            if evaluator.use_document_evaluation and 'document_evaluation' in result['details']:
                qa_item['supporting_evidence'] = [
                    {'sentence': sent, 'similarity': sim}
                    for sent, sim in result['details']['document_evaluation']['supporting_sentences']
                ]
            
            output_data['qa_pairs'].append(qa_item)
        
        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n结果已保存到: {args.output}")
        
        # 生成报告
        if not os.path.exists(args.report_dir):
            os.makedirs(args.report_dir)
        
        report_path = os.path.join(
            args.report_dir,
            f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        
        generate_report(results, selected_results, evaluator, report_path)
        logger.info(f"报告已保存到: {report_path}")
        
        # 显示摘要
        logger.info("\n" + "=" * 60)
        logger.info("评测完成！")
        logger.info("=" * 60)
        logger.info(f"总评测数量: {len(qa_pairs)}")
        logger.info(f"选中数量: {len(selected_results)}")
        logger.info(f"平均总分: {output_data['summary']['avg_total_score']:.3f}")
        
        if evaluator.use_document_evaluation:
            doc_summary = output_data['summary']['document_evaluation']
            logger.info("\n文档评测摘要:")
            logger.info(f"  - 平均相关性: {doc_summary['avg_relevance']:.3f}")
            logger.info(f"  - 平均准确性: {doc_summary['avg_accuracy']:.3f}")
            logger.info(f"  - 平均覆盖度: {doc_summary['avg_coverage']:.3f}")
            logger.info(f"  - 平均一致性: {doc_summary['avg_consistency']:.3f}")
        
    except Exception as e:
        logger.error(f"评测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_report(all_results: List[Dict], selected_results: List[Dict], 
                   evaluator: IntegratedQAEvaluator, report_path: str):
    """生成评测报告"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 问答对评测报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 概览
        f.write("## 评测概览\n\n")
        f.write(f"- 总评测数量: {len(all_results)}\n")
        f.write(f"- 选中数量: {len(selected_results)}\n")
        f.write(f"- 使用文档评测: {'是' if evaluator.use_document_evaluation else '否'}\n")
        if evaluator.use_document_evaluation:
            f.write(f"- 文档评测权重: {evaluator.doc_weight:.2f}\n")
            f.write(f"- QA评测权重: {evaluator.qa_weight:.2f}\n")
        f.write("\n")
        
        # 分数分布
        f.write("## 分数分布\n\n")
        all_scores = [r['scores']['total'] for r in all_results]
        selected_scores = [r['scores']['total'] for r in selected_results]
        
        f.write("### 所有问答对\n")
        f.write(f"- 最高分: {max(all_scores):.3f}\n")
        f.write(f"- 最低分: {min(all_scores):.3f}\n")
        f.write(f"- 平均分: {sum(all_scores) / len(all_scores):.3f}\n\n")
        
        f.write("### 选中的问答对\n")
        f.write(f"- 最高分: {max(selected_scores):.3f}\n")
        f.write(f"- 最低分: {min(selected_scores):.3f}\n")
        f.write(f"- 平均分: {sum(selected_scores) / len(selected_scores):.3f}\n\n")
        
        # Top 10 示例
        f.write("## Top 10 最佳问答对\n\n")
        for i, result in enumerate(selected_results[:10]):
            qa_pair = result['qa_pair']
            f.write(f"### {i+1}. {qa_pair.question}\n\n")
            f.write(f"**答案**: {qa_pair.answer}\n\n")
            f.write(f"**总分**: {result['scores']['total']:.3f}\n\n")
            
            if evaluator.use_document_evaluation:
                doc_scores = result['scores']['document_evaluation']
                f.write("**文档评测分数**:\n")
                f.write(f"- 相关性: {doc_scores['relevance']:.3f}\n")
                f.write(f"- 准确性: {doc_scores['accuracy']:.3f}\n")
                f.write(f"- 覆盖度: {doc_scores['coverage']:.3f}\n")
                f.write(f"- 一致性: {doc_scores['consistency']:.3f}\n\n")
                
                if 'document_evaluation' in result['details']:
                    supporting = result['details']['document_evaluation']['supporting_sentences']
                    if supporting:
                        f.write("**支持证据**:\n")
                        for sent, sim in supporting[:2]:
                            f.write(f"- {sent} (相似度: {sim:.3f})\n")
                        f.write("\n")
            
            f.write("---\n\n")


if __name__ == "__main__":
    main()