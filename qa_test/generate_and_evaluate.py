#!/usr/bin/env python3
"""
问答对生成与评测集成系统
支持从多个txt文件生成问答对并进行质量评测
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
import tempfile

# 添加父目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator import QAEvaluator, QAPair, EvaluationResult
from data_processor import DataProcessor
from knowledge_graph_builder_improved import KnowledgeGraphBuilder
from question_generator_optimized import QuestionGenerator


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """设置日志系统"""
    log_level_obj = getattr(logging, log_level.upper())
    
    # 创建logger
    logger = logging.getLogger('GenerateAndEvaluate')
    logger.setLevel(log_level_obj)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(log_level_obj)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # 文件处理器
    log_file = f'generate_evaluate_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
        description='从多个txt文件生成问答对并进行质量评测'
    )
    
    # 必需参数
    parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='包含txt文件的输入目录路径'
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
        help='最低分数阈值'
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
        '--skip-llm',
        action='store_true',
        help='跳过LLM评测（仅使用基于规则的评测）'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细日志'
    )
    
    parser.add_argument(
        '--kg-config',
        help='知识图谱构建配置文件路径'
    )
    
    parser.add_argument(
        '--qg-config',
        help='问题生成配置文件路径'
    )
    
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='保存中间结果（知识图谱、所有生成的问答对等）'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        help='最多处理的txt文件数量'
    )
    
    return parser.parse_args()


def load_txt_files(input_dir: str, max_files: Optional[int] = None) -> List[Path]:
    """加载目录中的所有txt文件"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    txt_files = list(input_path.glob("*.txt"))
    
    if not txt_files:
        raise ValueError(f"在 {input_dir} 中没有找到txt文件")
    
    # 按文件名排序
    txt_files.sort()
    
    # 限制文件数量
    if max_files and len(txt_files) > max_files:
        txt_files = txt_files[:max_files]
    
    return txt_files


def generate_qa_from_texts(txt_files: List[Path], 
                          kg_config: Dict,
                          qg_config: Dict,
                          logger: logging.Logger) -> List[Dict]:
    """从文本文件生成问答对"""
    all_qa_pairs = []
    
    # 初始化知识图谱构建器
    logger.info("初始化知识图谱构建器...")
    kg_builder = KnowledgeGraphBuilder(kg_config)
    
    # 初始化问题生成器
    logger.info("初始化问题生成器...")
    question_generator = QuestionGenerator(qg_config)
    
    # 处理每个文件
    for i, txt_file in enumerate(txt_files, 1):
        logger.info(f"[{i}/{len(txt_files)}] 处理文件: {txt_file.name}")
        
        try:
            # 读取文本内容
            with open(txt_file, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            if not text_content.strip():
                logger.warning(f"文件 {txt_file.name} 为空，跳过")
                continue
            
            # 构建知识图谱
            logger.info(f"  构建知识图谱...")
            knowledge_graph = kg_builder.build_from_text(text_content)
            
            if not knowledge_graph or knowledge_graph.number_of_nodes() == 0:
                logger.warning(f"  无法从 {txt_file.name} 构建知识图谱，跳过")
                continue
            
            logger.info(f"  知识图谱包含 {knowledge_graph.number_of_nodes()} 个节点，"
                       f"{knowledge_graph.number_of_edges()} 条边")
            
            # 提取子图
            logger.info(f"  提取子图...")
            subgraphs = kg_builder.extract_subgraphs(knowledge_graph)
            logger.info(f"  提取了 {len(subgraphs)} 个子图")
            
            # 生成问题
            logger.info(f"  生成问答对...")
            qa_pairs = question_generator.generate_questions(subgraphs)
            
            # 为每个问答对添加来源信息
            for qa_pair in qa_pairs:
                qa_pair['source_file'] = txt_file.name
                qa_pair['source_path'] = str(txt_file)
            
            all_qa_pairs.extend(qa_pairs)
            logger.info(f"  生成了 {len(qa_pairs)} 个问答对")
            
        except Exception as e:
            logger.error(f"处理文件 {txt_file.name} 时出错: {str(e)}", exc_info=True)
            continue
    
    logger.info(f"总共生成了 {len(all_qa_pairs)} 个问答对")
    return all_qa_pairs


def convert_to_qa_pairs(qa_data: List[Dict]) -> List[QAPair]:
    """将字典格式的问答对转换为QAPair对象"""
    qa_pairs = []
    for item in qa_data:
        qa_pair = QAPair(
            question=item.get('question', ''),
            answer=item.get('answer', ''),
            metadata=item.get('metadata', {})
        )
        # 添加额外信息到metadata
        if 'source_file' in item:
            qa_pair.metadata['source_file'] = item['source_file']
        if 'source_path' in item:
            qa_pair.metadata['source_path'] = item['source_path']
        if 'type' in item:
            qa_pair.metadata['type'] = item['type']
        if 'difficulty' in item:
            qa_pair.metadata['difficulty'] = item['difficulty']
        
        qa_pairs.append(qa_pair)
    
    return qa_pairs


def save_evaluation_results(results: List[EvaluationResult], 
                          output_path: str,
                          save_details: bool = True):
    """保存评测结果"""
    output_data = []
    
    for result in results:
        item = {
            'question': result.qa_pair.question,
            'answer': result.qa_pair.answer,
            'total_score': result.total_score,
            'metadata': result.qa_pair.metadata
        }
        
        if save_details:
            item['scores'] = {
                'relevance': result.relevance_score,
                'completeness': result.completeness_score,
                'clarity': result.clarity_score,
                'accuracy': result.accuracy_score,
                'depth': result.depth_score,
                'llm_score': result.llm_score
            }
            item['metrics'] = result.metrics
            item['issues'] = result.issues
            item['suggestions'] = result.suggestions
        
        output_data.append(item)
    
    # 保存为JSON格式
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def generate_evaluation_report(results: List[EvaluationResult],
                             evaluator: QAEvaluator,
                             report_dir: str,
                             config: Dict) -> tuple:
    """生成评测报告"""
    os.makedirs(report_dir, exist_ok=True)
    
    # 统计信息
    total_scores = [r.total_score for r in results]
    
    # 按来源文件分组统计
    source_stats = {}
    for result in results:
        source_file = result.qa_pair.metadata.get('source_file', 'unknown')
        if source_file not in source_stats:
            source_stats[source_file] = {
                'count': 0,
                'scores': [],
                'types': {}
            }
        
        source_stats[source_file]['count'] += 1
        source_stats[source_file]['scores'].append(result.total_score)
        
        q_type = result.qa_pair.metadata.get('type', 'unknown')
        if q_type not in source_stats[source_file]['types']:
            source_stats[source_file]['types'][q_type] = 0
        source_stats[source_file]['types'][q_type] += 1
    
    # 计算每个文件的平均分
    for source_file in source_stats:
        scores = source_stats[source_file]['scores']
        source_stats[source_file]['avg_score'] = sum(scores) / len(scores) if scores else 0
        source_stats[source_file]['max_score'] = max(scores) if scores else 0
        source_stats[source_file]['min_score'] = min(scores) if scores else 0
    
    # 创建报告
    report = {
        'evaluation_time': datetime.now().isoformat(),
        'total_evaluated': len(results),
        'config': config,
        'statistics': {
            'mean_score': sum(total_scores) / len(total_scores) if total_scores else 0,
            'std_score': evaluator._calculate_std(total_scores),
            'max_score': max(total_scores) if total_scores else 0,
            'min_score': min(total_scores) if total_scores else 0
        },
        'source_statistics': source_stats,
        'score_distribution': evaluator._get_score_distribution(results),
        'quality_insights': evaluator._get_quality_insights(results)
    }
    
    # 保存JSON报告
    report_path = os.path.join(report_dir, f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 生成文本摘要
    summary_path = os.path.join(report_dir, f'evaluation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("问答对生成与质量评测报告摘要\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理文件数: {len(source_stats)}\n")
        f.write(f"生成问答对总数: {report['total_evaluated']}\n")
        f.write(f"平均得分: {report['statistics']['mean_score']:.3f}\n")
        f.write(f"标准差: {report['statistics']['std_score']:.3f}\n")
        f.write(f"最高分: {report['statistics']['max_score']:.3f}\n")
        f.write(f"最低分: {report['statistics']['min_score']:.3f}\n\n")
        
        f.write("各文件生成情况:\n")
        for source_file, stats in source_stats.items():
            f.write(f"\n{source_file}:\n")
            f.write(f"  生成数量: {stats['count']}\n")
            f.write(f"  平均分: {stats['avg_score']:.3f}\n")
            f.write(f"  最高分: {stats['max_score']:.3f}\n")
            f.write(f"  最低分: {stats['min_score']:.3f}\n")
            f.write(f"  问题类型分布:\n")
            for q_type, count in stats['types'].items():
                f.write(f"    {q_type}: {count}\n")
        
        f.write("\n总体分数分布:\n")
        for range_str, count in report['score_distribution'].items():
            f.write(f"  {range_str}: {count}\n")
    
    return report_path, summary_path


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logging(log_level)
    
    logger.info("问答对生成与评测系统启动")
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出文件: {args.output}")
    logger.info(f"配置文件: {args.config}")
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 加载知识图谱和问题生成配置
        kg_config = {}
        qg_config = {}
        
        if args.kg_config:
            with open(args.kg_config, 'r', encoding='utf-8') as f:
                kg_config = yaml.safe_load(f)
        else:
            # 使用默认配置
            kg_config = {
                'llm': config.get('llm', {}),
                'extraction': {
                    'entity_types': ['技术', '产品', '材料', '工艺', '设备', '参数', '应用'],
                    'relation_types': ['使用', '包含', '具有', '用于', '基于', '影响', '优化'],
                    'max_entities_per_chunk': 10,
                    'chunk_size': 500,
                    'chunk_overlap': 50
                }
            }
        
        if args.qg_config:
            with open(args.qg_config, 'r', encoding='utf-8') as f:
                qg_config = yaml.safe_load(f)
        else:
            # 使用默认配置
            qg_config = {
                'llm': config.get('llm', {}),
                'question_generation': {
                    'questions_per_subgraph': 3,
                    'question_types': {
                        'factual': {'enabled': True, 'weight': 0.4},
                        'comparison': {'enabled': True, 'weight': 0.2},
                        'reasoning': {'enabled': True, 'weight': 0.2},
                        'multi_hop': {'enabled': True, 'weight': 0.2}
                    },
                    'difficulty_distribution': {
                        'easy': 0.3,
                        'medium': 0.5,
                        'hard': 0.2
                    }
                }
            }
        
        # 覆盖配置中的某些参数
        if args.min_score is not None:
            config['thresholds']['min_total_score'] = args.min_score
        
        if args.skip_llm:
            config['weights']['llm_score'] = 0.0
            logger.info("已跳过LLM评测")
        
        # 加载txt文件
        logger.info("正在加载txt文件...")
        txt_files = load_txt_files(args.input_dir, args.max_files)
        logger.info(f"找到 {len(txt_files)} 个txt文件")
        
        # 生成问答对
        logger.info("开始生成问答对...")
        all_qa_pairs = generate_qa_from_texts(txt_files, kg_config, qg_config, logger)
        
        if not all_qa_pairs:
            logger.error("未能生成任何问答对")
            sys.exit(1)
        
        # 保存所有生成的问答对（如果需要）
        if args.save_intermediate:
            all_qa_path = args.output.replace('.json', '_all_generated.json')
            with open(all_qa_path, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
            logger.info(f"所有生成的问答对已保存到: {all_qa_path}")
        
        # 初始化数据处理器
        data_processor = DataProcessor(config.get('preprocessing', {}))
        
        # 数据验证和清洗
        valid_qa_data, invalid_qa_data = data_processor.batch_validate(all_qa_pairs)
        if invalid_qa_data:
            logger.warning(f"发现 {len(invalid_qa_data)} 个无效问答对")
            if args.save_intermediate:
                invalid_path = args.output.replace('.json', '_invalid.json')
                data_processor.save_qa_pairs(invalid_qa_data, invalid_path)
        
        # 去重
        unique_qa_data = data_processor.deduplicate_qa_pairs(valid_qa_data)
        logger.info(f"去重后剩余 {len(unique_qa_data)} 个问答对")
        
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
        logger.info(f"处理完成！")
        logger.info(f"处理文件数: {len(txt_files)}")
        logger.info(f"生成问答对总数: {len(all_qa_pairs)}")
        logger.info(f"有效问答对数: {len(unique_qa_data)}")
        logger.info(f"评测数量: {len(results)}")
        logger.info(f"筛选后数量: {len(top_results)}")
        if top_results:
            avg_score = sum(r.total_score for r in top_results) / len(top_results)
            logger.info(f"筛选后平均分: {avg_score:.3f}")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("系统正常退出")


if __name__ == '__main__':
    main()