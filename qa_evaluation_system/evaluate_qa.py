#!/usr/bin/env python3
"""
Main entry point for QA evaluation system
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path

from qa_evaluator import QAEvaluator
from utils import load_qa_pairs, save_results

logger = logging.getLogger(__name__)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="评测问答对质量的综合系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评测JSON文件中的问答对
  python evaluate_qa.py --input data/qa_pairs.json
  
  # 使用自定义配置文件
  python evaluate_qa.py --input data/qa_pairs.json --config my_config.yaml
  
  # 只评测前100个样本
  python evaluate_qa.py --input data/qa_pairs.json --sample 100
  
  # 导出高质量问答对
  python evaluate_qa.py --input data/qa_pairs.json --export-top 500 --threshold 0.8
  
  # 评测单个问答对
  python evaluate_qa.py --question "什么是机器学习？" --answer "机器学习是..."
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str,
                            help='输入文件路径 (支持 JSON, JSONL, CSV, Excel)')
    input_group.add_argument('--question', '-q', type=str,
                            help='单个问题文本')
    
    parser.add_argument('--answer', '-a', type=str,
                       help='单个答案文本 (与 --question 一起使用)')
    
    # Output options
    parser.add_argument('--output', '-o', type=str,
                       help='输出文件路径 (默认: results/evaluation_results.json)')
    parser.add_argument('--format', '-f', choices=['json', 'csv', 'excel'],
                       default='json', help='输出格式 (默认: json)')
    
    # Configuration
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='配置文件路径 (默认: config.yaml)')
    
    # Sampling and filtering
    parser.add_argument('--sample', '-s', type=int,
                       help='随机抽样评测的数量')
    parser.add_argument('--export-top', type=int,
                       help='导出前N个高质量问答对')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='导出的最低质量阈值 (默认: 0.7)')
    
    # Processing options
    parser.add_argument('--no-parallel', action='store_true',
                       help='禁用并行处理')
    parser.add_argument('--workers', type=int, default=4,
                       help='并行处理的工作线程数 (默认: 4)')
    parser.add_argument('--no-cache', action='store_true',
                       help='禁用LLM响应缓存')
    parser.add_argument('--ensemble', action='store_true',
                       help='使用多模型集成评测')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细日志')
    parser.add_argument('--quiet', action='store_true',
                       help='只显示错误信息')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.question and not args.answer:
        parser.error("使用 --question 时必须提供 --answer")
    
    # Setup logging
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize evaluator
        evaluator = QAEvaluator(config_path=args.config)
        
        # Override config with command line options
        if args.workers:
            evaluator.config['processing']['num_workers'] = args.workers
        if args.no_cache:
            evaluator.config['llm']['enable_cache'] = False
        if args.ensemble:
            evaluator.config['advanced']['use_ensemble'] = True
        
        # Single QA pair evaluation
        if args.question:
            logger.info("评测单个问答对...")
            result = evaluator.evaluate_single(args.question, args.answer)
            
            # Print result
            print("\n评测结果:")
            print(f"总体评分: {result['overall_score']:.2f}")
            print(f"质量等级: {result['quality_level']}")
            print(f"\n详细评分:")
            for key, value in result['detailed_scores'].items():
                print(f"  {key}: {value:.2f}")
            
            if result.get('issues'):
                print(f"\n发现的问题:")
                for issue in result['issues']:
                    print(f"  - {issue}")
            
            if result.get('suggestions'):
                print(f"\n改进建议:")
                for suggestion in result['suggestions']:
                    print(f"  - {suggestion}")
            
            # Save if output specified
            if args.output:
                save_results([result], args.output, format=args.format)
                logger.info(f"结果已保存到: {args.output}")
        
        # File evaluation
        else:
            logger.info(f"评测文件: {args.input}")
            
            # Prepare output path
            if args.output:
                output_path = args.output
            else:
                output_dir = Path("results")
                output_dir.mkdir(exist_ok=True)
                output_path = str(output_dir / f"evaluation_results.{args.format}")
            
            # Evaluate file
            report = evaluator.evaluate_file(
                input_path=args.input,
                output_path=output_path,
                sample_size=args.sample,
                export_top=(args.export_top is not None)
            )
            
            # Print summary
            print("\n评测报告摘要:")
            print(f"评测问答对数量: {report['summary']['total_qa_pairs']}")
            print(f"平均得分: {report['summary']['average_score']:.3f}")
            print(f"中位数得分: {report['summary']['median_score']:.3f}")
            print(f"标准差: {report['summary']['std_score']:.3f}")
            
            print("\n质量分布:")
            for level, info in report['quality_distribution'].items():
                print(f"  {level}: {info['count']} ({info['percentage']:.1f}%)")
            
            print("\n各维度平均得分:")
            for dim, stats in report['dimension_analysis'].items():
                print(f"  {dim}: {stats['average']:.3f}")
            
            if report.get('common_issues'):
                print("\n常见问题 (前10):")
                for issue, count in list(report['common_issues'].items())[:10]:
                    print(f"  - {issue}: {count}次")
            
            print(f"\n详细结果已保存到: {output_path}")
            
            # Export top QA pairs if requested
            if args.export_top:
                # Load original QA pairs and results
                qa_pairs = load_qa_pairs(args.input)
                results = json.load(open(output_path))
                
                # Filter and sort
                qa_with_scores = []
                for qa, result in zip(qa_pairs, results):
                    if result['overall_score'] >= args.threshold:
                        qa_with_score = qa.copy()
                        qa_with_score['evaluation_score'] = result['overall_score']
                        qa_with_score['quality_level'] = result['quality_level']
                        qa_with_scores.append(qa_with_score)
                
                qa_with_scores.sort(key=lambda x: x['evaluation_score'], reverse=True)
                top_qa = qa_with_scores[:args.export_top]
                
                # Save top QA pairs
                export_path = Path("exports")
                export_path.mkdir(exist_ok=True)
                export_file = export_path / f"top_{args.export_top}_qa_pairs.json"
                
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(top_qa, f, ensure_ascii=False, indent=2)
                
                print(f"\n已导出 {len(top_qa)} 个高质量问答对到: {export_file}")
        
        logger.info("评测完成")
        
    except KeyboardInterrupt:
        logger.info("用户中断执行")
        sys.exit(1)
    except Exception as e:
        logger.error(f"执行出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()