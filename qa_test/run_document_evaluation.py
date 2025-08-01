#!/usr/bin/env python3
"""
运行基于文档的问答对评测
使用文档作为知识库来评测问答对的质量
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_based_evaluator import DocumentBasedEvaluator


def main():
    parser = argparse.ArgumentParser(
        description='基于文档内容评测问答对质量',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用websailor_domain的文档评测问答对
  python run_document_evaluation.py \\
    --documents ../websailor_domain/input_texts \\
    --input ../websailor_domain_dataset/qa_pairs.json \\
    --output evaluated_qa_pairs.json \\
    --top-k 100

  # 使用自定义文档目录
  python run_document_evaluation.py \\
    --documents /path/to/your/documents \\
    --input your_qa_pairs.json \\
    --output best_qa_pairs.json \\
    --top-k 50
        """
    )
    
    parser.add_argument(
        '--documents', '-d',
        required=True,
        help='文档目录路径（包含txt/md文件作为知识库）'
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='输入问答对JSON文件路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出评测结果文件路径'
    )
    
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=100,
        help='选择Top-K个最佳问答对（默认: 100）'
    )
    
    parser.add_argument(
        '--model',
        default='paraphrase-multilingual-mpnet-base-v2',
        help='句子嵌入模型名称（默认: paraphrase-multilingual-mpnet-base-v2）'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细输出'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 检查文档目录是否存在
    if not os.path.exists(args.documents):
        print(f"错误: 文档目录不存在: {args.documents}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 60)
    print("基于文档的问答对评测")
    print("=" * 60)
    print(f"文档目录: {args.documents}")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"Top-K: {args.top_k}")
    print(f"模型: {args.model}")
    print("=" * 60)
    
    try:
        # 创建评测器
        print("\n初始化评测器...")
        evaluator = DocumentBasedEvaluator(
            document_dir=args.documents,
            model_name=args.model
        )
        
        # 运行评测
        print("\n开始评测问答对...")
        evaluator.evaluate_qa_file(
            qa_file_path=args.input,
            output_path=args.output,
            top_k=args.top_k
        )
        
        # 读取并显示评测摘要
        with open(args.output, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print("\n" + "=" * 60)
        print("评测完成！")
        print("=" * 60)
        print(f"总评测数量: {results['total_evaluated']}")
        print(f"选中数量: {results['selected_count']}")
        print("\n评测摘要:")
        summary = results['evaluation_summary']
        print(f"  - 平均相关性: {summary['avg_relevance']:.3f}")
        print(f"  - 平均准确性: {summary['avg_accuracy']:.3f}")
        print(f"  - 平均覆盖度: {summary['avg_coverage']:.3f}")
        print(f"  - 平均一致性: {summary['avg_consistency']:.3f}")
        print(f"  - 平均总分: {summary['avg_total']:.3f}")
        
        # 显示Top 5问答对
        if args.verbose and results['qa_pairs']:
            print("\nTop 5 最佳问答对:")
            for i, qa in enumerate(results['qa_pairs'][:5]):
                print(f"\n{i+1}. 问题: {qa['question']}")
                print(f"   答案: {qa['answer'][:100]}...")
                print(f"   总分: {qa['scores']['total']:.3f}")
                if qa.get('supporting_evidence'):
                    print("   支持证据:")
                    for evidence in qa['supporting_evidence'][:2]:
                        print(f"     - {evidence['sentence'][:80]}... (相似度: {evidence['similarity']:.3f})")
        
        print(f"\n结果已保存到: {args.output}")
        print(f"详细报告已保存到: {args.output.replace('.json', '_report.md')}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()