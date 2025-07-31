#!/usr/bin/env python3
"""
Example usage of the QA Evaluation System
"""

import json
from qa_evaluator import QAEvaluator

def example_single_evaluation():
    """Example: Evaluate a single QA pair"""
    print("=== 单个问答对评测示例 ===\n")
    
    # Initialize evaluator
    evaluator = QAEvaluator()
    
    # Example QA pair
    question = "什么是深度学习？"
    answer = """深度学习是机器学习的一个子领域，它基于人工神经网络的学习算法。
    深度学习模型使用多层神经网络来逐步提取数据的高级特征。
    与传统机器学习方法相比，深度学习能够自动学习特征表示，
    在图像识别、自然语言处理、语音识别等领域取得了突破性进展。
    常见的深度学习架构包括卷积神经网络(CNN)、循环神经网络(RNN)、
    生成对抗网络(GAN)和Transformer等。"""
    
    # Evaluate
    result = evaluator.evaluate_single(question, answer)
    
    # Display results
    print(f"问题: {question}")
    print(f"答案: {answer[:100]}...\n")
    print(f"总体评分: {result['overall_score']:.2f}")
    print(f"质量等级: {result['quality_level']}")
    print("\n详细评分:")
    for key, value in result['detailed_scores'].items():
        print(f"  - {key}: {value:.2f}")
    
    if result.get('issues'):
        print("\n发现的问题:")
        for issue in result['issues']:
            print(f"  - {issue}")
    
    if result.get('suggestions'):
        print("\n改进建议:")
        for suggestion in result['suggestions']:
            print(f"  - {suggestion}")


def example_batch_evaluation():
    """Example: Evaluate multiple QA pairs from file"""
    print("\n\n=== 批量评测示例 ===\n")
    
    # Initialize evaluator
    evaluator = QAEvaluator()
    
    # Load sample QA pairs
    qa_pairs = [
        {
            "question": "Python中如何定义函数？",
            "answer": "在Python中，使用def关键字定义函数。基本语法是：def function_name(parameters): 后面跟缩进的函数体。"
        },
        {
            "question": "什么是API？",
            "answer": "API（应用程序编程接口）是一组定义和协议，用于构建和集成应用软件。它允许不同的软件组件相互通信。"
        },
        {
            "question": "如何学习编程？",
            "answer": "学编程"  # 故意设置一个低质量答案
        }
    ]
    
    # Evaluate batch
    results = evaluator.evaluate_batch(qa_pairs, show_progress=False)
    
    # Display summary
    print(f"评测了 {len(results)} 个问答对\n")
    
    # Statistics
    stats = evaluator.get_statistics(results)
    print(f"平均得分: {stats['average_score']:.3f}")
    print(f"最高得分: {stats['max_score']:.3f}")
    print(f"最低得分: {stats['min_score']:.3f}")
    
    print("\n质量分布:")
    for level, info in stats['quality_distribution'].items():
        print(f"  - {level}: {info['count']} ({info['percentage']:.1f}%)")
    
    # Show individual results
    print("\n各问答对评测结果:")
    for i, (qa, result) in enumerate(zip(qa_pairs, results)):
        print(f"\n{i+1}. 问题: {qa['question']}")
        print(f"   评分: {result['overall_score']:.2f} ({result['quality_level']})")
        if result['overall_score'] < 0.6:
            print(f"   主要问题: {', '.join(result['issues'][:2])}")


def example_file_evaluation():
    """Example: Evaluate QA pairs from JSON file"""
    print("\n\n=== 文件评测示例 ===\n")
    
    # Initialize evaluator
    evaluator = QAEvaluator()
    
    # Evaluate sample file
    input_file = "examples/sample_qa.json"
    output_file = "results/sample_evaluation.json"
    
    print(f"评测文件: {input_file}")
    
    try:
        report = evaluator.evaluate_file(
            input_path=input_file,
            output_path=output_file,
            sample_size=5,  # Only evaluate first 5 for demo
            export_top=False
        )
        
        print("\n评测报告:")
        print(f"总数: {report['summary']['total_qa_pairs']}")
        print(f"平均分: {report['summary']['average_score']:.3f}")
        print(f"标准差: {report['summary']['std_score']:.3f}")
        
        print(f"\n结果已保存到: {output_file}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        print("请确保在 qa_evaluation_system 目录下运行此脚本")


def example_custom_config():
    """Example: Use custom configuration"""
    print("\n\n=== 自定义配置示例 ===\n")
    
    # Create custom config
    custom_config = {
        'weights': {
            'overall': {
                'question_quality': 0.4,  # 提高问题质量权重
                'answer_quality': 0.4,
                'qa_relevance': 0.2
            }
        },
        'thresholds': {
            'excellent': 0.85,  # 更严格的阈值
            'good': 0.70,
            'medium': 0.55,
            'poor': 0.40
        }
    }
    
    # Save custom config
    with open('custom_config.yaml', 'w') as f:
        import yaml
        yaml.dump(custom_config, f)
    
    # Use custom config
    evaluator = QAEvaluator(config_path='custom_config.yaml')
    
    result = evaluator.evaluate_single(
        "什么是云计算？",
        "云计算是通过互联网提供计算服务。"
    )
    
    print(f"使用自定义配置的评测结果:")
    print(f"评分: {result['overall_score']:.2f}")
    print(f"等级: {result['quality_level']}")
    
    # Clean up
    import os
    os.remove('custom_config.yaml')


def main():
    """Run all examples"""
    print("QA评测系统使用示例\n")
    print("=" * 50)
    
    # Run examples
    example_single_evaluation()
    example_batch_evaluation()
    example_file_evaluation()
    example_custom_config()
    
    print("\n\n所有示例运行完成！")
    print("\n提示:")
    print("1. 确保已安装所有依赖: pip install -r requirements.txt")
    print("2. 确保已下载模型: python download_models.py")
    print("3. 配置API密钥: 复制 .env.example 为 .env 并填入密钥")
    print("4. 运行主程序: python evaluate_qa.py --help")


if __name__ == "__main__":
    main()