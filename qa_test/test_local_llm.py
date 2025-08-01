#!/usr/bin/env python3
"""
测试本地大模型集成
用于验证qa_test系统的本地LLM功能是否正常工作
"""

import json
import logging
import sys
from pathlib import Path

# 添加qa_test到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from llm_evaluator import LLMEvaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_local_llm():
    """测试本地LLM评测功能"""
    
    # 测试配置
    test_config = {
        'provider': 'local',
        'api_type': 'ollama',  # 可以改为 'vllm' 或 'openai-compatible'
        'model': 'qwen2.5:7b',  # 根据您安装的模型调整
        'endpoint_url': 'http://localhost:11434/api/chat',
        'temperature': 0.3,
        'max_tokens': 500,
        'timeout': 60,
        'cache_enabled': False  # 测试时禁用缓存
    }
    
    # 测试问答对
    test_qa_pairs = [
        {
            "question": "什么是机器学习？",
            "answer": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习和改进，而无需明确编程。通过识别数据中的模式，机器学习算法可以做出决策和预测。"
        },
        {
            "question": "Python中的列表和元组有什么区别？",
            "answer": "列表是可变的，可以修改其内容；元组是不可变的，创建后不能修改。列表使用方括号[]，元组使用圆括号()。"
        },
        {
            "question": "如何预防感冒？",
            "answer": "多喝水，保持充足睡眠。"  # 故意提供一个简短的答案来测试评分
        }
    ]
    
    print("=" * 60)
    print("本地大模型集成测试")
    print("=" * 60)
    print(f"\n配置信息：")
    print(f"- API类型: {test_config['api_type']}")
    print(f"- 模型: {test_config['model']}")
    print(f"- 端点: {test_config['endpoint_url']}")
    print()
    
    try:
        # 创建评测器
        print("正在初始化LLM评测器...")
        evaluator = LLMEvaluator(test_config)
        print("✓ 评测器初始化成功")
        
        # 测试每个问答对
        for i, qa in enumerate(test_qa_pairs, 1):
            print(f"\n测试问答对 {i}/{len(test_qa_pairs)}:")
            print(f"问题: {qa['question']}")
            print(f"答案: {qa['answer'][:50]}{'...' if len(qa['answer']) > 50 else ''}")
            
            try:
                # 执行评测
                print("正在评测...")
                result = evaluator.evaluate(qa['question'], qa['answer'])
                
                # 显示结果
                print("\n评测结果:")
                print(f"- 相关性: {result.get('relevance', 0):.2f}")
                print(f"- 准确性: {result.get('accuracy', 0):.2f}")
                print(f"- 完整性: {result.get('completeness', 0):.2f}")
                print(f"- 清晰度: {result.get('clarity', 0):.2f}")
                print(f"- 深度: {result.get('depth', 0):.2f}")
                print(f"- 总分: {result.get('score', 0):.2f}")
                
                if 'reason' in result:
                    print(f"- 评价: {result['reason']}")
                
                print("-" * 40)
                
            except Exception as e:
                print(f"✗ 评测失败: {e}")
                
        print("\n✓ 所有测试完成！")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        print("\n可能的原因：")
        print("1. 本地LLM服务未启动")
        print("2. 端口配置错误")
        print("3. 模型未安装")
        print("\n请参考 LOCAL_LLM_SETUP.md 进行设置")
        return False
    
    return True

def check_service_availability():
    """检查本地LLM服务是否可用"""
    import requests
    
    print("\n检查本地LLM服务状态...")
    
    # 检查Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✓ Ollama服务正在运行")
            if models:
                print(f"  可用模型: {', '.join([m['name'] for m in models[:5]])}")
            else:
                print("  ⚠ 未找到已安装的模型，请运行: ollama pull qwen2.5:7b")
        else:
            print("✗ Ollama服务响应异常")
    except:
        print("✗ 无法连接到Ollama服务 (http://localhost:11434)")
    
    # 检查vLLM/OpenAI兼容服务
    try:
        response = requests.get("http://localhost:8000/v1/models", timeout=5)
        if response.status_code == 200:
            print(f"✓ OpenAI兼容服务正在运行 (可能是vLLM、FastChat等)")
        else:
            print("✗ OpenAI兼容服务响应异常")
    except:
        print("✗ 无法连接到OpenAI兼容服务 (http://localhost:8000)")

if __name__ == "__main__":
    # 检查服务状态
    check_service_availability()
    
    # 运行测试
    print("\n" + "=" * 60)
    success = test_local_llm()
    
    if not success:
        sys.exit(1)