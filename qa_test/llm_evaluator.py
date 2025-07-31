"""
大语言模型评测模块
使用LLM评估问答对的质量
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
import anthropic
import requests


class LLMEvaluatorBase(ABC):
    """LLM评测器基类"""
    
    @abstractmethod
    def evaluate(self, question: str, answer: str) -> Dict:
        """评测问答对"""
        pass


class OpenAIEvaluator(LLMEvaluatorBase):
    """OpenAI GPT评测器"""
    
    def __init__(self, config: Dict):
        """
        初始化OpenAI评测器
        
        Args:
            config: 配置字典，包含api_key, model等
        """
        self.api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.3)
        self.max_tokens = config.get('max_tokens', 500)
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        openai.api_key = self.api_key
        self.logger = logging.getLogger('OpenAIEvaluator')
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def evaluate(self, question: str, answer: str) -> Dict:
        """
        使用GPT评测问答对
        
        Args:
            question: 问题
            answer: 答案
            
        Returns:
            评测结果字典
        """
        # 构建评测prompt
        prompt = self._build_evaluation_prompt(question, answer)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # 解析响应
            result_text = response.choices[0].message['content']
            result = self._parse_evaluation_result(result_text)
            
            # 添加元数据
            result['model'] = self.model
            result['raw_response'] = result_text
            
            return result
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个专业的问答质量评估专家。请评估给定的问答对的质量，从以下几个维度进行评分：

1. 相关性（0-1分）：答案是否直接回答了问题
2. 准确性（0-1分）：答案内容是否准确无误
3. 完整性（0-1分）：答案是否完整地回答了问题的所有方面
4. 清晰度（0-1分）：答案表达是否清晰易懂
5. 深度（0-1分）：答案是否提供了足够的细节和深度

请以JSON格式返回评分结果，包括各维度分数和总体评分（0-1分），以及简短的评价理由。"""
    
    def _build_evaluation_prompt(self, question: str, answer: str) -> str:
        """构建评测提示词"""
        return f"""请评估以下问答对的质量：

问题：{question}

答案：{answer}

请按照指定格式返回JSON评分结果。"""
    
    def _parse_evaluation_result(self, result_text: str) -> Dict:
        """解析评测结果"""
        try:
            # 尝试提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
            else:
                # 如果没有找到JSON，尝试直接解析
                result_json = json.loads(result_text)
            
            # 确保包含必要字段
            required_fields = ['relevance', 'accuracy', 'completeness', 'clarity', 'depth']
            for field in required_fields:
                if field not in result_json:
                    result_json[field] = 0.5  # 默认值
            
            # 计算总分
            if 'score' not in result_json:
                scores = [result_json.get(field, 0.5) for field in required_fields]
                result_json['score'] = sum(scores) / len(scores)
            
            return result_json
            
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON from response: {result_text}")
            # 返回默认评分
            return {
                'relevance': 0.5,
                'accuracy': 0.5,
                'completeness': 0.5,
                'clarity': 0.5,
                'depth': 0.5,
                'score': 0.5,
                'reason': 'Failed to parse evaluation result'
            }


class AnthropicEvaluator(LLMEvaluatorBase):
    """Anthropic Claude评测器"""
    
    def __init__(self, config: Dict):
        """
        初始化Anthropic评测器
        
        Args:
            config: 配置字典
        """
        self.api_key = config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
        self.model = config.get('model', 'claude-3-opus-20240229')
        self.max_tokens = config.get('max_tokens', 500)
        
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.logger = logging.getLogger('AnthropicEvaluator')
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def evaluate(self, question: str, answer: str) -> Dict:
        """使用Claude评测问答对"""
        prompt = self._build_evaluation_prompt(question, answer)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system=self._get_system_prompt()
            )
            
            result_text = response.content[0].text
            result = self._parse_evaluation_result(result_text)
            
            result['model'] = self.model
            result['raw_response'] = result_text
            
            return result
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个专业的问答质量评估专家。请评估给定的问答对的质量，从以下几个维度进行评分：

1. 相关性（0-1分）：答案是否直接回答了问题
2. 准确性（0-1分）：答案内容是否准确无误
3. 完整性（0-1分）：答案是否完整地回答了问题的所有方面
4. 清晰度（0-1分）：答案表达是否清晰易懂
5. 深度（0-1分）：答案是否提供了足够的细节和深度

请以JSON格式返回评分结果，包括各维度分数和总体评分（0-1分），以及简短的评价理由。"""
    
    def _build_evaluation_prompt(self, question: str, answer: str) -> str:
        """构建评测提示词"""
        return f"""请评估以下问答对的质量：

问题：{question}

答案：{answer}

请按照指定格式返回JSON评分结果。"""
    
    def _parse_evaluation_result(self, result_text: str) -> Dict:
        """解析评测结果"""
        try:
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
            else:
                result_json = json.loads(result_text)
            
            required_fields = ['relevance', 'accuracy', 'completeness', 'clarity', 'depth']
            for field in required_fields:
                if field not in result_json:
                    result_json[field] = 0.5
            
            if 'score' not in result_json:
                scores = [result_json.get(field, 0.5) for field in required_fields]
                result_json['score'] = sum(scores) / len(scores)
            
            return result_json
            
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON from response: {result_text}")
            return {
                'relevance': 0.5,
                'accuracy': 0.5,
                'completeness': 0.5,
                'clarity': 0.5,
                'depth': 0.5,
                'score': 0.5,
                'reason': 'Failed to parse evaluation result'
            }


class LocalLLMEvaluator(LLMEvaluatorBase):
    """本地LLM评测器（通过API）"""
    
    def __init__(self, config: Dict):
        """
        初始化本地LLM评测器
        
        Args:
            config: 配置字典，包含endpoint_url等
        """
        self.endpoint_url = config.get('endpoint_url', 'http://localhost:8000/v1/chat/completions')
        self.model = config.get('model', 'local-model')
        self.timeout = config.get('timeout', 30)
        self.logger = logging.getLogger('LocalLLMEvaluator')
    
    def evaluate(self, question: str, answer: str) -> Dict:
        """使用本地LLM评测问答对"""
        prompt = self._build_evaluation_prompt(question, answer)
        
        try:
            response = requests.post(
                self.endpoint_url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result_data = response.json()
            
            result_text = result_data['choices'][0]['message']['content']
            result = self._parse_evaluation_result(result_text)
            
            result['model'] = self.model
            result['raw_response'] = result_text
            
            return result
            
        except Exception as e:
            self.logger.error(f"Local LLM API error: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个专业的问答质量评估专家。请评估给定的问答对的质量，从以下几个维度进行评分：

1. 相关性（0-1分）：答案是否直接回答了问题
2. 准确性（0-1分）：答案内容是否准确无误
3. 完整性（0-1分）：答案是否完整地回答了问题的所有方面
4. 清晰度（0-1分）：答案表达是否清晰易懂
5. 深度（0-1分）：答案是否提供了足够的细节和深度

请以JSON格式返回评分结果。"""
    
    def _build_evaluation_prompt(self, question: str, answer: str) -> str:
        """构建评测提示词"""
        return f"""请评估以下问答对的质量：

问题：{question}

答案：{answer}

请按照指定格式返回JSON评分结果。"""
    
    def _parse_evaluation_result(self, result_text: str) -> Dict:
        """解析评测结果"""
        try:
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
            else:
                result_json = json.loads(result_text)
            
            required_fields = ['relevance', 'accuracy', 'completeness', 'clarity', 'depth']
            for field in required_fields:
                if field not in result_json:
                    result_json[field] = 0.5
            
            if 'score' not in result_json:
                scores = [result_json.get(field, 0.5) for field in required_fields]
                result_json['score'] = sum(scores) / len(scores)
            
            return result_json
            
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON from response: {result_text}")
            return {
                'relevance': 0.5,
                'accuracy': 0.5,
                'completeness': 0.5,
                'clarity': 0.5,
                'depth': 0.5,
                'score': 0.5,
                'reason': 'Failed to parse evaluation result'
            }


class LLMEvaluator:
    """LLM评测器工厂类"""
    
    def __init__(self, config: Dict):
        """
        初始化LLM评测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.provider = config.get('provider', 'openai')
        self.evaluator = self._create_evaluator()
        self.cache = {}
        self.cache_enabled = config.get('cache_enabled', True)
    
    def _create_evaluator(self) -> LLMEvaluatorBase:
        """创建具体的评测器实例"""
        if self.provider == 'openai':
            return OpenAIEvaluator(self.config)
        elif self.provider == 'anthropic':
            return AnthropicEvaluator(self.config)
        elif self.provider == 'local':
            return LocalLLMEvaluator(self.config)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def evaluate(self, question: str, answer: str) -> Dict:
        """
        评测问答对
        
        Args:
            question: 问题
            answer: 答案
            
        Returns:
            评测结果字典
        """
        # 检查缓存
        cache_key = f"{self.provider}:{question}:{answer}"
        if self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key]
        
        # 执行评测
        result = self.evaluator.evaluate(question, answer)
        
        # 缓存结果
        if self.cache_enabled:
            self.cache[cache_key] = result
        
        return result
    
    def batch_evaluate(self, qa_pairs: List[Tuple[str, str]], batch_size: int = 5) -> List[Dict]:
        """
        批量评测问答对
        
        Args:
            qa_pairs: 问答对列表 [(question, answer), ...]
            batch_size: 批次大小
            
        Returns:
            评测结果列表
        """
        results = []
        
        for i in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[i:i + batch_size]
            batch_results = []
            
            for question, answer in batch:
                try:
                    result = self.evaluate(question, answer)
                    batch_results.append(result)
                except Exception as e:
                    # 记录错误但继续处理
                    batch_results.append({
                        'score': 0.0,
                        'error': str(e),
                        'relevance': 0.0,
                        'accuracy': 0.0,
                        'completeness': 0.0,
                        'clarity': 0.0,
                        'depth': 0.0
                    })
            
            results.extend(batch_results)
            
            # 避免速率限制
            if i + batch_size < len(qa_pairs):
                time.sleep(1)
        
        return results