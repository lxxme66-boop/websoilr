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
from tenacity import retry, stop_after_attempt, wait_exponential
import requests

# Optional imports - only import if needed
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


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
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Please install with: pip install openai")
            
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
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Please install with: pip install anthropic")
            
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
    """本地LLM评测器（支持多种本地LLM服务）"""
    
    def __init__(self, config: Dict):
        """
        初始化本地LLM评测器
        
        Args:
            config: 配置字典，包含endpoint_url, api_type等
        """
        # 支持的API类型：ollama, vllm, openai-compatible
        self.api_type = config.get('api_type', 'openai-compatible')
        
        # 根据API类型设置默认端点
        default_endpoints = {
            'ollama': 'http://localhost:11434/api/chat',
            'vllm': 'http://localhost:8000/v1/chat/completions',
            'openai-compatible': 'http://localhost:8000/v1/chat/completions'
        }
        
        self.endpoint_url = config.get('endpoint_url', default_endpoints.get(self.api_type))
        self.model = config.get('model', 'qwen2.5:7b')  # 默认使用Qwen2.5
        self.timeout = config.get('timeout', 60)  # 增加超时时间
        self.temperature = config.get('temperature', 0.3)
        self.max_tokens = config.get('max_tokens', 500)
        
        # API密钥（某些本地服务可能需要）
        self.api_key = config.get('api_key', '')
        
        # 额外的请求头
        self.headers = config.get('headers', {})
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'
        
        self.logger = logging.getLogger('LocalLLMEvaluator')
        self.logger.info(f"Initialized {self.api_type} evaluator with endpoint: {self.endpoint_url}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def evaluate(self, question: str, answer: str) -> Dict:
        """使用本地LLM评测问答对"""
        prompt = self._build_evaluation_prompt(question, answer)
        
        try:
            if self.api_type == 'ollama':
                response = self._call_ollama_api(prompt)
            else:
                response = self._call_openai_compatible_api(prompt)
            
            result_text = self._extract_response_text(response)
            result = self._parse_evaluation_result(result_text)
            
            result['model'] = self.model
            result['raw_response'] = result_text
            result['api_type'] = self.api_type
            
            return result
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Request timeout after {self.timeout}s")
            raise Exception(f"Local LLM request timeout. Please check if the service is running at {self.endpoint_url}")
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Cannot connect to {self.endpoint_url}")
            raise Exception(f"Cannot connect to local LLM service at {self.endpoint_url}. Please ensure the service is running.")
        except Exception as e:
            self.logger.error(f"Local LLM API error: {e}")
            raise
    
    def _call_ollama_api(self, prompt: str) -> Dict:
        """调用Ollama API"""
        response = requests.post(
            self.endpoint_url,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            },
            headers=self.headers,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def _call_openai_compatible_api(self, prompt: str) -> Dict:
        """调用OpenAI兼容的API（vLLM, FastChat等）"""
        response = requests.post(
            self.endpoint_url,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            },
            headers={**self.headers, 'Content-Type': 'application/json'},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def _extract_response_text(self, response: Dict) -> str:
        """从API响应中提取文本"""
        if self.api_type == 'ollama':
            # Ollama返回格式
            if 'message' in response and 'content' in response['message']:
                return response['message']['content']
            elif 'response' in response:
                return response['response']
        
        # OpenAI兼容格式
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                return choice['message']['content']
            elif 'text' in choice:
                return choice['text']
        
        # 如果无法解析，返回整个响应
        return str(response)
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个专业的问答质量评估专家。请评估给定的问答对的质量，从以下几个维度进行评分：

1. 相关性（0-1分）：答案是否直接回答了问题
2. 准确性（0-1分）：答案内容是否准确无误
3. 完整性（0-1分）：答案是否完整地回答了问题的所有方面
4. 清晰度（0-1分）：答案表达是否清晰易懂
5. 深度（0-1分）：答案是否提供了足够的细节和深度

请以JSON格式返回评分结果，格式如下：
{
    "relevance": 0.9,
    "accuracy": 0.8,
    "completeness": 0.85,
    "clarity": 0.9,
    "depth": 0.7,
    "score": 0.84,
    "reason": "答案准确回答了问题，表达清晰，但在某些细节上还可以更深入"
}"""
    
    def _build_evaluation_prompt(self, question: str, answer: str) -> str:
        """构建评测提示词"""
        return f"""请评估以下问答对的质量：

问题：{question}

答案：{answer}

请严格按照JSON格式返回评分结果，包含relevance、accuracy、completeness、clarity、depth、score和reason字段。"""
    
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