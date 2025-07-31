"""
LLM-based evaluation module for QA pairs
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor
import openai
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

from utils import Cache, create_cache_key

logger = logging.getLogger(__name__)
load_dotenv()


@dataclass
class LLMResponse:
    """LLM evaluation response"""
    question_quality: float
    answer_quality: float
    relevance: float
    detailed_feedback: Dict[str, Any]
    raw_response: str


class LLMEvaluator:
    """LLM-based evaluator for QA pairs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config.get('llm', {})
        
        # Initialize LLM clients
        self._init_clients()
        
        # Initialize cache
        if self.llm_config.get('enable_cache', True):
            cache_dir = self.llm_config.get('cache_dir', '.cache/llm_responses')
            self.cache = Cache(cache_dir)
        else:
            self.cache = None
        
        # Load evaluation prompts
        self.prompts = self._load_prompts()
    
    def _init_clients(self):
        """Initialize LLM API clients"""
        self.clients = {}
        api_config = self.llm_config.get('api_config', {})
        
        # OpenAI
        if 'openai' in api_config:
            try:
                openai_config = api_config['openai']
                api_key = openai_config.get('api_key', os.getenv('OPENAI_API_KEY'))
                if api_key:
                    self.clients['openai'] = OpenAI(
                        api_key=api_key,
                        base_url=openai_config.get('base_url')
                    )
                    logger.info("Initialized OpenAI client")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        # Anthropic
        if 'anthropic' in api_config:
            try:
                anthropic_config = api_config['anthropic']
                api_key = anthropic_config.get('api_key', os.getenv('ANTHROPIC_API_KEY'))
                if api_key:
                    self.clients['anthropic'] = anthropic.Anthropic(api_key=api_key)
                    logger.info("Initialized Anthropic client")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
        
        # DeepSeek
        if 'deepseek' in api_config:
            try:
                deepseek_config = api_config['deepseek']
                api_key = deepseek_config.get('api_key', os.getenv('DEEPSEEK_API_KEY'))
                if api_key:
                    self.clients['deepseek'] = OpenAI(
                        api_key=api_key,
                        base_url=deepseek_config.get('base_url', 'https://api.deepseek.com/v1')
                    )
                    logger.info("Initialized DeepSeek client")
            except Exception as e:
                logger.error(f"Failed to initialize DeepSeek client: {e}")
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load evaluation prompts"""
        # Default prompts
        prompts = {
            'comprehensive': """你是一个专业的问答对质量评估专家。请从以下维度评估这个问答对的质量：

问题：{question}
答案：{answer}

请从以下方面进行评估：

1. 问题质量（0-10分）：
   - 问题的清晰度和具体性
   - 问题的逻辑性和合理性
   - 问题的信息价值
   - 语法和表达的规范性

2. 答案质量（0-10分）：
   - 答案的准确性和正确性
   - 答案的完整性和深度
   - 答案的条理性和逻辑性
   - 语言表达的专业性和清晰度

3. 问答相关性（0-10分）：
   - 答案是否直接回答了问题
   - 答案对问题的覆盖程度
   - 信息的一致性和匹配度

请以JSON格式输出评估结果：
{{
    "question_quality": <分数>,
    "answer_quality": <分数>,
    "relevance": <分数>,
    "question_feedback": {{
        "strengths": ["优点1", "优点2"],
        "weaknesses": ["缺点1", "缺点2"],
        "suggestions": ["建议1", "建议2"]
    }},
    "answer_feedback": {{
        "strengths": ["优点1", "优点2"],
        "weaknesses": ["缺点1", "缺点2"],
        "suggestions": ["建议1", "建议2"]
    }},
    "overall_assessment": "总体评价"
}}""",

            'question_quality': """请评估以下问题的质量：

问题：{question}

评估标准：
1. 清晰度：问题表述是否清楚明确，没有歧义
2. 具体性：问题是否具体，而非过于宽泛
3. 逻辑性：问题是否符合逻辑，合理可答
4. 价值性：问题是否有实际意义和价值
5. 规范性：语法、用词是否规范准确

请给出0-10的评分，并说明理由。输出JSON格式：
{{
    "score": <分数>,
    "clarity": <清晰度评分>,
    "specificity": <具体性评分>,
    "logic": <逻辑性评分>,
    "value": <价值性评分>,
    "grammar": <规范性评分>,
    "feedback": "详细反馈"
}}""",

            'answer_quality': """请评估以下答案的质量：

问题：{question}
答案：{answer}

评估标准：
1. 准确性：答案内容是否准确无误
2. 完整性：答案是否完整回答了问题
3. 深度：答案是否有足够的深度和细节
4. 条理性：答案组织是否有条理
5. 专业性：用语是否专业准确

请给出0-10的评分，并说明理由。输出JSON格式：
{{
    "score": <分数>,
    "accuracy": <准确性评分>,
    "completeness": <完整性评分>,
    "depth": <深度评分>,
    "organization": <条理性评分>,
    "professionalism": <专业性评分>,
    "feedback": "详细反馈"
}}""",

            'relevance': """请评估答案与问题的相关性：

问题：{question}
答案：{answer}

评估要点：
1. 答案是否直接回答了问题
2. 答案是否偏离主题
3. 答案的信息是否与问题匹配
4. 是否存在答非所问的情况

请给出0-10的相关性评分，并说明理由。输出JSON格式：
{{
    "score": <分数>,
    "direct_answer": <是否直接回答>,
    "topic_alignment": <主题一致性>,
    "information_match": <信息匹配度>,
    "feedback": "详细反馈"
}}"""
        }
        
        # Load custom prompts if available
        custom_prompts_dir = self.config.get('advanced', {}).get('custom_prompts_dir')
        if custom_prompts_dir and os.path.exists(custom_prompts_dir):
            for prompt_file in os.listdir(custom_prompts_dir):
                if prompt_file.endswith('.txt'):
                    prompt_name = prompt_file[:-4]
                    with open(os.path.join(custom_prompts_dir, prompt_file), 'r', encoding='utf-8') as f:
                        prompts[prompt_name] = f.read()
        
        return prompts
    
    def evaluate_qa_pair(self, question: str, answer: str, 
                        use_cache: bool = True) -> Dict[str, Any]:
        """Evaluate a single QA pair using LLM"""
        # Check cache
        if use_cache and self.cache:
            cache_key = create_cache_key({'question': question, 'answer': answer, 'type': 'llm_eval'})
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug("Using cached LLM evaluation")
                return cached_result
        
        # Get evaluation from primary model
        primary_model = self.llm_config.get('primary_model', 'gpt-4')
        result = self._evaluate_with_model(question, answer, primary_model)
        
        # If primary fails, try fallback models
        if result is None:
            fallback_models = self.llm_config.get('fallback_models', [])
            for model in fallback_models:
                result = self._evaluate_with_model(question, answer, model)
                if result is not None:
                    break
        
        # If all models fail, return default scores
        if result is None:
            logger.error("All LLM evaluations failed, using default scores")
            result = self._get_default_scores()
        
        # Cache result
        if use_cache and self.cache and result:
            self.cache.set(cache_key, result)
        
        return result
    
    def _evaluate_with_model(self, question: str, answer: str, 
                           model_name: str) -> Optional[Dict[str, Any]]:
        """Evaluate with a specific model"""
        try:
            # Determine which client to use
            if model_name.startswith('gpt'):
                client = self.clients.get('openai')
                provider = 'openai'
            elif model_name.startswith('claude'):
                client = self.clients.get('anthropic')
                provider = 'anthropic'
            elif model_name.startswith('deepseek'):
                client = self.clients.get('deepseek')
                provider = 'deepseek'
            else:
                logger.error(f"Unknown model: {model_name}")
                return None
            
            if not client:
                logger.error(f"No client available for {provider}")
                return None
            
            # Get comprehensive evaluation
            prompt = self.prompts['comprehensive'].format(
                question=question,
                answer=answer
            )
            
            response = self._call_llm(client, provider, model_name, prompt)
            if not response:
                return None
            
            # Parse response
            result = self._parse_llm_response(response)
            
            # Normalize scores to 0-1 range
            if result:
                result['question_quality'] = result.get('question_quality', 5) / 10
                result['answer_quality'] = result.get('answer_quality', 5) / 10
                result['relevance'] = result.get('relevance', 5) / 10
                result['model_used'] = model_name
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating with {model_name}: {e}")
            return None
    
    def _call_llm(self, client: Any, provider: str, model: str, 
                 prompt: str, max_retries: int = None) -> Optional[str]:
        """Call LLM API with retries"""
        max_retries = max_retries or self.llm_config.get('max_retries', 3)
        retry_delay = self.llm_config.get('retry_delay', 1.0)
        
        for attempt in range(max_retries):
            try:
                if provider == 'openai' or provider == 'deepseek':
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "你是一个专业的问答质量评估专家。"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.llm_config.get('temperature', 0.1),
                        max_tokens=self.llm_config.get('max_tokens', 1500),
                        top_p=self.llm_config.get('top_p', 0.95)
                    )
                    return response.choices[0].message.content
                
                elif provider == 'anthropic':
                    response = client.messages.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.llm_config.get('max_tokens', 1500),
                        temperature=self.llm_config.get('temperature', 0.1)
                    )
                    return response.content[0].text
                
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"All LLM call attempts failed for {model}")
                    return None
        
        return None
    
    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response to extract scores and feedback"""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                return result
            else:
                # Fallback: try to parse the entire response
                result = json.loads(response)
                return result
                
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            # Try to extract scores using regex as fallback
            import re
            
            result = {}
            
            # Extract scores
            patterns = {
                'question_quality': r'问题质量[：:]\s*(\d+(?:\.\d+)?)',
                'answer_quality': r'答案质量[：:]\s*(\d+(?:\.\d+)?)',
                'relevance': r'相关性[：:]\s*(\d+(?:\.\d+)?)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, response)
                if match:
                    result[key] = float(match.group(1))
            
            if result:
                result['raw_response'] = response
                return result
            
            return None
    
    def _get_default_scores(self) -> Dict[str, Any]:
        """Get default scores when LLM evaluation fails"""
        return {
            'question_quality': 0.5,
            'answer_quality': 0.5,
            'relevance': 0.5,
            'question_feedback': {
                'strengths': [],
                'weaknesses': ['无法进行LLM评估'],
                'suggestions': []
            },
            'answer_feedback': {
                'strengths': [],
                'weaknesses': ['无法进行LLM评估'],
                'suggestions': []
            },
            'overall_assessment': '由于技术原因无法进行LLM评估，使用默认分数',
            'evaluation_failed': True
        }
    
    def batch_evaluate(self, qa_pairs: List[Tuple[str, str]], 
                      parallel: bool = True,
                      max_workers: int = 4) -> List[Dict[str, Any]]:
        """Evaluate multiple QA pairs"""
        if parallel and len(qa_pairs) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for question, answer in qa_pairs:
                    future = executor.submit(self.evaluate_qa_pair, question, answer)
                    futures.append(future)
                
                results = [future.result() for future in futures]
        else:
            results = []
            for question, answer in qa_pairs:
                result = self.evaluate_qa_pair(question, answer)
                results.append(result)
        
        return results
    
    def evaluate_with_ensemble(self, question: str, answer: str) -> Dict[str, Any]:
        """Evaluate using multiple models and combine results"""
        models = [self.llm_config.get('primary_model')] + self.llm_config.get('fallback_models', [])
        models = [m for m in models if m]  # Remove None values
        
        all_results = []
        for model in models[:3]:  # Limit to 3 models for cost control
            result = self._evaluate_with_model(question, answer, model)
            if result and not result.get('evaluation_failed'):
                all_results.append(result)
        
        if not all_results:
            return self._get_default_scores()
        
        # Combine results
        ensemble_method = self.config.get('advanced', {}).get('ensemble_method', 'weighted_average')
        
        if ensemble_method == 'weighted_average':
            # Simple average for now
            combined = {
                'question_quality': np.mean([r['question_quality'] for r in all_results]),
                'answer_quality': np.mean([r['answer_quality'] for r in all_results]),
                'relevance': np.mean([r['relevance'] for r in all_results]),
                'models_used': [r.get('model_used') for r in all_results],
                'ensemble_method': ensemble_method
            }
            
            # Aggregate feedback
            all_q_strengths = []
            all_q_weaknesses = []
            all_a_strengths = []
            all_a_weaknesses = []
            
            for r in all_results:
                if 'question_feedback' in r:
                    all_q_strengths.extend(r['question_feedback'].get('strengths', []))
                    all_q_weaknesses.extend(r['question_feedback'].get('weaknesses', []))
                if 'answer_feedback' in r:
                    all_a_strengths.extend(r['answer_feedback'].get('strengths', []))
                    all_a_weaknesses.extend(r['answer_feedback'].get('weaknesses', []))
            
            combined['question_feedback'] = {
                'strengths': list(set(all_q_strengths))[:3],
                'weaknesses': list(set(all_q_weaknesses))[:3]
            }
            combined['answer_feedback'] = {
                'strengths': list(set(all_a_strengths))[:3],
                'weaknesses': list(set(all_a_weaknesses))[:3]
            }
            
            return combined
        
        else:
            # Default to first result
            return all_results[0]