"""
改进的核心评测引擎模块
使用改进的评分器，更好地对齐专家偏好
"""

import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

from llm_evaluator import LLMEvaluator
from nlp_metrics import NLPMetrics
from semantic_analyzer import SemanticAnalyzer
from scorer_improved import ImprovedScorer
from data_processor import DataProcessor


@dataclass
class QAPair:
    """问答对数据结构"""
    question: str
    answer: str
    id: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class EvaluationResult:
    """评测结果数据结构"""
    qa_pair: QAPair
    scores: Dict[str, float]
    total_score: float
    details: Dict[str, any]


class ImprovedQAEvaluator:
    """改进的问答对质量评测器"""
    
    def __init__(self, config: Dict):
        """
        初始化评测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = self._setup_logger()
        
        # 初始化各个评测组件
        self.llm_evaluator = LLMEvaluator(config.get('llm', {}))
        self.nlp_metrics = NLPMetrics()
        self.semantic_analyzer = SemanticAnalyzer(config.get('semantic', {}))
        self.scorer = ImprovedScorer(config.get('weights'))  # 使用改进的评分器
        self.data_processor = DataProcessor()
        
        # 评测配置
        self.batch_size = config.get('evaluation', {}).get('batch_size', 10)
        self.max_workers = config.get('evaluation', {}).get('max_workers', 4)
        self.cache_enabled = config.get('evaluation', {}).get('cache_enabled', True)
        
        # 结果缓存
        self.cache = {} if self.cache_enabled else None
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger('ImprovedQAEvaluator')
        logger.setLevel(logging.INFO)
        return logger
    
    def evaluate_single(self, qa_pair: QAPair) -> EvaluationResult:
        """
        评测单个问答对
        
        Args:
            qa_pair: 问答对
            
        Returns:
            评测结果
        """
        # 检查缓存
        cache_key = f"{qa_pair.question}:{qa_pair.answer}"
        if self.cache_enabled and cache_key in self.cache:
            self.logger.debug(f"Using cached result for QA pair {qa_pair.id}")
            return self.cache[cache_key]
        
        # 收集各维度评分
        scores = {}
        details = {}
        
        try:
            # 1. LLM评测
            llm_result = self.llm_evaluator.evaluate(
                qa_pair.question, 
                qa_pair.answer
            )
            scores['llm_score'] = llm_result['score']
            details['llm_details'] = llm_result
            
            # 2. 语义相似度
            semantic_score = self.semantic_analyzer.compute_similarity(
                qa_pair.question,
                qa_pair.answer
            )
            scores['semantic_similarity'] = semantic_score['score']
            details['semantic_details'] = semantic_score
            
            # 3. NLP指标
            nlp_scores = self.nlp_metrics.compute_all_metrics(
                qa_pair.question,
                qa_pair.answer
            )
            scores.update(nlp_scores)
            details['nlp_details'] = nlp_scores
            
            # 4. 答案质量评估（基于多个因素）
            answer_quality = self._assess_answer_quality(qa_pair, scores)
            scores['answer_quality'] = answer_quality['score']
            details['quality_details'] = answer_quality
            
            # 5. 计算综合得分（使用改进的评分方法）
            qa_dict = {
                'question': qa_pair.question,
                'answer': qa_pair.answer
            }
            total_score = self.scorer.compute_total_score(scores, qa_dict)
            
            # 创建评测结果
            result = EvaluationResult(
                qa_pair=qa_pair,
                scores=scores,
                total_score=total_score,
                details=details
            )
            
            # 缓存结果
            if self.cache_enabled:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating QA pair {qa_pair.id}: {e}")
            # 返回默认低分结果
            return EvaluationResult(
                qa_pair=qa_pair,
                scores={'error': 0.0},
                total_score=0.0,
                details={'error': str(e)}
            )
    
    def evaluate_batch(self, qa_pairs: List[QAPair], 
                      show_progress: bool = True) -> List[EvaluationResult]:
        """
        批量评测问答对
        
        Args:
            qa_pairs: 问答对列表
            show_progress: 是否显示进度条
            
        Returns:
            评测结果列表
        """
        results = []
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_qa = {
                executor.submit(self.evaluate_single, qa_pair): qa_pair
                for qa_pair in qa_pairs
            }
            
            # 收集结果
            iterator = as_completed(future_to_qa)
            if show_progress:
                iterator = tqdm(iterator, total=len(qa_pairs), desc="Evaluating")
            
            for future in iterator:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    qa_pair = future_to_qa[future]
                    self.logger.error(f"Failed to evaluate {qa_pair.id}: {e}")
                    # 添加失败结果
                    results.append(EvaluationResult(
                        qa_pair=qa_pair,
                        scores={'error': 0.0},
                        total_score=0.0,
                        details={'error': str(e)}
                    ))
        
        return results
    
    def _assess_answer_quality(self, qa_pair: QAPair, 
                              existing_scores: Dict[str, float]) -> Dict:
        """
        评估答案质量（改进版）
        考虑更多专家偏好的因素
        """
        quality_factors = {
            'length_appropriateness': 0.0,
            'information_density': 0.0,
            'structure_clarity': 0.0,
            'keyword_coverage': 0.0
        }
        
        # 1. 长度适当性（专家偏好简洁的答案）
        answer_len = len(qa_pair.answer)
        question_len = len(qa_pair.question)
        ratio = answer_len / max(question_len, 1)
        
        if ratio < 1.5:
            quality_factors['length_appropriateness'] = 0.5
        elif 1.5 <= ratio <= 3:
            quality_factors['length_appropriateness'] = 1.0
        elif 3 < ratio <= 5:
            quality_factors['length_appropriateness'] = 0.8
        elif 5 < ratio <= 7:
            quality_factors['length_appropriateness'] = 0.6
        else:
            quality_factors['length_appropriateness'] = 0.4
        
        # 2. 信息密度（避免冗余）
        sentences = [s.strip() for s in qa_pair.answer.split('。') if s.strip()]
        if sentences:
            unique_sentences = len(set(sentences))
            quality_factors['information_density'] = unique_sentences / len(sentences)
        else:
            quality_factors['information_density'] = 0.5
        
        # 3. 结构清晰度
        has_numbering = any(line.strip().startswith(('1', '2', '3', '①', '②', '③')) 
                           for line in qa_pair.answer.split('\n'))
        has_sections = '解决' in qa_pair.answer or '步骤' in qa_pair.answer
        
        if has_numbering and has_sections:
            quality_factors['structure_clarity'] = 1.0
        elif has_numbering or has_sections:
            quality_factors['structure_clarity'] = 0.8
        else:
            quality_factors['structure_clarity'] = 0.6
        
        # 4. 关键词覆盖（已有）
        quality_factors['keyword_coverage'] = existing_scores.get('keyword_coverage', 0.5)
        
        # 计算总体质量分数
        weights = {
            'length_appropriateness': 0.3,
            'information_density': 0.3,
            'structure_clarity': 0.2,
            'keyword_coverage': 0.2
        }
        
        overall_score = sum(
            quality_factors[factor] * weight 
            for factor, weight in weights.items()
        )
        
        return {
            'score': overall_score,
            'factors': quality_factors,
            'overall': overall_score
        }
    
    def filter_results(self, results: List[EvaluationResult], 
                      filters: Dict) -> List[EvaluationResult]:
        """
        根据条件过滤结果
        
        Args:
            results: 评测结果列表
            filters: 过滤条件
            
        Returns:
            过滤后的结果
        """
        return self.scorer.apply_filters(
            [self._result_to_dict(r) for r in results],
            filters
        )
    
    def rank_results(self, results: List[EvaluationResult]) -> List[Dict]:
        """
        对结果进行排序（使用改进的排序方法）
        
        Args:
            results: 评测结果列表
            
        Returns:
            排序后的结果
        """
        # 转换为字典格式
        dict_results = []
        for r in results:
            result_dict = self._result_to_dict(r)
            dict_results.append(result_dict)
        
        # 使用改进的排序方法
        ranked = self.scorer.rank_qa_pairs(dict_results)
        return ranked
    
    def _result_to_dict(self, result: EvaluationResult) -> Dict:
        """将评测结果转换为字典格式"""
        return {
            'question': result.qa_pair.question,
            'answer': result.qa_pair.answer,
            'id': result.qa_pair.id,
            'total_score': result.total_score,
            'scores': result.scores,
            'details': result.details
        }
    
    def generate_report(self, results: List[EvaluationResult]) -> Dict:
        """
        生成评测报告（增强版）
        
        Args:
            results: 评测结果列表
            
        Returns:
            评测报告
        """
        # 转换结果格式
        dict_results = [self._result_to_dict(r) for r in results]
        
        # 基础统计
        statistics = self.scorer.calculate_statistics(dict_results)
        
        # 分数分布
        distribution = self.scorer.get_score_distribution(dict_results)
        
        # 异常值检测
        outliers = self.scorer.identify_outliers(dict_results)
        
        # 专家对齐报告
        alignment_report = self.scorer.get_expert_alignment_report(dict_results)
        
        # 质量洞察
        quality_insights = self._generate_quality_insights(dict_results)
        
        report = {
            'total_evaluated': len(results),
            'statistics': statistics,
            'score_distribution': distribution,
            'quality_insights': quality_insights,
            'outliers': outliers,
            'expert_alignment': alignment_report
        }
        
        return report
    
    def _generate_quality_insights(self, results: List[Dict]) -> List[Dict]:
        """生成质量洞察"""
        insights = []
        
        # 找出各维度的最佳和最差样例
        metrics = ['llm_score', 'semantic_similarity', 'answer_quality']
        
        for metric in metrics:
            # 按该维度排序
            sorted_by_metric = sorted(
                results,
                key=lambda x: x.get('scores', {}).get(metric, 0),
                reverse=True
            )
            
            if sorted_by_metric:
                insights.append({
                    'metric': metric,
                    'best_example': {
                        'question': sorted_by_metric[0]['question'][:100] + '...',
                        'score': sorted_by_metric[0]['scores'].get(metric, 0)
                    },
                    'worst_example': {
                        'question': sorted_by_metric[-1]['question'][:100] + '...',
                        'score': sorted_by_metric[-1]['scores'].get(metric, 0)
                    }
                })
        
        return insights