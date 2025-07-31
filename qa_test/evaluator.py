"""
核心评测引擎模块
实现问答对质量的多维度评估
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
from scorer import Scorer
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


class QAEvaluator:
    """问答对质量评测器"""
    
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
        self.nlp_metrics = NLPMetrics(config.get('nlp', {}))
        self.semantic_analyzer = SemanticAnalyzer(config.get('semantic', {}))
        self.scorer = Scorer(config.get('weights', {}))
        self.data_processor = DataProcessor(config.get('preprocessing', {}))
        
        # 评测配置
        self.batch_size = config.get('batch_size', 10)
        self.max_workers = config.get('max_workers', 4)
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache = {} if self.cache_enabled else None
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('QAEvaluator')
        logger.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # 文件处理器
        fh = logging.FileHandler('qa_evaluation.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
    
    def evaluate_single(self, qa_pair: QAPair) -> EvaluationResult:
        """
        评测单个问答对
        
        Args:
            qa_pair: 问答对
            
        Returns:
            评测结果
        """
        self.logger.debug(f"Evaluating QA pair: {qa_pair.id}")
        
        # 检查缓存
        cache_key = f"{qa_pair.question}:{qa_pair.answer}"
        if self.cache_enabled and cache_key in self.cache:
            self.logger.debug(f"Using cached result for {qa_pair.id}")
            return self.cache[cache_key]
        
        # 数据预处理
        processed_qa = self.data_processor.process_qa_pair(qa_pair)
        
        # 执行多维度评测
        scores = {}
        details = {}
        
        # 1. LLM评测
        try:
            llm_result = self.llm_evaluator.evaluate(
                processed_qa.question, 
                processed_qa.answer
            )
            scores['llm_score'] = llm_result['score']
            details['llm_details'] = llm_result
        except Exception as e:
            self.logger.error(f"LLM evaluation failed: {e}")
            scores['llm_score'] = 0.0
            details['llm_details'] = {'error': str(e)}
        
        # 2. 语义相似度
        try:
            semantic_score = self.semantic_analyzer.compute_similarity(
                processed_qa.question,
                processed_qa.answer
            )
            scores['semantic_similarity'] = semantic_score
            details['semantic_details'] = {
                'score': semantic_score,
                'method': 'sentence-transformers'
            }
        except Exception as e:
            self.logger.error(f"Semantic analysis failed: {e}")
            scores['semantic_similarity'] = 0.0
            details['semantic_details'] = {'error': str(e)}
        
        # 3. NLP指标
        try:
            nlp_scores = self.nlp_metrics.compute_metrics(
                processed_qa.question,
                processed_qa.answer
            )
            scores.update(nlp_scores)
            details['nlp_details'] = nlp_scores
        except Exception as e:
            self.logger.error(f"NLP metrics computation failed: {e}")
            details['nlp_details'] = {'error': str(e)}
        
        # 4. 答案质量评估
        try:
            quality_scores = self._evaluate_answer_quality(
                processed_qa.question,
                processed_qa.answer
            )
            scores['answer_quality'] = quality_scores['overall']
            details['quality_details'] = quality_scores
        except Exception as e:
            self.logger.error(f"Quality evaluation failed: {e}")
            scores['answer_quality'] = 0.0
            details['quality_details'] = {'error': str(e)}
        
        # 计算综合得分
        total_score = self.scorer.compute_total_score(scores)
        
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
    
    def evaluate_batch(self, qa_pairs: List[QAPair]) -> List[EvaluationResult]:
        """
        批量评测问答对
        
        Args:
            qa_pairs: 问答对列表
            
        Returns:
            评测结果列表
        """
        self.logger.info(f"Starting batch evaluation of {len(qa_pairs)} QA pairs")
        
        results = []
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_qa = {
                executor.submit(self.evaluate_single, qa_pair): qa_pair 
                for qa_pair in qa_pairs
            }
            
            # 收集结果
            with tqdm(total=len(qa_pairs), desc="Evaluating QA pairs") as pbar:
                for future in as_completed(future_to_qa):
                    qa_pair = future_to_qa[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Failed to evaluate {qa_pair.id}: {e}")
                        # 创建失败结果
                        results.append(EvaluationResult(
                            qa_pair=qa_pair,
                            scores={},
                            total_score=0.0,
                            details={'error': str(e)}
                        ))
                    finally:
                        pbar.update(1)
        
        self.logger.info(f"Batch evaluation completed")
        return results
    
    def _evaluate_answer_quality(self, question: str, answer: str) -> Dict[str, float]:
        """
        评估答案质量
        
        Args:
            question: 问题
            answer: 答案
            
        Returns:
            质量评分字典
        """
        quality_scores = {}
        
        # 1. 答案长度适中性
        answer_length = len(answer)
        if answer_length < 20:
            length_score = 0.3
        elif answer_length > 500:
            length_score = 0.7
        else:
            # 理想长度范围内
            length_score = 1.0
        quality_scores['length_appropriateness'] = length_score
        
        # 2. 信息密度（非重复性）
        words = answer.split()
        unique_words = set(words)
        if len(words) > 0:
            diversity_score = len(unique_words) / len(words)
        else:
            diversity_score = 0.0
        quality_scores['information_density'] = diversity_score
        
        # 3. 结构清晰度（是否有分段、列表等）
        structure_score = 0.5  # 基础分
        if '\n' in answer:  # 有换行
            structure_score += 0.2
        if any(marker in answer for marker in ['1.', '2.', '•', '-']):  # 有列表
            structure_score += 0.3
        quality_scores['structure_clarity'] = min(structure_score, 1.0)
        
        # 4. 问题关键词覆盖率
        question_keywords = set(self.data_processor.extract_keywords(question))
        answer_keywords = set(self.data_processor.extract_keywords(answer))
        if question_keywords:
            coverage = len(question_keywords & answer_keywords) / len(question_keywords)
        else:
            coverage = 0.5
        quality_scores['keyword_coverage'] = coverage
        
        # 计算综合质量分
        quality_scores['overall'] = np.mean(list(quality_scores.values()))
        
        return quality_scores
    
    def select_top_k(self, results: List[EvaluationResult], k: int) -> List[EvaluationResult]:
        """
        选择得分最高的K个问答对
        
        Args:
            results: 所有评测结果
            k: 选择数量
            
        Returns:
            Top-K结果列表
        """
        # 按总分排序
        sorted_results = sorted(results, key=lambda x: x.total_score, reverse=True)
        
        # 应用阈值过滤
        min_score = self.config.get('thresholds', {}).get('min_total_score', 0.7)
        filtered_results = [r for r in sorted_results if r.total_score >= min_score]
        
        # 返回Top-K
        top_k_results = filtered_results[:k]
        self.logger.info(f"Selected {len(top_k_results)} QA pairs from {len(results)} total")
        
        return top_k_results
    
    def generate_report(self, results: List[EvaluationResult]) -> Dict:
        """
        生成评测报告
        
        Args:
            results: 评测结果列表
            
        Returns:
            报告字典
        """
        report = {
            'total_evaluated': len(results),
            'statistics': {},
            'score_distribution': {},
            'quality_insights': []
        }
        
        if not results:
            return report
        
        # 计算统计信息
        all_scores = [r.total_score for r in results]
        report['statistics'] = {
            'mean_score': np.mean(all_scores),
            'std_score': np.std(all_scores),
            'min_score': np.min(all_scores),
            'max_score': np.max(all_scores),
            'median_score': np.median(all_scores)
        }
        
        # 分数分布
        score_ranges = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        for low, high in score_ranges:
            count = sum(1 for s in all_scores if low <= s < high)
            report['score_distribution'][f'{low}-{high}'] = count
        
        # 质量洞察
        # 找出各维度得分最高和最低的样本
        for metric in ['llm_score', 'semantic_similarity', 'answer_quality']:
            valid_results = [r for r in results if metric in r.scores]
            if valid_results:
                best = max(valid_results, key=lambda x: x.scores[metric])
                worst = min(valid_results, key=lambda x: x.scores[metric])
                
                report['quality_insights'].append({
                    'metric': metric,
                    'best_example': {
                        'question': best.qa_pair.question[:100] + '...',
                        'score': best.scores[metric]
                    },
                    'worst_example': {
                        'question': worst.qa_pair.question[:100] + '...',
                        'score': worst.scores[metric]
                    }
                })
        
        return report