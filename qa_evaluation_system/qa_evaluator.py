"""
Core QA Evaluator module that integrates all evaluation methods
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils import (
    load_config, setup_logging, load_qa_pairs, save_results,
    weighted_average, get_quality_level, normalize_score,
    export_top_qa_pairs, create_evaluation_report
)
from rule_checker import RuleChecker
from nlp_metrics import NLPMetrics
from llm_evaluator import LLMEvaluator

logger = logging.getLogger(__name__)


class QAEvaluator:
    """Main QA evaluation system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the QA evaluator"""
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        setup_logging(self.config)
        
        # Initialize components
        self._init_components()
        
        logger.info("QA Evaluator initialized successfully")
    
    def _init_components(self):
        """Initialize evaluation components"""
        # Rule checker
        self.rule_checker = RuleChecker(self.config)
        logger.info("Initialized rule checker")
        
        # NLP metrics calculator
        self.nlp_metrics = NLPMetrics(self.config)
        logger.info("Initialized NLP metrics")
        
        # LLM evaluator
        self.llm_evaluator = LLMEvaluator(self.config)
        logger.info("Initialized LLM evaluator")
        
        # Get weights
        self.weights = self.config.get('weights', {})
        self.overall_weights = self.weights.get('overall', {
            'question_quality': 0.35,
            'answer_quality': 0.40,
            'qa_relevance': 0.25
        })
    
    def evaluate_single(self, question: str, answer: str, 
                       use_ensemble: bool = None) -> Dict[str, Any]:
        """Evaluate a single QA pair"""
        logger.debug(f"Evaluating QA pair - Q: {question[:50]}...")
        
        # Rule-based checks
        rule_results = self.rule_checker.check_qa_pair(question, answer)
        
        # NLP metrics
        nlp_results = self.nlp_metrics.calculate_metrics(question, answer)
        
        # LLM evaluation
        use_ensemble = use_ensemble if use_ensemble is not None else \
                      self.config.get('advanced', {}).get('use_ensemble', False)
        
        if use_ensemble:
            llm_results = self.llm_evaluator.evaluate_with_ensemble(question, answer)
        else:
            llm_results = self.llm_evaluator.evaluate_qa_pair(question, answer)
        
        # Combine all results
        combined_results = self._combine_results(rule_results, nlp_results, llm_results)
        
        # Add original QA pair
        combined_results['question'] = question
        combined_results['answer'] = answer
        
        return combined_results
    
    def _combine_results(self, rule_results: Dict[str, Any],
                        nlp_results: Dict[str, Any],
                        llm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from different evaluation methods"""
        # Extract scores
        rule_score = rule_results.get('rule_score', 0.5)
        nlp_score = nlp_results.get('nlp_score', 0.5)
        
        # Question quality score
        question_quality_scores = {
            'rule': rule_results['question_checks']['details'].get('quality', {}).get('score', 0.5),
            'llm': llm_results.get('question_quality', 0.5),
            'nlp': nlp_results.get('question_coherence', 0.5)
        }
        
        question_quality = weighted_average(
            question_quality_scores,
            {'rule': 0.2, 'llm': 0.6, 'nlp': 0.2}
        )
        
        # Answer quality score
        answer_quality_scores = {
            'rule': rule_results['answer_checks']['details'].get('quality', {}).get('score', 0.5),
            'llm': llm_results.get('answer_quality', 0.5),
            'nlp': (nlp_results['readability'].get('normalized_score', 0.5) +
                   nlp_results['informativeness']['score']) / 2
        }
        
        answer_quality = weighted_average(
            answer_quality_scores,
            {'rule': 0.2, 'llm': 0.6, 'nlp': 0.2}
        )
        
        # Relevance score
        relevance_scores = {
            'llm': llm_results.get('relevance', 0.5),
            'nlp': nlp_results['answer_relevance']['score'],
            'rule': 1.0 if rule_results['qa_checks']['passed'] else 0.7
        }
        
        qa_relevance = weighted_average(
            relevance_scores,
            {'llm': 0.5, 'nlp': 0.3, 'rule': 0.2}
        )
        
        # Overall score
        overall_score = weighted_average(
            {
                'question_quality': question_quality,
                'answer_quality': answer_quality,
                'qa_relevance': qa_relevance
            },
            self.overall_weights
        )
        
        # Compile issues and suggestions
        issues = []
        suggestions = []
        
        # From rule checker
        issues.extend(rule_results.get('issues', []))
        issues.extend(rule_results.get('warnings', []))
        
        # From LLM
        if 'question_feedback' in llm_results:
            issues.extend(llm_results['question_feedback'].get('weaknesses', []))
            suggestions.extend(llm_results['question_feedback'].get('suggestions', []))
        
        if 'answer_feedback' in llm_results:
            issues.extend(llm_results['answer_feedback'].get('weaknesses', []))
            suggestions.extend(llm_results['answer_feedback'].get('suggestions', []))
        
        # Remove duplicates
        issues = list(set(issues))[:5]  # Top 5 issues
        suggestions = list(set(suggestions))[:3]  # Top 3 suggestions
        
        # Get quality level
        quality_level = get_quality_level(overall_score, self.config.get('thresholds', {}))
        
        return {
            'overall_score': normalize_score(overall_score),
            'quality_level': quality_level,
            'detailed_scores': {
                'question_quality': normalize_score(question_quality),
                'answer_quality': normalize_score(answer_quality),
                'qa_relevance': normalize_score(qa_relevance),
                'rule_score': normalize_score(rule_score),
                'nlp_score': normalize_score(nlp_score)
            },
            'component_scores': {
                'rule_results': rule_results,
                'nlp_results': nlp_results,
                'llm_results': llm_results
            },
            'issues': issues,
            'suggestions': suggestions
        }
    
    def evaluate_batch(self, qa_pairs: List[Dict[str, str]], 
                      parallel: bool = True,
                      show_progress: bool = True) -> List[Dict[str, Any]]:
        """Evaluate multiple QA pairs"""
        logger.info(f"Starting batch evaluation of {len(qa_pairs)} QA pairs")
        
        processing_config = self.config.get('processing', {})
        num_workers = processing_config.get('num_workers', 4)
        
        results = []
        
        if parallel and len(qa_pairs) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_qa = {}
                for i, qa in enumerate(qa_pairs):
                    future = executor.submit(
                        self.evaluate_single,
                        qa['question'],
                        qa['answer']
                    )
                    future_to_qa[future] = (i, qa)
                
                # Process completed tasks
                if show_progress:
                    pbar = tqdm(total=len(qa_pairs), desc="Evaluating QA pairs")
                
                for future in as_completed(future_to_qa):
                    idx, qa = future_to_qa[future]
                    try:
                        result = future.result()
                        # Add metadata
                        result['id'] = qa.get('id', idx)
                        result['metadata'] = qa.get('metadata', {})
                        results.append((idx, result))
                    except Exception as e:
                        logger.error(f"Error evaluating QA pair {idx}: {e}")
                        if not processing_config.get('skip_on_error', True):
                            raise
                        # Add error result
                        error_result = self._get_error_result(qa['question'], qa['answer'], str(e))
                        error_result['id'] = qa.get('id', idx)
                        error_result['metadata'] = qa.get('metadata', {})
                        results.append((idx, error_result))
                    
                    if show_progress:
                        pbar.update(1)
                
                if show_progress:
                    pbar.close()
            
            # Sort results by original index
            results.sort(key=lambda x: x[0])
            results = [r[1] for r in results]
        
        else:
            # Sequential processing
            if show_progress:
                qa_pairs = tqdm(qa_pairs, desc="Evaluating QA pairs")
            
            for i, qa in enumerate(qa_pairs):
                try:
                    result = self.evaluate_single(qa['question'], qa['answer'])
                    result['id'] = qa.get('id', i)
                    result['metadata'] = qa.get('metadata', {})
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating QA pair {i}: {e}")
                    if not processing_config.get('skip_on_error', True):
                        raise
                    error_result = self._get_error_result(qa['question'], qa['answer'], str(e))
                    error_result['id'] = qa.get('id', i)
                    error_result['metadata'] = qa.get('metadata', {})
                    results.append(error_result)
        
        logger.info(f"Completed evaluation of {len(results)} QA pairs")
        return results
    
    def _get_error_result(self, question: str, answer: str, error_msg: str) -> Dict[str, Any]:
        """Get default result for error cases"""
        return {
            'question': question,
            'answer': answer,
            'overall_score': 0.0,
            'quality_level': '评测失败',
            'detailed_scores': {
                'question_quality': 0.0,
                'answer_quality': 0.0,
                'qa_relevance': 0.0,
                'rule_score': 0.0,
                'nlp_score': 0.0
            },
            'issues': [f'评测过程出错: {error_msg}'],
            'suggestions': ['请检查输入数据格式并重试'],
            'error': True
        }
    
    def evaluate_file(self, input_path: str, output_path: str = None,
                     sample_size: int = None,
                     export_top: bool = True) -> Dict[str, Any]:
        """Evaluate QA pairs from file"""
        # Load QA pairs
        qa_pairs = load_qa_pairs(input_path)
        logger.info(f"Loaded {len(qa_pairs)} QA pairs from {input_path}")
        
        # Sample if requested
        if sample_size and sample_size < len(qa_pairs):
            import random
            qa_pairs = random.sample(qa_pairs, sample_size)
            logger.info(f"Sampled {sample_size} QA pairs for evaluation")
        
        # Evaluate
        results = self.evaluate_batch(qa_pairs)
        
        # Save results
        if output_path:
            output_format = self.config.get('output', {}).get('format', 'json')
            save_results(results, output_path, format=output_format)
            logger.info(f"Saved evaluation results to {output_path}")
        
        # Export top QA pairs if requested
        if export_top:
            export_config = self.config.get('output', {})
            threshold = export_config.get('min_export_score', 0.7)
            top_percentage = export_config.get('export_top_percentage', 0.3)
            export_path = f"{export_config.get('export_dir', 'exports')}/top_qa_pairs.json"
            
            num_exported = export_top_qa_pairs(
                qa_pairs, results, export_path,
                threshold=threshold,
                top_percentage=top_percentage
            )
            logger.info(f"Exported {num_exported} top quality QA pairs to {export_path}")
        
        # Create evaluation report
        report_path = f"{self.config.get('output', {}).get('report_dir', 'results')}/evaluation_report.json"
        report = create_evaluation_report(results, report_path)
        
        return report
    
    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from evaluation results"""
        scores = [r['overall_score'] for r in results]
        
        stats = {
            'total_evaluated': len(results),
            'average_score': np.mean(scores),
            'median_score': np.median(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'quality_distribution': {},
            'score_percentiles': {
                '25th': np.percentile(scores, 25),
                '50th': np.percentile(scores, 50),
                '75th': np.percentile(scores, 75),
                '90th': np.percentile(scores, 90),
                '95th': np.percentile(scores, 95)
            }
        }
        
        # Quality distribution
        for result in results:
            level = result['quality_level']
            stats['quality_distribution'][level] = stats['quality_distribution'].get(level, 0) + 1
        
        # Convert counts to percentages
        total = len(results)
        for level in stats['quality_distribution']:
            count = stats['quality_distribution'][level]
            stats['quality_distribution'][level] = {
                'count': count,
                'percentage': count / total * 100
            }
        
        return stats