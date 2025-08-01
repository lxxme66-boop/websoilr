"""
改进的综合评分模块
针对专家评测结果进行优化，调整权重和评分策略
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np


class ImprovedScorer:
    """改进的综合评分器，更贴近专家评测标准"""
    
    def __init__(self, weights: Dict[str, float] = None, expert_alignment: bool = True):
        """
        初始化评分器
        
        Args:
            weights: 各维度权重字典
            expert_alignment: 是否启用专家对齐模式
        """
        self.logger = logging.getLogger('ImprovedScorer')
        self.expert_alignment = expert_alignment
        
        # 根据专家评测结果调整的默认权重
        if expert_alignment:
            self.default_weights = {
                'llm_score': 0.5,           # 提高LLM评测权重（专家更看重内容质量）
                'semantic_similarity': 0.15,  # 降低语义相似度权重（可能与参考答案措辞不同）
                'answer_quality': 0.25,      # 提高答案质量权重
                'fluency': 0.05,            # 降低流畅度权重（专业内容可能牺牲流畅性）
                'keyword_coverage': 0.05     # 降低关键词覆盖权重
            }
        else:
            self.default_weights = {
                'llm_score': 0.4,
                'semantic_similarity': 0.2,
                'answer_quality': 0.2,
                'fluency': 0.1,
                'keyword_coverage': 0.1
            }
        
        # 使用提供的权重或默认权重
        self.weights = weights or self.default_weights
        
        # 验证权重
        self._validate_weights()
        
        # 专家评测对齐参数
        self.expert_params = {
            'llm_score_threshold': 0.85,      # LLM高分阈值
            'quality_boost_factor': 1.1,      # 质量提升因子
            'technical_accuracy_weight': 0.3,  # 技术准确性权重
            'completeness_weight': 0.2        # 完整性权重
        }
    
    def _validate_weights(self):
        """验证权重配置"""
        total_weight = sum(self.weights.values())
        
        # 如果权重和不为1，进行归一化
        if abs(total_weight - 1.0) > 0.001:
            self.logger.warning(f"Weights sum to {total_weight}, normalizing...")
            for key in self.weights:
                self.weights[key] /= total_weight
    
    def compute_total_score(self, scores: Dict[str, float]) -> float:
        """
        计算综合得分（改进版）
        
        Args:
            scores: 各维度得分字典
            
        Returns:
            综合得分 (0-1)
        """
        # 基础加权得分计算
        total_score = 0.0
        used_weights = 0.0
        
        for metric, weight in self.weights.items():
            if metric in scores:
                score = scores[metric]
                # 确保分数在0-1范围内
                score = max(0.0, min(1.0, score))
                total_score += score * weight
                used_weights += weight
            else:
                self.logger.debug(f"Metric '{metric}' not found in scores")
        
        # 如果有缺失的指标，重新归一化
        if used_weights > 0 and used_weights < 1.0:
            total_score /= used_weights
        
        # 专家对齐调整
        if self.expert_alignment:
            total_score = self._apply_expert_adjustments(total_score, scores)
        
        return float(total_score)
    
    def _apply_expert_adjustments(self, base_score: float, scores: Dict[str, float]) -> float:
        """
        应用专家评测对齐调整
        
        Args:
            base_score: 基础得分
            scores: 各维度得分
            
        Returns:
            调整后的得分
        """
        adjusted_score = base_score
        
        # 1. LLM高分奖励：如果LLM评分很高，说明内容质量好
        if 'llm_score' in scores and scores['llm_score'] >= self.expert_params['llm_score_threshold']:
            boost = (scores['llm_score'] - self.expert_params['llm_score_threshold']) * 0.5
            adjusted_score += boost * 0.1
            self.logger.debug(f"Applied LLM high score boost: +{boost * 0.1:.3f}")
        
        # 2. 答案质量调整：专家更看重答案的实际质量
        if 'answer_quality' in scores:
            quality_factor = scores['answer_quality'] / 0.7  # 以0.7为基准
            if quality_factor > 1:
                quality_boost = (quality_factor - 1) * 0.05
                adjusted_score += quality_boost
                self.logger.debug(f"Applied quality boost: +{quality_boost:.3f}")
        
        # 3. 低语义相似度惩罚减轻：专业答案可能与参考答案措辞不同
        if 'semantic_similarity' in scores and scores['semantic_similarity'] < 0.6:
            # 如果其他指标都好，减轻语义相似度低的惩罚
            if scores.get('llm_score', 0) > 0.8 and scores.get('answer_quality', 0) > 0.7:
                penalty_reduction = (0.6 - scores['semantic_similarity']) * 0.3
                adjusted_score += penalty_reduction * 0.1
                self.logger.debug(f"Reduced semantic similarity penalty: +{penalty_reduction * 0.1:.3f}")
        
        # 4. 技术领域特殊处理：技术准确性比文字流畅更重要
        if self._is_technical_content(scores):
            # 降低流畅度的影响
            if 'fluency' in scores and scores['fluency'] < 0.7:
                fluency_penalty_reduction = (0.7 - scores['fluency']) * 0.5
                adjusted_score += fluency_penalty_reduction * 0.05
                self.logger.debug(f"Reduced fluency penalty for technical content: +{fluency_penalty_reduction * 0.05:.3f}")
        
        # 确保得分在合理范围内
        adjusted_score = max(0.0, min(1.0, adjusted_score))
        
        return adjusted_score
    
    def _is_technical_content(self, scores: Dict[str, float]) -> bool:
        """
        判断是否为技术内容
        基于关键词覆盖率和LLM评分判断
        """
        # 如果关键词覆盖率高且LLM评分高，可能是技术内容
        keyword_coverage = scores.get('keyword_coverage', 0)
        llm_score = scores.get('llm_score', 0)
        
        return keyword_coverage > 0.4 and llm_score > 0.75
    
    def compute_expert_aligned_score(self, scores: Dict[str, float], 
                                   question: str = None, answer: str = None) -> Tuple[float, Dict[str, float]]:
        """
        计算与专家评测对齐的得分
        
        Args:
            scores: 各维度得分
            question: 问题文本（用于额外分析）
            answer: 答案文本（用于额外分析）
            
        Returns:
            (总分, 调整详情)
        """
        # 基础得分
        base_score = self.compute_total_score(scores)
        
        adjustments = {
            'base_score': base_score,
            'adjustments': []
        }
        
        # 分析问题复杂度
        if question:
            complexity_score = self._analyze_question_complexity(question)
            if complexity_score > 0.7:
                # 复杂问题的答案应该得到更高评价
                complexity_bonus = complexity_score * 0.05
                base_score += complexity_bonus
                adjustments['adjustments'].append({
                    'type': 'complexity_bonus',
                    'value': complexity_bonus,
                    'reason': 'Complex question answered well'
                })
        
        # 分析答案的专业性
        if answer:
            professionalism_score = self._analyze_answer_professionalism(answer)
            if professionalism_score > 0.8:
                prof_bonus = professionalism_score * 0.03
                base_score += prof_bonus
                adjustments['adjustments'].append({
                    'type': 'professionalism_bonus',
                    'value': prof_bonus,
                    'reason': 'Highly professional answer'
                })
        
        # 确保最终得分在合理范围
        final_score = max(0.0, min(1.0, base_score))
        adjustments['final_score'] = final_score
        
        return final_score, adjustments
    
    def _analyze_question_complexity(self, question: str) -> float:
        """分析问题复杂度"""
        complexity_indicators = [
            '分析', '解释', '比较', '评估', '原因', '如何', '为什么',
            '详细', '步骤', '方案', '解决', '优化', '设计', '实现'
        ]
        
        # 计算复杂度指标
        indicator_count = sum(1 for ind in complexity_indicators if ind in question)
        question_length = len(question)
        
        # 综合评分
        complexity_score = min(1.0, (indicator_count * 0.15) + (min(question_length, 200) / 200 * 0.3))
        
        return complexity_score
    
    def _analyze_answer_professionalism(self, answer: str) -> float:
        """分析答案专业性"""
        professional_indicators = [
            'TFT', 'EPD', 'OLED', 'LCD', '显示', '技术', '工艺', '参数',
            '性能', '优化', '解决方案', '分析', '检测', '测试', '标准'
        ]
        
        # 专业术语密度
        term_count = sum(1 for term in professional_indicators if term in answer)
        answer_length = len(answer)
        
        # 结构化程度（包含序号、步骤等）
        structure_score = 0
        if any(marker in answer for marker in ['1.', '2.', '3.', '步骤', '首先', '其次', '最后']):
            structure_score = 0.3
        
        # 综合评分
        professionalism_score = min(1.0, 
            (term_count * 0.1) + 
            (min(answer_length, 500) / 500 * 0.2) + 
            structure_score
        )
        
        return professionalism_score
    
    def get_top_k_expert_aligned(self, evaluation_results: List[Dict], k: int = 10) -> List[Dict]:
        """
        获取与专家评测最对齐的Top-K结果
        
        Args:
            evaluation_results: 评测结果列表
            k: 返回的数量
            
        Returns:
            Top-K结果列表
        """
        # 重新计算专家对齐得分
        for result in evaluation_results:
            scores = result.get('scores', {})
            question = result.get('question', '')
            answer = result.get('answer', '')
            
            # 计算专家对齐得分
            expert_score, adjustments = self.compute_expert_aligned_score(
                scores, question, answer
            )
            
            result['expert_aligned_score'] = expert_score
            result['score_adjustments'] = adjustments
        
        # 按专家对齐得分排序
        sorted_results = sorted(
            evaluation_results,
            key=lambda x: x.get('expert_aligned_score', 0),
            reverse=True
        )
        
        return sorted_results[:k]