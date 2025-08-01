"""
改进的综合评分模块
增加了更多与专家评价一致的评分维度
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import re
from collections import Counter


class ImprovedScorer:
    """改进的综合评分器"""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        初始化评分器
        
        Args:
            weights: 各维度权重字典
        """
        self.logger = logging.getLogger('ImprovedScorer')
        
        # 改进的权重配置 - 更符合专家偏好
        self.default_weights = {
            'llm_score': 0.20,          # 降低LLM评测权重
            'semantic_similarity': 0.08,  # 降低语义相似度权重
            'answer_quality': 0.15,      # 降低答案质量权重
            'fluency': 0.05,            # 降低流畅度权重
            'keyword_coverage': 0.05,    # 降低关键词权重
            'conciseness': 0.12,        # 新增：简洁性
            'structure_clarity': 0.08,   # 新增：结构清晰度
            'actionability': 0.08,       # 新增：可操作性
            'document_validity': 0.19    # 新增：文档有效性（最重要）
        }
        
        # 使用提供的权重或默认权重
        self.weights = weights or self.default_weights
        
        # 验证权重
        self._validate_weights()
    
    def _validate_weights(self):
        """验证权重配置"""
        total_weight = sum(self.weights.values())
        
        # 如果权重和不为1，进行归一化
        if abs(total_weight - 1.0) > 0.001:
            self.logger.warning(f"Weights sum to {total_weight}, normalizing...")
            for key in self.weights:
                self.weights[key] /= total_weight
    
    def compute_conciseness_score(self, answer: str, question: str) -> float:
        """
        计算简洁性得分
        简洁但信息丰富的答案得分高
        """
        # 计算答案长度
        answer_length = len(answer)
        question_length = len(question)
        
        # 理想长度比例（答案是问题的2-4倍）
        length_ratio = answer_length / max(question_length, 1)
        
        # 检测重复内容
        sentences = re.split(r'[。！？\n]', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 计算句子重复度
        repetition_score = 1.0
        if len(sentences) > 1:
            unique_sentences = len(set(sentences))
            repetition_score = unique_sentences / len(sentences)
        
        # 检测段落重复
        paragraphs = answer.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        if len(paragraphs) > 1:
            # 简单的段落相似度检测
            para_repetition = len(set(paragraphs)) / len(paragraphs)
            repetition_score = (repetition_score + para_repetition) / 2
        
        # 长度得分
        if length_ratio < 1:
            length_score = 0.3  # 太短
        elif length_ratio < 2:
            length_score = 0.7  # 稍短
        elif length_ratio <= 4:
            length_score = 1.0  # 理想长度
        elif length_ratio <= 6:
            length_score = 0.8  # 稍长
        elif length_ratio <= 8:
            length_score = 0.6  # 偏长
        else:
            length_score = 0.4  # 太长
        
        # 综合得分
        conciseness_score = length_score * 0.6 + repetition_score * 0.4
        
        return float(conciseness_score)
    
    def compute_structure_clarity_score(self, answer: str) -> float:
        """
        计算结构清晰度得分
        有明确结构、分点说明的答案得分高
        """
        score = 0.5  # 基础分
        
        # 检查是否有编号列表
        numbered_pattern = r'^\s*\d+[\.\)、]'
        if re.findall(numbered_pattern, answer, re.MULTILINE):
            score += 0.2
        
        # 检查是否有明确的步骤标记
        step_keywords = ['首先', '其次', '然后', '最后', '第一', '第二', '步骤']
        step_count = sum(1 for keyword in step_keywords if keyword in answer)
        if step_count >= 2:
            score += 0.15
        
        # 检查是否有清晰的段落结构
        paragraphs = answer.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        if 2 <= len(paragraphs) <= 5:
            score += 0.1
        
        # 检查是否有总结或解决方案标记
        solution_keywords = ['解决方案', '解决步骤', '总结', '结论', '建议']
        if any(keyword in answer for keyword in solution_keywords):
            score += 0.05
        
        return min(1.0, float(score))
    
    def compute_actionability_score(self, answer: str) -> float:
        """
        计算可操作性得分
        提供具体、可执行步骤的答案得分高
        """
        score = 0.3  # 基础分
        
        # 检查动作词
        action_verbs = ['检查', '测试', '验证', '更换', '调整', '优化', '分析', 
                       '确认', '排查', '修复', '设置', '配置', '安装', '执行']
        action_count = sum(1 for verb in action_verbs if verb in answer)
        score += min(0.3, action_count * 0.05)
        
        # 检查具体技术术语和参数
        technical_pattern = r'[A-Za-z]+[-_]?[A-Za-z]*|TFT|EPD|SMT|ZnO'
        technical_matches = re.findall(technical_pattern, answer)
        if len(technical_matches) >= 3:
            score += 0.2
        
        # 检查是否有具体的数值或条件
        number_pattern = r'\d+[\.\d]*\s*[%℃度倍]'
        if re.findall(number_pattern, answer):
            score += 0.1
        
        # 检查是否有明确的因果关系描述
        causal_keywords = ['因为', '由于', '导致', '所以', '因此', '可能是']
        if any(keyword in answer for keyword in causal_keywords):
            score += 0.1
        
        return min(1.0, float(score))
    
    def compute_enhanced_scores(self, qa_pair: Dict) -> Dict[str, float]:
        """
        计算增强的评分维度
        
        Args:
            qa_pair: 包含question和answer的字典
            
        Returns:
            增强评分字典
        """
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        
        enhanced_scores = {
            'conciseness': self.compute_conciseness_score(answer, question),
            'structure_clarity': self.compute_structure_clarity_score(answer),
            'actionability': self.compute_actionability_score(answer)
        }
        
        return enhanced_scores
    
    def compute_total_score(self, scores: Dict[str, float], qa_pair: Dict = None) -> float:
        """
        计算综合得分
        
        Args:
            scores: 各维度得分字典
            qa_pair: 问答对（用于计算额外维度）
            
        Returns:
            综合得分 (0-1)
        """
        # 如果提供了问答对，计算额外维度
        if qa_pair:
            enhanced_scores = self.compute_enhanced_scores(qa_pair)
            scores.update(enhanced_scores)
        
        total_score = 0.0
        used_weights = 0.0
        
        # 计算加权得分
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
        
        return float(total_score)
    
    def apply_expert_alignment_penalty(self, result: Dict) -> float:
        """
        应用专家对齐惩罚
        对过长、重复、结构混乱的答案进行惩罚
        """
        answer = result.get('answer', '')
        penalty = 0.0
        
        # 长度惩罚
        if len(answer) > 800:
            penalty += 0.1
        if len(answer) > 1200:
            penalty += 0.1
        
        # 重复惩罚（检查连续重复的句子）
        sentences = re.split(r'[。！？]', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        for i in range(1, len(sentences)):
            if sentences[i] == sentences[i-1]:
                penalty += 0.05
        
        # 结构混乱惩罚（过多的换行或格式不一致）
        newline_count = answer.count('\n')
        if newline_count > 15:
            penalty += 0.05
        
        return min(0.3, penalty)  # 最大惩罚不超过0.3
    
    def rank_qa_pairs(self, evaluation_results: List[Dict]) -> List[Dict]:
        """
        对问答对进行排序（考虑专家偏好）
        
        Args:
            evaluation_results: 评测结果列表
            
        Returns:
            排序后的结果列表
        """
        # 计算每个结果的总分
        for result in evaluation_results:
            # 获取问答对信息
            qa_pair = {
                'question': result.get('question', ''),
                'answer': result.get('answer', '')
            }
            
            # 计算基础总分
            base_score = self.compute_total_score(result.get('scores', {}), qa_pair)
            
            # 应用专家对齐惩罚
            penalty = self.apply_expert_alignment_penalty(result)
            
            # 最终得分
            result['total_score'] = max(0.0, base_score - penalty)
            result['base_score'] = base_score
            result['penalty'] = penalty
        
        # 按总分降序排序
        sorted_results = sorted(
            evaluation_results,
            key=lambda x: x.get('total_score', 0),
            reverse=True
        )
        
        return sorted_results
    
    def get_expert_alignment_report(self, evaluation_results: List[Dict]) -> Dict:
        """
        生成专家对齐报告
        分析当前评分与专家偏好的差异
        """
        report = {
            'alignment_metrics': {},
            'recommendations': []
        }
        
        # 分析各维度得分分布
        score_distributions = {}
        for result in evaluation_results:
            scores = result.get('scores', {})
            for metric, score in scores.items():
                if metric not in score_distributions:
                    score_distributions[metric] = []
                score_distributions[metric].append(score)
        
        # 计算各维度的统计信息
        for metric, scores in score_distributions.items():
            if scores:
                report['alignment_metrics'][metric] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'correlation_with_expert': self._estimate_expert_correlation(metric, scores)
                }
        
        # 生成改进建议
        if 'llm_score' in report['alignment_metrics']:
            if report['alignment_metrics']['llm_score']['mean'] > 0.85:
                report['recommendations'].append(
                    "LLM评分普遍偏高，建议调整LLM评测标准或降低其权重"
                )
        
        if 'semantic_similarity' in report['alignment_metrics']:
            if report['alignment_metrics']['semantic_similarity']['std'] < 0.1:
                report['recommendations'].append(
                    "语义相似度区分度不足，建议使用更精细的语义模型"
                )
        
        return report
    
    def _estimate_expert_correlation(self, metric: str, scores: List[float]) -> float:
        """
        估计某个指标与专家评价的相关性
        这里使用简化的启发式方法
        """
        # 基于经验的相关性估计
        correlation_map = {
            'llm_score': 0.6,           # LLM分数与专家评价中度相关
            'semantic_similarity': 0.4,  # 语义相似度相关性较低
            'answer_quality': 0.7,       # 答案质量相关性较高
            'fluency': 0.3,             # 流畅度相关性低
            'keyword_coverage': 0.3,     # 关键词覆盖相关性低
            'conciseness': 0.8,         # 简洁性高度相关
            'structure_clarity': 0.7,    # 结构清晰度相关性高
            'actionability': 0.8        # 可操作性高度相关
        }
        
        return correlation_map.get(metric, 0.5)