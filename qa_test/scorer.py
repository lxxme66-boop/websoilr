"""
综合评分模块
实现多维度评分的加权计算
"""

import logging
from typing import Dict, List, Optional
import numpy as np


class Scorer:
    """综合评分器"""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        初始化评分器
        
        Args:
            weights: 各维度权重字典
        """
        self.logger = logging.getLogger('Scorer')
        
        # 默认权重配置
        self.default_weights = {
            'llm_score': 0.4,          # LLM评测权重
            'semantic_similarity': 0.2,  # 语义相似度权重
            'answer_quality': 0.2,      # 答案质量权重
            'fluency': 0.1,            # 流畅度权重
            'keyword_coverage': 0.1     # 关键词覆盖权重
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
    
    def compute_total_score(self, scores: Dict[str, float]) -> float:
        """
        计算综合得分
        
        Args:
            scores: 各维度得分字典
            
        Returns:
            综合得分 (0-1)
        """
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
    
    def compute_weighted_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        计算各维度的加权得分
        
        Args:
            scores: 原始得分字典
            
        Returns:
            加权得分字典
        """
        weighted_scores = {}
        
        for metric, weight in self.weights.items():
            if metric in scores:
                score = max(0.0, min(1.0, scores[metric]))
                weighted_scores[metric] = score * weight
        
        return weighted_scores
    
    def rank_qa_pairs(self, evaluation_results: List[Dict]) -> List[Dict]:
        """
        对问答对进行排序
        
        Args:
            evaluation_results: 评测结果列表
            
        Returns:
            排序后的结果列表
        """
        # 计算每个结果的总分
        for result in evaluation_results:
            if 'total_score' not in result:
                result['total_score'] = self.compute_total_score(result.get('scores', {}))
        
        # 按总分降序排序
        sorted_results = sorted(
            evaluation_results,
            key=lambda x: x.get('total_score', 0),
            reverse=True
        )
        
        return sorted_results
    
    def apply_filters(self, evaluation_results: List[Dict], filters: Dict) -> List[Dict]:
        """
        应用过滤条件
        
        Args:
            evaluation_results: 评测结果列表
            filters: 过滤条件字典
            
        Returns:
            过滤后的结果列表
        """
        filtered_results = []
        
        for result in evaluation_results:
            # 检查是否满足所有过滤条件
            passes_filter = True
            
            # 总分过滤
            if 'min_total_score' in filters:
                if result.get('total_score', 0) < filters['min_total_score']:
                    passes_filter = False
            
            # 各维度分数过滤
            scores = result.get('scores', {})
            for metric, min_score in filters.items():
                if metric.startswith('min_') and metric != 'min_total_score':
                    metric_name = metric[4:]  # 去掉'min_'前缀
                    if metric_name in scores and scores[metric_name] < min_score:
                        passes_filter = False
                        break
            
            if passes_filter:
                filtered_results.append(result)
        
        self.logger.info(f"Filtered {len(evaluation_results)} to {len(filtered_results)} results")
        return filtered_results
    
    def calculate_statistics(self, evaluation_results: List[Dict]) -> Dict:
        """
        计算评测结果的统计信息
        
        Args:
            evaluation_results: 评测结果列表
            
        Returns:
            统计信息字典
        """
        if not evaluation_results:
            return {}
        
        # 提取所有分数
        all_scores = {}
        for result in evaluation_results:
            scores = result.get('scores', {})
            for metric, score in scores.items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)
        
        # 计算统计信息
        statistics = {}
        
        # 总分统计
        total_scores = [r.get('total_score', 0) for r in evaluation_results]
        statistics['total_score'] = {
            'mean': float(np.mean(total_scores)),
            'std': float(np.std(total_scores)),
            'min': float(np.min(total_scores)),
            'max': float(np.max(total_scores)),
            'median': float(np.median(total_scores)),
            'percentiles': {
                '25': float(np.percentile(total_scores, 25)),
                '50': float(np.percentile(total_scores, 50)),
                '75': float(np.percentile(total_scores, 75)),
                '90': float(np.percentile(total_scores, 90))
            }
        }
        
        # 各维度统计
        for metric, scores in all_scores.items():
            if scores:
                statistics[metric] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'median': float(np.median(scores))
                }
        
        return statistics
    
    def get_score_distribution(self, evaluation_results: List[Dict], bins: int = 10) -> Dict:
        """
        获取分数分布
        
        Args:
            evaluation_results: 评测结果列表
            bins: 分箱数量
            
        Returns:
            分数分布字典
        """
        if not evaluation_results:
            return {}
        
        total_scores = [r.get('total_score', 0) for r in evaluation_results]
        
        # 计算直方图
        hist, bin_edges = np.histogram(total_scores, bins=bins, range=(0, 1))
        
        # 构建分布字典
        distribution = {
            'bins': [],
            'counts': [],
            'percentages': []
        }
        
        total_count = len(total_scores)
        for i in range(len(hist)):
            bin_range = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
            distribution['bins'].append(bin_range)
            distribution['counts'].append(int(hist[i]))
            distribution['percentages'].append(float(hist[i] / total_count * 100))
        
        return distribution
    
    def identify_outliers(self, evaluation_results: List[Dict], threshold: float = 2.0) -> Dict:
        """
        识别异常值
        
        Args:
            evaluation_results: 评测结果列表
            threshold: 异常值阈值（标准差的倍数）
            
        Returns:
            异常值信息字典
        """
        if not evaluation_results:
            return {}
        
        outliers = {
            'high_performers': [],
            'low_performers': [],
            'metric_outliers': {}
        }
        
        # 总分异常值
        total_scores = [r.get('total_score', 0) for r in evaluation_results]
        mean_score = np.mean(total_scores)
        std_score = np.std(total_scores)
        
        for i, result in enumerate(evaluation_results):
            score = result.get('total_score', 0)
            z_score = (score - mean_score) / std_score if std_score > 0 else 0
            
            if z_score > threshold:
                outliers['high_performers'].append({
                    'index': i,
                    'score': score,
                    'z_score': z_score
                })
            elif z_score < -threshold:
                outliers['low_performers'].append({
                    'index': i,
                    'score': score,
                    'z_score': z_score
                })
        
        # 各维度异常值
        all_scores = {}
        for result in evaluation_results:
            scores = result.get('scores', {})
            for metric, score in scores.items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)
        
        for metric, scores in all_scores.items():
            if len(scores) > 1:
                mean = np.mean(scores)
                std = np.std(scores)
                if std > 0:
                    outlier_indices = []
                    for i, score in enumerate(scores):
                        z_score = (score - mean) / std
                        if abs(z_score) > threshold:
                            outlier_indices.append(i)
                    
                    if outlier_indices:
                        outliers['metric_outliers'][metric] = outlier_indices
        
        return outliers
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        更新权重配置
        
        Args:
            new_weights: 新的权重字典
        """
        self.weights.update(new_weights)
        self._validate_weights()
        self.logger.info(f"Weights updated: {self.weights}")
    
    def get_weight_sensitivity(self, evaluation_results: List[Dict], metric: str, 
                             variations: List[float] = None) -> Dict:
        """
        分析权重敏感性
        
        Args:
            evaluation_results: 评测结果列表
            metric: 要分析的指标
            variations: 权重变化范围
            
        Returns:
            敏感性分析结果
        """
        if not variations:
            variations = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        original_weights = self.weights.copy()
        sensitivity_results = []
        
        for variation in variations:
            # 创建临时权重
            temp_weights = original_weights.copy()
            temp_weights[metric] = variation
            
            # 重新归一化其他权重
            other_weight_sum = sum(w for k, w in temp_weights.items() if k != metric)
            if other_weight_sum > 0:
                scale_factor = (1 - variation) / other_weight_sum
                for k in temp_weights:
                    if k != metric:
                        temp_weights[k] *= scale_factor
            
            # 计算新的总分
            self.weights = temp_weights
            new_scores = []
            for result in evaluation_results:
                new_score = self.compute_total_score(result.get('scores', {}))
                new_scores.append(new_score)
            
            sensitivity_results.append({
                'weight': variation,
                'mean_score': float(np.mean(new_scores)),
                'std_score': float(np.std(new_scores))
            })
        
        # 恢复原始权重
        self.weights = original_weights
        
        return {
            'metric': metric,
            'results': sensitivity_results
        }