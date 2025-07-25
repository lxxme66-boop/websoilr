#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Synthesizer - 数据综合器
将所有组件的输出整合成最终的数据集

该模块实现：
1. 数据整合：将问答对、轨迹、知识图谱等组件输出整合
2. 质量过滤：根据配置的质量标准过滤数据
3. 数据增强：通过释义、负样本生成等方式增强数据
4. 格式标准化：将数据转换为标准格式
"""

import logging
import random
import re
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
import json

from .obfuscation_processor import ObfuscatedQuestionAnswer
from .trajectory_generator import ReasoningTrajectory
from .knowledge_graph_builder import KnowledgeGraph
from .subgraph_sampler import Subgraph


class DatasetEntry:
    """数据集条目类"""
    
    def __init__(self, qa_pair: ObfuscatedQuestionAnswer, trajectory: ReasoningTrajectory,
                 entry_id: str = None, metadata: Dict[str, Any] = None):
        """
        初始化数据集条目
        
        Args:
            qa_pair: 问答对
            trajectory: 推理轨迹
            entry_id: 条目ID
            metadata: 元数据
        """
        self.qa_pair = qa_pair
        self.trajectory = trajectory
        self.entry_id = entry_id or f"entry_{random.randint(1000, 9999)}"
        self.metadata = metadata or {}
        
        # 计算质量分数
        self.quality_score = self._compute_quality_score()
    
    def _compute_quality_score(self) -> float:
        """计算质量分数"""
        # 基于多个因素计算质量分数
        factors = {
            "question_clarity": self._assess_question_clarity(),
            "answer_completeness": self._assess_answer_completeness(),
            "reasoning_coherence": self._assess_reasoning_coherence(),
            "complexity_appropriateness": self._assess_complexity_appropriateness()
        }
        
        # 加权平均
        weights = {
            "question_clarity": 0.25,
            "answer_completeness": 0.25,
            "reasoning_coherence": 0.3,
            "complexity_appropriateness": 0.2
        }
        
        quality_score = sum(factors[factor] * weights[factor] for factor in factors)
        return min(max(quality_score, 0.0), 1.0)  # 限制在0-1范围
    
    def _assess_question_clarity(self) -> float:
        """评估问题清晰度"""
        question = self.qa_pair.obfuscated_question
        
        # 基于问题长度、语法完整性等评估
        if len(question) < 10:
            return 0.3
        elif len(question) > 200:
            return 0.6
        
        # 检查是否包含问号
        has_question_mark = "？" in question or "?" in question
        
        # 检查是否包含关键疑问词
        question_words = ["什么", "哪些", "如何", "为什么", "是否", "多少"]
        has_question_word = any(word in question for word in question_words)
        
        clarity_score = 0.5
        if has_question_mark:
            clarity_score += 0.2
        if has_question_word:
            clarity_score += 0.3
        
        return clarity_score
    
    def _assess_answer_completeness(self) -> float:
        """评估答案完整性"""
        answer = self.qa_pair.obfuscated_answer
        
        # 基于答案长度、信息量等评估
        if len(answer) < 5:
            return 0.2
        elif len(answer) > 500:
            return 0.7
        
        # 检查是否包含具体信息
        has_specific_info = len(re.findall(r'\w+', answer)) > 3
        
        completeness_score = 0.5
        if has_specific_info:
            completeness_score += 0.3
        
        # 检查答案是否与问题相关
        question_entities = set(re.findall(r'\w+', self.qa_pair.obfuscated_question.lower()))
        answer_entities = set(re.findall(r'\w+', answer.lower()))
        overlap = len(question_entities.intersection(answer_entities))
        
        if overlap > 0:
            completeness_score += 0.2
        
        return completeness_score
    
    def _assess_reasoning_coherence(self) -> float:
        """评估推理连贯性"""
        if not self.trajectory or not self.trajectory.steps:
            return 0.3
        
        # 基于推理步骤的逻辑性评估
        coherence_score = 0.5
        
        # 检查步骤数量
        num_steps = len(self.trajectory.steps)
        if 3 <= num_steps <= 8:
            coherence_score += 0.2
        
        # 检查平均置信度
        avg_confidence = self.trajectory.features.get("avg_confidence", 0)
        if avg_confidence > 0.7:
            coherence_score += 0.3
        
        return coherence_score
    
    def _assess_complexity_appropriateness(self) -> float:
        """评估复杂度适当性"""
        # 基于问题类型和复杂度匹配度评估
        question_type = self.qa_pair.question_type
        complexity_score = self.qa_pair.features.get("complexity_score", 1.0)
        
        # 不同问题类型的期望复杂度范围
        expected_complexity = {
            "factual_single": (1.0, 2.5),
            "factual_multi": (2.0, 4.0),
            "comparative": (2.5, 4.5),
            "aggregative": (3.0, 5.0),
            "reasoning": (3.5, 5.0)
        }
        
        min_expected, max_expected = expected_complexity.get(question_type, (1.0, 5.0))
        
        if min_expected <= complexity_score <= max_expected:
            return 1.0
        elif complexity_score < min_expected:
            return 0.5 + (complexity_score / min_expected) * 0.5
        else:
            return 0.5 + (max_expected / complexity_score) * 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "entry_id": self.entry_id,
            "question": self.qa_pair.obfuscated_question,
            "answer": self.qa_pair.obfuscated_answer,
            "question_type": self.qa_pair.question_type,
            "obfuscation_type": self.qa_pair.obfuscation_type,
            "trajectory": self.trajectory.to_dict() if self.trajectory else None,
            "quality_score": self.quality_score,
            "features": {
                "qa_features": self.qa_pair.obfuscation_features,
                "trajectory_features": self.trajectory.features if self.trajectory else {}
            },
            "metadata": self.metadata
        }


class DataSynthesizer:
    """数据综合器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据综合器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.synthesis_config = config.get("data_synthesis", {})
        
        # 质量过滤配置
        self.quality_filters = self.synthesis_config.get("quality_filters", {})
        self.min_quality_score = self.quality_filters.get("coherence_threshold", 0.7)
        
        # 数据增强配置
        self.augmentation_config = self.synthesis_config.get("augmentation", {})
        
        logging.info("数据综合器初始化完成")
    
    def synthesize_dataset(self, qa_pairs: List[ObfuscatedQuestionAnswer],
                          trajectories: List[ReasoningTrajectory],
                          knowledge_graph: KnowledgeGraph,
                          subgraphs: List[Subgraph]) -> Dict[str, Any]:
        """
        综合生成最终数据集
        
        Args:
            qa_pairs: 问答对列表
            trajectories: 推理轨迹列表
            knowledge_graph: 知识图谱
            subgraphs: 子图列表
            
        Returns:
            综合后的数据集
        """
        logging.info("开始数据综合...")
        
        # 1. 配对问答对和轨迹
        dataset_entries = self._pair_qa_and_trajectories(qa_pairs, trajectories)
        
        # 2. 质量过滤
        filtered_entries = self._apply_quality_filters(dataset_entries)
        
        # 3. 数据增强
        if self.augmentation_config.get("paraphrase_questions", False):
            filtered_entries = self._apply_question_paraphrasing(filtered_entries)
        
        if self.augmentation_config.get("generate_negative_samples", False):
            negative_samples = self._generate_negative_samples(filtered_entries)
            filtered_entries.extend(negative_samples)
        
        if self.augmentation_config.get("add_context_variations", False):
            filtered_entries = self._add_context_variations(filtered_entries)
        
        # 4. 格式标准化
        standardized_qa_pairs = [entry.to_dict() for entry in filtered_entries]
        standardized_trajectories = [entry.trajectory.to_dict() for entry in filtered_entries if entry.trajectory]
        
        # 5. 生成统计信息
        statistics = self._generate_statistics(
            standardized_qa_pairs, standardized_trajectories, knowledge_graph, subgraphs
        )
        
        # 6. 构建最终数据集
        final_dataset = {
            "qa_pairs": standardized_qa_pairs,
            "trajectories": standardized_trajectories,
            "knowledge_graph": knowledge_graph.to_dict(),
            "statistics": statistics,
            "metadata": {
                "generation_config": self.config,
                "total_entries": len(standardized_qa_pairs),
                "quality_threshold": self.min_quality_score
            }
        }
        
        logging.info(f"数据综合完成，最终数据集包含 {len(standardized_qa_pairs)} 个条目")
        return final_dataset
    
    def _pair_qa_and_trajectories(self, qa_pairs: List[ObfuscatedQuestionAnswer],
                                 trajectories: List[ReasoningTrajectory]) -> List[DatasetEntry]:
        """配对问答对和推理轨迹"""
        entries = []
        
        # 创建轨迹索引（基于问题匹配）
        trajectory_index = {}
        for trajectory in trajectories:
            question = trajectory.qa_pair.obfuscated_question
            trajectory_index[question] = trajectory
        
        # 配对
        for qa_pair in qa_pairs:
            question = qa_pair.obfuscated_question
            trajectory = trajectory_index.get(question)
            
            entry = DatasetEntry(qa_pair, trajectory)
            entries.append(entry)
        
        return entries
    
    def _apply_quality_filters(self, entries: List[DatasetEntry]) -> List[DatasetEntry]:
        """应用质量过滤"""
        filtered_entries = []
        
        for entry in entries:
            # 检查质量分数
            if entry.quality_score < self.min_quality_score:
                continue
            
            # 检查问题长度
            question_length = len(entry.qa_pair.obfuscated_question)
            min_length = self.quality_filters.get("min_question_length", 10)
            max_length = self.quality_filters.get("max_question_length", 200)
            
            if not (min_length <= question_length <= max_length):
                continue
            
            # 检查答案长度
            answer_length = len(entry.qa_pair.obfuscated_answer)
            min_answer_length = self.quality_filters.get("min_answer_length", 5)
            max_answer_length = self.quality_filters.get("max_answer_length", 500)
            
            if not (min_answer_length <= answer_length <= max_answer_length):
                continue
            
            filtered_entries.append(entry)
        
        logging.info(f"质量过滤完成：{len(entries)} -> {len(filtered_entries)}")
        return filtered_entries
    
    def _apply_question_paraphrasing(self, entries: List[DatasetEntry]) -> List[DatasetEntry]:
        """应用问题释义"""
        enhanced_entries = []
        
        for entry in entries:
            enhanced_entries.append(entry)
            
            # 为部分条目生成释义版本
            if random.random() < 0.3:  # 30%的概率生成释义
                paraphrased_question = self._paraphrase_question(entry.qa_pair.obfuscated_question)
                if paraphrased_question != entry.qa_pair.obfuscated_question:
                    # 创建新的问答对
                    paraphrased_qa = ObfuscatedQuestionAnswer(
                        original_qa=entry.qa_pair.original_qa,
                        obfuscated_question=paraphrased_question,
                        obfuscated_answer=entry.qa_pair.obfuscated_answer,
                        obfuscation_type=entry.qa_pair.obfuscation_type + "+paraphrase",
                        obfuscation_metadata=entry.qa_pair.obfuscation_metadata
                    )
                    
                    paraphrased_entry = DatasetEntry(
                        qa_pair=paraphrased_qa,
                        trajectory=entry.trajectory,
                        entry_id=entry.entry_id + "_paraphrase",
                        metadata={"is_paraphrase": True, "original_entry": entry.entry_id}
                    )
                    enhanced_entries.append(paraphrased_entry)
        
        logging.info(f"问题释义完成：{len(entries)} -> {len(enhanced_entries)}")
        return enhanced_entries
    
    def _generate_negative_samples(self, entries: List[DatasetEntry]) -> List[DatasetEntry]:
        """生成负样本"""
        negative_samples = []
        
        # 随机选择一些条目生成负样本
        sample_size = min(len(entries) // 10, 50)  # 最多生成10%的负样本
        selected_entries = random.sample(entries, sample_size)
        
        for entry in selected_entries:
            # 方法1: 错误答案配对
            wrong_answer = self._generate_wrong_answer(entry, entries)
            if wrong_answer:
                negative_qa = ObfuscatedQuestionAnswer(
                    original_qa=entry.qa_pair.original_qa,
                    obfuscated_question=entry.qa_pair.obfuscated_question,
                    obfuscated_answer=wrong_answer,
                    obfuscation_type=entry.qa_pair.obfuscation_type + "+negative",
                    obfuscation_metadata={"is_negative": True}
                )
                
                negative_entry = DatasetEntry(
                    qa_pair=negative_qa,
                    trajectory=None,  # 负样本不提供推理轨迹
                    entry_id=entry.entry_id + "_negative",
                    metadata={"is_negative": True, "original_entry": entry.entry_id}
                )
                negative_samples.append(negative_entry)
        
        logging.info(f"负样本生成完成：生成 {len(negative_samples)} 个负样本")
        return negative_samples
    
    def _add_context_variations(self, entries: List[DatasetEntry]) -> List[DatasetEntry]:
        """添加上下文变体"""
        enhanced_entries = []
        
        for entry in entries:
            enhanced_entries.append(entry)
            
            # 为部分条目添加上下文变体
            if random.random() < 0.2:  # 20%的概率添加变体
                context_variation = self._create_context_variation(entry)
                if context_variation:
                    enhanced_entries.append(context_variation)
        
        logging.info(f"上下文变体生成完成：{len(entries)} -> {len(enhanced_entries)}")
        return enhanced_entries
    
    def _paraphrase_question(self, question: str) -> str:
        """释义问题"""
        # 简单的释义策略
        paraphrase_patterns = [
            (r"什么是", "请问什么是"),
            (r"如何", "怎样"),
            (r"为什么", "为何"),
            (r"哪些", "有哪些"),
            (r"是否", "是不是"),
            (r"有什么", "具有什么"),
            (r"的特点", "的特征"),
            (r"的优势", "的优点")
        ]
        
        paraphrased = question
        for pattern, replacement in paraphrase_patterns:
            if random.random() < 0.5:  # 50%概率应用每个模式
                paraphrased = re.sub(pattern, replacement, paraphrased)
        
        return paraphrased
    
    def _generate_wrong_answer(self, entry: DatasetEntry, all_entries: List[DatasetEntry]) -> Optional[str]:
        """生成错误答案"""
        # 从其他条目中随机选择答案作为错误答案
        other_entries = [e for e in all_entries if e.entry_id != entry.entry_id]
        if other_entries:
            wrong_entry = random.choice(other_entries)
            return wrong_entry.qa_pair.obfuscated_answer
        
        return None
    
    def _create_context_variation(self, entry: DatasetEntry) -> Optional[DatasetEntry]:
        """创建上下文变体"""
        # 添加不同的上下文前缀
        context_prefixes = [
            "在TCL工业领域中，",
            "根据相关资料，",
            "从技术角度来看，",
            "在当前市场环境下，"
        ]
        
        prefix = random.choice(context_prefixes)
        varied_question = prefix + entry.qa_pair.obfuscated_question
        
        varied_qa = ObfuscatedQuestionAnswer(
            original_qa=entry.qa_pair.original_qa,
            obfuscated_question=varied_question,
            obfuscated_answer=entry.qa_pair.obfuscated_answer,
            obfuscation_type=entry.qa_pair.obfuscation_type + "+context_variation",
            obfuscation_metadata={"context_prefix": prefix}
        )
        
        varied_entry = DatasetEntry(
            qa_pair=varied_qa,
            trajectory=entry.trajectory,
            entry_id=entry.entry_id + "_context_var",
            metadata={"is_context_variation": True, "original_entry": entry.entry_id}
        )
        
        return varied_entry
    
    def _generate_statistics(self, qa_pairs: List[Dict[str, Any]], 
                           trajectories: List[Dict[str, Any]],
                           knowledge_graph: KnowledgeGraph,
                           subgraphs: List[Subgraph]) -> Dict[str, Any]:
        """生成统计信息"""
        stats = {
            "dataset_overview": {
                "total_qa_pairs": len(qa_pairs),
                "total_trajectories": len(trajectories),
                "total_entities": len(knowledge_graph.entities),
                "total_relations": len(knowledge_graph.relations),
                "total_subgraphs": len(subgraphs)
            },
            "question_type_distribution": self._analyze_question_types(qa_pairs),
            "obfuscation_type_distribution": self._analyze_obfuscation_types(qa_pairs),
            "subgraph_topology_distribution": self._analyze_subgraph_topologies(subgraphs),
            "quality_distribution": self._analyze_quality_distribution(qa_pairs),
            "complexity_distribution": self._analyze_complexity_distribution(qa_pairs),
            "trajectory_type_distribution": self._analyze_trajectory_types(trajectories),
            "entity_type_distribution": self._analyze_entity_types(knowledge_graph),
            "relation_type_distribution": self._analyze_relation_types(knowledge_graph)
        }
        
        return stats
    
    def _analyze_question_types(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析问题类型分布"""
        type_counts = Counter(qa["question_type"] for qa in qa_pairs)
        return dict(type_counts)
    
    def _analyze_obfuscation_types(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析模糊化类型分布"""
        type_counts = Counter(qa["obfuscation_type"] for qa in qa_pairs)
        return dict(type_counts)
    
    def _analyze_subgraph_topologies(self, subgraphs: List[Subgraph]) -> Dict[str, int]:
        """分析子图拓扑分布"""
        topology_counts = Counter(subgraph.topology_type for subgraph in subgraphs)
        return dict(topology_counts)
    
    def _analyze_quality_distribution(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析质量分布"""
        quality_ranges = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0
        }
        
        for qa in qa_pairs:
            quality_score = qa.get("quality_score", 0)
            if quality_score < 0.2:
                quality_ranges["0.0-0.2"] += 1
            elif quality_score < 0.4:
                quality_ranges["0.2-0.4"] += 1
            elif quality_score < 0.6:
                quality_ranges["0.4-0.6"] += 1
            elif quality_score < 0.8:
                quality_ranges["0.6-0.8"] += 1
            else:
                quality_ranges["0.8-1.0"] += 1
        
        return quality_ranges
    
    def _analyze_complexity_distribution(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析复杂度分布"""
        complexity_ranges = {
            "简单(1-2)": 0,
            "中等(2-3)": 0,
            "复杂(3-4)": 0,
            "很复杂(4-5)": 0
        }
        
        for qa in qa_pairs:
            features = qa.get("features", {}).get("qa_features", {})
            complexity = features.get("reasoning_difficulty", 1.0)
            
            if complexity < 2:
                complexity_ranges["简单(1-2)"] += 1
            elif complexity < 3:
                complexity_ranges["中等(2-3)"] += 1
            elif complexity < 4:
                complexity_ranges["复杂(3-4)"] += 1
            else:
                complexity_ranges["很复杂(4-5)"] += 1
        
        return complexity_ranges
    
    def _analyze_trajectory_types(self, trajectories: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析轨迹类型分布"""
        type_counts = Counter(traj["trajectory_type"] for traj in trajectories)
        return dict(type_counts)
    
    def _analyze_entity_types(self, knowledge_graph: KnowledgeGraph) -> Dict[str, int]:
        """分析实体类型分布"""
        type_counts = defaultdict(int)
        for entity in knowledge_graph.entities.values():
            entity_type = entity.entity_type
            type_counts[entity_type] += 1
        
        return dict(type_counts)
    
    def _analyze_relation_types(self, knowledge_graph: KnowledgeGraph) -> Dict[str, int]:
        """分析关系类型分布"""
        type_counts = defaultdict(int)
        for relation in knowledge_graph.relations:
            relation_type = relation.relation
            type_counts[relation_type] += 1
        
        return dict(type_counts)