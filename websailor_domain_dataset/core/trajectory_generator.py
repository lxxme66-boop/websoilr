#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Generator - 推理轨迹生成器
为模糊化后的问答对生成详细的推理轨迹

该模块实现：
1. 逐步推理轨迹：生成从问题到答案的逐步推理过程
2. 图遍历路径：基于知识图谱的路径探索轨迹
3. 证据收集：收集支持答案的证据链
4. 假设验证：验证推理假设的过程轨迹
"""

import logging
import random
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional
from collections import deque

from .obfuscation_processor import ObfuscatedQuestionAnswer
from .knowledge_graph_builder import KnowledgeGraph


class ReasoningStep:
    """推理步骤类"""
    
    def __init__(self, step_id: int, step_type: str, description: str, 
                 entities: List[str] = None, relations: List[str] = None,
                 confidence: float = 1.0, evidence: str = None):
        """
        初始化推理步骤
        
        Args:
            step_id: 步骤ID
            step_type: 步骤类型
            description: 步骤描述
            entities: 涉及的实体
            relations: 涉及的关系
            confidence: 置信度
            evidence: 支持证据
        """
        self.step_id = step_id
        self.step_type = step_type
        self.description = description
        self.entities = entities or []
        self.relations = relations or []
        self.confidence = confidence
        self.evidence = evidence
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "description": self.description,
            "entities": self.entities,
            "relations": self.relations,
            "confidence": self.confidence,
            "evidence": self.evidence
        }


class ReasoningTrajectory:
    """推理轨迹类"""
    
    def __init__(self, qa_pair: ObfuscatedQuestionAnswer, trajectory_type: str):
        """
        初始化推理轨迹
        
        Args:
            qa_pair: 问答对
            trajectory_type: 轨迹类型
        """
        self.qa_pair = qa_pair
        self.trajectory_type = trajectory_type
        self.steps = []
        self.metadata = {}
        
        # 计算轨迹特征
        self._compute_features()
    
    def add_step(self, step: ReasoningStep):
        """添加推理步骤"""
        self.steps.append(step)
        self._compute_features()
    
    def _compute_features(self):
        """计算轨迹特征"""
        self.features = {
            "num_steps": len(self.steps),
            "avg_confidence": sum(step.confidence for step in self.steps) / len(self.steps) if self.steps else 0,
            "complexity_score": self._compute_complexity_score(),
            "reasoning_depth": self._compute_reasoning_depth(),
            "evidence_strength": self._compute_evidence_strength()
        }
    
    def _compute_complexity_score(self) -> float:
        """计算复杂度分数"""
        if not self.steps:
            return 0.0
        
        # 基于步骤数量、实体数量、关系数量计算复杂度
        num_steps = len(self.steps)
        total_entities = sum(len(step.entities) for step in self.steps)
        total_relations = sum(len(step.relations) for step in self.steps)
        
        complexity = (num_steps * 0.3 + total_entities * 0.4 + total_relations * 0.3)
        return min(complexity / 10.0, 5.0)  # 归一化到0-5
    
    def _compute_reasoning_depth(self) -> int:
        """计算推理深度"""
        # 推理深度等于步骤数量
        return len(self.steps)
    
    def _compute_evidence_strength(self) -> float:
        """计算证据强度"""
        if not self.steps:
            return 0.0
        
        # 基于有证据的步骤比例和平均置信度
        steps_with_evidence = sum(1 for step in self.steps if step.evidence)
        evidence_ratio = steps_with_evidence / len(self.steps)
        avg_confidence = self.features.get("avg_confidence", 0)
        
        return (evidence_ratio * 0.6 + avg_confidence * 0.4)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "question": self.qa_pair.obfuscated_question,
            "answer": self.qa_pair.obfuscated_answer,
            "trajectory_type": self.trajectory_type,
            "steps": [step.to_dict() for step in self.steps],
            "features": self.features,
            "metadata": self.metadata
        }


class TrajectoryGenerator:
    """推理轨迹生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化推理轨迹生成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.trajectory_config = config.get("trajectory_generation", {})
        
        # 加载轨迹类型
        self.trajectory_types = self.trajectory_config.get("trajectory_types", [])
        self.max_trajectory_length = self.trajectory_config.get("max_trajectory_length", 10)
        self.include_intermediate_steps = self.trajectory_config.get("include_intermediate_steps", True)
        self.include_confidence_scores = self.trajectory_config.get("include_confidence_scores", True)
        
        logging.info("推理轨迹生成器初始化完成")
    
    def generate_trajectories(self, qa_pairs: List[ObfuscatedQuestionAnswer], 
                            knowledge_graph: KnowledgeGraph) -> List[ReasoningTrajectory]:
        """
        为问答对生成推理轨迹
        
        Args:
            qa_pairs: 模糊化后的问答对列表
            knowledge_graph: 知识图谱
            
        Returns:
            推理轨迹列表
        """
        logging.info(f"开始为 {len(qa_pairs)} 个问答对生成推理轨迹...")
        
        trajectories = []
        
        for qa_pair in qa_pairs:
            try:
                # 为每个问答对生成轨迹
                trajectory = self._generate_single_trajectory(qa_pair, knowledge_graph)
                if trajectory:
                    trajectories.append(trajectory)
            except Exception as e:
                logging.warning(f"生成推理轨迹时出错: {e}")
        
        logging.info(f"推理轨迹生成完成，共生成 {len(trajectories)} 个轨迹")
        return trajectories
    
    def _generate_single_trajectory(self, qa_pair: ObfuscatedQuestionAnswer, 
                                   knowledge_graph: KnowledgeGraph) -> Optional[ReasoningTrajectory]:
        """为单个问答对生成推理轨迹"""
        # 根据问题类型选择合适的轨迹类型
        trajectory_type = self._select_trajectory_type(qa_pair)
        
        # 创建轨迹对象
        trajectory = ReasoningTrajectory(qa_pair, trajectory_type)
        
        # 根据轨迹类型生成步骤
        if trajectory_type == "step_by_step_reasoning":
            self._generate_step_by_step_trajectory(trajectory, knowledge_graph)
        elif trajectory_type == "graph_traversal_path":
            self._generate_graph_traversal_trajectory(trajectory, knowledge_graph)
        elif trajectory_type == "evidence_collection":
            self._generate_evidence_collection_trajectory(trajectory, knowledge_graph)
        elif trajectory_type == "hypothesis_validation":
            self._generate_hypothesis_validation_trajectory(trajectory, knowledge_graph)
        else:
            # 默认生成逐步推理轨迹
            self._generate_step_by_step_trajectory(trajectory, knowledge_graph)
        
        return trajectory if trajectory.steps else None
    
    def _select_trajectory_type(self, qa_pair: ObfuscatedQuestionAnswer) -> str:
        """根据问答对特征选择轨迹类型"""
        question_type = qa_pair.question_type
        subgraph_topology = qa_pair.subgraph.topology_type
        
        # 根据问题类型和子图拓扑选择合适的轨迹类型
        if question_type == "factual_single":
            return "step_by_step_reasoning"
        elif question_type == "factual_multi":
            return "graph_traversal_path"
        elif question_type == "comparative":
            return "evidence_collection"
        elif question_type == "aggregative":
            return "evidence_collection"
        elif question_type == "reasoning":
            return "hypothesis_validation"
        else:
            return random.choice(self.trajectory_types)
    
    def _generate_step_by_step_trajectory(self, trajectory: ReasoningTrajectory, 
                                         knowledge_graph: KnowledgeGraph):
        """生成逐步推理轨迹"""
        qa_pair = trajectory.qa_pair
        
        # 步骤1: 问题理解
        step1 = ReasoningStep(
            step_id=1,
            step_type="question_understanding",
            description=f"理解问题：{qa_pair.obfuscated_question}",
            confidence=0.9
        )
        trajectory.add_step(step1)
        
        # 步骤2: 实体识别
        entities_in_question = self._extract_entities_from_text(
            qa_pair.obfuscated_question, qa_pair.subgraph
        )
        step2 = ReasoningStep(
            step_id=2,
            step_type="entity_identification",
            description=f"识别问题中的关键实体：{', '.join(entities_in_question)}",
            entities=entities_in_question,
            confidence=0.8
        )
        trajectory.add_step(step2)
        
        # 步骤3: 知识检索
        relevant_knowledge = self._retrieve_relevant_knowledge(
            entities_in_question, knowledge_graph
        )
        step3 = ReasoningStep(
            step_id=3,
            step_type="knowledge_retrieval",
            description=f"检索相关知识：找到 {len(relevant_knowledge)} 条相关信息",
            entities=list(relevant_knowledge.keys()),
            confidence=0.7,
            evidence=str(relevant_knowledge)[:200] + "..."
        )
        trajectory.add_step(step3)
        
        # 步骤4: 推理过程
        reasoning_process = self._generate_reasoning_process(qa_pair, relevant_knowledge)
        step4 = ReasoningStep(
            step_id=4,
            step_type="reasoning_process",
            description=reasoning_process,
            confidence=0.8
        )
        trajectory.add_step(step4)
        
        # 步骤5: 答案生成
        step5 = ReasoningStep(
            step_id=5,
            step_type="answer_generation",
            description=f"生成答案：{qa_pair.obfuscated_answer}",
            confidence=0.9
        )
        trajectory.add_step(step5)
    
    def _generate_graph_traversal_trajectory(self, trajectory: ReasoningTrajectory, 
                                           knowledge_graph: KnowledgeGraph):
        """生成图遍历路径轨迹"""
        qa_pair = trajectory.qa_pair
        
        # 步骤1: 起始节点确定
        start_entities = self._extract_entities_from_text(
            qa_pair.obfuscated_question, qa_pair.subgraph
        )
        if not start_entities:
            return
        
        start_entity = start_entities[0]
        step1 = ReasoningStep(
            step_id=1,
            step_type="start_node_identification",
            description=f"确定起始节点：{start_entity}",
            entities=[start_entity],
            confidence=0.9
        )
        trajectory.add_step(step1)
        
        # 步骤2-N: 图遍历过程
        current_entity = start_entity
        visited_entities = {start_entity}
        step_id = 2
        
        # 在子图中进行有限的遍历
        max_hops = min(self.max_trajectory_length - 2, 5)
        
        for hop in range(max_hops):
            if current_entity not in qa_pair.subgraph.graph:
                break
            
            neighbors = list(qa_pair.subgraph.graph.neighbors(current_entity))
            unvisited_neighbors = [n for n in neighbors if n not in visited_entities]
            
            if not unvisited_neighbors:
                break
            
            next_entity = random.choice(unvisited_neighbors)
            edge_data = qa_pair.subgraph.graph.get_edge_data(current_entity, next_entity, {})
            relation = edge_data.get("relation", "相关")
            
            step = ReasoningStep(
                step_id=step_id,
                step_type="graph_traversal",
                description=f"从 {current_entity} 通过 {relation} 关系到达 {next_entity}",
                entities=[current_entity, next_entity],
                relations=[relation],
                confidence=0.7,
                evidence=f"图中存在边：{current_entity} --{relation}--> {next_entity}"
            )
            trajectory.add_step(step)
            
            visited_entities.add(next_entity)
            current_entity = next_entity
            step_id += 1
        
        # 最后一步: 答案推导
        final_step = ReasoningStep(
            step_id=step_id,
            step_type="answer_derivation",
            description=f"基于遍历路径推导答案：{qa_pair.obfuscated_answer}",
            entities=list(visited_entities),
            confidence=0.8
        )
        trajectory.add_step(final_step)
    
    def _generate_evidence_collection_trajectory(self, trajectory: ReasoningTrajectory, 
                                               knowledge_graph: KnowledgeGraph):
        """生成证据收集轨迹"""
        qa_pair = trajectory.qa_pair
        
        # 步骤1: 证据需求分析
        step1 = ReasoningStep(
            step_id=1,
            step_type="evidence_requirement_analysis",
            description="分析回答问题所需的证据类型",
            confidence=0.9
        )
        trajectory.add_step(step1)
        
        # 步骤2: 证据收集
        entities = self._extract_entities_from_text(qa_pair.obfuscated_question, qa_pair.subgraph)
        evidence_pieces = []
        
        for i, entity in enumerate(entities[:3]):  # 限制实体数量
            if entity in qa_pair.subgraph.graph:
                neighbors = list(qa_pair.subgraph.graph.neighbors(entity))
                for neighbor in neighbors[:2]:  # 限制邻居数量
                    edge_data = qa_pair.subgraph.graph.get_edge_data(entity, neighbor, {})
                    relation = edge_data.get("relation", "相关")
                    evidence = f"{entity} 与 {neighbor} 存在 {relation} 关系"
                    evidence_pieces.append(evidence)
                    
                    step = ReasoningStep(
                        step_id=len(trajectory.steps) + 1,
                        step_type="evidence_collection",
                        description=f"收集证据：{evidence}",
                        entities=[entity, neighbor],
                        relations=[relation],
                        confidence=0.8,
                        evidence=evidence
                    )
                    trajectory.add_step(step)
        
        # 步骤N: 证据综合
        final_step = ReasoningStep(
            step_id=len(trajectory.steps) + 1,
            step_type="evidence_synthesis",
            description=f"综合 {len(evidence_pieces)} 条证据得出答案：{qa_pair.obfuscated_answer}",
            confidence=0.8,
            evidence="; ".join(evidence_pieces[:3])
        )
        trajectory.add_step(final_step)
    
    def _generate_hypothesis_validation_trajectory(self, trajectory: ReasoningTrajectory, 
                                                 knowledge_graph: KnowledgeGraph):
        """生成假设验证轨迹"""
        qa_pair = trajectory.qa_pair
        
        # 步骤1: 假设提出
        hypothesis = self._generate_hypothesis(qa_pair)
        step1 = ReasoningStep(
            step_id=1,
            step_type="hypothesis_generation",
            description=f"提出假设：{hypothesis}",
            confidence=0.7
        )
        trajectory.add_step(step1)
        
        # 步骤2: 假设分解
        sub_hypotheses = self._decompose_hypothesis(hypothesis, qa_pair)
        step2 = ReasoningStep(
            step_id=2,
            step_type="hypothesis_decomposition",
            description=f"将假设分解为 {len(sub_hypotheses)} 个子假设",
            confidence=0.8
        )
        trajectory.add_step(step2)
        
        # 步骤3-N: 子假设验证
        for i, sub_hypothesis in enumerate(sub_hypotheses):
            validation_result = self._validate_sub_hypothesis(sub_hypothesis, qa_pair, knowledge_graph)
            step = ReasoningStep(
                step_id=len(trajectory.steps) + 1,
                step_type="sub_hypothesis_validation",
                description=f"验证子假设 {i+1}：{sub_hypothesis} - {validation_result['result']}",
                confidence=validation_result['confidence'],
                evidence=validation_result['evidence']
            )
            trajectory.add_step(step)
        
        # 最后一步: 最终验证
        final_step = ReasoningStep(
            step_id=len(trajectory.steps) + 1,
            step_type="final_validation",
            description=f"基于子假设验证结果，确认答案：{qa_pair.obfuscated_answer}",
            confidence=0.9
        )
        trajectory.add_step(final_step)
    
    def _extract_entities_from_text(self, text: str, subgraph) -> List[str]:
        """从文本中提取实体"""
        entities = []
        for node in subgraph.nodes:
            if node in text:
                entities.append(node)
        return entities
    
    def _retrieve_relevant_knowledge(self, entities: List[str], 
                                   knowledge_graph: KnowledgeGraph) -> Dict[str, List[str]]:
        """检索相关知识"""
        relevant_knowledge = {}
        
        for entity in entities:
            if entity in knowledge_graph.graph:
                neighbors = list(knowledge_graph.graph.neighbors(entity))
                relevant_knowledge[entity] = neighbors[:3]  # 限制数量
        
        return relevant_knowledge
    
    def _generate_reasoning_process(self, qa_pair: ObfuscatedQuestionAnswer, 
                                  relevant_knowledge: Dict[str, List[str]]) -> str:
        """生成推理过程描述"""
        if qa_pair.question_type == "factual_single":
            return "通过直接查询知识图谱中的关系信息来回答问题"
        elif qa_pair.question_type == "factual_multi":
            return "通过多跳推理，沿着知识图谱中的关系链找到答案"
        elif qa_pair.question_type == "comparative":
            return "通过比较不同实体的属性和关系来分析差异"
        elif qa_pair.question_type == "aggregative":
            return "通过聚合多个相关实体的信息来统计答案"
        elif qa_pair.question_type == "reasoning":
            return "通过逻辑推理和假设验证来得出结论"
        else:
            return "通过分析相关知识进行推理"
    
    def _generate_hypothesis(self, qa_pair: ObfuscatedQuestionAnswer) -> str:
        """生成假设"""
        if qa_pair.question_type == "reasoning":
            return f"假设：问题的答案是 {qa_pair.obfuscated_answer}"
        else:
            return f"假设：通过分析相关实体关系可以找到答案"
    
    def _decompose_hypothesis(self, hypothesis: str, qa_pair: ObfuscatedQuestionAnswer) -> List[str]:
        """分解假设"""
        sub_hypotheses = [
            "相关实体在知识图谱中存在",
            "实体间存在有意义的关系",
            "这些关系支持预期的答案"
        ]
        return sub_hypotheses
    
    def _validate_sub_hypothesis(self, sub_hypothesis: str, qa_pair: ObfuscatedQuestionAnswer, 
                               knowledge_graph: KnowledgeGraph) -> Dict[str, Any]:
        """验证子假设"""
        # 简化的验证逻辑
        entities = self._extract_entities_from_text(qa_pair.obfuscated_question, qa_pair.subgraph)
        
        if "实体在知识图谱中存在" in sub_hypothesis:
            exists_count = sum(1 for entity in entities if entity in knowledge_graph.graph)
            confidence = exists_count / len(entities) if entities else 0
            result = "通过" if confidence > 0.5 else "部分通过"
            evidence = f"{exists_count}/{len(entities)} 个实体在知识图谱中找到"
        
        elif "存在有意义的关系" in sub_hypothesis:
            relation_count = 0
            for entity in entities:
                if entity in qa_pair.subgraph.graph:
                    relation_count += len(list(qa_pair.subgraph.graph.neighbors(entity)))
            
            confidence = min(relation_count / 10.0, 1.0)  # 归一化
            result = "通过" if confidence > 0.3 else "部分通过"
            evidence = f"找到 {relation_count} 个相关关系"
        
        else:
            confidence = 0.7
            result = "通过"
            evidence = "基于领域知识验证"
        
        return {
            "result": result,
            "confidence": confidence,
            "evidence": evidence
        }