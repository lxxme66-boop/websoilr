"""
推理轨迹生成器
生成从问题到答案的推理路径，包括正确和错误的推理示例
"""

import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import networkx as nx
import random
from collections import deque

from .obfuscation_processor import ObfuscatedQAPair


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_type: str  # entity_identification, relation_traversal, etc.
    description: str
    entities_involved: List[str]
    relations_involved: List[str]
    result: Any
    is_correct: bool = True
    

@dataclass
class ReasoningTrajectory:
    """推理轨迹"""
    qa_pair: ObfuscatedQAPair
    steps: List[ReasoningStep]
    reasoning_pattern: str  # deductive, inductive, etc.
    is_successful: bool
    final_answer: str
    metadata: Dict[str, Any]


class TrajectoryGenerator:
    """
    推理轨迹生成器
    为每个问答对生成详细的推理步骤
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.reasoning_patterns = config.get('reasoning_patterns', [])
        self.step_types = config.get('step_types', [])
        self.max_steps = config.get('max_steps', 10)
        self.include_negative = config.get('include_negative_examples', True)
        self.error_rate = config.get('error_injection_rate', 0.1)
        
    def generate_trajectories(
        self, 
        qa_pairs: List[ObfuscatedQAPair], 
        knowledge_graph: nx.Graph
    ) -> List[ReasoningTrajectory]:
        """
        为问答对生成推理轨迹
        """
        trajectories = []
        
        for qa_pair in qa_pairs:
            # 生成正确的推理轨迹
            correct_trajectory = self._generate_correct_trajectory(
                qa_pair, knowledge_graph
            )
            trajectories.append(correct_trajectory)
            
            # 生成错误的推理轨迹（用于对比学习）
            if self.include_negative and random.random() < 0.3:
                error_trajectory = self._generate_error_trajectory(
                    qa_pair, knowledge_graph
                )
                trajectories.append(error_trajectory)
                
        self.logger.info(f"Generated {len(trajectories)} reasoning trajectories")
        
        return trajectories
        
    def _generate_correct_trajectory(
        self, 
        qa_pair: ObfuscatedQAPair, 
        knowledge_graph: nx.Graph
    ) -> ReasoningTrajectory:
        """生成正确的推理轨迹"""
        steps = []
        
        # 根据问题类型选择推理模式
        reasoning_pattern = self._select_reasoning_pattern(qa_pair)
        
        # 根据证据路径生成推理步骤
        if qa_pair.question_type == 'single_hop':
            steps = self._generate_single_hop_steps(qa_pair, knowledge_graph)
        elif qa_pair.question_type == 'multi_hop':
            steps = self._generate_multi_hop_steps(qa_pair, knowledge_graph)
        elif qa_pair.question_type == 'comparison':
            steps = self._generate_comparison_steps(qa_pair, knowledge_graph)
        elif qa_pair.question_type == 'aggregation':
            steps = self._generate_aggregation_steps(qa_pair, knowledge_graph)
        elif qa_pair.question_type == 'constraint':
            steps = self._generate_constraint_steps(qa_pair, knowledge_graph)
            
        trajectory = ReasoningTrajectory(
            qa_pair=qa_pair,
            steps=steps,
            reasoning_pattern=reasoning_pattern,
            is_successful=True,
            final_answer=qa_pair.answer,
            metadata={
                'num_steps': len(steps),
                'question_type': qa_pair.question_type
            }
        )
        
        return trajectory
        
    def _generate_error_trajectory(
        self, 
        qa_pair: ObfuscatedQAPair, 
        knowledge_graph: nx.Graph
    ) -> ReasoningTrajectory:
        """生成包含错误的推理轨迹"""
        # 先生成正确的步骤
        correct_steps = self._generate_correct_trajectory(
            qa_pair, knowledge_graph
        ).steps
        
        # 随机注入错误
        error_steps = []
        error_injected = False
        
        for i, step in enumerate(correct_steps):
            if not error_injected and random.random() < self.error_rate:
                # 注入错误
                error_step = self._inject_error_into_step(
                    step, qa_pair, knowledge_graph
                )
                error_steps.append(error_step)
                error_injected = True
                
                # 后续步骤也可能受影响
                remaining_steps = self._generate_error_propagation(
                    error_step, correct_steps[i+1:], knowledge_graph
                )
                error_steps.extend(remaining_steps)
                break
            else:
                error_steps.append(step)
                
        # 生成错误答案
        wrong_answer = self._generate_wrong_answer(qa_pair, knowledge_graph)
        
        trajectory = ReasoningTrajectory(
            qa_pair=qa_pair,
            steps=error_steps,
            reasoning_pattern=self._select_reasoning_pattern(qa_pair),
            is_successful=False,
            final_answer=wrong_answer,
            metadata={
                'num_steps': len(error_steps),
                'error_type': 'reasoning_error',
                'correct_answer': qa_pair.answer
            }
        )
        
        return trajectory
        
    def _select_reasoning_pattern(self, qa_pair: ObfuscatedQAPair) -> str:
        """根据问题类型选择推理模式"""
        type_to_pattern = {
            'single_hop': 'deductive',
            'multi_hop': 'deductive',
            'comparison': 'analogical',
            'aggregation': 'inductive',
            'constraint': 'abductive'
        }
        
        return type_to_pattern.get(qa_pair.question_type, 'deductive')
        
    def _generate_single_hop_steps(
        self, 
        qa_pair: ObfuscatedQAPair, 
        knowledge_graph: nx.Graph
    ) -> List[ReasoningStep]:
        """生成单跳推理步骤"""
        steps = []
        
        # 步骤1：识别问题中的实体
        if qa_pair.evidence_path:
            source, relation, target = qa_pair.evidence_path[0]
            
            steps.append(ReasoningStep(
                step_type='entity_identification',
                description=f"识别问题中的关键实体：{source}",
                entities_involved=[source],
                relations_involved=[],
                result=source,
                is_correct=True
            ))
            
            # 步骤2：识别关系类型
            steps.append(ReasoningStep(
                step_type='relation_identification',
                description=f"确定查询的关系类型：{relation}",
                entities_involved=[source],
                relations_involved=[relation],
                result=relation,
                is_correct=True
            ))
            
            # 步骤3：遍历关系找到答案
            steps.append(ReasoningStep(
                step_type='relation_traversal',
                description=f"通过{relation}关系从{source}找到{target}",
                entities_involved=[source, target],
                relations_involved=[relation],
                result=target,
                is_correct=True
            ))
            
        return steps
        
    def _generate_multi_hop_steps(
        self, 
        qa_pair: ObfuscatedQAPair, 
        knowledge_graph: nx.Graph
    ) -> List[ReasoningStep]:
        """生成多跳推理步骤"""
        steps = []
        
        # 步骤1：识别起始实体
        if qa_pair.evidence_path:
            first_hop = qa_pair.evidence_path[0]
            steps.append(ReasoningStep(
                step_type='entity_identification',
                description=f"识别起始实体：{first_hop[0]}",
                entities_involved=[first_hop[0]],
                relations_involved=[],
                result=first_hop[0],
                is_correct=True
            ))
            
            # 为每一跳生成步骤
            current_entity = first_hop[0]
            for i, (source, relation, target) in enumerate(qa_pair.evidence_path):
                # 识别下一步关系
                steps.append(ReasoningStep(
                    step_type='relation_identification',
                    description=f"第{i+1}跳：识别{relation}关系",
                    entities_involved=[current_entity],
                    relations_involved=[relation],
                    result=relation,
                    is_correct=True
                ))
                
                # 遍历关系
                steps.append(ReasoningStep(
                    step_type='relation_traversal',
                    description=f"通过{relation}从{source}到达{target}",
                    entities_involved=[source, target],
                    relations_involved=[relation],
                    result=target,
                    is_correct=True
                ))
                
                current_entity = target
                
            # 最终答案确认
            steps.append(ReasoningStep(
                step_type='answer_verification',
                description=f"确认最终答案：{current_entity}",
                entities_involved=[current_entity],
                relations_involved=[],
                result=current_entity,
                is_correct=True
            ))
            
        return steps
        
    def _generate_comparison_steps(
        self, 
        qa_pair: ObfuscatedQAPair, 
        knowledge_graph: nx.Graph
    ) -> List[ReasoningStep]:
        """生成比较推理步骤"""
        steps = []
        
        # 从元数据中获取比较的实体
        entities = qa_pair.metadata.get('entities', [])
        if len(entities) >= 2:
            entity1, entity2 = entities[0], entities[1]
            
            # 步骤1：识别比较对象
            steps.append(ReasoningStep(
                step_type='entity_identification',
                description=f"识别比较对象：{entity1} 和 {entity2}",
                entities_involved=[entity1, entity2],
                relations_involved=[],
                result=[entity1, entity2],
                is_correct=True
            ))
            
            # 步骤2：提取第一个实体的属性
            steps.append(ReasoningStep(
                step_type='attribute_extraction',
                description=f"提取{entity1}的属性",
                entities_involved=[entity1],
                relations_involved=['has_attribute'],
                result={'entity': entity1, 'attributes': []},
                is_correct=True
            ))
            
            # 步骤3：提取第二个实体的属性
            steps.append(ReasoningStep(
                step_type='attribute_extraction',
                description=f"提取{entity2}的属性",
                entities_involved=[entity2],
                relations_involved=['has_attribute'],
                result={'entity': entity2, 'attributes': []},
                is_correct=True
            ))
            
            # 步骤4：比较分析
            steps.append(ReasoningStep(
                step_type='comparison_analysis',
                description=f"比较{entity1}和{entity2}的异同",
                entities_involved=[entity1, entity2],
                relations_involved=[],
                result=qa_pair.answer,
                is_correct=True
            ))
            
        return steps
        
    def _generate_aggregation_steps(
        self, 
        qa_pair: ObfuscatedQAPair, 
        knowledge_graph: nx.Graph
    ) -> List[ReasoningStep]:
        """生成聚合推理步骤"""
        steps = []
        
        # 步骤1：识别聚合目标
        steps.append(ReasoningStep(
            step_type='aggregation_target_identification',
            description="识别需要聚合的目标类型",
            entities_involved=[],
            relations_involved=[],
            result='aggregation_target',
            is_correct=True
        ))
        
        # 步骤2：收集相关实体
        steps.append(ReasoningStep(
            step_type='entity_collection',
            description="收集满足条件的所有实体",
            entities_involved=[],
            relations_involved=[],
            result=[],
            is_correct=True
        ))
        
        # 步骤3：执行聚合计算
        aggregation_type = qa_pair.metadata.get('aggregation_type', 'count')
        if aggregation_type == 'count':
            steps.append(ReasoningStep(
                step_type='aggregation_computation',
                description="计算实体数量",
                entities_involved=[],
                relations_involved=[],
                result=qa_pair.answer,
                is_correct=True
            ))
        elif aggregation_type == 'list':
            steps.append(ReasoningStep(
                step_type='aggregation_computation',
                description="生成实体列表",
                entities_involved=[],
                relations_involved=[],
                result=qa_pair.answer,
                is_correct=True
            ))
            
        return steps
        
    def _generate_constraint_steps(
        self, 
        qa_pair: ObfuscatedQAPair, 
        knowledge_graph: nx.Graph
    ) -> List[ReasoningStep]:
        """生成约束推理步骤"""
        steps = []
        
        constraints = qa_pair.metadata.get('constraints', [])
        
        # 步骤1：识别约束条件
        steps.append(ReasoningStep(
            step_type='constraint_identification',
            description=f"识别{len(constraints)}个约束条件",
            entities_involved=[],
            relations_involved=[],
            result=constraints,
            is_correct=True
        ))
        
        # 步骤2：为每个约束条件查找候选
        for i, constraint in enumerate(constraints):
            steps.append(ReasoningStep(
                step_type='constraint_checking',
                description=f"检查约束{i+1}: {constraint}",
                entities_involved=[],
                relations_involved=[constraint.get('type', '')],
                result=[],
                is_correct=True
            ))
            
        # 步骤3：找出满足所有约束的实体
        steps.append(ReasoningStep(
            step_type='constraint_intersection',
            description="找出满足所有约束条件的实体",
            entities_involved=[],
            relations_involved=[],
            result=qa_pair.answer,
            is_correct=True
        ))
        
        return steps
        
    def _inject_error_into_step(
        self, 
        step: ReasoningStep, 
        qa_pair: ObfuscatedQAPair, 
        knowledge_graph: nx.Graph
    ) -> ReasoningStep:
        """向推理步骤注入错误"""
        error_types = [
            'wrong_entity',
            'wrong_relation',
            'wrong_direction',
            'incomplete_traversal'
        ]
        
        error_type = random.choice(error_types)
        
        if error_type == 'wrong_entity':
            # 选择错误的实体
            all_entities = list(knowledge_graph.nodes())
            wrong_entity = random.choice(all_entities)
            
            return ReasoningStep(
                step_type=step.step_type,
                description=f"{step.description} [错误：选择了{wrong_entity}]",
                entities_involved=[wrong_entity],
                relations_involved=step.relations_involved,
                result=wrong_entity,
                is_correct=False
            )
            
        elif error_type == 'wrong_relation':
            # 选择错误的关系
            wrong_relation = random.choice(self.config.get('relation_types', []))
            
            return ReasoningStep(
                step_type=step.step_type,
                description=f"{step.description} [错误：使用了{wrong_relation}关系]",
                entities_involved=step.entities_involved,
                relations_involved=[wrong_relation],
                result=None,
                is_correct=False
            )
            
        else:
            # 其他错误类型
            return ReasoningStep(
                step_type=step.step_type,
                description=f"{step.description} [推理错误]",
                entities_involved=step.entities_involved,
                relations_involved=step.relations_involved,
                result=None,
                is_correct=False
            )
            
    def _generate_error_propagation(
        self, 
        error_step: ReasoningStep, 
        remaining_steps: List[ReasoningStep], 
        knowledge_graph: nx.Graph
    ) -> List[ReasoningStep]:
        """生成错误传播后的步骤"""
        # 简化实现：后续步骤都标记为错误
        propagated_steps = []
        
        for step in remaining_steps:
            propagated_steps.append(ReasoningStep(
                step_type=step.step_type,
                description=f"{step.description} [受前序错误影响]",
                entities_involved=step.entities_involved,
                relations_involved=step.relations_involved,
                result=None,
                is_correct=False
            ))
            
        return propagated_steps
        
    def _generate_wrong_answer(
        self, 
        qa_pair: ObfuscatedQAPair, 
        knowledge_graph: nx.Graph
    ) -> str:
        """生成错误答案"""
        # 根据问题类型生成合理的错误答案
        if qa_pair.question_type == 'single_hop':
            # 选择一个相似类型的实体
            correct_answer = qa_pair.answer
            if knowledge_graph.has_node(correct_answer):
                correct_type = knowledge_graph.nodes[correct_answer].get('type')
                candidates = [
                    node for node in knowledge_graph.nodes()
                    if knowledge_graph.nodes[node].get('type') == correct_type
                    and node != correct_answer
                ]
                if candidates:
                    return random.choice(candidates)
                    
        elif qa_pair.question_type in ['aggregation', 'count']:
            # 生成错误的数字
            try:
                correct_num = int(qa_pair.answer)
                wrong_num = correct_num + random.randint(-3, 3)
                if wrong_num < 0:
                    wrong_num = 0
                return str(wrong_num)
            except:
                pass
                
        # 默认返回"未知"
        return "未知"