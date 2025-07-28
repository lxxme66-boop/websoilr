"""
推理轨迹生成器
生成多样化的推理路径，包括正确路径和死胡同
"""

import random
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
from collections import deque


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: int
    content: str
    step_type: str  # observation, hypothesis, verification, conclusion
    confidence: float
    entities_involved: List[str]
    relations_used: List[str]


@dataclass
class Trajectory:
    """推理轨迹"""
    trajectory_id: str
    qa_pair: Any  # 关联的问答对
    reasoning_pattern: str  # deductive, inductive, abductive, analogical
    steps: List[ReasoningStep]
    is_correct: bool  # 是否是正确路径
    final_answer: str
    confidence_score: float
    alternative_paths: List[List[ReasoningStep]]  # 可选路径
    metadata: Dict[str, Any]


class TrajectoryGenerator:
    """
    推理轨迹生成器
    生成从问题到答案的推理路径
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.reasoning_patterns = config.get('reasoning_patterns', [])
        self.include_dead_ends = config.get('include_dead_ends', True)
        self.dead_end_ratio = config.get('dead_end_ratio', 0.15)
        self.max_trajectory_length = config.get('max_trajectory_length', 8)
        self.min_trajectory_length = config.get('min_trajectory_length', 3)
        self.include_confidence_scores = config.get('include_confidence_scores', True)
        
    def generate_trajectories(self, qa_pairs: List[Any], 
                            knowledge_graph: nx.Graph) -> List[Trajectory]:
        """
        为问答对生成推理轨迹
        
        Args:
            qa_pairs: 问答对列表（可能已经过模糊化）
            knowledge_graph: 知识图谱
            
        Returns:
            推理轨迹列表
        """
        self.logger.info(f"开始为{len(qa_pairs)}个问答对生成推理轨迹...")
        
        trajectories = []
        
        for qa in qa_pairs:
            # 根据问题类型选择推理模式
            pattern = self._select_reasoning_pattern(qa)
            
            # 生成主要轨迹
            main_trajectory = self._generate_trajectory(qa, knowledge_graph, pattern)
            
            # 生成替代路径
            alternative_paths = self._generate_alternative_paths(
                qa, knowledge_graph, pattern
            )
            main_trajectory.alternative_paths = alternative_paths
            
            trajectories.append(main_trajectory)
            
            # 根据配置生成死胡同
            if self.include_dead_ends and random.random() < self.dead_end_ratio:
                dead_end = self._generate_dead_end(qa, knowledge_graph, pattern)
                trajectories.append(dead_end)
        
        self.logger.info(f"生成了{len(trajectories)}条推理轨迹")
        return trajectories
    
    def _select_reasoning_pattern(self, qa: Any) -> str:
        """根据问题类型选择推理模式"""
        # 获取问题类型
        question_type = getattr(qa, 'question_type', 'factual')
        
        # 基于问题类型的推理模式偏好
        pattern_preferences = {
            'factual': ['deductive', 'analogical'],
            'reasoning': ['deductive', 'inductive'],
            'multi_hop': ['deductive', 'abductive'],
            'comparative': ['analogical', 'inductive']
        }
        
        preferred_patterns = pattern_preferences.get(question_type, ['deductive'])
        
        # 从配置的模式中选择
        available_patterns = [p['name'] for p in self.reasoning_patterns]
        valid_patterns = [p for p in preferred_patterns if p in available_patterns]
        
        if valid_patterns:
            # 根据权重选择
            pattern_weights = {p['name']: p['weight'] 
                             for p in self.reasoning_patterns 
                             if p['name'] in valid_patterns}
            
            return self._weighted_choice(pattern_weights)
        
        # 默认使用演绎推理
        return 'deductive'
    
    def _generate_trajectory(self, qa: Any, knowledge_graph: nx.Graph,
                           pattern: str) -> Trajectory:
        """生成推理轨迹"""
        # 获取推理模式配置
        pattern_config = next(
            (p for p in self.reasoning_patterns if p['name'] == pattern),
            None
        )
        
        if not pattern_config:
            pattern_config = {'steps': ['observation', 'hypothesis', 'conclusion']}
        
        # 生成推理步骤
        steps = []
        step_id = 0
        
        # 获取问题中的实体
        entities = getattr(qa, 'entities', [])
        
        for step_type in pattern_config['steps']:
            step = self._generate_step(
                step_id, step_type, qa, knowledge_graph, 
                entities, previous_steps=steps
            )
            steps.append(step)
            step_id += 1
        
        # 确保轨迹长度在范围内
        while len(steps) < self.min_trajectory_length:
            # 添加额外的推理步骤
            step = self._generate_step(
                step_id, 'verification', qa, knowledge_graph,
                entities, previous_steps=steps
            )
            steps.append(step)
            step_id += 1
        
        # 计算置信度
        confidence = self._calculate_trajectory_confidence(steps)
        
        # 获取最终答案
        final_answer = getattr(qa, 'obfuscated_answer', getattr(qa, 'answer', ''))
        
        return Trajectory(
            trajectory_id=f"traj_{id(qa)}_{pattern}",
            qa_pair=qa,
            reasoning_pattern=pattern,
            steps=steps,
            is_correct=True,
            final_answer=final_answer,
            confidence_score=confidence,
            alternative_paths=[],
            metadata={'pattern_config': pattern_config}
        )
    
    def _generate_step(self, step_id: int, step_type: str,
                      qa: Any, knowledge_graph: nx.Graph,
                      entities: List[str],
                      previous_steps: List[ReasoningStep]) -> ReasoningStep:
        """生成单个推理步骤"""
        # 根据步骤类型生成内容
        if step_type == 'observation':
            content = self._generate_observation(qa, entities, knowledge_graph)
        elif step_type == 'hypothesis':
            content = self._generate_hypothesis(qa, entities, previous_steps)
        elif step_type == 'verification':
            content = self._generate_verification(qa, entities, knowledge_graph, previous_steps)
        elif step_type == 'conclusion':
            content = self._generate_conclusion(qa, previous_steps)
        elif step_type == 'premise':
            content = self._generate_premise(qa, entities, knowledge_graph)
        elif step_type == 'rule_application':
            content = self._generate_rule_application(qa, entities, previous_steps)
        elif step_type == 'pattern_recognition':
            content = self._generate_pattern_recognition(qa, entities, previous_steps)
        elif step_type == 'generalization':
            content = self._generate_generalization(qa, previous_steps)
        elif step_type == 'source_case':
            content = self._generate_source_case(qa, entities, knowledge_graph)
        elif step_type == 'mapping':
            content = self._generate_mapping(qa, entities, previous_steps)
        elif step_type == 'target_application':
            content = self._generate_target_application(qa, previous_steps)
        else:
            content = f"执行{step_type}步骤"
        
        # 提取涉及的实体和关系
        entities_involved = self._extract_entities_from_content(content, entities)
        relations_used = self._extract_relations_from_content(content, knowledge_graph)
        
        # 计算步骤置信度
        confidence = self._calculate_step_confidence(step_type, len(previous_steps))
        
        return ReasoningStep(
            step_id=step_id,
            content=content,
            step_type=step_type,
            confidence=confidence,
            entities_involved=entities_involved,
            relations_used=relations_used
        )
    
    def _generate_observation(self, qa: Any, entities: List[str],
                            knowledge_graph: nx.Graph) -> str:
        """生成观察步骤"""
        question = getattr(qa, 'obfuscated_question', getattr(qa, 'question', ''))
        
        # 提取问题中的关键信息
        if entities:
            entity = random.choice(entities)
            if entity in knowledge_graph:
                # 获取实体的一些属性
                attrs = knowledge_graph.nodes[entity]
                if attrs:
                    attr_key = random.choice([k for k in attrs.keys() 
                                            if k not in ['id', 'type']])
                    return f"观察到：{entity}的{attr_key}是{attrs[attr_key]}，这是问题的关键信息。"
        
        return f"观察到问题询问关于{entities[0] if entities else '某个对象'}的信息。"
    
    def _generate_hypothesis(self, qa: Any, entities: List[str],
                           previous_steps: List[ReasoningStep]) -> str:
        """生成假设步骤"""
        # 基于前面的观察生成假设
        if previous_steps:
            prev_content = previous_steps[-1].content
            
            if entities and len(entities) >= 2:
                return f"基于观察，假设{entities[0]}和{entities[1]}之间存在某种关联。"
            elif entities:
                return f"假设{entities[0]}是解决问题的关键因素。"
        
        return "提出初步假设：问题的答案需要通过分析实体间的关系得出。"
    
    def _generate_verification(self, qa: Any, entities: List[str],
                             knowledge_graph: nx.Graph,
                             previous_steps: List[ReasoningStep]) -> str:
        """生成验证步骤"""
        # 验证假设
        if entities and len(entities) >= 2:
            entity1, entity2 = entities[0], entities[1]
            
            if entity1 in knowledge_graph and entity2 in knowledge_graph:
                # 检查是否有路径
                try:
                    path = nx.shortest_path(knowledge_graph, entity1, entity2)
                    if len(path) > 2:
                        intermediate = path[1]
                        return f"验证发现：{entity1}通过{intermediate}与{entity2}相连。"
                    else:
                        return f"验证确认：{entity1}和{entity2}直接相关。"
                except nx.NetworkXNoPath:
                    return f"验证发现：{entity1}和{entity2}之间没有直接联系，需要寻找其他路径。"
        
        return "通过知识图谱验证了假设的合理性。"
    
    def _generate_conclusion(self, qa: Any,
                           previous_steps: List[ReasoningStep]) -> str:
        """生成结论步骤"""
        # 基于前面的步骤生成结论
        answer = getattr(qa, 'obfuscated_answer', getattr(qa, 'answer', ''))
        
        # 提取答案的核心内容
        if len(answer) > 100:
            # 简化长答案
            return f"综合以上分析，得出结论：{answer[:50]}..."
        
        return f"因此，最终答案是：{answer}"
    
    def _generate_premise(self, qa: Any, entities: List[str],
                        knowledge_graph: nx.Graph) -> str:
        """生成前提（演绎推理）"""
        if entities and entities[0] in knowledge_graph:
            entity = entities[0]
            entity_type = knowledge_graph.nodes[entity].get('type', '实体')
            return f"前提：所有的{entity_type}都具有某些共同特征。{entity}是一个{entity_type}。"
        
        return "建立推理前提：根据已知信息和领域知识。"
    
    def _generate_rule_application(self, qa: Any, entities: List[str],
                                 previous_steps: List[ReasoningStep]) -> str:
        """应用规则（演绎推理）"""
        rules = [
            "如果A导致B，B导致C，那么A间接导致C",
            "如果某个条件满足，则相应的结果必然发生",
            "根据因果关系链，可以推导出最终影响"
        ]
        
        rule = random.choice(rules)
        return f"应用推理规则：{rule}。"
    
    def _generate_pattern_recognition(self, qa: Any, entities: List[str],
                                    previous_steps: List[ReasoningStep]) -> str:
        """识别模式（归纳推理）"""
        if len(entities) >= 3:
            return f"识别到模式：{entities[0]}、{entities[1]}和{entities[2]}都表现出相似的特征。"
        
        return "通过观察多个案例，识别出共同的模式。"
    
    def _generate_generalization(self, qa: Any,
                               previous_steps: List[ReasoningStep]) -> str:
        """生成概括（归纳推理）"""
        return "基于观察到的模式，可以概括出一般性规律。"
    
    def _generate_source_case(self, qa: Any, entities: List[str],
                            knowledge_graph: nx.Graph) -> str:
        """生成源案例（类比推理）"""
        if entities:
            entity = entities[0]
            # 寻找相似案例
            if entity in knowledge_graph:
                entity_type = knowledge_graph.nodes[entity].get('type', '')
                similar_nodes = [
                    n for n in knowledge_graph.nodes()
                    if n != entity and
                    knowledge_graph.nodes[n].get('type', '') == entity_type
                ]
                
                if similar_nodes:
                    similar = random.choice(similar_nodes)
                    return f"参考案例：{similar}的情况与当前问题相似。"
        
        return "找到一个相关的参考案例进行类比。"
    
    def _generate_mapping(self, qa: Any, entities: List[str],
                        previous_steps: List[ReasoningStep]) -> str:
        """生成映射（类比推理）"""
        return "建立源案例和目标问题之间的对应关系。"
    
    def _generate_target_application(self, qa: Any,
                                   previous_steps: List[ReasoningStep]) -> str:
        """应用到目标（类比推理）"""
        return "将类比得到的结论应用到当前问题。"
    
    def _generate_alternative_paths(self, qa: Any, knowledge_graph: nx.Graph,
                                  pattern: str) -> List[List[ReasoningStep]]:
        """生成替代推理路径"""
        alternative_paths = []
        
        # 生成1-2条替代路径
        num_alternatives = random.randint(1, 2)
        
        for i in range(num_alternatives):
            # 使用不同的推理模式或步骤顺序
            alt_pattern = self._get_alternative_pattern(pattern)
            alt_trajectory = self._generate_trajectory(qa, knowledge_graph, alt_pattern)
            alternative_paths.append(alt_trajectory.steps)
        
        return alternative_paths
    
    def _generate_dead_end(self, qa: Any, knowledge_graph: nx.Graph,
                         pattern: str) -> Trajectory:
        """生成死胡同轨迹"""
        # 生成一个看似合理但最终错误的推理路径
        trajectory = self._generate_trajectory(qa, knowledge_graph, pattern)
        
        # 修改为死胡同
        trajectory.is_correct = False
        
        # 修改最后的步骤
        if trajectory.steps:
            last_step = trajectory.steps[-1]
            last_step.content = "经过验证，这个推理路径存在逻辑错误，需要重新考虑。"
            last_step.confidence = 0.3
        
        # 错误的最终答案
        trajectory.final_answer = "无法得出确定的结论"
        trajectory.confidence_score *= 0.5
        
        return trajectory
    
    def _extract_entities_from_content(self, content: str,
                                     available_entities: List[str]) -> List[str]:
        """从内容中提取涉及的实体"""
        entities_involved = []
        
        for entity in available_entities:
            if entity in content:
                entities_involved.append(entity)
        
        return entities_involved
    
    def _extract_relations_from_content(self, content: str,
                                      knowledge_graph: nx.Graph) -> List[str]:
        """从内容中提取使用的关系"""
        relations = []
        
        # 简单的关键词匹配
        relation_keywords = [
            '导致', '影响', '包含', '属于', '依赖',
            '需要', '使用', '生产', '检测', '维护'
        ]
        
        for keyword in relation_keywords:
            if keyword in content:
                relations.append(keyword)
        
        return relations
    
    def _calculate_step_confidence(self, step_type: str,
                                 num_previous_steps: int) -> float:
        """计算步骤置信度"""
        # 基础置信度
        base_confidence = {
            'observation': 0.9,
            'premise': 0.85,
            'hypothesis': 0.7,
            'verification': 0.8,
            'conclusion': 0.9,
            'rule_application': 0.85,
            'pattern_recognition': 0.75,
            'generalization': 0.7,
            'source_case': 0.8,
            'mapping': 0.75,
            'target_application': 0.8
        }
        
        confidence = base_confidence.get(step_type, 0.7)
        
        # 随着步骤增加，置信度略微下降
        confidence *= (0.98 ** num_previous_steps)
        
        # 添加一些随机性
        confidence += random.uniform(-0.05, 0.05)
        
        return max(0.5, min(1.0, confidence))
    
    def _calculate_trajectory_confidence(self, steps: List[ReasoningStep]) -> float:
        """计算整体轨迹置信度"""
        if not steps:
            return 0.5
        
        # 计算所有步骤的平均置信度
        avg_confidence = sum(step.confidence for step in steps) / len(steps)
        
        # 考虑轨迹长度的影响
        length_factor = 1.0
        if len(steps) < self.min_trajectory_length:
            length_factor = 0.8
        elif len(steps) > self.max_trajectory_length:
            length_factor = 0.9
        
        return avg_confidence * length_factor
    
    def _weighted_choice(self, choices: Dict[str, float]) -> str:
        """根据权重随机选择"""
        total = sum(choices.values())
        r = random.uniform(0, total)
        
        cumulative = 0
        for choice, weight in choices.items():
            cumulative += weight
            if r <= cumulative:
                return choice
        
        return list(choices.keys())[0]
    
    def _get_alternative_pattern(self, original_pattern: str) -> str:
        """获取替代推理模式"""
        alternatives = {
            'deductive': ['inductive', 'abductive'],
            'inductive': ['deductive', 'analogical'],
            'abductive': ['deductive', 'inductive'],
            'analogical': ['inductive', 'deductive']
        }
        
        alt_patterns = alternatives.get(original_pattern, ['deductive'])
        return random.choice(alt_patterns)