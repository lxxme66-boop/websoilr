"""
推理轨迹生成器
为问题和答案生成详细的推理过程，支持多跳推理和复杂逻辑链
"""

import logging
import random
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class TrajectoryGenerator:
    """
    推理轨迹生成器
    
    功能：
    1. 为每个QA对生成详细的推理轨迹
    2. 支持多种推理模式：演绎、归纳、类比
    3. 生成中间推理步骤和证据链
    4. 提供推理过程的可解释性
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trajectory_types = config['trajectory_types']
        self.max_steps = config['max_trajectory_steps']
        self.include_evidence = config['include_evidence']
        
        # TCL工业领域推理模板
        self.reasoning_templates = {
            'deductive': [
                "基于{premise}，可以推断{conclusion}",
                "由于{condition}，因此{result}",
                "根据{rule}，当{situation}时，{outcome}"
            ],
            'inductive': [
                "观察到{observations}，可以归纳出{pattern}",
                "通过分析{cases}，发现{generalization}",
                "基于{evidence}，可以得出{hypothesis}"
            ],
            'analogical': [
                "类似于{analogy_source}，{target}也{similarity}",
                "参照{reference_case}，{current_case}应该{prediction}",
                "借鉴{past_experience}，{current_situation}可以{solution}"
            ],
            'causal': [
                "{cause}导致{effect}",
                "由于{factor}的影响，{entity}发生{change}",
                "{condition}是{outcome}的直接原因"
            ]
        }
    
    def generate_trajectory(self, question: Dict[str, Any], subgraph: nx.Graph) -> Dict[str, Any]:
        """
        为问题生成推理轨迹
        
        Args:
            question: 问题数据
            subgraph: 相关子图
            
        Returns:
            Dict: 推理轨迹数据
        """
        logger.debug(f"为问题生成推理轨迹: {question['question']}")
        
        question_type = question['type']
        source_nodes = question.get('source_nodes', [])
        source_edges = question.get('source_edges', [])
        
        # 根据问题类型选择推理策略
        if question_type == 'factual':
            trajectory = self._generate_factual_trajectory(question, subgraph)
        elif question_type == 'relational':
            trajectory = self._generate_relational_trajectory(question, subgraph)
        elif question_type == 'multi_hop':
            trajectory = self._generate_multi_hop_trajectory(question, subgraph)
        elif question_type == 'comparative':
            trajectory = self._generate_comparative_trajectory(question, subgraph)
        elif question_type == 'reasoning':
            trajectory = self._generate_reasoning_trajectory(question, subgraph)
        elif question_type == 'path_finding':
            trajectory = self._generate_path_finding_trajectory(question, subgraph)
        else:
            trajectory = self._generate_default_trajectory(question, subgraph)
        
        # 添加轨迹元数据
        trajectory.update({
            'question_id': question.get('question_id', ''),
            'question_type': question_type,
            'total_steps': len(trajectory.get('steps', [])),
            'reasoning_complexity': self._calculate_complexity(trajectory),
            'confidence_score': self._calculate_confidence(trajectory, subgraph)
        })
        
        return trajectory
    
    def _generate_factual_trajectory(self, question: Dict[str, Any], subgraph: nx.Graph) -> Dict[str, Any]:
        """生成事实查询的推理轨迹"""
        steps = []
        evidence = []
        
        source_nodes = question.get('source_nodes', [])
        
        if source_nodes:
            target_node = source_nodes[0]
            node_data = subgraph.nodes.get(target_node, {})
            
            # 步骤1：识别目标实体
            steps.append({
                'step_number': 1,
                'action': 'identify_target',
                'description': f"识别目标实体: {target_node}",
                'reasoning': f"问题询问关于{target_node}的信息",
                'entities_involved': [target_node],
                'relations_used': []
            })
            
            # 步骤2：检索实体属性
            if 'attributes' in node_data:
                for attr_key, attr_value in node_data['attributes'].items():
                    steps.append({
                        'step_number': len(steps) + 1,
                        'action': 'retrieve_attribute',
                        'description': f"检索{target_node}的{attr_key}属性",
                        'reasoning': f"从知识图谱中查找{target_node}的{attr_key}信息",
                        'entities_involved': [target_node],
                        'relations_used': [],
                        'result': attr_value
                    })
                    
                    evidence.append({
                        'type': 'attribute',
                        'entity': target_node,
                        'attribute': attr_key,
                        'value': attr_value,
                        'source': 'knowledge_graph'
                    })
            
            # 步骤3：形成答案
            steps.append({
                'step_number': len(steps) + 1,
                'action': 'formulate_answer',
                'description': f"基于检索到的信息形成答案",
                'reasoning': f"整合{target_node}的相关属性信息",
                'entities_involved': [target_node],
                'relations_used': [],
                'result': question.get('answer', '')
            })
        
        return {
            'trajectory_type': 'factual_retrieval',
            'reasoning_mode': 'direct_lookup',
            'steps': steps,
            'evidence': evidence,
            'final_answer': question.get('answer', ''),
            'reasoning_chain': ' -> '.join([step['action'] for step in steps])
        }
    
    def _generate_relational_trajectory(self, question: Dict[str, Any], subgraph: nx.Graph) -> Dict[str, Any]:
        """生成关系查询的推理轨迹"""
        steps = []
        evidence = []
        
        source_nodes = question.get('source_nodes', [])
        source_edges = question.get('source_edges', [])
        
        if len(source_nodes) >= 2:
            entity1, entity2 = source_nodes[0], source_nodes[1]
            
            # 步骤1：识别相关实体
            steps.append({
                'step_number': 1,
                'action': 'identify_entities',
                'description': f"识别相关实体: {entity1} 和 {entity2}",
                'reasoning': f"问题询问{entity1}和{entity2}之间的关系",
                'entities_involved': [entity1, entity2],
                'relations_used': []
            })
            
            # 步骤2：查找直接关系
            if subgraph.has_edge(entity1, entity2):
                edge_data = subgraph.get_edge_data(entity1, entity2)
                relation = edge_data.get('relation', '相关')
                
                steps.append({
                    'step_number': 2,
                    'action': 'find_direct_relation',
                    'description': f"查找{entity1}和{entity2}的直接关系",
                    'reasoning': f"在知识图谱中发现{entity1}与{entity2}之间存在{relation}关系",
                    'entities_involved': [entity1, entity2],
                    'relations_used': [relation],
                    'result': relation
                })
                
                evidence.append({
                    'type': 'direct_relation',
                    'source': entity1,
                    'target': entity2,
                    'relation': relation,
                    'confidence': 1.0
                })
            
            # 步骤3：形成答案
            steps.append({
                'step_number': len(steps) + 1,
                'action': 'formulate_answer',
                'description': f"基于发现的关系形成答案",
                'reasoning': f"将找到的关系信息转换为自然语言答案",
                'entities_involved': [entity1, entity2],
                'relations_used': [],
                'result': question.get('answer', '')
            })
        
        return {
            'trajectory_type': 'relational_query',
            'reasoning_mode': 'relation_lookup',
            'steps': steps,
            'evidence': evidence,
            'final_answer': question.get('answer', ''),
            'reasoning_chain': ' -> '.join([step['action'] for step in steps])
        }
    
    def _generate_multi_hop_trajectory(self, question: Dict[str, Any], subgraph: nx.Graph) -> Dict[str, Any]:
        """生成多跳推理的推理轨迹"""
        steps = []
        evidence = []
        
        source_nodes = question.get('source_nodes', [])
        
        if len(source_nodes) >= 3:  # 多跳推理至少需要3个节点
            path = source_nodes
            
            # 步骤1：识别推理路径
            steps.append({
                'step_number': 1,
                'action': 'identify_reasoning_path',
                'description': f"识别从{path[0]}到{path[-1]}的推理路径",
                'reasoning': f"问题需要通过多步推理连接{path[0]}和{path[-1]}",
                'entities_involved': path,
                'relations_used': []
            })
            
            # 步骤2-N：逐步推理
            for i in range(len(path) - 1):
                current_entity = path[i]
                next_entity = path[i + 1]
                
                # 查找两个实体间的关系
                relation = '相关'
                if subgraph.has_edge(current_entity, next_entity):
                    edge_data = subgraph.get_edge_data(current_entity, next_entity)
                    relation = edge_data.get('relation', '相关')
                
                steps.append({
                    'step_number': len(steps) + 1,
                    'action': 'intermediate_reasoning',
                    'description': f"推理步骤{i+1}: {current_entity} -> {next_entity}",
                    'reasoning': f"基于{relation}关系，从{current_entity}推导到{next_entity}",
                    'entities_involved': [current_entity, next_entity],
                    'relations_used': [relation],
                    'intermediate_result': f"{current_entity} {relation} {next_entity}"
                })
                
                evidence.append({
                    'type': 'intermediate_step',
                    'step': i + 1,
                    'source': current_entity,
                    'target': next_entity,
                    'relation': relation,
                    'reasoning': f"中间推理步骤，连接{current_entity}和{next_entity}"
                })
            
            # 最后步骤：综合推理结果
            steps.append({
                'step_number': len(steps) + 1,
                'action': 'synthesize_result',
                'description': f"综合多步推理结果",
                'reasoning': f"将所有中间步骤组合，得出从{path[0]}到{path[-1]}的完整推理链",
                'entities_involved': path,
                'relations_used': [],
                'result': question.get('answer', '')
            })
        
        return {
            'trajectory_type': 'multi_hop_reasoning',
            'reasoning_mode': 'chain_inference',
            'steps': steps,
            'evidence': evidence,
            'final_answer': question.get('answer', ''),
            'reasoning_chain': ' -> '.join([step['action'] for step in steps]),
            'hop_count': len(source_nodes) - 1 if source_nodes else 0
        }
    
    def _generate_comparative_trajectory(self, question: Dict[str, Any], subgraph: nx.Graph) -> Dict[str, Any]:
        """生成比较分析的推理轨迹"""
        steps = []
        evidence = []
        
        source_nodes = question.get('source_nodes', [])
        
        if len(source_nodes) >= 2:
            entity1, entity2 = source_nodes[0], source_nodes[1]
            
            # 步骤1：识别比较对象
            steps.append({
                'step_number': 1,
                'action': 'identify_comparison_targets',
                'description': f"识别比较对象: {entity1} vs {entity2}",
                'reasoning': f"问题要求比较{entity1}和{entity2}",
                'entities_involved': [entity1, entity2],
                'relations_used': []
            })
            
            # 步骤2：收集比较维度
            comparison_dimensions = []
            
            # 基于度数比较
            degree1 = subgraph.degree(entity1)
            degree2 = subgraph.degree(entity2)
            
            steps.append({
                'step_number': 2,
                'action': 'collect_comparison_data',
                'description': f"收集比较数据",
                'reasoning': f"分析{entity1}和{entity2}的各项指标",
                'entities_involved': [entity1, entity2],
                'relations_used': [],
                'data': {
                    entity1: {'degree': degree1},
                    entity2: {'degree': degree2}
                }
            })
            
            comparison_dimensions.append({
                'dimension': 'connectivity',
                'entity1_value': degree1,
                'entity2_value': degree2,
                'comparison': 'higher' if degree1 > degree2 else 'lower' if degree1 < degree2 else 'equal'
            })
            
            evidence.extend([
                {
                    'type': 'comparison_data',
                    'entity': entity1,
                    'dimension': 'connectivity',
                    'value': degree1
                },
                {
                    'type': 'comparison_data',
                    'entity': entity2,
                    'dimension': 'connectivity',
                    'value': degree2
                }
            ])
            
            # 步骤3：执行比较分析
            steps.append({
                'step_number': 3,
                'action': 'perform_comparison',
                'description': f"执行比较分析",
                'reasoning': f"基于收集的数据比较{entity1}和{entity2}",
                'entities_involved': [entity1, entity2],
                'relations_used': [],
                'comparison_results': comparison_dimensions
            })
            
            # 步骤4：得出结论
            steps.append({
                'step_number': 4,
                'action': 'draw_conclusion',
                'description': f"得出比较结论",
                'reasoning': f"基于比较分析结果形成最终答案",
                'entities_involved': [entity1, entity2],
                'relations_used': [],
                'result': question.get('answer', '')
            })
        
        return {
            'trajectory_type': 'comparative_analysis',
            'reasoning_mode': 'comparison',
            'steps': steps,
            'evidence': evidence,
            'final_answer': question.get('answer', ''),
            'reasoning_chain': ' -> '.join([step['action'] for step in steps])
        }
    
    def _generate_reasoning_trajectory(self, question: Dict[str, Any], subgraph: nx.Graph) -> Dict[str, Any]:
        """生成推理判断的推理轨迹"""
        steps = []
        evidence = []
        
        source_nodes = question.get('source_nodes', [])
        
        if len(source_nodes) >= 2:
            cause_entity, effect_entity = source_nodes[0], source_nodes[1]
            
            # 步骤1：识别因果关系
            steps.append({
                'step_number': 1,
                'action': 'identify_causal_relationship',
                'description': f"识别{cause_entity}和{effect_entity}的因果关系",
                'reasoning': f"问题涉及{cause_entity}对{effect_entity}的影响",
                'entities_involved': [cause_entity, effect_entity],
                'relations_used': []
            })
            
            # 步骤2：分析因果机制
            relation = '影响'
            if subgraph.has_edge(cause_entity, effect_entity):
                edge_data = subgraph.get_edge_data(cause_entity, effect_entity)
                relation = edge_data.get('relation', '影响')
            
            steps.append({
                'step_number': 2,
                'action': 'analyze_causal_mechanism',
                'description': f"分析因果机制",
                'reasoning': f"{cause_entity}通过{relation}关系影响{effect_entity}",
                'entities_involved': [cause_entity, effect_entity],
                'relations_used': [relation],
                'mechanism': f"{cause_entity} -> {relation} -> {effect_entity}"
            })
            
            evidence.append({
                'type': 'causal_link',
                'cause': cause_entity,
                'effect': effect_entity,
                'mechanism': relation,
                'strength': 'strong' if relation in ['导致', '引起'] else 'moderate'
            })
            
            # 步骤3：推理结果
            steps.append({
                'step_number': 3,
                'action': 'infer_result',
                'description': f"推理影响结果",
                'reasoning': f"基于因果机制推断{cause_entity}变化对{effect_entity}的影响",
                'entities_involved': [cause_entity, effect_entity],
                'relations_used': [relation],
                'result': question.get('answer', '')
            })
        
        return {
            'trajectory_type': 'causal_reasoning',
            'reasoning_mode': 'causal_inference',
            'steps': steps,
            'evidence': evidence,
            'final_answer': question.get('answer', ''),
            'reasoning_chain': ' -> '.join([step['action'] for step in steps])
        }
    
    def _generate_path_finding_trajectory(self, question: Dict[str, Any], subgraph: nx.Graph) -> Dict[str, Any]:
        """生成路径查找的推理轨迹"""
        steps = []
        evidence = []
        
        source_nodes = question.get('source_nodes', [])
        
        if len(source_nodes) >= 2:
            start_node = source_nodes[0]
            end_node = source_nodes[-1]
            
            # 步骤1：确定起点和终点
            steps.append({
                'step_number': 1,
                'action': 'define_endpoints',
                'description': f"确定路径起点和终点",
                'reasoning': f"需要找到从{start_node}到{end_node}的路径",
                'entities_involved': [start_node, end_node],
                'relations_used': []
            })
            
            # 步骤2：搜索路径
            if nx.has_path(subgraph, start_node, end_node):
                path = nx.shortest_path(subgraph, start_node, end_node)
                
                steps.append({
                    'step_number': 2,
                    'action': 'search_path',
                    'description': f"搜索最短路径",
                    'reasoning': f"在知识图谱中搜索从{start_node}到{end_node}的最短路径",
                    'entities_involved': path,
                    'relations_used': [],
                    'path_found': path
                })
                
                # 步骤3：验证路径
                path_relations = []
                for i in range(len(path) - 1):
                    if subgraph.has_edge(path[i], path[i+1]):
                        edge_data = subgraph.get_edge_data(path[i], path[i+1])
                        relation = edge_data.get('relation', '连接')
                        path_relations.append(relation)
                
                steps.append({
                    'step_number': 3,
                    'action': 'verify_path',
                    'description': f"验证路径有效性",
                    'reasoning': f"确认路径中每个连接都有效",
                    'entities_involved': path,
                    'relations_used': path_relations,
                    'path_details': list(zip(path[:-1], path_relations, path[1:]))
                })
                
                for i, (source, relation, target) in enumerate(zip(path[:-1], path_relations, path[1:])):
                    evidence.append({
                        'type': 'path_segment',
                        'segment_number': i + 1,
                        'source': source,
                        'target': target,
                        'relation': relation
                    })
            
            # 步骤4：构建答案
            steps.append({
                'step_number': len(steps) + 1,
                'action': 'construct_answer',
                'description': f"构建路径描述答案",
                'reasoning': f"将找到的路径转换为自然语言描述",
                'entities_involved': source_nodes,
                'relations_used': [],
                'result': question.get('answer', '')
            })
        
        return {
            'trajectory_type': 'path_finding',
            'reasoning_mode': 'graph_traversal',
            'steps': steps,
            'evidence': evidence,
            'final_answer': question.get('answer', ''),
            'reasoning_chain': ' -> '.join([step['action'] for step in steps])
        }
    
    def _generate_default_trajectory(self, question: Dict[str, Any], subgraph: nx.Graph) -> Dict[str, Any]:
        """生成默认推理轨迹"""
        steps = [
            {
                'step_number': 1,
                'action': 'analyze_question',
                'description': f"分析问题类型和要求",
                'reasoning': f"理解问题的具体需求",
                'entities_involved': question.get('source_nodes', []),
                'relations_used': []
            },
            {
                'step_number': 2,
                'action': 'retrieve_information',
                'description': f"检索相关信息",
                'reasoning': f"从知识图谱中获取相关数据",
                'entities_involved': question.get('source_nodes', []),
                'relations_used': []
            },
            {
                'step_number': 3,
                'action': 'formulate_answer',
                'description': f"形成答案",
                'reasoning': f"基于检索到的信息构建答案",
                'entities_involved': question.get('source_nodes', []),
                'relations_used': [],
                'result': question.get('answer', '')
            }
        ]
        
        return {
            'trajectory_type': 'general_reasoning',
            'reasoning_mode': 'basic_inference',
            'steps': steps,
            'evidence': [],
            'final_answer': question.get('answer', ''),
            'reasoning_chain': ' -> '.join([step['action'] for step in steps])
        }
    
    def _calculate_complexity(self, trajectory: Dict[str, Any]) -> float:
        """计算推理复杂度"""
        steps = trajectory.get('steps', [])
        evidence = trajectory.get('evidence', [])
        
        # 基于步骤数量和证据数量计算复杂度
        step_complexity = len(steps) / self.max_steps
        evidence_complexity = len(evidence) / 10  # 假设最大证据数为10
        
        # 基于推理类型调整复杂度
        trajectory_type = trajectory.get('trajectory_type', 'general_reasoning')
        type_weights = {
            'factual_retrieval': 0.2,
            'relational_query': 0.4,
            'multi_hop_reasoning': 0.8,
            'comparative_analysis': 0.6,
            'causal_reasoning': 0.7,
            'path_finding': 0.5,
            'general_reasoning': 0.3
        }
        
        type_complexity = type_weights.get(trajectory_type, 0.3)
        
        # 综合复杂度
        total_complexity = (step_complexity + evidence_complexity + type_complexity) / 3
        return min(total_complexity, 1.0)
    
    def _calculate_confidence(self, trajectory: Dict[str, Any], subgraph: nx.Graph) -> float:
        """计算推理置信度"""
        steps = trajectory.get('steps', [])
        evidence = trajectory.get('evidence', [])
        
        # 基于证据数量和质量计算置信度
        evidence_score = min(len(evidence) / 5, 1.0)  # 假设5个证据为满分
        
        # 基于推理步骤的完整性
        step_score = 1.0 if len(steps) >= 3 else len(steps) / 3
        
        # 基于图结构支持度
        entities_involved = set()
        for step in steps:
            entities_involved.update(step.get('entities_involved', []))
        
        graph_support = len(entities_involved.intersection(set(subgraph.nodes()))) / max(len(entities_involved), 1)
        
        # 综合置信度
        confidence = (evidence_score + step_score + graph_support) / 3
        return min(confidence, 1.0)