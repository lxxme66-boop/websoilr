"""
问题生成器 - WebSailor核心思想实现
基于子图中节点与关系,设计 QA 问题
覆盖多种问题类型：事实查询、推理判断、多跳推理、比较分析等
"""

import logging
import random
import json
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

class QuestionGenerator:
    """
    问题生成器 - WebSailor的核心组件
    
    核心思想：
    1. 基于子图中的节点与关系设计多样化的QA问题
    2. 覆盖多种问题类型：
       - 事实查询（直接从图中获取答案）
       - 推理判断（需要多步推理）
       - 比较分析（涉及多个实体的比较）
       - 路径查找（寻找实体间的关系路径）
       - 属性查询（查询实体的特定属性）
    3. 确保问题的多样性和复杂度分布
    """
    
    def __init__(self, config: Dict[str, Any], templates: Dict[str, Any]):
        self.config = config
        self.templates = templates
        self.question_types = config['question_types']
        self.min_difficulty = config['min_difficulty']
        self.max_difficulty = config['max_difficulty']
        self.questions_per_subgraph = config['questions_per_subgraph']
        
        # TCL工业领域特定的问题模板
        self.tcl_templates = {
            'factual': [
                "{entity}的{attribute}是什么？",
                "哪个{entity_type}具有{attribute}特性？",
                "{entity}属于什么{category}？",
                "{entity}的主要{feature}有哪些？"
            ],
            'relational': [
                "{entity1}和{entity2}之间有什么关系？",
                "{entity1}如何{relation}{entity2}？",
                "什么{entity_type}能够{action}{target}？",
                "{entity1}对{entity2}产生了什么{effect}？"
            ],
            'multi_hop': [
                "从{start_entity}到{end_entity}需要经过哪些{intermediate_type}？",
                "{entity1}通过什么方式最终{relation}{entity2}？",
                "要实现{goal}，需要哪些{resource_type}的支持？",
                "{problem}的解决方案涉及哪些{component_type}？"
            ],
            'comparative': [
                "{entity1}和{entity2}在{aspect}方面有什么不同？",
                "相比{entity1}，{entity2}的优势是什么？",
                "哪个{entity_type}在{metric}方面表现更好？",
                "{category}中最{superlative}的是什么？"
            ],
            'reasoning': [
                "如果{condition}，那么{entity}会发生什么变化？",
                "为什么{entity1}会{relation}{entity2}？",
                "{phenomenon}的根本原因是什么？",
                "如何解决{problem}？"
            ],
            'path_finding': [
                "从{start}到{end}的最短路径是什么？",
                "{entity1}如何影响到{entity2}？",
                "{input}经过哪些步骤变成{output}？",
                "{cause}是如何导致{effect}的？"
            ]
        }
        
        # TCL工业领域词汇映射
        self.domain_vocab = {
            'entity_types': ['产品', '技术', '工艺', '材料', '设备', '质量指标', '性能参数'],
            'relations': ['包含', '依赖', '影响', '改进', '应用于', '导致', '解决', '优化'],
            'attributes': ['性能', '质量', '效率', '成本', '可靠性', '稳定性', '精度'],
            'actions': ['生产', '测试', '优化', '改进', '应用', '控制', '监测'],
            'categories': ['类型', '等级', '规格', '标准', '系列', '版本']
        }
    
    def generate_questions(self, subgraph_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        为给定子图生成多种类型的问题
        
        Args:
            subgraph_data: 子图数据，包含图结构和元数据
            
        Returns:
            List[Dict]: 生成的问题列表
        """
        subgraph = subgraph_data['subgraph']
        scenario_features = subgraph_data['scenario_features']
        
        logger.debug(f"为大小为 {len(subgraph.nodes())} 的子图生成问题")
        
        questions = []
        
        # 为每种问题类型生成问题
        for question_type in self.question_types:
            type_questions = self._generate_questions_by_type(
                subgraph, scenario_features, question_type
            )
            questions.extend(type_questions)
        
        # 确保问题数量符合配置
        if len(questions) > self.questions_per_subgraph:
            questions = random.sample(questions, self.questions_per_subgraph)
        elif len(questions) < self.questions_per_subgraph:
            # 如果问题不够，重复生成一些问题
            additional_needed = self.questions_per_subgraph - len(questions)
            for _ in range(additional_needed):
                question_type = random.choice(self.question_types)
                additional_questions = self._generate_questions_by_type(
                    subgraph, scenario_features, question_type
                )
                if additional_questions:
                    questions.append(random.choice(additional_questions))
        
        # 添加问题元数据
        for i, question in enumerate(questions):
            question.update({
                'question_id': f"{subgraph_data.get('strategy', 'unknown')}_{i}",
                'subgraph_size': len(subgraph.nodes()),
                'subgraph_edges': len(subgraph.edges()),
                'difficulty': self._estimate_difficulty(question, subgraph),
                'required_hops': self._count_required_hops(question, subgraph),
                'involved_entities': self._extract_involved_entities(question, subgraph)
            })
        
        logger.debug(f"成功生成 {len(questions)} 个问题")
        return questions
    
    def _generate_questions_by_type(self, subgraph: nx.Graph, scenario_features: Dict, 
                                   question_type: str) -> List[Dict[str, Any]]:
        """根据问题类型生成问题"""
        if question_type == 'factual':
            return self._generate_factual_questions(subgraph, scenario_features)
        elif question_type == 'relational':
            return self._generate_relational_questions(subgraph, scenario_features)
        elif question_type == 'multi_hop':
            return self._generate_multi_hop_questions(subgraph, scenario_features)
        elif question_type == 'comparative':
            return self._generate_comparative_questions(subgraph, scenario_features)
        elif question_type == 'reasoning':
            return self._generate_reasoning_questions(subgraph, scenario_features)
        elif question_type == 'path_finding':
            return self._generate_path_finding_questions(subgraph, scenario_features)
        else:
            return []
    
    def _generate_factual_questions(self, subgraph: nx.Graph, scenario_features: Dict) -> List[Dict[str, Any]]:
        """生成事实查询问题 - 直接从图中获取答案"""
        questions = []
        
        # 基于节点属性的问题
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            node_type = node_data.get('type', '实体')
            
            # 属性查询问题
            if 'attributes' in node_data:
                for attr_key, attr_value in node_data['attributes'].items():
                    template = random.choice(self.tcl_templates['factual'])
                    question_text = template.format(
                        entity=node,
                        attribute=attr_key,
                        entity_type=node_type,
                        category=random.choice(self.domain_vocab['categories']),
                        feature=random.choice(self.domain_vocab['attributes'])
                    )
                    
                    questions.append({
                        'question': question_text,
                        'answer': str(attr_value),
                        'type': 'factual',
                        'subtype': 'attribute_query',
                        'source_nodes': [node],
                        'source_edges': [],
                        'reasoning_steps': [f"查询{node}的{attr_key}属性"]
                    })
        
        # 基于节点类型的问题
        node_types = defaultdict(list)
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node].get('type', 'unknown')
            node_types[node_type].append(node)
        
        for node_type, nodes in node_types.items():
            if len(nodes) > 1:
                template = "有哪些{entity_type}？"
                question_text = template.format(entity_type=node_type)
                
                questions.append({
                    'question': question_text,
                    'answer': '、'.join(nodes),
                    'type': 'factual',
                    'subtype': 'enumeration',
                    'source_nodes': nodes,
                    'source_edges': [],
                    'reasoning_steps': [f"枚举所有{node_type}类型的实体"]
                })
        
        return questions[:3]  # 限制数量
    
    def _generate_relational_questions(self, subgraph: nx.Graph, scenario_features: Dict) -> List[Dict[str, Any]]:
        """生成关系查询问题 - 查询实体间的直接关系"""
        questions = []
        
        # 基于边的关系问题
        for source, target, edge_data in subgraph.edges(data=True):
            relation = edge_data.get('relation', '相关')
            
            template = random.choice(self.tcl_templates['relational'])
            question_text = template.format(
                entity1=source,
                entity2=target,
                relation=relation,
                entity_type=subgraph.nodes[source].get('type', '实体'),
                action=random.choice(self.domain_vocab['actions']),
                target=target,
                effect=random.choice(self.domain_vocab['attributes'])
            )
            
            questions.append({
                'question': question_text,
                'answer': f"{source}{relation}{target}",
                'type': 'relational',
                'subtype': 'direct_relation',
                'source_nodes': [source, target],
                'source_edges': [(source, target)],
                'reasoning_steps': [f"查询{source}和{target}之间的{relation}关系"]
            })
        
        # 基于度数的关系问题
        degrees = dict(subgraph.degree())
        high_degree_node = max(degrees, key=degrees.get)
        neighbors = list(subgraph.neighbors(high_degree_node))
        
        if neighbors:
            question_text = f"{high_degree_node}与哪些实体有关系？"
            answer = '、'.join(neighbors)
            
            questions.append({
                'question': question_text,
                'answer': answer,
                'type': 'relational',
                'subtype': 'neighbor_query',
                'source_nodes': [high_degree_node] + neighbors,
                'source_edges': [(high_degree_node, n) for n in neighbors],
                'reasoning_steps': [f"查询{high_degree_node}的所有邻居节点"]
            })
        
        return questions[:2]  # 限制数量
    
    def _generate_multi_hop_questions(self, subgraph: nx.Graph, scenario_features: Dict) -> List[Dict[str, Any]]:
        """生成多跳推理问题 - 需要通过多个关系进行推理"""
        questions = []
        
        # 利用隐含路径生成多跳问题
        implicit_paths = scenario_features.get('implicit_paths', [])
        
        for path in implicit_paths[:2]:  # 最多2个路径
            if len(path) >= 3:
                start_node = path[0]
                end_node = path[-1]
                intermediate_nodes = path[1:-1]
                
                template = random.choice(self.tcl_templates['multi_hop'])
                question_text = template.format(
                    start_entity=start_node,
                    end_entity=end_node,
                    intermediate_type=random.choice(self.domain_vocab['entity_types']),
                    entity1=start_node,
                    entity2=end_node,
                    relation=random.choice(self.domain_vocab['relations']),
                    goal=f"{start_node}到{end_node}的连接",
                    resource_type=random.choice(self.domain_vocab['entity_types']),
                    problem=f"{start_node}相关问题",
                    component_type=random.choice(self.domain_vocab['entity_types'])
                )
                
                # 构建推理步骤
                reasoning_steps = []
                for i in range(len(path) - 1):
                    edge_data = subgraph.get_edge_data(path[i], path[i+1])
                    relation = edge_data.get('relation', '相关') if edge_data else '相关'
                    reasoning_steps.append(f"{path[i]} {relation} {path[i+1]}")
                
                questions.append({
                    'question': question_text,
                    'answer': ' -> '.join(path),
                    'type': 'multi_hop',
                    'subtype': 'path_reasoning',
                    'source_nodes': path,
                    'source_edges': [(path[i], path[i+1]) for i in range(len(path)-1)],
                    'reasoning_steps': reasoning_steps
                })
        
        # 基于多跳关系生成问题
        multi_hop_relations = scenario_features.get('multi_hop_relations', [])
        for relation_chain in multi_hop_relations[:1]:
            # 解析关系链
            parts = relation_chain.split('->')
            if len(parts) >= 3:
                question_text = f"{parts[0]}如何间接影响{parts[-1]}？"
                answer = f"通过{' -> '.join(parts)}的关系链"
                
                questions.append({
                    'question': question_text,
                    'answer': answer,
                    'type': 'multi_hop',
                    'subtype': 'indirect_influence',
                    'source_nodes': [p.split('-')[0] for p in parts if '-' in p],
                    'source_edges': [],
                    'reasoning_steps': [f"分析关系链: {relation_chain}"]
                })
        
        return questions
    
    def _generate_comparative_questions(self, subgraph: nx.Graph, scenario_features: Dict) -> List[Dict[str, Any]]:
        """生成比较分析问题 - 涉及多个实体的比较"""
        questions = []
        
        # 基于同类型实体的比较
        node_types = defaultdict(list)
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node].get('type', 'unknown')
            node_types[node_type].append(node)
        
        for node_type, nodes in node_types.items():
            if len(nodes) >= 2:
                entity1, entity2 = random.sample(nodes, 2)
                
                template = random.choice(self.tcl_templates['comparative'])
                question_text = template.format(
                    entity1=entity1,
                    entity2=entity2,
                    aspect=random.choice(self.domain_vocab['attributes']),
                    entity_type=node_type,
                    metric=random.choice(self.domain_vocab['attributes']),
                    category=node_type,
                    superlative='重要的'
                )
                
                # 基于度数进行比较
                degree1 = subgraph.degree(entity1)
                degree2 = subgraph.degree(entity2)
                
                if degree1 > degree2:
                    answer = f"{entity1}在连接性方面优于{entity2}（连接数：{degree1} vs {degree2}）"
                elif degree2 > degree1:
                    answer = f"{entity2}在连接性方面优于{entity1}（连接数：{degree2} vs {degree1}）"
                else:
                    answer = f"{entity1}和{entity2}在连接性方面相当（连接数都为{degree1}）"
                
                questions.append({
                    'question': question_text,
                    'answer': answer,
                    'type': 'comparative',
                    'subtype': 'entity_comparison',
                    'source_nodes': [entity1, entity2],
                    'source_edges': [],
                    'reasoning_steps': [
                        f"比较{entity1}的连接数: {degree1}",
                        f"比较{entity2}的连接数: {degree2}",
                        "得出比较结论"
                    ]
                })
        
        return questions[:1]  # 限制数量
    
    def _generate_reasoning_questions(self, subgraph: nx.Graph, scenario_features: Dict) -> List[Dict[str, Any]]:
        """生成推理判断问题 - 需要逻辑推理"""
        questions = []
        
        # 基于因果关系的推理
        potential_targets = scenario_features.get('potential_targets', [])
        
        for target in potential_targets[:1]:
            neighbors = list(subgraph.neighbors(target))
            if neighbors:
                cause_entity = random.choice(neighbors)
                edge_data = subgraph.get_edge_data(cause_entity, target)
                relation = edge_data.get('relation', '影响') if edge_data else '影响'
                
                template = random.choice(self.tcl_templates['reasoning'])
                question_text = template.format(
                    condition=f"{cause_entity}发生变化",
                    entity=target,
                    entity1=cause_entity,
                    entity2=target,
                    relation=relation,
                    phenomenon=f"{target}的状态变化",
                    problem=f"{target}相关问题"
                )
                
                answer = f"由于{cause_entity}和{target}之间存在{relation}关系，{cause_entity}的变化会导致{target}发生相应变化"
                
                questions.append({
                    'question': question_text,
                    'answer': answer,
                    'type': 'reasoning',
                    'subtype': 'causal_reasoning',
                    'source_nodes': [cause_entity, target],
                    'source_edges': [(cause_entity, target)],
                    'reasoning_steps': [
                        f"识别{cause_entity}和{target}的{relation}关系",
                        f"推理{cause_entity}变化对{target}的影响",
                        "得出因果结论"
                    ]
                })
        
        return questions
    
    def _generate_path_finding_questions(self, subgraph: nx.Graph, scenario_features: Dict) -> List[Dict[str, Any]]:
        """生成路径查找问题 - 寻找实体间的关系路径"""
        questions = []
        
        # 利用隐含路径
        implicit_paths = scenario_features.get('implicit_paths', [])
        
        for path in implicit_paths[:1]:
            if len(path) >= 3:
                start_node = path[0]
                end_node = path[-1]
                
                template = random.choice(self.tcl_templates['path_finding'])
                question_text = template.format(
                    start=start_node,
                    end=end_node,
                    entity1=start_node,
                    entity2=end_node,
                    input=start_node,
                    output=end_node,
                    cause=start_node,
                    effect=end_node
                )
                
                # 构建路径描述
                path_description = []
                for i in range(len(path) - 1):
                    edge_data = subgraph.get_edge_data(path[i], path[i+1])
                    relation = edge_data.get('relation', '连接') if edge_data else '连接'
                    path_description.append(f"{path[i]} -{relation}-> {path[i+1]}")
                
                answer = "; ".join(path_description)
                
                questions.append({
                    'question': question_text,
                    'answer': answer,
                    'type': 'path_finding',
                    'subtype': 'shortest_path',
                    'source_nodes': path,
                    'source_edges': [(path[i], path[i+1]) for i in range(len(path)-1)],
                    'reasoning_steps': [
                        f"寻找从{start_node}到{end_node}的路径",
                        f"确定最短路径: {' -> '.join(path)}",
                        "描述路径中的每个关系"
                    ]
                })
        
        return questions
    
    def _estimate_difficulty(self, question: Dict[str, Any], subgraph: nx.Graph) -> int:
        """估算问题难度（1-5级）"""
        difficulty = 1
        
        # 基于问题类型调整难度
        type_difficulty = {
            'factual': 1,
            'relational': 2,
            'multi_hop': 4,
            'comparative': 3,
            'reasoning': 4,
            'path_finding': 3
        }
        difficulty = type_difficulty.get(question['type'], 1)
        
        # 基于涉及的节点数量调整
        involved_nodes = len(question.get('source_nodes', []))
        if involved_nodes > 3:
            difficulty += 1
        
        # 基于推理步骤数量调整
        reasoning_steps = len(question.get('reasoning_steps', []))
        if reasoning_steps > 2:
            difficulty += 1
        
        return min(difficulty, 5)
    
    def _count_required_hops(self, question: Dict[str, Any], subgraph: nx.Graph) -> int:
        """计算问题需要的推理跳数"""
        source_edges = question.get('source_edges', [])
        if not source_edges:
            return 0
        
        # 计算涉及的最长路径
        max_hops = 0
        source_nodes = question.get('source_nodes', [])
        
        for i in range(len(source_nodes)):
            for j in range(i+1, len(source_nodes)):
                if nx.has_path(subgraph, source_nodes[i], source_nodes[j]):
                    path_length = nx.shortest_path_length(subgraph, source_nodes[i], source_nodes[j])
                    max_hops = max(max_hops, path_length)
        
        return max_hops
    
    def _extract_involved_entities(self, question: Dict[str, Any], subgraph: nx.Graph) -> List[str]:
        """提取问题涉及的实体"""
        entities = set()
        
        # 从source_nodes提取
        source_nodes = question.get('source_nodes', [])
        entities.update(source_nodes)
        
        # 从问题文本中提取（简单的实体识别）
        question_text = question['question']
        for node in subgraph.nodes():
            if node in question_text:
                entities.add(node)
        
        return list(entities)