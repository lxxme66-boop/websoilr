"""
问题生成器 - WebSailor核心模块
基于子图中节点与关系，设计QA问题
覆盖多种问题类型：单跳、多跳、比较、聚合、约束等
"""

import random
import logging
from typing import List, Dict, Tuple, Any, Set, Optional
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import re
from string import Template

from .subgraph_sampler import Subgraph


@dataclass
class QAPair:
    """问答对数据结构"""
    question: str
    answer: str
    question_type: str
    subgraph: Subgraph
    evidence_path: List[Tuple[str, str, str]]  # (source, relation, target)
    difficulty: float
    metadata: Dict[str, Any]


class QuestionGenerator:
    """
    WebSailor问题生成器
    基于子图生成多样化的问题
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging
        self.question_types = config.get('question_types', [])
        self.answer_config = config.get('answer_generation', {})
        
        # 预编译模板
        self._compile_templates()
        
    def _compile_templates(self):
        """预编译问题模板"""
        for qt in self.question_types:
            qt['compiled_templates'] = [
                Template(template) for template in qt.get('templates', [])
            ]
            
    def generate_from_subgraphs(self, subgraphs: List[Subgraph]) -> List[QAPair]:
        """
        从子图集合生成问答对
        这是WebSailor的核心功能之一
        """
        all_qa_pairs = []
        
        for subgraph in subgraphs:
            # 根据子图拓扑类型选择合适的问题类型
            suitable_question_types = self._select_suitable_question_types(subgraph)
            
            for question_type in suitable_question_types:
                qa_pairs = self._generate_questions_by_type(
                    subgraph, 
                    question_type
                )
                all_qa_pairs.extend(qa_pairs)
                
        # 去重和质量过滤
        all_qa_pairs = self._deduplicate_qa_pairs(all_qa_pairs)
        all_qa_pairs = self._filter_quality(all_qa_pairs)
        
        self.logger.info(f"Generated {len(all_qa_pairs)} QA pairs from {len(subgraphs)} subgraphs")
        
        return all_qa_pairs
        
    def _select_suitable_question_types(self, subgraph: Subgraph) -> List[Dict]:
        """根据子图特征选择合适的问题类型"""
        suitable_types = []
        
        # 根据拓扑类型和图特征选择
        graph = subgraph.graph
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        for qt in self.question_types:
            qt_type = qt['type']
            
            # 单跳问题适合所有拓扑
            if qt_type == 'single_hop' and num_edges >= 1:
                suitable_types.append(qt)
                
            # 多跳问题需要路径存在
            elif qt_type == 'multi_hop' and num_edges >= 2:
                # 检查是否存在多跳路径
                if self._has_multi_hop_paths(graph):
                    suitable_types.append(qt)
                    
            # 比较问题需要多个相似实体
            elif qt_type == 'comparison' and num_nodes >= 3:
                if self._has_comparable_entities(graph):
                    suitable_types.append(qt)
                    
            # 聚合问题适合星型和网状拓扑
            elif qt_type == 'aggregation':
                if subgraph.topology_type in ['star', 'mesh'] and num_nodes >= 4:
                    suitable_types.append(qt)
                    
            # 约束问题需要多个属性
            elif qt_type == 'constraint':
                if self._has_multiple_attributes(graph) and num_nodes >= 3:
                    suitable_types.append(qt)
                    
        return suitable_types
        
    def _generate_questions_by_type(
        self, 
        subgraph: Subgraph, 
        question_type: Dict
    ) -> List[QAPair]:
        """根据问题类型生成具体问题"""
        qt_type = question_type['type']
        
        if qt_type == 'single_hop':
            return self._generate_single_hop_questions(subgraph, question_type)
        elif qt_type == 'multi_hop':
            return self._generate_multi_hop_questions(subgraph, question_type)
        elif qt_type == 'comparison':
            return self._generate_comparison_questions(subgraph, question_type)
        elif qt_type == 'aggregation':
            return self._generate_aggregation_questions(subgraph, question_type)
        elif qt_type == 'constraint':
            return self._generate_constraint_questions(subgraph, question_type)
        else:
            self.logger.warning(f"Unknown question type: {qt_type}")
            return []
            
    def _generate_single_hop_questions(
        self, 
        subgraph: Subgraph, 
        question_type: Dict
    ) -> List[QAPair]:
        """生成单跳问题"""
        qa_pairs = []
        graph = subgraph.graph
        templates = question_type.get('compiled_templates', [])
        
        # 遍历所有边
        for source, target, data in graph.edges(data=True):
            relation = data.get('relation', 'related_to')
            
            # 尝试每个模板
            for template in templates:
                try:
                    # 正向问题
                    question = self._fill_template(
                        template,
                        entity=source,
                        relation=relation
                    )
                    
                    answer = target
                    evidence_path = [(source, relation, target)]
                    
                    qa_pairs.append(QAPair(
                        question=question,
                        answer=answer,
                        question_type='single_hop',
                        subgraph=subgraph,
                        evidence_path=evidence_path,
                        difficulty=0.2,
                        metadata={
                            'hop_count': 1,
                            'direction': 'forward'
                        }
                    ))
                    
                    # 反向问题（如果适用）
                    if self._is_reversible_relation(relation):
                        reverse_question = self._generate_reverse_question(
                            template, target, relation
                        )
                        if reverse_question:
                            qa_pairs.append(QAPair(
                                question=reverse_question,
                                answer=source,
                                question_type='single_hop',
                                subgraph=subgraph,
                                evidence_path=[(target, f"reverse_{relation}", source)],
                                difficulty=0.3,
                                metadata={
                                    'hop_count': 1,
                                    'direction': 'reverse'
                                }
                            ))
                except Exception as e:
                    self.logger.debug(f"Failed to generate single-hop question: {e}")
                    
        return qa_pairs[:10]  # 限制每个子图的问题数量
        
    def _generate_multi_hop_questions(
        self, 
        subgraph: Subgraph, 
        question_type: Dict
    ) -> List[QAPair]:
        """生成多跳问题"""
        qa_pairs = []
        graph = subgraph.graph
        templates = question_type.get('compiled_templates', [])
        
        # 找出所有2-4跳的路径
        paths = self._find_multi_hop_paths(graph, min_hops=2, max_hops=4)
        
        for path in paths[:20]:  # 限制路径数量
            if len(path) < 3:  # 至少需要3个节点（2跳）
                continue
                
            # 构建问题
            start_entity = path[0]
            end_entity = path[-1]
            
            # 提取路径中的关系
            relations = []
            evidence_path = []
            
            for i in range(len(path) - 1):
                if graph.has_edge(path[i], path[i+1]):
                    edge_data = graph[path[i]][path[i+1]]
                    relation = edge_data.get('relation', 'related_to')
                    relations.append(relation)
                    evidence_path.append((path[i], relation, path[i+1]))
                    
            # 选择合适的模板
            template = random.choice(templates)
            
            try:
                question = self._create_multi_hop_question(
                    template, start_entity, relations, path
                )
                
                qa_pairs.append(QAPair(
                    question=question,
                    answer=end_entity,
                    question_type='multi_hop',
                    subgraph=subgraph,
                    evidence_path=evidence_path,
                    difficulty=0.3 + 0.1 * len(path),
                    metadata={
                        'hop_count': len(path) - 1,
                        'path': path,
                        'relations': relations
                    }
                ))
            except Exception as e:
                self.logger.debug(f"Failed to generate multi-hop question: {e}")
                
        return qa_pairs
        
    def _generate_comparison_questions(
        self, 
        subgraph: Subgraph, 
        question_type: Dict
    ) -> List[QAPair]:
        """生成比较类问题"""
        qa_pairs = []
        graph = subgraph.graph
        templates = question_type.get('compiled_templates', [])
        
        # 找出可比较的实体对
        comparable_pairs = self._find_comparable_entity_pairs(graph)
        
        for entity1, entity2, common_attrs in comparable_pairs[:10]:
            template = random.choice(templates)
            
            try:
                # 生成比较问题
                question = self._fill_template(
                    template,
                    entity1=entity1,
                    entity2=entity2
                )
                
                # 生成比较答案
                answer, evidence = self._generate_comparison_answer(
                    graph, entity1, entity2, common_attrs
                )
                
                qa_pairs.append(QAPair(
                    question=question,
                    answer=answer,
                    question_type='comparison',
                    subgraph=subgraph,
                    evidence_path=evidence,
                    difficulty=0.5,
                    metadata={
                        'entities': [entity1, entity2],
                        'common_attributes': list(common_attrs)
                    }
                ))
            except Exception as e:
                self.logger.debug(f"Failed to generate comparison question: {e}")
                
        return qa_pairs
        
    def _generate_aggregation_questions(
        self, 
        subgraph: Subgraph, 
        question_type: Dict
    ) -> List[QAPair]:
        """生成聚合类问题"""
        qa_pairs = []
        graph = subgraph.graph
        templates = question_type.get('compiled_templates', [])
        
        # 找出可聚合的模式
        aggregation_patterns = self._find_aggregation_patterns(graph)
        
        for pattern in aggregation_patterns[:10]:
            template = random.choice(templates)
            
            try:
                # 根据模式类型生成问题
                if pattern['type'] == 'count':
                    question = self._create_count_question(
                        template, pattern
                    )
                    answer = str(pattern['count'])
                elif pattern['type'] == 'list':
                    question = self._create_list_question(
                        template, pattern
                    )
                    answer = ', '.join(pattern['items'])
                else:
                    continue
                    
                qa_pairs.append(QAPair(
                    question=question,
                    answer=answer,
                    question_type='aggregation',
                    subgraph=subgraph,
                    evidence_path=pattern['evidence'],
                    difficulty=0.4,
                    metadata={
                        'aggregation_type': pattern['type'],
                        'pattern': pattern
                    }
                ))
            except Exception as e:
                self.logger.debug(f"Failed to generate aggregation question: {e}")
                
        return qa_pairs
        
    def _generate_constraint_questions(
        self, 
        subgraph: Subgraph, 
        question_type: Dict
    ) -> List[QAPair]:
        """生成约束类问题"""
        qa_pairs = []
        graph = subgraph.graph
        templates = question_type.get('compiled_templates', [])
        
        # 找出满足多个约束的实体
        constraint_patterns = self._find_constraint_patterns(graph)
        
        for pattern in constraint_patterns[:10]:
            template = random.choice(templates)
            
            try:
                # 生成约束问题
                question = self._create_constraint_question(
                    template, pattern
                )
                
                # 找出满足所有约束的实体
                satisfying_entities = pattern['satisfying_entities']
                
                if satisfying_entities:
                    answer = ', '.join(satisfying_entities)
                else:
                    answer = "没有实体满足所有约束条件"
                    
                qa_pairs.append(QAPair(
                    question=question,
                    answer=answer,
                    question_type='constraint',
                    subgraph=subgraph,
                    evidence_path=pattern['evidence'],
                    difficulty=0.6,
                    metadata={
                        'constraints': pattern['constraints'],
                        'num_constraints': len(pattern['constraints'])
                    }
                ))
            except Exception as e:
                self.logger.debug(f"Failed to generate constraint question: {e}")
                
        return qa_pairs
        
    # 辅助方法
    def _has_multi_hop_paths(self, graph: nx.Graph) -> bool:
        """检查图中是否存在多跳路径"""
        for node in graph.nodes():
            if nx.single_source_shortest_path_length(graph, node, cutoff=2):
                return True
        return False
        
    def _has_comparable_entities(self, graph: nx.Graph) -> bool:
        """检查是否有可比较的实体"""
        # 简单检查：是否有共享邻居的节点
        for node1 in graph.nodes():
            neighbors1 = set(graph.neighbors(node1))
            for node2 in graph.nodes():
                if node1 != node2:
                    neighbors2 = set(graph.neighbors(node2))
                    if neighbors1 & neighbors2:  # 有共同邻居
                        return True
        return False
        
    def _has_multiple_attributes(self, graph: nx.Graph) -> bool:
        """检查实体是否有多个属性"""
        for node in graph.nodes():
            if graph.degree(node) >= 2:
                return True
        return False
        
    def _fill_template(self, template: Template, **kwargs) -> str:
        """填充模板生成问题"""
        try:
            return template.safe_substitute(**kwargs)
        except Exception as e:
            self.logger.debug(f"Template filling failed: {e}")
            return template.template
            
    def _is_reversible_relation(self, relation: str) -> bool:
        """判断关系是否可逆"""
        reversible_relations = [
            'manufactures', 'developed_by', 'contains_component',
            'cooperates_with', 'located_in'
        ]
        return relation in reversible_relations
        
    def _generate_reverse_question(
        self, 
        template: Template, 
        entity: str, 
        relation: str
    ) -> Optional[str]:
        """生成反向问题"""
        reverse_relations = {
            'manufactures': '被...制造',
            'developed_by': '开发了',
            'contains_component': '是...的组件',
            'located_in': '包含'
        }
        
        if relation in reverse_relations:
            return f"{entity}{reverse_relations[relation]}什么？"
        return None
        
    def _find_multi_hop_paths(
        self, 
        graph: nx.Graph, 
        min_hops: int = 2, 
        max_hops: int = 4
    ) -> List[List[str]]:
        """找出多跳路径"""
        paths = []
        nodes = list(graph.nodes())
        
        for start in nodes:
            for end in nodes:
                if start == end:
                    continue
                    
                try:
                    # 找出所有简单路径
                    simple_paths = list(nx.all_simple_paths(
                        graph, start, end, cutoff=max_hops
                    ))
                    
                    for path in simple_paths:
                        if min_hops <= len(path) - 1 <= max_hops:
                            paths.append(path)
                except nx.NetworkXNoPath:
                    continue
                    
        # 随机采样避免过多
        if len(paths) > 50:
            paths = random.sample(paths, 50)
            
        return paths
        
    def _create_multi_hop_question(
        self, 
        template: Template, 
        start_entity: str, 
        relations: List[str], 
        path: List[str]
    ) -> str:
        """创建多跳问题"""
        # 简化版本，实际可以更复杂
        if len(relations) == 2:
            return f"{start_entity}的{relations[0]}的{relations[1]}是什么？"
        elif len(relations) == 3:
            return f"{start_entity}的{relations[0]}的{relations[1]}的{relations[2]}是什么？"
        else:
            return f"从{start_entity}出发，经过{len(relations)}步关系后到达什么实体？"
            
    def _find_comparable_entity_pairs(
        self, 
        graph: nx.Graph
    ) -> List[Tuple[str, str, Set[str]]]:
        """找出可比较的实体对"""
        pairs = []
        nodes = list(graph.nodes())
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # 找出共同属性
                neighbors1 = set(graph.neighbors(node1))
                neighbors2 = set(graph.neighbors(node2))
                common = neighbors1 & neighbors2
                
                if common:
                    pairs.append((node1, node2, common))
                    
        return pairs[:20]  # 限制数量
        
    def _generate_comparison_answer(
        self, 
        graph: nx.Graph, 
        entity1: str, 
        entity2: str, 
        common_attrs: Set[str]
    ) -> Tuple[str, List[Tuple[str, str, str]]]:
        """生成比较答案"""
        differences = []
        evidence = []
        
        # 比较共同属性的值
        for attr in common_attrs:
            if graph.has_edge(entity1, attr):
                rel1 = graph[entity1][attr].get('relation', 'has')
                evidence.append((entity1, rel1, attr))
                
            if graph.has_edge(entity2, attr):
                rel2 = graph[entity2][attr].get('relation', 'has')
                evidence.append((entity2, rel2, attr))
                
        # 找出独有属性
        neighbors1 = set(graph.neighbors(entity1))
        neighbors2 = set(graph.neighbors(entity2))
        
        unique1 = neighbors1 - neighbors2
        unique2 = neighbors2 - neighbors1
        
        answer_parts = []
        if unique1:
            answer_parts.append(f"{entity1}独有: {', '.join(unique1)}")
        if unique2:
            answer_parts.append(f"{entity2}独有: {', '.join(unique2)}")
        if common_attrs:
            answer_parts.append(f"共同拥有: {', '.join(common_attrs)}")
            
        return '; '.join(answer_parts), evidence
        
    def _find_aggregation_patterns(self, graph: nx.Graph) -> List[Dict]:
        """找出可聚合的模式"""
        patterns = []
        
        # 计数模式：统计某种关系的数量
        relation_counts = defaultdict(lambda: defaultdict(int))
        
        for source, target, data in graph.edges(data=True):
            relation = data.get('relation', 'related_to')
            relation_counts[relation][source] += 1
            
        # 生成计数模式
        for relation, entity_counts in relation_counts.items():
            for entity, count in entity_counts.items():
                if count >= 2:
                    evidence = [
                        (entity, relation, target)
                        for _, target, d in graph.edges(entity, data=True)
                        if d.get('relation') == relation
                    ]
                    
                    patterns.append({
                        'type': 'count',
                        'entity': entity,
                        'relation': relation,
                        'count': count,
                        'evidence': evidence
                    })
                    
        # 列表模式：列出满足条件的所有实体
        # 例如：所有使用某技术的产品
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if len(neighbors) >= 3:
                patterns.append({
                    'type': 'list',
                    'center': node,
                    'items': neighbors,
                    'evidence': [
                        (node, graph[node][n].get('relation', 'related_to'), n)
                        for n in neighbors
                    ]
                })
                
        return patterns
        
    def _find_constraint_patterns(self, graph: nx.Graph) -> List[Dict]:
        """找出约束模式"""
        patterns = []
        
        # 找出有多个属性的实体
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if len(neighbors) >= 2:
                # 生成约束组合
                constraints = []
                evidence = []
                
                for neighbor in neighbors[:3]:  # 限制约束数量
                    if graph.has_edge(node, neighbor):
                        relation = graph[node][neighbor].get('relation', 'has')
                        constraints.append({
                            'type': relation,
                            'value': neighbor
                        })
                        evidence.append((node, relation, neighbor))
                        
                if len(constraints) >= 2:
                    patterns.append({
                        'constraints': constraints,
                        'satisfying_entities': [node],
                        'evidence': evidence
                    })
                    
        return patterns
        
    def _create_count_question(self, template: Template, pattern: Dict) -> str:
        """创建计数问题"""
        return f"{pattern['entity']}有多少个{pattern['relation']}？"
        
    def _create_list_question(self, template: Template, pattern: Dict) -> str:
        """创建列表问题"""
        return f"列出所有与{pattern['center']}相关的实体。"
        
    def _create_constraint_question(self, template: Template, pattern: Dict) -> str:
        """创建约束问题"""
        constraints_str = '且'.join([
            f"{c['type']}是{c['value']}" 
            for c in pattern['constraints']
        ])
        return f"哪些实体{constraints_str}？"
        
    def _deduplicate_qa_pairs(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """去除重复的问答对"""
        seen = set()
        unique_pairs = []
        
        for pair in qa_pairs:
            # 使用问题和答案的组合作为唯一标识
            key = (pair.question, pair.answer)
            if key not in seen:
                seen.add(key)
                unique_pairs.append(pair)
                
        return unique_pairs
        
    def _filter_quality(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """过滤低质量的问答对"""
        filtered = []
        
        for pair in qa_pairs:
            # 检查问题长度
            if len(pair.question) < 5 or len(pair.question) > 200:
                continue
                
            # 检查答案
            if not pair.answer or len(pair.answer) > 500:
                continue
                
            # 检查证据路径
            if not pair.evidence_path:
                continue
                
            filtered.append(pair)
            
        return filtered