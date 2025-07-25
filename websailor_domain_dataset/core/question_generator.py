#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Question Generator - 问题生成器 (WebSailor核心思想2)
基于子图中节点与关系，设计QA问题，覆盖多种问题类型

WebSailor核心思想2：问题生成
- 基于子图中节点与关系,设计 QA 问题
- 覆盖多种问题类型：单跳事实查询、多跳推理、比较分析、聚合统计、复杂推理
- 根据子图拓扑结构选择合适的问题类型

该模块实现：
1. 单跳事实查询：直接查询某个实体的属性
2. 多跳事实查询：需要通过多个节点推理
3. 比较查询：比较多个实体的异同
4. 聚合查询：统计或汇总信息
5. 推理查询：需要逻辑推理的复杂问题
"""

import logging
import random
import re
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import networkx as nx

from .subgraph_sampler import Subgraph


class QuestionAnswer:
    """问答对类"""
    
    def __init__(self, question: str, answer: str, question_type: str, 
                 subgraph: Subgraph, metadata: Dict[str, Any] = None):
        """
        初始化问答对
        
        Args:
            question: 问题文本
            answer: 答案文本
            question_type: 问题类型
            subgraph: 相关子图
            metadata: 元数据信息
        """
        self.question = question
        self.answer = answer
        self.question_type = question_type
        self.subgraph = subgraph
        self.metadata = metadata or {}
        
        # 计算问题特征
        self._compute_features()
    
    def _compute_features(self):
        """计算问题特征"""
        self.features = {
            "question_length": len(self.question),
            "answer_length": len(self.answer),
            "num_entities_in_question": self._count_entities_in_text(self.question),
            "num_entities_in_answer": self._count_entities_in_text(self.answer),
            "complexity_score": self._compute_complexity_score(),
            "subgraph_topology": self.subgraph.topology_type,
            "subgraph_size": self.subgraph.num_nodes
        }
    
    def _count_entities_in_text(self, text: str) -> int:
        """统计文本中的实体数量"""
        entity_count = 0
        for node in self.subgraph.nodes:
            if node in text:
                entity_count += 1
        return entity_count
    
    def _compute_complexity_score(self) -> float:
        """计算问题复杂度分数"""
        # 基于多个因素计算复杂度
        base_score = 1.0
        
        # 问题类型权重
        type_weights = {
            "factual_single": 1.0,
            "factual_multi": 2.0,
            "comparative": 2.5,
            "aggregative": 3.0,
            "reasoning": 4.0
        }
        
        type_weight = type_weights.get(self.question_type, 1.0)
        
        # 子图复杂度
        subgraph_complexity = (self.subgraph.num_nodes * 0.1 + 
                             self.subgraph.num_edges * 0.15)
        
        # 问题长度影响
        length_factor = min(len(self.question) / 100, 2.0)
        
        return base_score * type_weight * (1 + subgraph_complexity) * length_factor
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "features": self.features,
            "metadata": self.metadata,
            "subgraph_info": {
                "topology_type": self.subgraph.topology_type,
                "num_nodes": self.subgraph.num_nodes,
                "num_edges": self.subgraph.num_edges
            }
        }


class QuestionGenerator:
    """问题生成器 - WebSailor核心思想2"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化问题生成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.question_config = config.get("question_generation", {})
        self.domain_config = config.get("tcl_industry_domain", {})
        
        # 加载问题类型配置
        self.question_types = self._load_question_types()
        
        # 加载领域特定信息
        self.entity_types = self.domain_config.get("entity_types", [])
        self.relation_types = self.domain_config.get("relation_types", [])
        self.focus_areas = self.domain_config.get("focus_areas", [])
        
        logging.info("问题生成器初始化完成")
    
    def _load_question_types(self) -> Dict[str, Dict[str, Any]]:
        """加载问题类型配置"""
        question_types = {}
        
        for type_config in self.question_config.get("question_types", []):
            name = type_config["name"]
            question_types[name] = {
                "description": type_config["description"],
                "weight": type_config["weight"],
                "templates": type_config["templates"],
                "generator": self._get_generator_function(name)
            }
        
        return question_types
    
    def _get_generator_function(self, question_type: str):
        """获取对应的问题生成函数"""
        generator_mapping = {
            "factual_single": self._generate_factual_single_question,
            "factual_multi": self._generate_factual_multi_question,
            "comparative": self._generate_comparative_question,
            "aggregative": self._generate_aggregative_question,
            "reasoning": self._generate_reasoning_question
        }
        
        return generator_mapping.get(question_type, self._generate_factual_single_question)
    
    def generate_questions(self, subgraph: Subgraph, 
                          question_types: List[str] = None,
                          num_questions_per_subgraph: int = 3) -> List[QuestionAnswer]:
        """
        为子图生成问题 - WebSailor核心思想2的实现
        
        Args:
            subgraph: 输入子图
            question_types: 要生成的问题类型列表
            num_questions_per_subgraph: 每个子图生成的问题数量
            
        Returns:
            生成的问答对列表
        """
        if subgraph.num_nodes == 0:
            return []
        
        qa_pairs = []
        
        # 根据子图拓扑选择合适的问题类型
        suitable_types = self._select_suitable_question_types(subgraph, question_types)
        
        # 为每种问题类型生成问题
        type_counts = self._allocate_question_counts(num_questions_per_subgraph, suitable_types)
        
        for q_type, count in type_counts.items():
            if count > 0 and q_type in self.question_types:
                try:
                    type_qa_pairs = self._generate_questions_of_type(
                        subgraph, q_type, count
                    )
                    qa_pairs.extend(type_qa_pairs)
                except Exception as e:
                    logging.warning(f"生成 {q_type} 类型问题时出错: {e}")
        
        return qa_pairs
    
    def _select_suitable_question_types(self, subgraph: Subgraph, 
                                       question_types: List[str] = None) -> List[str]:
        """根据子图拓扑选择合适的问题类型"""
        if question_types:
            return question_types
        
        suitable_types = []
        topology = subgraph.topology_type
        
        # 根据拓扑类型选择适合的问题类型
        if topology == "star_topology":
            # 星形拓扑适合聚合和比较问题
            suitable_types = ["factual_single", "aggregative", "comparative"]
        elif topology == "path_topology":
            # 路径拓扑适合多跳推理问题
            suitable_types = ["factual_multi", "reasoning", "factual_single"]
        elif topology == "cluster_topology":
            # 簇拓扑适合复杂推理和比较问题
            suitable_types = ["reasoning", "comparative", "aggregative"]
        elif topology == "tree_topology":
            # 树形拓扑适合层次化推理问题
            suitable_types = ["reasoning", "factual_multi", "aggregative"]
        else:
            # 默认类型
            suitable_types = ["factual_single", "factual_multi", "comparative"]
        
        return suitable_types
    
    def _allocate_question_counts(self, total_count: int, 
                                 question_types: List[str]) -> Dict[str, int]:
        """分配每种问题类型的生成数量"""
        if not question_types:
            return {}
        
        # 获取权重
        type_weights = {}
        total_weight = 0
        
        for q_type in question_types:
            if q_type in self.question_types:
                weight = self.question_types[q_type]["weight"]
                type_weights[q_type] = weight
                total_weight += weight
        
        # 按权重分配
        type_counts = {}
        allocated_count = 0
        
        for q_type, weight in type_weights.items():
            count = int(total_count * weight / total_weight) if total_weight > 0 else 0
            type_counts[q_type] = count
            allocated_count += count
        
        # 处理余数
        remaining = total_count - allocated_count
        types_list = list(type_weights.keys())
        for i in range(remaining):
            if types_list:
                q_type = types_list[i % len(types_list)]
                type_counts[q_type] += 1
        
        return type_counts
    
    def _generate_questions_of_type(self, subgraph: Subgraph, 
                                   question_type: str, count: int) -> List[QuestionAnswer]:
        """生成指定类型的问题"""
        if question_type not in self.question_types:
            return []
        
        generator_func = self.question_types[question_type]["generator"]
        qa_pairs = []
        
        max_attempts = count * 5
        attempts = 0
        
        while len(qa_pairs) < count and attempts < max_attempts:
            try:
                qa_pair = generator_func(subgraph)
                if qa_pair:
                    qa_pairs.append(qa_pair)
            except Exception as e:
                logging.warning(f"生成问题时出错: {e}")
            
            attempts += 1
        
        return qa_pairs
    
    def _generate_factual_single_question(self, subgraph: Subgraph) -> Optional[QuestionAnswer]:
        """
        生成单跳事实查询问题
        
        特点：直接查询某个实体的属性
        适用于：所有拓扑类型
        """
        if subgraph.num_nodes == 0:
            return None
        
        # 随机选择一个实体
        target_entity = random.choice(subgraph.nodes)
        entity_data = subgraph.graph.nodes.get(target_entity, {})
        entity_type = entity_data.get("type", "实体")
        
        # 获取实体的属性或邻居
        neighbors = list(subgraph.graph.neighbors(target_entity))
        
        if not neighbors:
            return None
        
        # 选择一个相关实体作为答案
        answer_entity = random.choice(neighbors)
        answer_data = subgraph.graph.nodes.get(answer_entity, {})
        
        # 获取关系信息
        edge_data = subgraph.graph.get_edge_data(target_entity, answer_entity, {})
        relation = edge_data.get("relation", "相关")
        
        # 选择问题模板
        templates = self.question_types["factual_single"]["templates"]
        template = random.choice(templates)
        
        # 生成问题和答案
        question = template.format(
            entity=target_entity,
            attribute=self._get_attribute_name(relation, entity_type)
        )
        
        answer = f"{answer_entity}"
        
        # 添加上下文信息
        if "contexts" in edge_data:
            contexts = edge_data["contexts"]
            if contexts:
                answer += f"。根据相关信息：{contexts[0][:100]}..."
        
        metadata = {
            "target_entity": target_entity,
            "answer_entity": answer_entity,
            "relation": relation,
            "entity_type": entity_type
        }
        
        return QuestionAnswer(question, answer, "factual_single", subgraph, metadata)
    
    def _generate_factual_multi_question(self, subgraph: Subgraph) -> Optional[QuestionAnswer]:
        """
        生成多跳事实查询问题
        
        特点：需要通过多个节点推理
        适用于：路径拓扑、树形拓扑
        """
        if subgraph.num_nodes < 3:
            return None
        
        # 寻找多跳路径
        nodes = subgraph.nodes
        start_entity = random.choice(nodes)
        
        # 找到距离较远的节点
        try:
            distances = nx.single_source_shortest_path_length(subgraph.graph, start_entity)
            distant_nodes = [node for node, dist in distances.items() if dist >= 2]
            
            if not distant_nodes:
                return None
            
            end_entity = random.choice(distant_nodes)
            
            # 获取路径
            path = nx.shortest_path(subgraph.graph, start_entity, end_entity)
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        
        if len(path) < 3:
            return None
        
        # 构建问题
        intermediate_entity = path[1]
        
        # 获取关系信息
        relation1 = subgraph.graph.get_edge_data(start_entity, intermediate_entity, {}).get("relation", "相关")
        relation2 = subgraph.graph.get_edge_data(intermediate_entity, end_entity, {}).get("relation", "相关")
        
        templates = self.question_types["factual_multi"]["templates"]
        template = random.choice(templates)
        
        question = template.format(
            entity1=start_entity,
            relation1=relation1,
            entity2=intermediate_entity,
            attribute=self._get_attribute_name(relation2, "实体")
        )
        
        answer = f"{end_entity}"
        
        # 添加推理路径说明
        path_description = " -> ".join([f"{path[i]}({subgraph.graph.get_edge_data(path[i], path[i+1], {}).get('relation', '相关')})" 
                                      for i in range(len(path)-1)]) + f" -> {path[-1]}"
        answer += f"。推理路径：{path_description}"
        
        metadata = {
            "start_entity": start_entity,
            "end_entity": end_entity,
            "reasoning_path": path,
            "path_length": len(path),
            "intermediate_entities": path[1:-1]
        }
        
        return QuestionAnswer(question, answer, "factual_multi", subgraph, metadata)
    
    def _generate_comparative_question(self, subgraph: Subgraph) -> Optional[QuestionAnswer]:
        """
        生成比较查询问题
        
        特点：比较多个实体的异同
        适用于：星形拓扑、簇拓扑
        """
        if subgraph.num_nodes < 2:
            return None
        
        # 选择两个实体进行比较
        entities = random.sample(subgraph.nodes, 2)
        entity1, entity2 = entities
        
        entity1_data = subgraph.graph.nodes.get(entity1, {})
        entity2_data = subgraph.graph.nodes.get(entity2, {})
        
        entity1_type = entity1_data.get("type", "实体")
        entity2_type = entity2_data.get("type", "实体")
        
        # 获取比较维度
        comparison_aspect = self._get_comparison_aspect(entity1_type, entity2_type)
        
        templates = self.question_types["comparative"]["templates"]
        template = random.choice(templates)
        
        question = template.format(
            entity1=entity1,
            entity2=entity2,
            aspect=comparison_aspect,
            attribute=comparison_aspect
        )
        
        # 生成比较答案
        answer = self._generate_comparison_answer(subgraph, entity1, entity2, comparison_aspect)
        
        metadata = {
            "compared_entities": [entity1, entity2],
            "comparison_aspect": comparison_aspect,
            "entity_types": [entity1_type, entity2_type]
        }
        
        return QuestionAnswer(question, answer, "comparative", subgraph, metadata)
    
    def _generate_aggregative_question(self, subgraph: Subgraph) -> Optional[QuestionAnswer]:
        """
        生成聚合查询问题
        
        特点：统计或汇总信息
        适用于：星形拓扑、簇拓扑
        """
        if subgraph.num_nodes < 3:
            return None
        
        # 统计实体类型
        entity_types = defaultdict(list)
        for node in subgraph.nodes:
            node_data = subgraph.graph.nodes.get(node, {})
            node_type = node_data.get("type", "实体")
            entity_types[node_type].append(node)
        
        # 选择数量较多的实体类型
        suitable_types = {k: v for k, v in entity_types.items() if len(v) >= 2}
        
        if not suitable_types:
            return None
        
        target_type = random.choice(list(suitable_types.keys()))
        entities_of_type = suitable_types[target_type]
        
        templates = self.question_types["aggregative"]["templates"]
        template = random.choice(templates)
        
        # 随机选择聚合类型
        aggregation_types = ["数量", "类型", "关系"]
        agg_type = random.choice(aggregation_types)
        
        if agg_type == "数量":
            question = template.format(
                category=target_type,
                entity_type=target_type,
                domain="TCL工业领域"
            )
            answer = f"共有{len(entities_of_type)}个{target_type}：{', '.join(entities_of_type)}"
            
        elif agg_type == "关系":
            # 统计关系类型
            relations = set()
            for entity in entities_of_type:
                for neighbor in subgraph.graph.neighbors(entity):
                    edge_data = subgraph.graph.get_edge_data(entity, neighbor, {})
                    relation = edge_data.get("relation", "相关")
                    relations.add(relation)
            
            question = f"{target_type}实体相关的所有关系类型有哪些？"
            answer = f"相关关系类型包括：{', '.join(relations)}"
        
        else:  # 类型
            question = f"在当前知识图谱中，有哪些类型的实体？"
            all_types = list(entity_types.keys())
            answer = f"包含以下类型的实体：{', '.join(all_types)}"
        
        metadata = {
            "aggregation_type": agg_type,
            "target_entity_type": target_type,
            "entities_count": len(entities_of_type),
            "entities_list": entities_of_type
        }
        
        return QuestionAnswer(question, answer, "aggregative", subgraph, metadata)
    
    def _generate_reasoning_question(self, subgraph: Subgraph) -> Optional[QuestionAnswer]:
        """
        生成推理查询问题
        
        特点：需要逻辑推理的复杂问题
        适用于：簇拓扑、树形拓扑
        """
        if subgraph.num_nodes < 3:
            return None
        
        # 寻找有趣的推理模式
        reasoning_patterns = self._identify_reasoning_patterns(subgraph)
        
        if not reasoning_patterns:
            return None
        
        pattern = random.choice(reasoning_patterns)
        pattern_type = pattern["type"]
        
        templates = self.question_types["reasoning"]["templates"]
        template = random.choice(templates)
        
        if pattern_type == "causal":
            # 因果推理
            cause_entity = pattern["cause"]
            effect_entity = pattern["effect"]
            relation = pattern["relation"]
            
            question = template.format(
                condition1=f"{cause_entity}具有某种特性",
                entity=effect_entity,
                action=f"产生{relation}关系"
            )
            
            answer = f"根据知识图谱中的关系，{cause_entity}通过{relation}关系影响{effect_entity}。"
            
        elif pattern_type == "transitive":
            # 传递推理
            entities = pattern["entities"]
            relations = pattern["relations"]
            
            question = template.format(
                entity1=entities[0],
                attribute1=relations[0],
                entity2=entities[1],
                attribute2=relations[1]
            )
            
            reasoning_chain = " -> ".join([f"{entities[i]}({relations[i] if i < len(relations) else ''})" 
                                         for i in range(len(entities))])
            answer = f"通过传递推理：{reasoning_chain}，可以推断出相关联系。"
            
        else:  # default
            # 默认推理
            central_nodes = subgraph.get_central_nodes(1)
            if central_nodes:
                central_entity = central_nodes[0]
                neighbors = list(subgraph.graph.neighbors(central_entity))
                
                question = f"为什么{central_entity}在当前场景中起到重要作用？"
                answer = f"{central_entity}连接了{len(neighbors)}个相关实体：{', '.join(neighbors[:3])}等，是信息汇聚的关键节点。"
            else:
                return None
        
        metadata = {
            "reasoning_type": pattern_type,
            "reasoning_pattern": pattern,
            "complexity_level": "high"
        }
        
        return QuestionAnswer(question, answer, "reasoning", subgraph, metadata)
    
    def _get_attribute_name(self, relation: str, entity_type: str) -> str:
        """根据关系和实体类型生成属性名"""
        attribute_mapping = {
            "开发": "开发的产品",
            "应用": "应用领域",
            "竞争": "竞争对手",
            "合作": "合作伙伴",
            "投资": "投资项目",
            "收购": "收购目标",
            "技术转移": "技术来源",
            "标准制定": "相关标准"
        }
        
        return attribute_mapping.get(relation, "相关信息")
    
    def _get_comparison_aspect(self, type1: str, type2: str) -> str:
        """获取比较维度"""
        if type1 == type2:
            if type1 == "产品":
                return "功能特性"
            elif type1 == "技术":
                return "技术先进性"
            elif type1 == "公司":
                return "市场地位"
            else:
                return "特点"
        else:
            return "应用场景"
    
    def _generate_comparison_answer(self, subgraph: Subgraph, 
                                   entity1: str, entity2: str, aspect: str) -> str:
        """生成比较答案"""
        # 获取两个实体的邻居和关系
        neighbors1 = set(subgraph.graph.neighbors(entity1))
        neighbors2 = set(subgraph.graph.neighbors(entity2))
        
        common_neighbors = neighbors1.intersection(neighbors2)
        unique_neighbors1 = neighbors1 - neighbors2
        unique_neighbors2 = neighbors2 - neighbors1
        
        answer_parts = []
        
        if common_neighbors:
            answer_parts.append(f"共同点：{entity1}和{entity2}都与{', '.join(list(common_neighbors)[:2])}等相关")
        
        if unique_neighbors1:
            answer_parts.append(f"{entity1}独有的关联：{', '.join(list(unique_neighbors1)[:2])}")
        
        if unique_neighbors2:
            answer_parts.append(f"{entity2}独有的关联：{', '.join(list(unique_neighbors2)[:2])}")
        
        if not answer_parts:
            answer_parts.append(f"{entity1}和{entity2}在{aspect}方面各有特点")
        
        return "；".join(answer_parts) + "。"
    
    def _identify_reasoning_patterns(self, subgraph: Subgraph) -> List[Dict[str, Any]]:
        """识别推理模式"""
        patterns = []
        
        # 寻找因果模式（A->B->C）
        for node in subgraph.nodes:
            out_neighbors = list(subgraph.graph.neighbors(node))
            for neighbor in out_neighbors:
                second_neighbors = [n for n in subgraph.graph.neighbors(neighbor) if n != node]
                if second_neighbors:
                    relation1 = subgraph.graph.get_edge_data(node, neighbor, {}).get("relation", "相关")
                    for second_neighbor in second_neighbors:
                        relation2 = subgraph.graph.get_edge_data(neighbor, second_neighbor, {}).get("relation", "相关")
                        patterns.append({
                            "type": "transitive",
                            "entities": [node, neighbor, second_neighbor],
                            "relations": [relation1, relation2]
                        })
        
        # 寻找因果关系模式
        for edge in subgraph.edges:
            node1, node2 = edge
            edge_data = subgraph.graph.get_edge_data(node1, node2, {})
            relation = edge_data.get("relation", "相关")
            
            if relation in ["开发", "应用", "投资", "收购"]:
                patterns.append({
                    "type": "causal",
                    "cause": node1,
                    "effect": node2,
                    "relation": relation
                })
        
        return patterns[:5]  # 限制数量