"""
子图采样器 - WebSailor核心模块
从整个知识图中抽取不同拓扑的子图作为问题候选基础
每个子图代表了一种"任务场景"：可能包含多个目标、干扰信息、隐含路径
"""

import random
import logging
from typing import List, Dict, Set, Tuple, Any
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import numpy as np


@dataclass
class Subgraph:
    """子图数据结构"""
    graph: nx.Graph
    topology_type: str
    center_nodes: List[str]
    metadata: Dict[str, Any]
    

class SubgraphSampler:
    """
    WebSailor子图采样器
    负责从知识图谱中采样不同拓扑结构的子图
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.sampling_strategies = config.get('sampling_strategies', [])
        self.num_subgraphs_per_strategy = config.get('num_subgraphs_per_strategy', 100)
        self.ensure_connectivity = config.get('ensure_connectivity', True)
        self.include_attributes = config.get('include_attributes', True)
        
    def sample_diverse_subgraphs(self, knowledge_graph: nx.Graph) -> List[Subgraph]:
        """
        从知识图谱中采样多样化的子图
        这是WebSailor的核心功能之一
        """
        all_subgraphs = []
        
        for strategy in self.sampling_strategies:
            strategy_name = strategy['name']
            self.logger.info(f"Sampling subgraphs using {strategy_name} strategy...")
            
            if strategy_name == 'star_topology':
                subgraphs = self._sample_star_topology(knowledge_graph, strategy)
            elif strategy_name == 'chain_topology':
                subgraphs = self._sample_chain_topology(knowledge_graph, strategy)
            elif strategy_name == 'tree_topology':
                subgraphs = self._sample_tree_topology(knowledge_graph, strategy)
            elif strategy_name == 'mesh_topology':
                subgraphs = self._sample_mesh_topology(knowledge_graph, strategy)
            else:
                self.logger.warning(f"Unknown sampling strategy: {strategy_name}")
                continue
                
            all_subgraphs.extend(subgraphs)
            self.logger.info(f"Sampled {len(subgraphs)} {strategy_name} subgraphs")
            
        # 去重和验证
        all_subgraphs = self._deduplicate_subgraphs(all_subgraphs)
        all_subgraphs = self._validate_subgraphs(all_subgraphs)
        
        self.logger.info(f"Total unique valid subgraphs sampled: {len(all_subgraphs)}")
        return all_subgraphs
        
    def _sample_star_topology(self, kg: nx.Graph, strategy: Dict) -> List[Subgraph]:
        """
        采样星型拓扑子图
        中心节点向外辐射，适合单实体多属性查询
        """
        subgraphs = []
        min_nodes = strategy.get('min_nodes', 5)
        max_nodes = strategy.get('max_nodes', 15)
        num_samples = int(self.num_subgraphs_per_strategy * strategy.get('weight', 0.25))
        
        # 选择高度数节点作为中心
        node_degrees = [(node, deg) for node, deg in kg.degree() if deg >= min_nodes - 1]
        if not node_degrees:
            return subgraphs
            
        # 按度数排序，优先选择连接较多的节点
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        candidate_centers = [node for node, _ in node_degrees[:len(node_degrees)//2]]
        
        for _ in range(min(num_samples, len(candidate_centers))):
            center = random.choice(candidate_centers)
            
            # 获取所有邻居
            neighbors = list(kg.neighbors(center))
            if len(neighbors) < min_nodes - 1:
                continue
                
            # 随机选择邻居数量
            num_neighbors = random.randint(min_nodes - 1, min(max_nodes - 1, len(neighbors)))
            selected_neighbors = random.sample(neighbors, num_neighbors)
            
            # 构建星型子图
            subgraph = kg.subgraph([center] + selected_neighbors).copy()
            
            # 添加中心到所有邻居的边（确保星型结构）
            for neighbor in selected_neighbors:
                if not subgraph.has_edge(center, neighbor):
                    # 从原图复制边属性
                    if kg.has_edge(center, neighbor):
                        subgraph.add_edge(center, neighbor, **kg[center][neighbor])
                        
            subgraphs.append(Subgraph(
                graph=subgraph,
                topology_type='star',
                center_nodes=[center],
                metadata={
                    'num_nodes': subgraph.number_of_nodes(),
                    'num_edges': subgraph.number_of_edges(),
                    'center_degree': kg.degree(center)
                }
            ))
            
        return subgraphs
        
    def _sample_chain_topology(self, kg: nx.Graph, strategy: Dict) -> List[Subgraph]:
        """
        采样链式拓扑子图
        适合生产流程、因果关系等线性推理场景
        """
        subgraphs = []
        min_length = strategy.get('min_length', 3)
        max_length = strategy.get('max_length', 8)
        num_samples = int(self.num_subgraphs_per_strategy * strategy.get('weight', 0.25))
        
        # 使用随机游走生成链
        all_nodes = list(kg.nodes())
        
        for _ in range(num_samples):
            if not all_nodes:
                break
                
            # 随机选择起始节点
            start_node = random.choice(all_nodes)
            chain_length = random.randint(min_length, max_length)
            
            # 执行随机游走
            chain = self._random_walk(kg, start_node, chain_length)
            
            if len(chain) < min_length:
                continue
                
            # 构建链式子图
            subgraph = nx.Graph()
            for i in range(len(chain) - 1):
                if kg.has_edge(chain[i], chain[i+1]):
                    subgraph.add_edge(chain[i], chain[i+1], **kg[chain[i]][chain[i+1]])
                    
            # 确保连通性
            if self.ensure_connectivity and not nx.is_connected(subgraph):
                continue
                
            subgraphs.append(Subgraph(
                graph=subgraph,
                topology_type='chain',
                center_nodes=[chain[0], chain[-1]],  # 起点和终点
                metadata={
                    'chain_length': len(chain),
                    'path': chain
                }
            ))
            
        return subgraphs
        
    def _sample_tree_topology(self, kg: nx.Graph, strategy: Dict) -> List[Subgraph]:
        """
        采样树形拓扑子图
        适合层级关系、组件结构等场景
        """
        subgraphs = []
        min_depth = strategy.get('min_depth', 2)
        max_depth = strategy.get('max_depth', 4)
        num_samples = int(self.num_subgraphs_per_strategy * strategy.get('weight', 0.25))
        
        for _ in range(num_samples):
            # 随机选择根节点
            root = random.choice(list(kg.nodes()))
            
            # BFS构建树
            tree = nx.Graph()
            visited = {root}
            current_level = [root]
            depth = 0
            
            while current_level and depth < max_depth:
                next_level = []
                for node in current_level:
                    neighbors = [n for n in kg.neighbors(node) if n not in visited]
                    
                    # 随机选择部分邻居作为子节点
                    if neighbors:
                        num_children = random.randint(1, min(3, len(neighbors)))
                        children = random.sample(neighbors, num_children)
                        
                        for child in children:
                            tree.add_edge(node, child, **kg[node][child])
                            visited.add(child)
                            next_level.append(child)
                            
                current_level = next_level
                depth += 1
                
            # 检查深度要求
            if depth < min_depth:
                continue
                
            subgraphs.append(Subgraph(
                graph=tree,
                topology_type='tree',
                center_nodes=[root],
                metadata={
                    'depth': depth,
                    'num_leaves': len([n for n in tree.nodes() if tree.degree(n) == 1])
                }
            ))
            
        return subgraphs
        
    def _sample_mesh_topology(self, kg: nx.Graph, strategy: Dict) -> List[Subgraph]:
        """
        采样网状拓扑子图
        包含多个交叉关系，适合复杂推理场景
        """
        subgraphs = []
        min_nodes = strategy.get('min_nodes', 6)
        max_nodes = strategy.get('max_nodes', 20)
        density = strategy.get('density', 0.3)
        num_samples = int(self.num_subgraphs_per_strategy * strategy.get('weight', 0.25))
        
        for _ in range(num_samples):
            # 随机选择种子节点
            seed = random.choice(list(kg.nodes()))
            
            # 使用BFS扩展到指定大小
            subgraph_nodes = {seed}
            candidates = set(kg.neighbors(seed))
            
            target_size = random.randint(min_nodes, max_nodes)
            
            while len(subgraph_nodes) < target_size and candidates:
                # 优先选择与现有节点连接较多的候选节点
                node_connections = []
                for candidate in candidates:
                    connections = len(set(kg.neighbors(candidate)) & subgraph_nodes)
                    node_connections.append((candidate, connections))
                    
                # 按连接数排序，倾向于选择连接更多的节点（增加网状密度）
                node_connections.sort(key=lambda x: x[1], reverse=True)
                
                # 使用概率选择，连接越多概率越大
                weights = [conn + 1 for _, conn in node_connections]
                selected = random.choices(
                    [node for node, _ in node_connections],
                    weights=weights,
                    k=1
                )[0]
                
                subgraph_nodes.add(selected)
                candidates.remove(selected)
                
                # 添加新候选
                new_candidates = set(kg.neighbors(selected)) - subgraph_nodes
                candidates.update(new_candidates)
                
            # 构建子图
            subgraph = kg.subgraph(subgraph_nodes).copy()
            
            # 检查密度
            actual_density = nx.density(subgraph)
            if actual_density < density * 0.7:  # 允许一定偏差
                continue
                
            # 找出关键节点（度数最高的几个）
            node_degrees = [(n, d) for n, d in subgraph.degree()]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            center_nodes = [n for n, _ in node_degrees[:3]]
            
            subgraphs.append(Subgraph(
                graph=subgraph,
                topology_type='mesh',
                center_nodes=center_nodes,
                metadata={
                    'density': actual_density,
                    'clustering_coefficient': nx.average_clustering(subgraph.to_undirected())
                }
            ))
            
        return subgraphs
        
    def _random_walk(self, graph: nx.Graph, start_node: str, max_length: int) -> List[str]:
        """执行随机游走"""
        path = [start_node]
        current = start_node
        
        for _ in range(max_length - 1):
            neighbors = list(graph.neighbors(current))
            if not neighbors:
                break
                
            # 避免立即返回
            if len(path) > 1 and path[-2] in neighbors and len(neighbors) > 1:
                neighbors.remove(path[-2])
                
            current = random.choice(neighbors)
            path.append(current)
            
        return path
        
    def _deduplicate_subgraphs(self, subgraphs: List[Subgraph]) -> List[Subgraph]:
        """去除重复的子图"""
        unique_subgraphs = []
        seen_hashes = set()
        
        for subgraph in subgraphs:
            # 使用节点集合和边集合的哈希作为唯一标识
            nodes_hash = hash(frozenset(subgraph.graph.nodes()))
            edges_hash = hash(frozenset(subgraph.graph.edges()))
            subgraph_hash = (nodes_hash, edges_hash)
            
            if subgraph_hash not in seen_hashes:
                seen_hashes.add(subgraph_hash)
                unique_subgraphs.append(subgraph)
                
        return unique_subgraphs
        
    def _validate_subgraphs(self, subgraphs: List[Subgraph]) -> List[Subgraph]:
        """验证子图的有效性"""
        valid_subgraphs = []
        
        for subgraph in subgraphs:
            # 检查连通性
            if self.ensure_connectivity and not nx.is_connected(subgraph.graph):
                continue
                
            # 检查最小节点数
            if subgraph.graph.number_of_nodes() < 2:
                continue
                
            # 检查是否有边
            if subgraph.graph.number_of_edges() < 1:
                continue
                
            valid_subgraphs.append(subgraph)
            
        return valid_subgraphs