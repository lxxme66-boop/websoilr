"""
子图采样器
实现WebSailor的核心思想：从整个知识图中抽取不同拓扑的子图作为问题候选基础
每个子图代表了一种"任务场景"：可能包含多个目标、干扰信息、隐含路径
"""

import logging
import random
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import networkx as nx
import numpy as np
from tqdm import tqdm
import community as community_louvain

logger = logging.getLogger(__name__)


class SubgraphSampler:
    """
    子图采样器
    WebSailor核心组件：通过采样不同拓扑结构的子图来创建多样化的问题场景
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_settings = config.get('data_settings', {})
        
        # 子图大小限制
        self.min_size = self.data_settings.get('min_subgraph_size', 5)
        self.max_size = self.data_settings.get('max_subgraph_size', 50)
        
        # 采样策略
        self.sampling_strategies = self.data_settings.get(
            'subgraph_sampling_strategies', 
            ['random_walk', 'bfs', 'community_based']
        )
        
        # 拓扑类型
        self.topology_types = [
            'chain',      # 链式：A->B->C->D
            'star',       # 星型：中心节点连接多个周边节点
            'tree',       # 树形：层次结构
            'cycle',      # 环形：包含循环
            'mixed',      # 混合：复杂拓扑
            'dense',      # 密集：高连接度
            'sparse'      # 稀疏：低连接度
        ]
        
    def sample_subgraphs(self, kg: nx.DiGraph, num_samples: int = 1000) -> List[nx.DiGraph]:
        """
        从知识图谱中采样多个子图
        
        Args:
            kg: 输入的知识图谱
            num_samples: 需要采样的子图数量
            
        Returns:
            List[nx.DiGraph]: 采样得到的子图列表
        """
        logger.info(f"开始从知识图谱中采样{num_samples}个子图...")
        
        subgraphs = []
        samples_per_strategy = num_samples // len(self.sampling_strategies)
        
        # 使用不同策略采样
        for strategy in self.sampling_strategies:
            logger.info(f"使用{strategy}策略采样{samples_per_strategy}个子图...")
            
            if strategy == 'random_walk':
                subgraphs.extend(self._random_walk_sampling(kg, samples_per_strategy))
            elif strategy == 'bfs':
                subgraphs.extend(self._bfs_sampling(kg, samples_per_strategy))
            elif strategy == 'community_based':
                subgraphs.extend(self._community_based_sampling(kg, samples_per_strategy))
            else:
                logger.warning(f"未知的采样策略: {strategy}")
                
        # 过滤和验证子图
        valid_subgraphs = self._filter_valid_subgraphs(subgraphs)
        
        # 为每个子图标注拓扑类型
        for subgraph in valid_subgraphs:
            topology = self._identify_topology(subgraph)
            subgraph.graph['topology'] = topology
            subgraph.graph['complexity'] = self._calculate_complexity(subgraph)
            
        logger.info(f"子图采样完成，共得到{len(valid_subgraphs)}个有效子图")
        return valid_subgraphs
        
    def _random_walk_sampling(self, kg: nx.DiGraph, num_samples: int) -> List[nx.DiGraph]:
        """
        随机游走采样
        WebSailor思想：通过随机游走创建包含隐含路径的子图
        """
        subgraphs = []
        nodes = list(kg.nodes())
        
        for _ in tqdm(range(num_samples), desc="随机游走采样"):
            # 随机选择起始节点
            start_node = random.choice(nodes)
            
            # 随机确定子图大小
            target_size = random.randint(self.min_size, self.max_size)
            
            # 执行随机游走
            visited = set()
            walk_nodes = [start_node]
            current = start_node
            visited.add(current)
            
            while len(visited) < target_size:
                # 获取邻居节点
                neighbors = list(kg.neighbors(current)) + list(kg.predecessors(current))
                unvisited_neighbors = [n for n in neighbors if n not in visited]
                
                if unvisited_neighbors:
                    # 随机选择下一个节点
                    next_node = random.choice(unvisited_neighbors)
                    walk_nodes.append(next_node)
                    visited.add(next_node)
                    current = next_node
                else:
                    # 如果没有未访问的邻居，随机跳转
                    unvisited = [n for n in nodes if n not in visited]
                    if unvisited:
                        current = random.choice(unvisited)
                        walk_nodes.append(current)
                        visited.add(current)
                    else:
                        break
                        
            # 构建子图
            subgraph = kg.subgraph(visited).copy()
            
            # 添加一些随机边以增加复杂性（WebSailor的干扰信息）
            if len(visited) > 3:
                self._add_random_edges(subgraph, probability=0.1)
                
            subgraphs.append(subgraph)
            
        return subgraphs
        
    def _bfs_sampling(self, kg: nx.DiGraph, num_samples: int) -> List[nx.DiGraph]:
        """
        广度优先采样
        WebSailor思想：创建以某个节点为中心的局部知识结构
        """
        subgraphs = []
        nodes = list(kg.nodes())
        
        for _ in tqdm(range(num_samples), desc="BFS采样"):
            # 随机选择中心节点
            center = random.choice(nodes)
            
            # 随机确定深度
            max_depth = random.randint(2, 4)
            
            # BFS遍历
            visited = {center}
            current_level = {center}
            
            for depth in range(max_depth):
                next_level = set()
                for node in current_level:
                    # 获取所有邻居
                    neighbors = set(kg.neighbors(node)) | set(kg.predecessors(node))
                    
                    # 随机选择部分邻居（控制子图大小）
                    if len(neighbors) > 5:
                        neighbors = set(random.sample(list(neighbors), 5))
                        
                    next_level.update(neighbors - visited)
                    
                visited.update(next_level)
                current_level = next_level
                
                # 检查大小限制
                if len(visited) >= self.max_size:
                    break
                    
            # 确保子图大小合适
            if len(visited) < self.min_size:
                # 添加更多节点
                remaining = list(set(nodes) - visited)
                if remaining:
                    additional = random.sample(
                        remaining, 
                        min(self.min_size - len(visited), len(remaining))
                    )
                    visited.update(additional)
                    
            # 构建子图
            subgraph = kg.subgraph(visited).copy()
            subgraphs.append(subgraph)
            
        return subgraphs
        
    def _community_based_sampling(self, kg: nx.DiGraph, num_samples: int) -> List[nx.DiGraph]:
        """
        基于社区的采样
        WebSailor思想：利用图的社区结构创建语义相关的子图
        """
        subgraphs = []
        
        # 转换为无向图进行社区检测
        undirected_kg = kg.to_undirected()
        
        # 使用Louvain算法检测社区
        try:
            partition = community_louvain.best_partition(undirected_kg)
            communities = defaultdict(list)
            
            for node, comm_id in partition.items():
                communities[comm_id].append(node)
                
            # 从每个社区采样
            samples_per_community = max(1, num_samples // len(communities))
            
            for comm_id, comm_nodes in tqdm(communities.items(), desc="社区采样"):
                for _ in range(min(samples_per_community, len(comm_nodes) // self.min_size)):
                    if len(comm_nodes) >= self.min_size:
                        # 从社区中采样节点
                        sample_size = random.randint(
                            self.min_size, 
                            min(self.max_size, len(comm_nodes))
                        )
                        sampled_nodes = random.sample(comm_nodes, sample_size)
                        
                        # 构建子图
                        subgraph = kg.subgraph(sampled_nodes).copy()
                        
                        # 添加一些跨社区的边（增加难度）
                        self._add_cross_community_edges(subgraph, kg, partition, probability=0.2)
                        
                        subgraphs.append(subgraph)
                        
        except Exception as e:
            logger.warning(f"社区检测失败: {e}，使用随机采样替代")
            # 降级到随机采样
            subgraphs.extend(self._random_walk_sampling(kg, num_samples))
            
        return subgraphs[:num_samples]
        
    def _add_random_edges(self, subgraph: nx.DiGraph, probability: float = 0.1):
        """
        添加随机边以增加复杂性
        WebSailor思想：通过添加干扰边增加推理难度
        """
        nodes = list(subgraph.nodes())
        num_nodes = len(nodes)
        
        if num_nodes < 2:
            return
            
        # 计算要添加的边数
        num_edges_to_add = int(num_nodes * (num_nodes - 1) * probability)
        
        for _ in range(num_edges_to_add):
            source = random.choice(nodes)
            target = random.choice(nodes)
            
            if source != target and not subgraph.has_edge(source, target):
                # 添加带有较低置信度的边
                subgraph.add_edge(
                    source, target,
                    type="推断",
                    confidence=0.3,
                    is_noise=True
                )
                
    def _add_cross_community_edges(self, subgraph: nx.DiGraph, kg: nx.DiGraph, 
                                   partition: Dict, probability: float = 0.2):
        """
        添加跨社区的边
        WebSailor思想：创建需要跨领域推理的复杂场景
        """
        nodes = list(subgraph.nodes())
        
        for node in nodes:
            if random.random() < probability:
                # 找到其他社区的节点
                node_community = partition.get(node, -1)
                other_nodes = [
                    n for n in kg.nodes() 
                    if partition.get(n, -1) != node_community and n not in nodes
                ]
                
                if other_nodes:
                    # 随机选择一个其他社区的节点
                    other_node = random.choice(other_nodes)
                    
                    # 检查原图中是否有路径
                    if nx.has_path(kg, node, other_node):
                        # 添加这个节点和边
                        subgraph.add_node(
                            other_node,
                            **kg.nodes[other_node]
                        )
                        subgraph.add_edge(
                            node, other_node,
                            type="跨域关联",
                            confidence=0.5,
                            is_cross_community=True
                        )
                        
    def _filter_valid_subgraphs(self, subgraphs: List[nx.DiGraph]) -> List[nx.DiGraph]:
        """
        过滤有效的子图
        确保子图满足WebSailor的要求：连通、大小合适、包含足够的信息
        """
        valid_subgraphs = []
        
        for subgraph in subgraphs:
            # 检查大小
            if self.min_size <= subgraph.number_of_nodes() <= self.max_size:
                # 检查连通性（转为无向图检查）
                undirected = subgraph.to_undirected()
                if nx.is_connected(undirected):
                    # 检查是否有足够的边
                    if subgraph.number_of_edges() >= subgraph.number_of_nodes() - 1:
                        valid_subgraphs.append(subgraph)
                        
        return valid_subgraphs
        
    def _identify_topology(self, subgraph: nx.DiGraph) -> str:
        """
        识别子图的拓扑类型
        用于后续根据不同拓扑生成不同类型的问题
        """
        num_nodes = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()
        
        # 计算度分布
        in_degrees = dict(subgraph.in_degree())
        out_degrees = dict(subgraph.out_degree())
        
        # 转为无向图计算一些指标
        undirected = subgraph.to_undirected()
        
        # 检查是否是链式
        if self._is_chain(subgraph):
            return 'chain'
            
        # 检查是否是星型
        if self._is_star(subgraph):
            return 'star'
            
        # 检查是否是树形
        if nx.is_tree(undirected):
            return 'tree'
            
        # 检查是否包含环
        if len(list(nx.simple_cycles(subgraph))) > 0:
            return 'cycle'
            
        # 根据密度判断
        density = nx.density(subgraph)
        if density > 0.5:
            return 'dense'
        elif density < 0.2:
            return 'sparse'
        else:
            return 'mixed'
            
    def _is_chain(self, subgraph: nx.DiGraph) -> bool:
        """检查是否是链式结构"""
        # 链式结构：每个节点最多一个入度和一个出度
        for node in subgraph.nodes():
            if subgraph.in_degree(node) > 1 or subgraph.out_degree(node) > 1:
                return False
                
        # 检查是否存在一条包含所有节点的路径
        undirected = subgraph.to_undirected()
        if nx.is_path_graph(undirected):
            return True
            
        return False
        
    def _is_star(self, subgraph: nx.DiGraph) -> bool:
        """检查是否是星型结构"""
        degrees = dict(subgraph.degree())
        max_degree = max(degrees.values())
        
        # 星型结构：存在一个中心节点连接大部分其他节点
        high_degree_nodes = [n for n, d in degrees.items() if d >= len(subgraph) * 0.6]
        
        return len(high_degree_nodes) == 1
        
    def _calculate_complexity(self, subgraph: nx.DiGraph) -> float:
        """
        计算子图的复杂度
        WebSailor思想：复杂度决定了问题的难度
        """
        # 考虑多个因素
        num_nodes = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()
        
        # 密度
        density = nx.density(subgraph)
        
        # 平均路径长度
        try:
            avg_path_length = nx.average_shortest_path_length(subgraph.to_undirected())
        except:
            avg_path_length = 1
            
        # 节点类型多样性
        node_types = set(nx.get_node_attributes(subgraph, 'type').values())
        type_diversity = len(node_types) / max(1, num_nodes)
        
        # 边类型多样性
        edge_types = set(nx.get_edge_attributes(subgraph, 'type').values())
        edge_diversity = len(edge_types) / max(1, num_edges)
        
        # 综合复杂度
        complexity = (
            0.2 * (num_nodes / self.max_size) +
            0.2 * density +
            0.2 * (avg_path_length / max(1, num_nodes)) +
            0.2 * type_diversity +
            0.2 * edge_diversity
        )
        
        return min(1.0, complexity)
        
    def save_subgraphs(self, subgraphs: List[nx.DiGraph], output_path: str):
        """保存采样的子图"""
        import json
        
        subgraphs_data = []
        
        for i, subgraph in enumerate(subgraphs):
            data = {
                'id': i,
                'topology': subgraph.graph.get('topology', 'unknown'),
                'complexity': subgraph.graph.get('complexity', 0),
                'num_nodes': subgraph.number_of_nodes(),
                'num_edges': subgraph.number_of_edges(),
                'nodes': [
                    {
                        'id': node,
                        'type': subgraph.nodes[node].get('type', ''),
                        'confidence': subgraph.nodes[node].get('confidence', 1.0)
                    }
                    for node in subgraph.nodes()
                ],
                'edges': [
                    {
                        'source': u,
                        'target': v,
                        'type': subgraph.edges[u, v].get('type', ''),
                        'confidence': subgraph.edges[u, v].get('confidence', 1.0),
                        'is_noise': subgraph.edges[u, v].get('is_noise', False)
                    }
                    for u, v in subgraph.edges()
                ]
            }
            subgraphs_data.append(data)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(subgraphs_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"已保存{len(subgraphs)}个子图到: {output_path}")