#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subgraph Sampler - 子图采样器 (WebSailor核心思想1)
从整个知识图中抽取不同拓扑的子图作为问题候选基础

WebSailor核心思想1：子图采样
- 从整个知识图中抽取不同拓扑的子图作为问题候选基础
- 每个子图代表了一种"任务场景":可能包含多个目标、干扰信息、隐含路径
- 支持多种拓扑结构：星形、路径、簇、树形等

该模块实现：
1. 星形拓扑采样：一个中心节点连接多个周边节点
2. 路径拓扑采样：线性连接的节点链
3. 簇拓扑采样：密集连接的节点群
4. 树形拓扑采样：层次结构的节点关系
"""

import logging
import random
import networkx as nx
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict, deque
import numpy as np

from .knowledge_graph_builder import KnowledgeGraph


class Subgraph:
    """子图类"""
    
    def __init__(self, graph: nx.Graph, topology_type: str, metadata: Dict[str, Any] = None):
        """
        初始化子图
        
        Args:
            graph: NetworkX图对象
            topology_type: 拓扑类型
            metadata: 元数据信息
        """
        self.graph = graph
        self.topology_type = topology_type
        self.metadata = metadata or {}
        self.nodes = list(graph.nodes())
        self.edges = list(graph.edges())
        self.num_nodes = len(self.nodes)
        self.num_edges = len(self.edges)
        
        # 计算子图特征
        self._compute_features()
    
    def _compute_features(self):
        """计算子图特征"""
        self.features = {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "density": self.num_edges / (self.num_nodes * (self.num_nodes - 1) / 2) if self.num_nodes > 1 else 0,
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.num_nodes if self.num_nodes > 0 else 0,
            "diameter": self._compute_diameter(),
            "clustering_coefficient": nx.average_clustering(self.graph) if self.num_nodes > 2 else 0
        }
    
    def _compute_diameter(self) -> int:
        """计算图的直径"""
        try:
            if nx.is_connected(self.graph):
                return nx.diameter(self.graph)
            else:
                # 对于非连通图，返回最大连通分量的直径
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                return nx.diameter(subgraph) if len(largest_cc) > 1 else 0
        except:
            return 0
    
    def get_central_nodes(self, k: int = 3) -> List[str]:
        """获取中心节点"""
        if self.num_nodes == 0:
            return []
        
        # 计算度中心性
        centrality = nx.degree_centrality(self.graph)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:k]]
    
    def get_boundary_nodes(self) -> List[str]:
        """获取边界节点（度数较小的节点）"""
        if self.num_nodes == 0:
            return []
        
        degrees = dict(self.graph.degree())
        min_degree = min(degrees.values())
        return [node for node, degree in degrees.items() if degree == min_degree]
    
    def get_paths_between_nodes(self, source: str, target: str) -> List[List[str]]:
        """获取两个节点之间的所有简单路径"""
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=5))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "topology_type": self.topology_type,
            "nodes": self.nodes,
            "edges": [(u, v, self.graph[u][v]) for u, v in self.edges],
            "features": self.features,
            "metadata": self.metadata
        }


class SubgraphSampler:
    """子图采样器 - WebSailor核心思想1"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化子图采样器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.sampling_config = config.get("subgraph_sampling", {})
        
        # 加载采样策略
        self.sampling_strategies = self._load_sampling_strategies()
        
        logging.info("子图采样器初始化完成")
    
    def _load_sampling_strategies(self) -> Dict[str, Dict[str, Any]]:
        """加载采样策略配置"""
        strategies = {}
        
        for strategy_config in self.sampling_config.get("sampling_strategies", []):
            name = strategy_config["name"]
            strategies[name] = {
                "description": strategy_config["description"],
                "weight": strategy_config["weight"],
                "parameters": strategy_config["parameters"],
                "sampler": self._get_sampler_function(name)
            }
        
        return strategies
    
    def _get_sampler_function(self, strategy_name: str):
        """获取对应的采样函数"""
        sampler_mapping = {
            "star_topology": self._sample_star_topology,
            "path_topology": self._sample_path_topology,
            "cluster_topology": self._sample_cluster_topology,
            "tree_topology": self._sample_tree_topology
        }
        
        return sampler_mapping.get(strategy_name, self._sample_random_subgraph)
    
    def sample_subgraphs(self, knowledge_graph: KnowledgeGraph, 
                        num_subgraphs: int, 
                        sampling_strategies: List[str] = None) -> List[Subgraph]:
        """
        采样子图 - WebSailor核心思想1的实现
        
        Args:
            knowledge_graph: 知识图谱
            num_subgraphs: 要采样的子图数量
            sampling_strategies: 采样策略列表
            
        Returns:
            采样得到的子图列表
        """
        logging.info(f"开始采样 {num_subgraphs} 个子图...")
        
        if not knowledge_graph.graph.nodes():
            logging.warning("知识图谱为空，无法采样子图")
            return []
        
        subgraphs = []
        
        # 根据权重分配每种策略的采样数量
        strategy_counts = self._allocate_sampling_counts(num_subgraphs, sampling_strategies)
        
        for strategy_name, count in strategy_counts.items():
            if count > 0:
                logging.info(f"使用 {strategy_name} 策略采样 {count} 个子图...")
                strategy_subgraphs = self._sample_with_strategy(
                    knowledge_graph, strategy_name, count
                )
                subgraphs.extend(strategy_subgraphs)
        
        # 随机打乱子图顺序
        random.shuffle(subgraphs)
        
        logging.info(f"子图采样完成，共采样 {len(subgraphs)} 个子图")
        return subgraphs
    
    def _allocate_sampling_counts(self, total_count: int, 
                                 strategies: List[str] = None) -> Dict[str, int]:
        """根据权重分配采样数量"""
        if strategies is None:
            strategies = list(self.sampling_strategies.keys())
        
        # 获取策略权重
        strategy_weights = {}
        total_weight = 0
        
        for strategy in strategies:
            if strategy in self.sampling_strategies:
                weight = self.sampling_strategies[strategy]["weight"]
                strategy_weights[strategy] = weight
                total_weight += weight
        
        # 按权重分配数量
        strategy_counts = {}
        allocated_count = 0
        
        for strategy, weight in strategy_weights.items():
            count = int(total_count * weight / total_weight)
            strategy_counts[strategy] = count
            allocated_count += count
        
        # 处理余数
        remaining = total_count - allocated_count
        strategies_list = list(strategy_weights.keys())
        for i in range(remaining):
            strategy = strategies_list[i % len(strategies_list)]
            strategy_counts[strategy] += 1
        
        return strategy_counts
    
    def _sample_with_strategy(self, knowledge_graph: KnowledgeGraph, 
                             strategy_name: str, count: int) -> List[Subgraph]:
        """使用指定策略采样子图"""
        if strategy_name not in self.sampling_strategies:
            logging.warning(f"未知的采样策略: {strategy_name}")
            return []
        
        strategy = self.sampling_strategies[strategy_name]
        sampler_func = strategy["sampler"]
        parameters = strategy["parameters"]
        
        subgraphs = []
        max_attempts = count * 10  # 最大尝试次数
        attempts = 0
        
        while len(subgraphs) < count and attempts < max_attempts:
            try:
                subgraph = sampler_func(knowledge_graph, parameters)
                if subgraph and subgraph.num_nodes > 0:
                    subgraphs.append(subgraph)
            except Exception as e:
                logging.warning(f"采样子图时出错: {e}")
            
            attempts += 1
        
        if len(subgraphs) < count:
            logging.warning(f"策略 {strategy_name} 只成功采样了 {len(subgraphs)}/{count} 个子图")
        
        return subgraphs
    
    def _sample_star_topology(self, knowledge_graph: KnowledgeGraph, 
                             parameters: Dict[str, Any]) -> Subgraph:
        """
        采样星形拓扑子图
        
        星形拓扑特点：
        - 一个中心节点连接多个周边节点
        - 中心节点是信息汇聚点，适合生成聚合类问题
        - 周边节点提供多样化的信息源
        """
        center_node_types = parameters.get("center_node_types", [])
        min_neighbors = parameters.get("min_neighbors", 3)
        max_neighbors = parameters.get("max_neighbors", 8)
        
        # 选择中心节点
        candidate_centers = []
        for node_name, node_data in knowledge_graph.graph.nodes(data=True):
            # 检查节点类型
            if center_node_types and node_data.get("type") not in center_node_types:
                continue
            
            # 检查度数
            degree = knowledge_graph.graph.degree(node_name)
            if degree >= min_neighbors:
                candidate_centers.append((node_name, degree))
        
        if not candidate_centers:
            return None
        
        # 选择度数较高的节点作为中心
        candidate_centers.sort(key=lambda x: x[1], reverse=True)
        center_node = candidate_centers[0][0]
        
        # 获取邻居节点
        neighbors = list(knowledge_graph.graph.neighbors(center_node))
        
        # 随机选择邻居节点
        num_neighbors = min(random.randint(min_neighbors, max_neighbors), len(neighbors))
        selected_neighbors = random.sample(neighbors, num_neighbors)
        
        # 构建星形子图
        subgraph_nodes = [center_node] + selected_neighbors
        subgraph = knowledge_graph.graph.subgraph(subgraph_nodes).copy()
        
        # 创建Subgraph对象
        metadata = {
            "center_node": center_node,
            "center_node_type": knowledge_graph.graph.nodes[center_node].get("type", "unknown"),
            "num_neighbors": len(selected_neighbors),
            "neighbor_types": [knowledge_graph.graph.nodes[node].get("type", "unknown") 
                             for node in selected_neighbors]
        }
        
        return Subgraph(subgraph, "star_topology", metadata)
    
    def _sample_path_topology(self, knowledge_graph: KnowledgeGraph, 
                             parameters: Dict[str, Any]) -> Subgraph:
        """
        采样路径拓扑子图
        
        路径拓扑特点：
        - 线性连接的节点链
        - 适合生成推理链问题和多跳查询
        - 可以包含分支，增加复杂性
        """
        min_path_length = parameters.get("min_path_length", 3)
        max_path_length = parameters.get("max_path_length", 6)
        include_branches = parameters.get("include_branches", True)
        
        # 随机选择起始节点
        nodes = list(knowledge_graph.graph.nodes())
        if not nodes:
            return None
        
        start_node = random.choice(nodes)
        
        # 使用BFS或DFS构建路径
        path_length = random.randint(min_path_length, max_path_length)
        path_nodes = self._build_path_from_node(
            knowledge_graph.graph, start_node, path_length
        )
        
        if len(path_nodes) < min_path_length:
            return None
        
        # 添加分支节点（如果启用）
        if include_branches:
            branch_nodes = self._add_branch_nodes(knowledge_graph.graph, path_nodes)
            path_nodes.extend(branch_nodes)
        
        # 构建路径子图
        subgraph = knowledge_graph.graph.subgraph(path_nodes).copy()
        
        # 创建Subgraph对象
        metadata = {
            "start_node": start_node,
            "end_node": path_nodes[-1] if len(path_nodes) > 1 else start_node,
            "path_length": len(path_nodes),
            "has_branches": include_branches and len(path_nodes) > path_length
        }
        
        return Subgraph(subgraph, "path_topology", metadata)
    
    def _sample_cluster_topology(self, knowledge_graph: KnowledgeGraph, 
                                parameters: Dict[str, Any]) -> Subgraph:
        """
        采样簇拓扑子图
        
        簇拓扑特点：
        - 密集连接的节点群
        - 节点间关系复杂，适合生成复杂推理问题
        - 包含多个目标和干扰信息
        """
        min_cluster_size = parameters.get("min_cluster_size", 4)
        max_cluster_size = parameters.get("max_cluster_size", 10)
        min_internal_edges = parameters.get("min_internal_edges", 3)
        
        # 寻找密集连接的区域
        # 使用社区检测算法找到簇
        try:
            communities = nx.community.greedy_modularity_communities(knowledge_graph.graph)
        except:
            # 如果社区检测失败，使用随机采样
            communities = [set(random.sample(list(knowledge_graph.graph.nodes()), 
                                           min(10, len(knowledge_graph.graph.nodes()))))]
        
        # 选择合适大小的社区
        suitable_communities = [
            community for community in communities 
            if min_cluster_size <= len(community) <= max_cluster_size
        ]
        
        if not suitable_communities:
            # 如果没有合适的社区，随机选择节点
            nodes = list(knowledge_graph.graph.nodes())
            cluster_size = random.randint(min_cluster_size, 
                                        min(max_cluster_size, len(nodes)))
            cluster_nodes = random.sample(nodes, cluster_size)
        else:
            community = random.choice(suitable_communities)
            cluster_nodes = list(community)
        
        # 构建簇子图
        subgraph = knowledge_graph.graph.subgraph(cluster_nodes).copy()
        
        # 检查内部边数
        if subgraph.number_of_edges() < min_internal_edges:
            return None
        
        # 创建Subgraph对象
        metadata = {
            "cluster_size": len(cluster_nodes),
            "internal_edges": subgraph.number_of_edges(),
            "density": subgraph.number_of_edges() / (len(cluster_nodes) * (len(cluster_nodes) - 1) / 2) if len(cluster_nodes) > 1 else 0
        }
        
        return Subgraph(subgraph, "cluster_topology", metadata)
    
    def _sample_tree_topology(self, knowledge_graph: KnowledgeGraph, 
                             parameters: Dict[str, Any]) -> Subgraph:
        """
        采样树形拓扑子图
        
        树形拓扑特点：
        - 层次结构的节点关系
        - 适合生成层次化推理问题
        - 有明确的根节点和叶节点
        """
        max_depth = parameters.get("max_depth", 4)
        min_branches = parameters.get("min_branches", 2)
        max_branches = parameters.get("max_branches", 5)
        
        # 随机选择根节点
        nodes = list(knowledge_graph.graph.nodes())
        if not nodes:
            return None
        
        root_node = random.choice(nodes)
        
        # 使用BFS构建树
        tree_nodes = self._build_tree_from_root(
            knowledge_graph.graph, root_node, max_depth, min_branches, max_branches
        )
        
        if len(tree_nodes) < 3:  # 至少需要根节点和两个子节点
            return None
        
        # 构建树子图
        subgraph = knowledge_graph.graph.subgraph(tree_nodes).copy()
        
        # 确保是树结构（无环）
        if not nx.is_tree(subgraph):
            # 如果有环，移除一些边使其成为树
            subgraph = self._make_tree(subgraph, root_node)
        
        # 创建Subgraph对象
        metadata = {
            "root_node": root_node,
            "tree_depth": self._compute_tree_depth(subgraph, root_node),
            "num_leaves": len([node for node in subgraph.nodes() if subgraph.degree(node) == 1 and node != root_node]),
            "branching_factor": self._compute_avg_branching_factor(subgraph, root_node)
        }
        
        return Subgraph(subgraph, "tree_topology", metadata)
    
    def _sample_random_subgraph(self, knowledge_graph: KnowledgeGraph, 
                               parameters: Dict[str, Any]) -> Subgraph:
        """随机采样子图（备用方法）"""
        nodes = list(knowledge_graph.graph.nodes())
        if not nodes:
            return None
        
        # 随机选择节点数量
        subgraph_size = random.randint(3, min(10, len(nodes)))
        selected_nodes = random.sample(nodes, subgraph_size)
        
        # 构建子图
        subgraph = knowledge_graph.graph.subgraph(selected_nodes).copy()
        
        metadata = {
            "sampling_method": "random",
            "selected_nodes": selected_nodes
        }
        
        return Subgraph(subgraph, "random", metadata)
    
    def _build_path_from_node(self, graph: nx.Graph, start_node: str, 
                             max_length: int) -> List[str]:
        """从指定节点构建路径"""
        path = [start_node]
        current_node = start_node
        visited = {start_node}
        
        for _ in range(max_length - 1):
            neighbors = [n for n in graph.neighbors(current_node) if n not in visited]
            if not neighbors:
                break
            
            next_node = random.choice(neighbors)
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
        
        return path
    
    def _add_branch_nodes(self, graph: nx.Graph, path_nodes: List[str]) -> List[str]:
        """为路径添加分支节点"""
        branch_nodes = []
        
        for node in path_nodes[:-1]:  # 不为最后一个节点添加分支
            neighbors = [n for n in graph.neighbors(node) if n not in path_nodes]
            if neighbors and random.random() < 0.3:  # 30%概率添加分支
                branch_node = random.choice(neighbors)
                branch_nodes.append(branch_node)
        
        return branch_nodes
    
    def _build_tree_from_root(self, graph: nx.Graph, root: str, max_depth: int, 
                             min_branches: int, max_branches: int) -> List[str]:
        """从根节点构建树"""
        tree_nodes = [root]
        queue = deque([(root, 0)])  # (node, depth)
        visited = {root}
        
        while queue:
            current_node, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # 获取未访问的邻居
            neighbors = [n for n in graph.neighbors(current_node) if n not in visited]
            
            if neighbors:
                # 随机选择分支数量
                num_branches = random.randint(
                    min(min_branches, len(neighbors)), 
                    min(max_branches, len(neighbors))
                )
                
                selected_neighbors = random.sample(neighbors, num_branches)
                
                for neighbor in selected_neighbors:
                    tree_nodes.append(neighbor)
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        return tree_nodes
    
    def _make_tree(self, graph: nx.Graph, root: str) -> nx.Graph:
        """将图转换为以root为根的树"""
        tree = nx.Graph()
        tree.add_node(root)
        visited = {root}
        queue = deque([root])
        
        while queue:
            current = queue.popleft()
            
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    tree.add_edge(current, neighbor)
                    # 复制节点和边的属性
                    if neighbor in graph.nodes:
                        tree.nodes[neighbor].update(graph.nodes[neighbor])
                    if graph.has_edge(current, neighbor):
                        tree.edges[current, neighbor].update(graph.edges[current, neighbor])
                    
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return tree
    
    def _compute_tree_depth(self, tree: nx.Graph, root: str) -> int:
        """计算树的深度"""
        if not tree.nodes():
            return 0
        
        max_depth = 0
        queue = deque([(root, 0)])
        visited = {root}
        
        while queue:
            node, depth = queue.popleft()
            max_depth = max(max_depth, depth)
            
            for neighbor in tree.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        return max_depth
    
    def _compute_avg_branching_factor(self, tree: nx.Graph, root: str) -> float:
        """计算平均分支因子"""
        if tree.number_of_nodes() <= 1:
            return 0
        
        internal_nodes = [node for node in tree.nodes() if tree.degree(node) > 1]
        if not internal_nodes:
            return 0
        
        total_branches = sum(tree.degree(node) - 1 for node in internal_nodes if node != root)
        total_branches += tree.degree(root)  # 根节点的所有边都是分支
        
        return total_branches / len(internal_nodes) if internal_nodes else 0