"""
子图采样器 - WebSailor核心模块
从知识图谱中采样不同拓扑结构的子图，每个子图代表一种"任务场景"
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Set, Tuple
import networkx as nx
from tqdm import tqdm
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class SubgraphSampler:
    """
    WebSailor核心：子图采样器
    从整个知识图中抽取不同拓扑的子图作为问题候选基础
    每个子图代表了一种"任务场景"：可能包含多个目标、干扰信息、隐含路径
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.sampling_config = config.get('subgraph_sampling', {})
        
        # 拓扑类型
        self.topology_types = self.sampling_config.get('topology_types', [])
        
        # 子图大小限制
        self.min_nodes = self.sampling_config['subgraph_sizes']['min_nodes']
        self.max_nodes = self.sampling_config['subgraph_sizes']['max_nodes']
        self.min_edges = self.sampling_config['subgraph_sizes']['min_edges']
        self.max_edges = self.sampling_config['subgraph_sizes']['max_edges']
        
        # 采样策略
        self.sampling_strategies = self.sampling_config.get('sampling_strategies', [])
        
    def sample_diverse_subgraphs(self, graph: nx.MultiDiGraph, 
                                num_samples: int) -> List[Dict]:
        """
        采样多样化的子图
        确保采样的子图具有不同的拓扑结构，增加数据集的多样性
        """
        logger.info(f"开始从知识图谱采样 {num_samples} 个子图...")
        
        subgraphs = []
        samples_per_topology = num_samples // len(self.topology_types)
        
        # 为每种拓扑类型采样
        for topology in self.topology_types:
            logger.info(f"采样 {topology} 类型子图...")
            
            topology_subgraphs = self._sample_topology_subgraphs(
                graph, topology, samples_per_topology
            )
            subgraphs.extend(topology_subgraphs)
        
        # 如果还需要更多样本，随机采样
        while len(subgraphs) < num_samples:
            strategy = random.choice(self.sampling_strategies)
            subgraph = self._sample_subgraph(graph, strategy)
            if subgraph:
                subgraphs.append(subgraph)
        
        logger.info(f"子图采样完成，共采样 {len(subgraphs)} 个子图")
        
        # 分析子图统计信息
        self._analyze_subgraphs(subgraphs)
        
        return subgraphs[:num_samples]
    
    def _sample_topology_subgraphs(self, graph: nx.MultiDiGraph, 
                                  topology: str, num_samples: int) -> List[Dict]:
        """根据特定拓扑类型采样子图"""
        subgraphs = []
        
        for _ in range(num_samples):
            if topology == "chain":
                subgraph = self._sample_chain_subgraph(graph)
            elif topology == "star":
                subgraph = self._sample_star_subgraph(graph)
            elif topology == "tree":
                subgraph = self._sample_tree_subgraph(graph)
            elif topology == "cycle":
                subgraph = self._sample_cycle_subgraph(graph)
            elif topology == "mixed":
                subgraph = self._sample_mixed_subgraph(graph)
            else:
                # 默认使用随机游走
                subgraph = self._sample_random_walk_subgraph(graph)
            
            if subgraph and self._validate_subgraph(subgraph):
                subgraphs.append(subgraph)
        
        return subgraphs
    
    def _sample_chain_subgraph(self, graph: nx.MultiDiGraph) -> Dict:
        """
        采样链式子图：A -> B -> C -> D
        适合生成需要多步推理的问题
        """
        # 随机选择起始节点
        start_node = random.choice(list(graph.nodes()))
        
        # 构建链式路径
        path = [start_node]
        current = start_node
        visited = {start_node}
        
        # 目标链长度
        target_length = random.randint(self.min_nodes, min(self.max_nodes, 6))
        
        while len(path) < target_length:
            # 获取当前节点的后继
            successors = [n for n in graph.successors(current) if n not in visited]
            
            if not successors:
                break
                
            # 选择下一个节点
            next_node = random.choice(successors)
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        # 构建子图
        subgraph = self._path_to_subgraph(graph, path)
        subgraph['topology'] = 'chain'
        subgraph['path'] = path
        
        return subgraph
    
    def _sample_star_subgraph(self, graph: nx.MultiDiGraph) -> Dict:
        """
        采样星型子图：中心节点连接多个周边节点
        适合生成比较类或聚合类问题
        """
        # 选择度数较高的节点作为中心
        node_degrees = [(n, graph.degree(n)) for n in graph.nodes()]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        
        # 从度数前20%的节点中选择
        top_nodes = node_degrees[:max(1, len(node_degrees) // 5)]
        center_node, _ = random.choice(top_nodes)
        
        # 获取中心节点的邻居
        neighbors = list(graph.predecessors(center_node)) + list(graph.successors(center_node))
        neighbors = list(set(neighbors))  # 去重
        
        # 随机选择一些邻居
        num_neighbors = min(len(neighbors), random.randint(3, self.max_nodes - 1))
        selected_neighbors = random.sample(neighbors, num_neighbors)
        
        # 构建子图
        nodes = [center_node] + selected_neighbors
        subgraph = self._nodes_to_subgraph(graph, nodes)
        subgraph['topology'] = 'star'
        subgraph['center'] = center_node
        
        return subgraph
    
    def _sample_tree_subgraph(self, graph: nx.MultiDiGraph) -> Dict:
        """
        采样树型子图：分层结构
        适合生成层次推理问题
        """
        # 选择根节点
        root = random.choice(list(graph.nodes()))
        
        # BFS构建树
        tree_nodes = [root]
        tree_edges = []
        visited = {root}
        queue = [(root, 0)]  # (node, depth)
        
        max_depth = 3
        
        while queue and len(tree_nodes) < self.max_nodes:
            current, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # 获取子节点
            children = [n for n in graph.successors(current) if n not in visited]
            
            # 随机选择一些子节点
            if children:
                num_children = min(len(children), random.randint(1, 3))
                selected_children = random.sample(children, num_children)
                
                for child in selected_children:
                    if len(tree_nodes) >= self.max_nodes:
                        break
                    tree_nodes.append(child)
                    tree_edges.append((current, child))
                    visited.add(child)
                    queue.append((child, depth + 1))
        
        # 构建子图
        subgraph = self._nodes_to_subgraph(graph, tree_nodes)
        subgraph['topology'] = 'tree'
        subgraph['root'] = root
        subgraph['tree_edges'] = tree_edges
        
        return subgraph
    
    def _sample_cycle_subgraph(self, graph: nx.MultiDiGraph) -> Dict:
        """
        采样环型子图：包含循环结构
        适合生成涉及循环依赖的问题
        """
        # 尝试找到一个简单环
        try:
            cycles = list(nx.simple_cycles(graph))
            if not cycles:
                # 如果没有环，退化为链式
                return self._sample_chain_subgraph(graph)
            
            # 选择一个合适大小的环
            suitable_cycles = [c for c in cycles if self.min_nodes <= len(c) <= self.max_nodes]
            
            if suitable_cycles:
                cycle = random.choice(suitable_cycles)
            else:
                # 选择最接近的环
                cycle = min(cycles, key=lambda c: abs(len(c) - self.min_nodes))
                
            # 可能添加一些额外节点
            nodes = list(cycle)
            for node in cycle[:2]:  # 为前两个节点添加一些邻居
                neighbors = list(graph.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in nodes and len(nodes) < self.max_nodes:
                        nodes.append(neighbor)
            
            # 构建子图
            subgraph = self._nodes_to_subgraph(graph, nodes)
            subgraph['topology'] = 'cycle'
            subgraph['cycle'] = cycle
            
            return subgraph
            
        except Exception as e:
            logger.warning(f"环型子图采样失败: {e}")
            return self._sample_chain_subgraph(graph)
    
    def _sample_mixed_subgraph(self, graph: nx.MultiDiGraph) -> Dict:
        """
        采样混合型子图：包含多种拓扑结构
        适合生成复杂的综合性问题
        """
        # 使用多种策略的组合
        strategies = ['random_walk', 'bfs_sampling', 'importance_sampling']
        strategy = random.choice(strategies)
        
        subgraph = self._sample_subgraph(graph, strategy)
        if subgraph:
            subgraph['topology'] = 'mixed'
            
        return subgraph
    
    def _sample_subgraph(self, graph: nx.MultiDiGraph, strategy: str) -> Dict:
        """根据策略采样子图"""
        if strategy == 'random_walk':
            return self._sample_random_walk_subgraph(graph)
        elif strategy == 'bfs_sampling':
            return self._sample_bfs_subgraph(graph)
        elif strategy == 'importance_sampling':
            return self._sample_importance_subgraph(graph)
        elif strategy == 'topology_guided':
            return self._sample_topology_guided_subgraph(graph)
        else:
            return self._sample_random_walk_subgraph(graph)
    
    def _sample_random_walk_subgraph(self, graph: nx.MultiDiGraph) -> Dict:
        """随机游走采样子图"""
        start_node = random.choice(list(graph.nodes()))
        
        nodes = {start_node}
        edges = []
        
        # 随机游走
        current = start_node
        walk_length = random.randint(self.min_nodes * 2, self.max_nodes * 3)
        
        for _ in range(walk_length):
            neighbors = list(graph.neighbors(current))
            if not neighbors:
                # 重新开始
                current = random.choice(list(nodes))
                continue
            
            next_node = random.choice(neighbors)
            nodes.add(next_node)
            
            # 添加边
            if graph.has_edge(current, next_node):
                edges.append((current, next_node))
            
            current = next_node
            
            if len(nodes) >= self.max_nodes:
                break
        
        return self._nodes_to_subgraph(graph, list(nodes))
    
    def _sample_bfs_subgraph(self, graph: nx.MultiDiGraph) -> Dict:
        """广度优先采样子图"""
        start_node = random.choice(list(graph.nodes()))
        
        nodes = [start_node]
        visited = {start_node}
        queue = [start_node]
        
        while queue and len(nodes) < self.max_nodes:
            current = queue.pop(0)
            
            # 获取所有邻居
            neighbors = list(set(list(graph.predecessors(current)) + 
                               list(graph.successors(current))))
            
            # 随机打乱邻居顺序
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in visited and len(nodes) < self.max_nodes:
                    nodes.append(neighbor)
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return self._nodes_to_subgraph(graph, nodes)
    
    def _sample_importance_subgraph(self, graph: nx.MultiDiGraph) -> Dict:
        """基于重要性采样子图"""
        # 计算节点重要性（使用PageRank）
        try:
            pagerank = nx.pagerank(graph)
        except:
            # 如果PageRank失败，使用度中心性
            pagerank = {n: graph.degree(n) for n in graph.nodes()}
        
        # 按重要性排序
        sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        
        # 选择重要节点
        important_nodes = [n for n, _ in sorted_nodes[:self.max_nodes // 2]]
        
        # 添加这些节点之间的连接节点
        nodes = set(important_nodes)
        for i in range(len(important_nodes)):
            for j in range(i + 1, len(important_nodes)):
                try:
                    # 尝试找到最短路径
                    path = nx.shortest_path(graph, important_nodes[i], important_nodes[j])
                    nodes.update(path[:3])  # 只添加路径的前几个节点
                except:
                    continue
                
                if len(nodes) >= self.max_nodes:
                    break
        
        return self._nodes_to_subgraph(graph, list(nodes)[:self.max_nodes])
    
    def _sample_topology_guided_subgraph(self, graph: nx.MultiDiGraph) -> Dict:
        """拓扑引导的采样"""
        # 随机选择一种拓扑类型
        topology = random.choice(self.topology_types)
        
        if topology == "chain":
            return self._sample_chain_subgraph(graph)
        elif topology == "star":
            return self._sample_star_subgraph(graph)
        elif topology == "tree":
            return self._sample_tree_subgraph(graph)
        elif topology == "cycle":
            return self._sample_cycle_subgraph(graph)
        else:
            return self._sample_mixed_subgraph(graph)
    
    def _path_to_subgraph(self, graph: nx.MultiDiGraph, path: List) -> Dict:
        """将路径转换为子图"""
        nodes = path
        edges = []
        
        for i in range(len(path) - 1):
            if graph.has_edge(path[i], path[i + 1]):
                edge_data = graph.get_edge_data(path[i], path[i + 1])
                edges.append({
                    'source': path[i],
                    'target': path[i + 1],
                    'relation': list(edge_data.values())[0].get('relation', 'unknown')
                })
        
        return self._create_subgraph_dict(graph, nodes, edges)
    
    def _nodes_to_subgraph(self, graph: nx.MultiDiGraph, nodes: List) -> Dict:
        """将节点列表转换为子图"""
        # 提取这些节点之间的所有边
        edges = []
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if graph.has_edge(node1, node2):
                    edge_data = graph.get_edge_data(node1, node2)
                    for key, data in edge_data.items():
                        edges.append({
                            'source': node1,
                            'target': node2,
                            'relation': data.get('relation', 'unknown')
                        })
                
                if graph.has_edge(node2, node1):
                    edge_data = graph.get_edge_data(node2, node1)
                    for key, data in edge_data.items():
                        edges.append({
                            'source': node2,
                            'target': node1,
                            'relation': data.get('relation', 'unknown')
                        })
        
        return self._create_subgraph_dict(graph, nodes, edges)
    
    def _create_subgraph_dict(self, graph: nx.MultiDiGraph, 
                             nodes: List, edges: List[Dict]) -> Dict:
        """创建子图字典"""
        # 获取节点属性
        node_data = []
        for node in nodes:
            data = graph.nodes[node]
            node_data.append({
                'id': node,
                'type': data.get('type', 'unknown'),
                'confidence': data.get('confidence', 1.0)
            })
        
        # 创建子图
        subgraph = {
            'nodes': node_data,
            'edges': edges,
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'node_types': [n['type'] for n in node_data],
            'relation_types': [e['relation'] for e in edges]
        }
        
        return subgraph
    
    def _validate_subgraph(self, subgraph: Dict) -> bool:
        """验证子图是否满足要求"""
        # 检查节点数量
        if not (self.min_nodes <= subgraph['num_nodes'] <= self.max_nodes):
            return False
        
        # 检查边数量
        if subgraph['num_edges'] < self.min_edges:
            return False
        
        # 检查连通性（简化检查）
        if subgraph['num_edges'] == 0 and subgraph['num_nodes'] > 1:
            return False
        
        return True
    
    def _analyze_subgraphs(self, subgraphs: List[Dict]):
        """分析子图统计信息"""
        topology_counts = defaultdict(int)
        node_counts = []
        edge_counts = []
        
        for subgraph in subgraphs:
            topology_counts[subgraph.get('topology', 'unknown')] += 1
            node_counts.append(subgraph['num_nodes'])
            edge_counts.append(subgraph['num_edges'])
        
        logger.info("子图统计信息:")
        logger.info(f"  拓扑类型分布: {dict(topology_counts)}")
        logger.info(f"  平均节点数: {np.mean(node_counts):.2f}")
        logger.info(f"  平均边数: {np.mean(edge_counts):.2f}")
        logger.info(f"  节点数范围: {min(node_counts)} - {max(node_counts)}")
        logger.info(f"  边数范围: {min(edge_counts)} - {max(edge_counts)}")
    
    def save_subgraphs(self, subgraphs: List[Dict], output_path: Path):
        """保存子图"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(subgraphs, f, ensure_ascii=False, indent=2)
        
        logger.info(f"子图已保存到: {output_path}")
    
    def load_subgraphs(self, input_path: Path) -> List[Dict]:
        """加载子图"""
        with open(input_path, 'r', encoding='utf-8') as f:
            subgraphs = json.load(f)
        
        logger.info(f"加载了 {len(subgraphs)} 个子图")
        return subgraphs