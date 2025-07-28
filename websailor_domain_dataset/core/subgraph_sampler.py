"""
子图采样器 - WebSailor核心思想实现
从整个知识图中抽取不同拓扑的子图作为问题候选基础
每个子图代表了一种"任务场景"：可能包含多个目标、干扰信息、隐含路径
"""

import logging
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Set
import networkx as nx
from collections import defaultdict, deque
import community as community_louvain

logger = logging.getLogger(__name__)

class SubgraphSampler:
    """
    子图采样器 - WebSailor的核心组件
    
    核心思想：
    1. 从整个知识图中抽取不同拓扑结构的子图
    2. 每个子图代表一种"任务场景"，包含：
       - 多个目标实体和关系
       - 干扰信息和噪声节点
       - 隐含的推理路径
    3. 支持多种采样策略以获得多样化的子图结构
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sampling_strategies = config['sampling_strategies']
        self.min_size = config['min_subgraph_size']
        self.max_size = config['max_subgraph_size']
        self.overlap_ratio = config['overlap_ratio']
        self.diversity_threshold = config['diversity_threshold']
        
        # 子图质量评估权重
        self.quality_weights = {
            'connectivity': 0.3,    # 连通性
            'diversity': 0.25,      # 节点类型多样性
            'centrality': 0.2,      # 中心性分布
            'path_complexity': 0.25 # 路径复杂度
        }
        
    def sample_subgraphs(self, knowledge_graph: nx.Graph, num_subgraphs: int) -> List[Dict[str, Any]]:
        """
        采样多个不同拓扑的子图
        
        Args:
            knowledge_graph: 完整的知识图谱
            num_subgraphs: 需要采样的子图数量
            
        Returns:
            List[Dict]: 子图列表，每个包含图结构和元数据
        """
        logger.info(f"开始采样 {num_subgraphs} 个子图，知识图谱节点数: {len(knowledge_graph.nodes)}")
        
        subgraphs = []
        sampled_node_sets = []  # 记录已采样的节点集合，避免重复
        
        # 预计算图的全局特征
        global_features = self._compute_global_features(knowledge_graph)
        
        for i in range(num_subgraphs):
            # 随机选择采样策略
            strategy = random.choice(self.sampling_strategies)
            
            # 采样子图
            subgraph_data = self._sample_single_subgraph(
                knowledge_graph, strategy, global_features, sampled_node_sets
            )
            
            if subgraph_data:
                subgraphs.append(subgraph_data)
                sampled_node_sets.append(set(subgraph_data['nodes']))
                
                if (i + 1) % 100 == 0:
                    logger.info(f"已采样 {i + 1}/{num_subgraphs} 个子图")
        
        logger.info(f"子图采样完成，成功采样 {len(subgraphs)} 个子图")
        return subgraphs
    
    def _sample_single_subgraph(self, knowledge_graph: nx.Graph, strategy: str, 
                               global_features: Dict, sampled_node_sets: List[Set]) -> Dict[str, Any]:
        """采样单个子图"""
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                # 根据策略采样节点
                if strategy == "random_walk":
                    nodes = self._random_walk_sampling(knowledge_graph)
                elif strategy == "breadth_first":
                    nodes = self._breadth_first_sampling(knowledge_graph)
                elif strategy == "depth_first":
                    nodes = self._depth_first_sampling(knowledge_graph)
                elif strategy == "community_based":
                    nodes = self._community_based_sampling(knowledge_graph, global_features)
                elif strategy == "centrality_based":
                    nodes = self._centrality_based_sampling(knowledge_graph, global_features)
                else:
                    nodes = self._random_walk_sampling(knowledge_graph)
                
                if not nodes or len(nodes) < self.min_size:
                    continue
                
                # 检查与已有子图的重叠度
                node_set = set(nodes)
                if self._check_overlap(node_set, sampled_node_sets):
                    continue
                
                # 构建子图
                subgraph = knowledge_graph.subgraph(nodes).copy()
                
                # 确保子图连通
                if not nx.is_connected(subgraph):
                    # 选择最大连通分量
                    largest_cc = max(nx.connected_components(subgraph), key=len)
                    subgraph = subgraph.subgraph(largest_cc).copy()
                    nodes = list(subgraph.nodes())
                
                # 评估子图质量
                quality_score = self._evaluate_subgraph_quality(subgraph, knowledge_graph)
                
                if quality_score < 0.3:  # 质量阈值
                    continue
                
                # 添加任务场景特征
                scenario_features = self._extract_scenario_features(subgraph, knowledge_graph)
                
                return {
                    'subgraph': subgraph,
                    'nodes': nodes,
                    'edges': list(subgraph.edges(data=True)),
                    'strategy': strategy,
                    'quality_score': quality_score,
                    'scenario_features': scenario_features,
                    'size': len(nodes),
                    'num_edges': len(subgraph.edges()),
                    'metadata': {
                        'node_types': self._get_node_type_distribution(subgraph),
                        'relation_types': self._get_relation_type_distribution(subgraph),
                        'centrality_stats': self._compute_centrality_stats(subgraph),
                        'path_stats': self._compute_path_stats(subgraph)
                    }
                }
                
            except Exception as e:
                logger.warning(f"采样第 {attempt + 1} 次尝试失败: {e}")
                continue
        
        logger.warning(f"使用策略 {strategy} 采样失败")
        return None
    
    def _random_walk_sampling(self, graph: nx.Graph) -> List[str]:
        """随机游走采样 - 获得连续的路径结构"""
        if len(graph.nodes()) == 0:
            return []
        
        # 随机选择起始节点
        start_node = random.choice(list(graph.nodes()))
        nodes = [start_node]
        current_node = start_node
        
        target_size = random.randint(self.min_size, min(self.max_size, len(graph.nodes())))
        
        for _ in range(target_size - 1):
            neighbors = list(graph.neighbors(current_node))
            if not neighbors:
                break
            
            # 带权重的随机选择（偏向度数高的节点）
            weights = [graph.degree(n) + 1 for n in neighbors]
            next_node = random.choices(neighbors, weights=weights)[0]
            
            if next_node not in nodes:
                nodes.append(next_node)
                current_node = next_node
            else:
                # 如果回到已访问节点，随机跳转
                current_node = random.choice(nodes)
        
        return nodes
    
    def _breadth_first_sampling(self, graph: nx.Graph) -> List[str]:
        """广度优先采样 - 获得扇形扩展结构"""
        if len(graph.nodes()) == 0:
            return []
        
        start_node = random.choice(list(graph.nodes()))
        visited = {start_node}
        queue = deque([start_node])
        nodes = [start_node]
        
        target_size = random.randint(self.min_size, min(self.max_size, len(graph.nodes())))
        
        while queue and len(nodes) < target_size:
            current = queue.popleft()
            neighbors = list(graph.neighbors(current))
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in visited and len(nodes) < target_size:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    nodes.append(neighbor)
        
        return nodes
    
    def _depth_first_sampling(self, graph: nx.Graph) -> List[str]:
        """深度优先采样 - 获得深层路径结构"""
        if len(graph.nodes()) == 0:
            return []
        
        start_node = random.choice(list(graph.nodes()))
        visited = {start_node}
        stack = [start_node]
        nodes = [start_node]
        
        target_size = random.randint(self.min_size, min(self.max_size, len(graph.nodes())))
        
        while stack and len(nodes) < target_size:
            current = stack.pop()
            neighbors = list(graph.neighbors(current))
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in visited and len(nodes) < target_size:
                    visited.add(neighbor)
                    stack.append(neighbor)
                    nodes.append(neighbor)
                    break  # 深度优先，只选择一个邻居
        
        return nodes
    
    def _community_based_sampling(self, graph: nx.Graph, global_features: Dict) -> List[str]:
        """基于社区的采样 - 获得密集连接的子结构"""
        communities = global_features.get('communities', {})
        if not communities:
            return self._random_walk_sampling(graph)
        
        # 随机选择一个社区
        community_id = random.choice(list(communities.keys()))
        community_nodes = communities[community_id]
        
        # 在社区内采样
        target_size = random.randint(self.min_size, min(self.max_size, len(community_nodes)))
        sampled_nodes = random.sample(community_nodes, target_size)
        
        # 可能添加一些社区外的节点作为干扰
        if random.random() < 0.3:  # 30%概率添加干扰节点
            other_nodes = [n for n in graph.nodes() if n not in community_nodes]
            if other_nodes:
                num_noise = random.randint(1, min(3, len(other_nodes)))
                noise_nodes = random.sample(other_nodes, num_noise)
                sampled_nodes.extend(noise_nodes)
        
        return sampled_nodes
    
    def _centrality_based_sampling(self, graph: nx.Graph, global_features: Dict) -> List[str]:
        """基于中心性的采样 - 获得包含重要节点的子图"""
        centrality = global_features.get('centrality', {})
        if not centrality:
            return self._random_walk_sampling(graph)
        
        # 按中心性排序
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        # 选择高中心性节点作为核心
        num_core = random.randint(1, 3)
        core_nodes = [node for node, _ in sorted_nodes[:num_core]]
        
        # 围绕核心节点扩展
        nodes = core_nodes.copy()
        target_size = random.randint(self.min_size, min(self.max_size, len(graph.nodes())))
        
        for core_node in core_nodes:
            neighbors = list(graph.neighbors(core_node))
            if neighbors:
                # 添加一些邻居节点
                num_neighbors = random.randint(1, min(4, len(neighbors)))
                selected_neighbors = random.sample(neighbors, num_neighbors)
                for neighbor in selected_neighbors:
                    if neighbor not in nodes and len(nodes) < target_size:
                        nodes.append(neighbor)
        
        return nodes
    
    def _compute_global_features(self, graph: nx.Graph) -> Dict[str, Any]:
        """计算图的全局特征"""
        features = {}
        
        try:
            # 计算中心性
            features['centrality'] = nx.betweenness_centrality(graph)
            
            # 检测社区
            if len(graph.edges()) > 0:
                partition = community_louvain.best_partition(graph)
                communities = defaultdict(list)
                for node, comm_id in partition.items():
                    communities[comm_id].append(node)
                features['communities'] = dict(communities)
            else:
                features['communities'] = {}
                
        except Exception as e:
            logger.warning(f"计算全局特征失败: {e}")
            features['centrality'] = {}
            features['communities'] = {}
        
        return features
    
    def _check_overlap(self, node_set: Set[str], sampled_node_sets: List[Set[str]]) -> bool:
        """检查与已有子图的重叠度"""
        for existing_set in sampled_node_sets:
            overlap = len(node_set.intersection(existing_set))
            overlap_ratio = overlap / min(len(node_set), len(existing_set))
            if overlap_ratio > self.overlap_ratio:
                return True
        return False
    
    def _evaluate_subgraph_quality(self, subgraph: nx.Graph, full_graph: nx.Graph) -> float:
        """评估子图质量"""
        if len(subgraph.nodes()) == 0:
            return 0.0
        
        scores = {}
        
        # 连通性评分
        if nx.is_connected(subgraph):
            scores['connectivity'] = 1.0
        else:
            largest_cc_size = len(max(nx.connected_components(subgraph), key=len))
            scores['connectivity'] = largest_cc_size / len(subgraph.nodes())
        
        # 节点类型多样性
        node_types = [subgraph.nodes[node].get('type', 'unknown') for node in subgraph.nodes()]
        unique_types = len(set(node_types))
        scores['diversity'] = min(unique_types / 5, 1.0)  # 假设最多5种类型
        
        # 中心性分布
        if len(subgraph.edges()) > 0:
            centrality = nx.betweenness_centrality(subgraph)
            centrality_std = np.std(list(centrality.values()))
            scores['centrality'] = min(centrality_std, 1.0)
        else:
            scores['centrality'] = 0.0
        
        # 路径复杂度
        if nx.is_connected(subgraph):
            avg_path_length = nx.average_shortest_path_length(subgraph)
            scores['path_complexity'] = min(avg_path_length / 5, 1.0)
        else:
            scores['path_complexity'] = 0.0
        
        # 加权总分
        total_score = sum(
            scores[key] * self.quality_weights[key] 
            for key in scores.keys()
        )
        
        return total_score
    
    def _extract_scenario_features(self, subgraph: nx.Graph, full_graph: nx.Graph) -> Dict[str, Any]:
        """提取任务场景特征 - WebSailor的核心思想"""
        features = {
            'potential_targets': [],      # 潜在目标实体
            'interference_nodes': [],     # 干扰节点
            'implicit_paths': [],         # 隐含路径
            'multi_hop_relations': [],    # 多跳关系
            'ambiguous_entities': []      # 模糊实体（用于后续模糊化处理）
        }
        
        # 识别潜在目标实体（高度数节点）
        degrees = dict(subgraph.degree())
        high_degree_nodes = [node for node, degree in degrees.items() if degree >= 2]
        features['potential_targets'] = high_degree_nodes[:3]  # 最多3个目标
        
        # 识别干扰节点（度数为1的叶子节点）
        leaf_nodes = [node for node, degree in degrees.items() if degree == 1]
        features['interference_nodes'] = leaf_nodes
        
        # 识别隐含路径
        if len(subgraph.nodes()) >= 3:
            nodes = list(subgraph.nodes())
            for i in range(min(5, len(nodes))):
                for j in range(i+2, min(i+5, len(nodes))):
                    if nx.has_path(subgraph, nodes[i], nodes[j]):
                        path = nx.shortest_path(subgraph, nodes[i], nodes[j])
                        if len(path) >= 3:  # 至少3跳
                            features['implicit_paths'].append(path)
        
        # 识别多跳关系
        for edge in subgraph.edges(data=True):
            source, target, data = edge
            relation_type = data.get('relation', '')
            if relation_type:
                # 查找2跳关系
                for intermediate in subgraph.neighbors(target):
                    if intermediate != source and subgraph.has_edge(intermediate, target):
                        two_hop_relation = f"{source}-{relation_type}->{target}-{subgraph[target][intermediate].get('relation', '')}->{intermediate}"
                        features['multi_hop_relations'].append(two_hop_relation)
        
        # 识别模糊实体（同类型的多个实体）
        entity_types = defaultdict(list)
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node].get('type', 'unknown')
            entity_types[node_type].append(node)
        
        for entity_type, entities in entity_types.items():
            if len(entities) >= 2:
                features['ambiguous_entities'].extend(entities)
        
        return features
    
    def _get_node_type_distribution(self, subgraph: nx.Graph) -> Dict[str, int]:
        """获取节点类型分布"""
        type_count = defaultdict(int)
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node].get('type', 'unknown')
            type_count[node_type] += 1
        return dict(type_count)
    
    def _get_relation_type_distribution(self, subgraph: nx.Graph) -> Dict[str, int]:
        """获取关系类型分布"""
        relation_count = defaultdict(int)
        for _, _, data in subgraph.edges(data=True):
            relation_type = data.get('relation', 'unknown')
            relation_count[relation_type] += 1
        return dict(relation_count)
    
    def _compute_centrality_stats(self, subgraph: nx.Graph) -> Dict[str, float]:
        """计算中心性统计"""
        if len(subgraph.edges()) == 0:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0}
        
        try:
            centrality = nx.betweenness_centrality(subgraph)
            values = list(centrality.values())
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values)
            }
        except:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0}
    
    def _compute_path_stats(self, subgraph: nx.Graph) -> Dict[str, float]:
        """计算路径统计"""
        if not nx.is_connected(subgraph):
            return {'avg_path_length': 0.0, 'diameter': 0.0}
        
        try:
            avg_path_length = nx.average_shortest_path_length(subgraph)
            diameter = nx.diameter(subgraph)
            return {
                'avg_path_length': avg_path_length,
                'diameter': diameter
            }
        except:
            return {'avg_path_length': 0.0, 'diameter': 0.0}