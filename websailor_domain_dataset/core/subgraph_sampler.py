"""
子图采样器 - WebSailor核心思想实现
从整个知识图中抽取不同拓扑的子图作为问题候选基础
每个子图代表了一种"任务场景"：可能包含多个目标、干扰信息、隐含路径
"""

import random
import logging
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import networkx as nx
import numpy as np
from dataclasses import dataclass


@dataclass
class Subgraph:
    """子图数据结构"""
    graph: nx.Graph
    topology_type: str  # star, chain, tree, cycle等
    central_nodes: List[str]  # 中心节点
    task_scenario: str  # 任务场景类型
    complexity_level: str  # 复杂度级别
    metadata: Dict[str, Any]  # 其他元数据


class SubgraphSampler:
    """
    WebSailor子图采样器
    核心思想：通过采样不同拓扑结构的子图来创建多样化的任务场景
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.sampling_strategies = config.get('sampling_strategies', [])
        self.total_subgraphs = config.get('total_subgraphs', 1000)
        self.ensure_coverage = config.get('ensure_coverage', True)
        self.balance_topology = config.get('balance_topology', True)
        
    def sample_subgraphs(self, knowledge_graph: nx.Graph) -> List[Subgraph]:
        """
        主采样函数：从知识图谱中采样子图
        
        Args:
            knowledge_graph: 完整的知识图谱
            
        Returns:
            采样得到的子图列表
        """
        self.logger.info(f"开始从{len(knowledge_graph.nodes)}个节点的图中采样子图...")
        
        subgraphs = []
        strategy_weights = [s['weight'] for s in self.sampling_strategies]
        
        # 确保覆盖所有节点
        if self.ensure_coverage:
            covered_nodes = set()
            target_coverage = 0.95  # 目标覆盖率
        
        # 按策略权重分配子图数量
        strategy_counts = self._distribute_counts(self.total_subgraphs, strategy_weights)
        
        for strategy, count in zip(self.sampling_strategies, strategy_counts):
            strategy_name = strategy['name']
            self.logger.info(f"使用{strategy_name}策略采样{count}个子图")
            
            if strategy_name == 'topology_based':
                strategy_subgraphs = self._topology_based_sampling(
                    knowledge_graph, count, strategy['params']
                )
            elif strategy_name == 'semantic_based':
                strategy_subgraphs = self._semantic_based_sampling(
                    knowledge_graph, count, strategy['params']
                )
            elif strategy_name == 'task_oriented':
                strategy_subgraphs = self._task_oriented_sampling(
                    knowledge_graph, count, strategy['params']
                )
            else:
                self.logger.warning(f"未知的采样策略: {strategy_name}")
                continue
                
            subgraphs.extend(strategy_subgraphs)
            
            # 更新覆盖的节点
            if self.ensure_coverage:
                for subgraph in strategy_subgraphs:
                    covered_nodes.update(subgraph.graph.nodes())
        
        # 检查覆盖率
        if self.ensure_coverage:
            coverage = len(covered_nodes) / len(knowledge_graph.nodes)
            self.logger.info(f"节点覆盖率: {coverage:.2%}")
            
            # 如果覆盖率不足，补充采样
            if coverage < target_coverage:
                additional_subgraphs = self._ensure_coverage_sampling(
                    knowledge_graph, covered_nodes, 
                    int(self.total_subgraphs * 0.1)  # 额外10%的子图
                )
                subgraphs.extend(additional_subgraphs)
        
        self.logger.info(f"总共采样了{len(subgraphs)}个子图")
        return subgraphs
    
    def _topology_based_sampling(self, graph: nx.Graph, count: int, 
                                params: Dict[str, Any]) -> List[Subgraph]:
        """基于拓扑结构的采样"""
        subgraphs = []
        patterns = params.get('include_patterns', ['star', 'chain', 'tree', 'cycle'])
        min_nodes = params.get('min_nodes', 3)
        max_nodes = params.get('max_nodes', 15)
        hop_limit = params.get('hop_limit', 3)
        
        # 平均分配每种拓扑模式
        pattern_counts = self._distribute_counts(count, [1] * len(patterns))
        
        for pattern, pattern_count in zip(patterns, pattern_counts):
            for _ in range(pattern_count):
                if pattern == 'star':
                    subgraph = self._sample_star_subgraph(
                        graph, min_nodes, max_nodes
                    )
                elif pattern == 'chain':
                    subgraph = self._sample_chain_subgraph(
                        graph, min_nodes, max_nodes
                    )
                elif pattern == 'tree':
                    subgraph = self._sample_tree_subgraph(
                        graph, min_nodes, max_nodes, hop_limit
                    )
                elif pattern == 'cycle':
                    subgraph = self._sample_cycle_subgraph(
                        graph, min_nodes, max_nodes
                    )
                else:
                    continue
                    
                if subgraph:
                    subgraphs.append(subgraph)
        
        return subgraphs
    
    def _sample_star_subgraph(self, graph: nx.Graph, 
                             min_nodes: int, max_nodes: int) -> Subgraph:
        """采样星形子图：一个中心节点连接多个周边节点"""
        # 选择度数较高的节点作为中心
        node_degrees = [(n, d) for n, d in graph.degree()]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        
        # 从前20%的高度节点中随机选择
        top_nodes = node_degrees[:max(1, len(node_degrees) // 5)]
        center = random.choice(top_nodes)[0]
        
        # 获取邻居节点
        neighbors = list(graph.neighbors(center))
        if len(neighbors) < min_nodes - 1:
            return None
            
        # 随机选择邻居
        num_neighbors = random.randint(
            min(min_nodes - 1, len(neighbors)),
            min(max_nodes - 1, len(neighbors))
        )
        selected_neighbors = random.sample(neighbors, num_neighbors)
        
        # 构建子图
        nodes = [center] + selected_neighbors
        subgraph = graph.subgraph(nodes).copy()
        
        return Subgraph(
            graph=subgraph,
            topology_type='star',
            central_nodes=[center],
            task_scenario='中心辐射型查询',
            complexity_level=self._estimate_complexity(subgraph),
            metadata={'center_degree': graph.degree(center)}
        )
    
    def _sample_chain_subgraph(self, graph: nx.Graph,
                              min_nodes: int, max_nodes: int) -> Subgraph:
        """采样链式子图：节点形成链状连接"""
        # 随机选择起始节点
        start_node = random.choice(list(graph.nodes()))
        
        # 通过随机游走构建链
        chain = [start_node]
        current = start_node
        visited = {start_node}
        
        chain_length = random.randint(min_nodes, max_nodes)
        
        for _ in range(chain_length - 1):
            neighbors = [n for n in graph.neighbors(current) if n not in visited]
            if not neighbors:
                break
                
            next_node = random.choice(neighbors)
            chain.append(next_node)
            visited.add(next_node)
            current = next_node
        
        if len(chain) < min_nodes:
            return None
            
        # 构建子图
        subgraph = graph.subgraph(chain).copy()
        
        # 确保是链状结构，移除多余的边
        chain_edges = [(chain[i], chain[i+1]) for i in range(len(chain)-1)]
        chain_graph = nx.Graph()
        chain_graph.add_nodes_from(chain)
        chain_graph.add_edges_from(chain_edges)
        
        # 复制节点属性
        for node in chain:
            chain_graph.nodes[node].update(graph.nodes[node])
        
        return Subgraph(
            graph=chain_graph,
            topology_type='chain',
            central_nodes=[chain[0], chain[-1]],  # 起点和终点
            task_scenario='序列推理型查询',
            complexity_level=self._estimate_complexity(chain_graph),
            metadata={'chain_length': len(chain)}
        )
    
    def _sample_tree_subgraph(self, graph: nx.Graph,
                             min_nodes: int, max_nodes: int,
                             hop_limit: int) -> Subgraph:
        """采样树形子图：从根节点开始的树状结构"""
        # 选择根节点
        root = random.choice(list(graph.nodes()))
        
        # BFS构建树
        tree_nodes = {root}
        tree_edges = []
        queue = [(root, 0)]  # (node, depth)
        
        while queue and len(tree_nodes) < max_nodes:
            current, depth = queue.pop(0)
            
            if depth >= hop_limit:
                continue
                
            neighbors = list(graph.neighbors(current))
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in tree_nodes and len(tree_nodes) < max_nodes:
                    tree_nodes.add(neighbor)
                    tree_edges.append((current, neighbor))
                    queue.append((neighbor, depth + 1))
        
        if len(tree_nodes) < min_nodes:
            return None
            
        # 构建树形子图
        tree_graph = nx.Graph()
        tree_graph.add_nodes_from(tree_nodes)
        tree_graph.add_edges_from(tree_edges)
        
        # 复制节点属性
        for node in tree_nodes:
            tree_graph.nodes[node].update(graph.nodes[node])
            
        return Subgraph(
            graph=tree_graph,
            topology_type='tree',
            central_nodes=[root],
            task_scenario='层次探索型查询',
            complexity_level=self._estimate_complexity(tree_graph),
            metadata={'tree_depth': hop_limit, 'root': root}
        )
    
    def _sample_cycle_subgraph(self, graph: nx.Graph,
                              min_nodes: int, max_nodes: int) -> Subgraph:
        """采样环形子图：包含循环的子图"""
        # 寻找图中的环
        try:
            cycles = nx.cycle_basis(graph)
            if not cycles:
                return None
                
            # 选择合适大小的环
            suitable_cycles = [c for c in cycles 
                             if min_nodes <= len(c) <= max_nodes]
            
            if not suitable_cycles:
                # 如果没有合适的环，尝试扩展小环
                small_cycles = [c for c in cycles if len(c) < min_nodes]
                if small_cycles:
                    cycle = random.choice(small_cycles)
                    # 扩展环
                    expanded = self._expand_cycle(graph, cycle, min_nodes, max_nodes)
                    if expanded:
                        cycle = expanded
                    else:
                        return None
                else:
                    return None
            else:
                cycle = random.choice(suitable_cycles)
            
            # 构建子图
            subgraph = graph.subgraph(cycle).copy()
            
            return Subgraph(
                graph=subgraph,
                topology_type='cycle',
                central_nodes=cycle[:2],  # 环上的前两个节点
                task_scenario='循环依赖型查询',
                complexity_level=self._estimate_complexity(subgraph),
                metadata={'cycle_length': len(cycle)}
            )
            
        except:
            return None
    
    def _semantic_based_sampling(self, graph: nx.Graph, count: int,
                                params: Dict[str, Any]) -> List[Subgraph]:
        """基于语义相关性的采样"""
        subgraphs = []
        coherence_threshold = params.get('coherence_threshold', 0.7)
        
        for _ in range(count):
            # 选择种子节点
            seed = random.choice(list(graph.nodes()))
            
            # 通过语义相似度扩展
            subgraph_nodes = self._expand_by_semantics(
                graph, seed, coherence_threshold
            )
            
            if len(subgraph_nodes) >= 3:  # 至少3个节点
                subgraph = graph.subgraph(subgraph_nodes).copy()
                
                subgraphs.append(Subgraph(
                    graph=subgraph,
                    topology_type='semantic_cluster',
                    central_nodes=[seed],
                    task_scenario='语义关联型查询',
                    complexity_level=self._estimate_complexity(subgraph),
                    metadata={'coherence_score': coherence_threshold}
                ))
        
        return subgraphs
    
    def _task_oriented_sampling(self, graph: nx.Graph, count: int,
                               params: Dict[str, Any]) -> List[Subgraph]:
        """面向任务的采样"""
        subgraphs = []
        task_types = params.get('task_types', [])
        complexity_levels = params.get('complexity_levels', ['简单', '中等', '复杂'])
        
        # 为每种任务类型分配数量
        task_counts = self._distribute_counts(count, [1] * len(task_types))
        
        for task_type, task_count in zip(task_types, task_counts):
            for _ in range(task_count):
                complexity = random.choice(complexity_levels)
                
                if task_type == '故障诊断':
                    subgraph = self._sample_fault_diagnosis_subgraph(
                        graph, complexity
                    )
                elif task_type == '工艺优化':
                    subgraph = self._sample_process_optimization_subgraph(
                        graph, complexity
                    )
                elif task_type == '质量控制':
                    subgraph = self._sample_quality_control_subgraph(
                        graph, complexity
                    )
                elif task_type == '设备维护':
                    subgraph = self._sample_maintenance_subgraph(
                        graph, complexity
                    )
                else:
                    continue
                    
                if subgraph:
                    subgraphs.append(subgraph)
        
        return subgraphs
    
    def _sample_fault_diagnosis_subgraph(self, graph: nx.Graph,
                                       complexity: str) -> Subgraph:
        """采样故障诊断相关的子图"""
        # 寻找包含"故障"、"原因"、"解决"等关键词的节点
        fault_nodes = [n for n in graph.nodes() 
                      if any(kw in str(n) for kw in ['故障', '异常', '错误', '问题'])]
        
        if not fault_nodes:
            return None
            
        # 选择故障节点
        fault_node = random.choice(fault_nodes)
        
        # 根据复杂度确定子图大小
        size_map = {'简单': 5, '中等': 10, '复杂': 15}
        max_size = size_map.get(complexity, 10)
        
        # 扩展子图：包含原因、影响、解决方案
        subgraph_nodes = self._expand_fault_subgraph(
            graph, fault_node, max_size
        )
        
        subgraph = graph.subgraph(subgraph_nodes).copy()
        
        return Subgraph(
            graph=subgraph,
            topology_type='fault_diagnosis',
            central_nodes=[fault_node],
            task_scenario='故障诊断',
            complexity_level=complexity,
            metadata={'fault_type': fault_node}
        )
    
    def _expand_fault_subgraph(self, graph: nx.Graph, fault_node: str,
                              max_size: int) -> Set[str]:
        """扩展故障诊断子图"""
        nodes = {fault_node}
        
        # 优先添加因果关系节点
        for neighbor in graph.neighbors(fault_node):
            edge_data = graph.get_edge_data(fault_node, neighbor, {})
            relation = edge_data.get('relation', '')
            
            if any(kw in relation for kw in ['导致', '原因', '引起', '造成']):
                nodes.add(neighbor)
            elif any(kw in relation for kw in ['解决', '修复', '处理', '维修']):
                nodes.add(neighbor)
        
        # 如果节点不够，通过BFS扩展
        if len(nodes) < max_size:
            queue = list(nodes)
            visited = set(nodes)
            
            while queue and len(nodes) < max_size:
                current = queue.pop(0)
                for neighbor in graph.neighbors(current):
                    if neighbor not in visited and len(nodes) < max_size:
                        nodes.add(neighbor)
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        return nodes
    
    def _estimate_complexity(self, subgraph: nx.Graph) -> str:
        """估计子图的复杂度"""
        num_nodes = len(subgraph.nodes())
        num_edges = len(subgraph.edges())
        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
        
        # 基于节点数、边数和平均度数判断复杂度
        if num_nodes <= 5 and avg_degree <= 2:
            return '简单'
        elif num_nodes <= 10 and avg_degree <= 3:
            return '中等'
        else:
            return '复杂'
    
    def _distribute_counts(self, total: int, weights: List[float]) -> List[int]:
        """根据权重分配数量"""
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        counts = (weights * total).astype(int)
        
        # 处理舍入误差
        diff = total - counts.sum()
        if diff > 0:
            # 将剩余的分配给权重最大的
            indices = np.argsort(weights)[::-1]
            for i in range(diff):
                counts[indices[i % len(indices)]] += 1
        
        return counts.tolist()
    
    def _expand_by_semantics(self, graph: nx.Graph, seed: str,
                           threshold: float) -> Set[str]:
        """基于语义相似度扩展节点集合"""
        nodes = {seed}
        candidates = set(graph.neighbors(seed))
        
        # 简化的语义相似度计算（实际应用中应使用词向量等方法）
        while candidates:
            best_candidate = None
            best_score = 0
            
            for candidate in candidates:
                # 计算与已有节点的平均相似度
                score = self._compute_semantic_score(
                    graph, candidate, nodes
                )
                
                if score > best_score and score >= threshold:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate:
                nodes.add(best_candidate)
                candidates.remove(best_candidate)
                # 添加新的候选
                new_neighbors = set(graph.neighbors(best_candidate))
                candidates.update(new_neighbors - nodes)
            else:
                break
        
        return nodes
    
    def _compute_semantic_score(self, graph: nx.Graph, 
                               candidate: str, existing_nodes: Set[str]) -> float:
        """计算语义相似度得分（简化版本）"""
        # 这里使用共同邻居数量作为相似度的简单度量
        # 实际应用中应该使用更复杂的语义相似度计算
        candidate_neighbors = set(graph.neighbors(candidate))
        
        total_score = 0
        for node in existing_nodes:
            node_neighbors = set(graph.neighbors(node))
            common = len(candidate_neighbors & node_neighbors)
            total = len(candidate_neighbors | node_neighbors)
            
            if total > 0:
                score = common / total
            else:
                score = 0
                
            total_score += score
        
        return total_score / len(existing_nodes) if existing_nodes else 0
    
    def _ensure_coverage_sampling(self, graph: nx.Graph,
                                 covered_nodes: Set[str],
                                 additional_count: int) -> List[Subgraph]:
        """确保覆盖未被采样的节点"""
        uncovered = set(graph.nodes()) - covered_nodes
        subgraphs = []
        
        while uncovered and len(subgraphs) < additional_count:
            # 选择一个未覆盖的节点
            seed = random.choice(list(uncovered))
            
            # 扩展一个小子图
            subgraph_nodes = {seed}
            neighbors = list(graph.neighbors(seed))
            
            if neighbors:
                # 添加一些邻居
                num_neighbors = min(4, len(neighbors))
                selected = random.sample(neighbors, num_neighbors)
                subgraph_nodes.update(selected)
            
            subgraph = graph.subgraph(subgraph_nodes).copy()
            
            subgraphs.append(Subgraph(
                graph=subgraph,
                topology_type='coverage',
                central_nodes=[seed],
                task_scenario='覆盖补充型查询',
                complexity_level='简单',
                metadata={'coverage_seed': seed}
            ))
            
            # 更新未覆盖集合
            uncovered -= subgraph_nodes
        
        return subgraphs
    
    def _sample_process_optimization_subgraph(self, graph: nx.Graph,
                                            complexity: str) -> Subgraph:
        """采样工艺优化相关的子图"""
        # 寻找工艺相关节点
        process_nodes = [n for n in graph.nodes() 
                        if any(kw in str(n) for kw in ['工艺', '流程', '步骤', '参数'])]
        
        if not process_nodes:
            return None
            
        seed = random.choice(process_nodes)
        size_map = {'简单': 6, '中等': 12, '复杂': 18}
        max_size = size_map.get(complexity, 12)
        
        # 构建包含输入、过程、输出的子图
        subgraph_nodes = self._expand_process_subgraph(graph, seed, max_size)
        subgraph = graph.subgraph(subgraph_nodes).copy()
        
        return Subgraph(
            graph=subgraph,
            topology_type='process_optimization',
            central_nodes=[seed],
            task_scenario='工艺优化',
            complexity_level=complexity,
            metadata={'process_type': seed}
        )
    
    def _sample_quality_control_subgraph(self, graph: nx.Graph,
                                       complexity: str) -> Subgraph:
        """采样质量控制相关的子图"""
        quality_nodes = [n for n in graph.nodes() 
                        if any(kw in str(n) for kw in ['质量', '标准', '检测', '合格'])]
        
        if not quality_nodes:
            return None
            
        seed = random.choice(quality_nodes)
        size_map = {'简单': 5, '中等': 10, '复杂': 15}
        max_size = size_map.get(complexity, 10)
        
        subgraph_nodes = self._expand_quality_subgraph(graph, seed, max_size)
        subgraph = graph.subgraph(subgraph_nodes).copy()
        
        return Subgraph(
            graph=subgraph,
            topology_type='quality_control',
            central_nodes=[seed],
            task_scenario='质量控制',
            complexity_level=complexity,
            metadata={'quality_aspect': seed}
        )
    
    def _sample_maintenance_subgraph(self, graph: nx.Graph,
                                   complexity: str) -> Subgraph:
        """采样设备维护相关的子图"""
        maintenance_nodes = [n for n in graph.nodes() 
                           if any(kw in str(n) for kw in ['维护', '保养', '设备', '检修'])]
        
        if not maintenance_nodes:
            return None
            
        seed = random.choice(maintenance_nodes)
        size_map = {'简单': 6, '中等': 11, '复杂': 16}
        max_size = size_map.get(complexity, 11)
        
        subgraph_nodes = self._expand_maintenance_subgraph(graph, seed, max_size)
        subgraph = graph.subgraph(subgraph_nodes).copy()
        
        return Subgraph(
            graph=subgraph,
            topology_type='maintenance',
            central_nodes=[seed],
            task_scenario='设备维护',
            complexity_level=complexity,
            metadata={'maintenance_type': seed}
        )
    
    def _expand_process_subgraph(self, graph: nx.Graph, seed: str,
                               max_size: int) -> Set[str]:
        """扩展工艺流程子图"""
        nodes = {seed}
        
        # 寻找上下游节点
        for neighbor in graph.neighbors(seed):
            edge_data = graph.get_edge_data(seed, neighbor, {})
            relation = edge_data.get('relation', '')
            
            if any(kw in relation for kw in ['输入', '前置', '需要', '依赖']):
                nodes.add(neighbor)
            elif any(kw in relation for kw in ['输出', '产生', '得到', '生成']):
                nodes.add(neighbor)
        
        # BFS扩展
        return self._bfs_expand(graph, nodes, max_size)
    
    def _expand_quality_subgraph(self, graph: nx.Graph, seed: str,
                               max_size: int) -> Set[str]:
        """扩展质量控制子图"""
        nodes = {seed}
        
        # 优先添加标准、参数、方法相关节点
        for neighbor in graph.neighbors(seed):
            if any(kw in str(neighbor) for kw in ['标准', '参数', '方法', '指标']):
                nodes.add(neighbor)
        
        return self._bfs_expand(graph, nodes, max_size)
    
    def _expand_maintenance_subgraph(self, graph: nx.Graph, seed: str,
                                   max_size: int) -> Set[str]:
        """扩展维护子图"""
        nodes = {seed}
        
        # 添加相关的设备、周期、方法节点
        for neighbor in graph.neighbors(seed):
            if any(kw in str(neighbor) for kw in ['设备', '周期', '方法', '工具']):
                nodes.add(neighbor)
        
        return self._bfs_expand(graph, nodes, max_size)
    
    def _bfs_expand(self, graph: nx.Graph, initial_nodes: Set[str],
                   max_size: int) -> Set[str]:
        """BFS扩展节点集合"""
        nodes = set(initial_nodes)
        queue = list(initial_nodes)
        visited = set(initial_nodes)
        
        while queue and len(nodes) < max_size:
            current = queue.pop(0)
            for neighbor in graph.neighbors(current):
                if neighbor not in visited and len(nodes) < max_size:
                    nodes.add(neighbor)
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return nodes
    
    def _expand_cycle(self, graph: nx.Graph, cycle: List[str],
                     min_nodes: int, max_nodes: int) -> List[str]:
        """扩展环形结构"""
        expanded = set(cycle)
        
        # 从环上的每个节点尝试扩展
        for node in cycle:
            if len(expanded) >= max_nodes:
                break
                
            neighbors = set(graph.neighbors(node)) - expanded
            if neighbors:
                # 添加一些邻居
                num_to_add = min(
                    len(neighbors),
                    max_nodes - len(expanded)
                )
                expanded.update(random.sample(list(neighbors), num_to_add))
        
        if len(expanded) >= min_nodes:
            return list(expanded)
        return None