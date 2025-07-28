"""
图处理工具函数
提供图可视化、度量计算、社区发现等功能
"""

import networkx as nx
from typing import List, Dict, Any, Tuple, Set, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pyvis.network import Network
import numpy as np
from collections import defaultdict
import logging


logger = logging.getLogger(__name__)


def visualize_graph(graph: nx.Graph, output_path: str = None,
                   layout: str = 'spring', interactive: bool = False) -> None:
    """
    可视化图
    
    Args:
        graph: NetworkX图对象
        output_path: 输出文件路径
        layout: 布局算法
        interactive: 是否生成交互式图
    """
    if interactive:
        _visualize_interactive(graph, output_path)
    else:
        _visualize_static(graph, output_path, layout)


def _visualize_static(graph: nx.Graph, output_path: str = None,
                     layout: str = 'spring') -> None:
    """生成静态图可视化"""
    plt.figure(figsize=(12, 8))
    
    # 选择布局算法
    if layout == 'spring':
        pos = nx.spring_layout(graph, k=1, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.spring_layout(graph)
    
    # 绘制节点
    node_colors = []
    for node in graph.nodes():
        node_type = graph.nodes[node].get('type', 'default')
        # 根据节点类型设置颜色
        color_map = {
            '产品': '#ff6b6b',
            '设备': '#4ecdc4',
            '工艺': '#45b7d1',
            '材料': '#f7b731',
            '参数': '#5f27cd',
            'default': '#95a5a6'
        }
        node_colors.append(color_map.get(node_type, color_map['default']))
    
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors,
                          node_size=500, alpha=0.8)
    
    # 绘制边
    nx.draw_networkx_edges(graph, pos, edge_color='gray',
                          alpha=0.5, width=1)
    
    # 绘制标签
    nx.draw_networkx_labels(graph, pos, font_size=10,
                           font_family='sans-serif')
    
    plt.title('Knowledge Graph Visualization')
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def _visualize_interactive(graph: nx.Graph, output_path: str = None) -> None:
    """生成交互式图可视化"""
    net = Network(height='750px', width='100%', bgcolor='#222222',
                  font_color='white')
    
    # 添加节点
    for node in graph.nodes():
        node_attrs = graph.nodes[node]
        node_type = node_attrs.get('type', 'default')
        
        # 设置节点颜色
        color_map = {
            '产品': '#ff6b6b',
            '设备': '#4ecdc4',
            '工艺': '#45b7d1',
            '材料': '#f7b731',
            '参数': '#5f27cd',
            'default': '#95a5a6'
        }
        color = color_map.get(node_type, color_map['default'])
        
        net.add_node(str(node), label=str(node), color=color,
                     title=f"Type: {node_type}")
    
    # 添加边
    for u, v, attrs in graph.edges(data=True):
        relation = attrs.get('relation', '')
        net.add_edge(str(u), str(v), title=relation)
    
    # 设置物理布局
    net.barnes_hut(gravity=-80000, central_gravity=0.3,
                   spring_length=250, spring_strength=0.001)
    
    # 保存或显示
    if output_path:
        net.save_graph(output_path)
    else:
        net.show('graph.html')


def calculate_graph_metrics(graph: nx.Graph) -> Dict[str, Any]:
    """
    计算图的各种度量指标
    
    Args:
        graph: NetworkX图对象
        
    Returns:
        包含各种度量指标的字典
    """
    metrics = {}
    
    # 基本统计
    metrics['num_nodes'] = graph.number_of_nodes()
    metrics['num_edges'] = graph.number_of_edges()
    metrics['density'] = nx.density(graph)
    
    # 度统计
    degrees = dict(graph.degree())
    metrics['avg_degree'] = np.mean(list(degrees.values()))
    metrics['max_degree'] = max(degrees.values())
    metrics['min_degree'] = min(degrees.values())
    
    # 连通性
    if graph.is_directed():
        metrics['is_weakly_connected'] = nx.is_weakly_connected(graph)
        metrics['is_strongly_connected'] = nx.is_strongly_connected(graph)
        metrics['num_weakly_connected_components'] = nx.number_weakly_connected_components(graph)
        metrics['num_strongly_connected_components'] = nx.number_strongly_connected_components(graph)
    else:
        metrics['is_connected'] = nx.is_connected(graph)
        metrics['num_connected_components'] = nx.number_connected_components(graph)
    
    # 路径统计
    if metrics.get('is_connected', False) or metrics.get('is_strongly_connected', False):
        metrics['diameter'] = nx.diameter(graph)
        metrics['radius'] = nx.radius(graph)
        metrics['avg_shortest_path_length'] = nx.average_shortest_path_length(graph)
    
    # 聚类系数
    metrics['avg_clustering_coefficient'] = nx.average_clustering(graph)
    
    # 中心性度量（只计算最大连通分量）
    if not (metrics.get('is_connected', False) or metrics.get('is_strongly_connected', False)):
        # 获取最大连通分量
        if graph.is_directed():
            largest_cc = max(nx.weakly_connected_components(graph), key=len)
        else:
            largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
    else:
        subgraph = graph
    
    # 计算中心性
    degree_centrality = nx.degree_centrality(subgraph)
    betweenness_centrality = nx.betweenness_centrality(subgraph)
    closeness_centrality = nx.closeness_centrality(subgraph)
    
    # 找出最中心的节点
    metrics['most_central_nodes'] = {
        'degree': max(degree_centrality, key=degree_centrality.get),
        'betweenness': max(betweenness_centrality, key=betweenness_centrality.get),
        'closeness': max(closeness_centrality, key=closeness_centrality.get)
    }
    
    return metrics


def find_communities(graph: nx.Graph, method: str = 'louvain') -> List[Set[Any]]:
    """
    发现图中的社区
    
    Args:
        graph: NetworkX图对象
        method: 社区发现算法
        
    Returns:
        社区列表，每个社区是节点集合
    """
    # 确保是无向图
    if graph.is_directed():
        graph = graph.to_undirected()
    
    if method == 'louvain':
        # 使用Louvain算法
        import community
        partition = community.best_partition(graph)
        
        # 将分区转换为社区列表
        communities = defaultdict(set)
        for node, comm_id in partition.items():
            communities[comm_id].add(node)
        
        return list(communities.values())
    
    elif method == 'label_propagation':
        # 使用标签传播算法
        communities = nx.algorithms.community.label_propagation_communities(graph)
        return list(communities)
    
    elif method == 'greedy_modularity':
        # 使用贪婪模块度算法
        communities = nx.algorithms.community.greedy_modularity_communities(graph)
        return list(communities)
    
    else:
        raise ValueError(f"不支持的社区发现方法: {method}")


def extract_paths(graph: nx.Graph, source: Any, target: Any,
                 k: int = 5, cutoff: int = None) -> List[List[Any]]:
    """
    提取两个节点之间的路径
    
    Args:
        graph: NetworkX图对象
        source: 源节点
        target: 目标节点
        k: 返回前k条最短路径
        cutoff: 路径长度截断
        
    Returns:
        路径列表
    """
    paths = []
    
    try:
        # 尝试找到k条最短路径
        for path in nx.shortest_simple_paths(graph, source, target):
            if cutoff and len(path) - 1 > cutoff:
                break
            paths.append(path)
            if len(paths) >= k:
                break
    except nx.NetworkXNoPath:
        logger.warning(f"节点{source}和{target}之间没有路径")
    except nx.NodeNotFound as e:
        logger.warning(f"节点不存在: {e}")
    
    return paths


def find_important_nodes(graph: nx.Graph, top_k: int = 10,
                        criteria: str = 'degree') -> List[Tuple[Any, float]]:
    """
    找出图中的重要节点
    
    Args:
        graph: NetworkX图对象
        top_k: 返回前k个重要节点
        criteria: 重要性标准
        
    Returns:
        (节点, 重要性分数)元组列表
    """
    if criteria == 'degree':
        # 按度排序
        importance = dict(graph.degree())
    
    elif criteria == 'betweenness':
        # 按介数中心性排序
        importance = nx.betweenness_centrality(graph)
    
    elif criteria == 'closeness':
        # 按接近中心性排序
        importance = nx.closeness_centrality(graph)
    
    elif criteria == 'pagerank':
        # 按PageRank排序
        importance = nx.pagerank(graph)
    
    elif criteria == 'eigenvector':
        # 按特征向量中心性排序
        try:
            importance = nx.eigenvector_centrality(graph, max_iter=1000)
        except:
            logger.warning("特征向量中心性计算失败，使用度中心性替代")
            importance = dict(graph.degree())
    
    else:
        raise ValueError(f"不支持的标准: {criteria}")
    
    # 排序并返回前k个
    sorted_nodes = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:top_k]


def extract_subgraph_by_nodes(graph: nx.Graph, nodes: List[Any],
                             include_neighbors: bool = False) -> nx.Graph:
    """
    根据节点列表提取子图
    
    Args:
        graph: 原始图
        nodes: 节点列表
        include_neighbors: 是否包含邻居节点
        
    Returns:
        子图
    """
    if include_neighbors:
        # 包含所有邻居
        extended_nodes = set(nodes)
        for node in nodes:
            if node in graph:
                extended_nodes.update(graph.neighbors(node))
        return graph.subgraph(extended_nodes).copy()
    else:
        # 只包含指定节点
        return graph.subgraph(nodes).copy()


def find_bridges(graph: nx.Graph) -> List[Tuple[Any, Any]]:
    """
    找出图中的桥（删除后会增加连通分量的边）
    
    Args:
        graph: NetworkX图对象
        
    Returns:
        桥的列表
    """
    if graph.is_directed():
        # 转换为无向图
        graph = graph.to_undirected()
    
    return list(nx.bridges(graph))


def find_articulation_points(graph: nx.Graph) -> Set[Any]:
    """
    找出图中的关节点（删除后会增加连通分量的节点）
    
    Args:
        graph: NetworkX图对象
        
    Returns:
        关节点集合
    """
    if graph.is_directed():
        # 转换为无向图
        graph = graph.to_undirected()
    
    return set(nx.articulation_points(graph))


def calculate_modularity(graph: nx.Graph, communities: List[Set[Any]]) -> float:
    """
    计算社区划分的模块度
    
    Args:
        graph: NetworkX图对象
        communities: 社区列表
        
    Returns:
        模块度值
    """
    if graph.is_directed():
        graph = graph.to_undirected()
    
    return nx.algorithms.community.modularity(graph, communities)


def find_cliques(graph: nx.Graph, min_size: int = 3) -> List[Set[Any]]:
    """
    找出图中的团（完全子图）
    
    Args:
        graph: NetworkX图对象
        min_size: 最小团大小
        
    Returns:
        团的列表
    """
    if graph.is_directed():
        graph = graph.to_undirected()
    
    cliques = []
    for clique in nx.find_cliques(graph):
        if len(clique) >= min_size:
            cliques.append(set(clique))
    
    return cliques


def graph_to_adjacency_matrix(graph: nx.Graph) -> Tuple[np.ndarray, List[Any]]:
    """
    将图转换为邻接矩阵
    
    Args:
        graph: NetworkX图对象
        
    Returns:
        (邻接矩阵, 节点列表)
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # 创建邻接矩阵
    adj_matrix = np.zeros((n, n))
    
    for u, v in graph.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj_matrix[i, j] = 1
        if not graph.is_directed():
            adj_matrix[j, i] = 1
    
    return adj_matrix, nodes