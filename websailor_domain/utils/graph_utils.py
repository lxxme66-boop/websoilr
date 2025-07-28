"""
图处理工具函数模块
提供图操作、分析、可视化等功能
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
import json
import logging
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

logger = logging.getLogger(__name__)

# 设置中文字体
def setup_chinese_font():
    """设置matplotlib中文字体"""
    try:
        # 尝试使用系统中文字体
        font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        chinese_fonts = [f for f in font_list if 'simhei' in f.lower() or 'simsun' in f.lower()]
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = ['SimHei']
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    except:
        logger.warning("无法设置中文字体，图表可能无法正确显示中文")
    
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def calculate_graph_metrics(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    计算图的各种指标
    """
    metrics = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'is_connected': nx.is_weakly_connected(graph),
        'num_components': nx.number_weakly_connected_components(graph),
    }
    
    # 度分布
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    
    metrics['avg_in_degree'] = np.mean(list(in_degrees.values())) if in_degrees else 0
    metrics['avg_out_degree'] = np.mean(list(out_degrees.values())) if out_degrees else 0
    metrics['max_in_degree'] = max(in_degrees.values()) if in_degrees else 0
    metrics['max_out_degree'] = max(out_degrees.values()) if out_degrees else 0
    
    # 中心性指标（对小图计算）
    if graph.number_of_nodes() < 1000:
        try:
            metrics['avg_betweenness_centrality'] = np.mean(
                list(nx.betweenness_centrality(graph).values())
            )
        except:
            metrics['avg_betweenness_centrality'] = 0.0
    
    # 最长路径（对小图计算）
    if graph.number_of_nodes() < 100:
        try:
            metrics['diameter'] = nx.diameter(graph.to_undirected())
        except:
            metrics['diameter'] = -1
    
    return metrics


def find_important_nodes(graph: nx.DiGraph, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    找出图中最重要的节点
    基于多种中心性指标的综合评分
    """
    if graph.number_of_nodes() == 0:
        return []
    
    # 计算各种中心性
    scores = defaultdict(float)
    
    # 度中心性
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    max_in = max(in_degree.values()) if in_degree else 1
    max_out = max(out_degree.values()) if out_degree else 1
    
    for node in graph.nodes():
        scores[node] += (in_degree.get(node, 0) / max_in + 
                        out_degree.get(node, 0) / max_out) * 0.5
    
    # PageRank（对较小的图）
    if graph.number_of_nodes() < 1000:
        try:
            pagerank = nx.pagerank(graph, max_iter=100)
            max_pr = max(pagerank.values()) if pagerank else 1
            for node, pr in pagerank.items():
                scores[node] += pr / max_pr * 0.3
        except:
            pass
    
    # 介数中心性（对更小的图）
    if graph.number_of_nodes() < 100:
        try:
            betweenness = nx.betweenness_centrality(graph)
            max_bet = max(betweenness.values()) if betweenness else 1
            for node, bet in betweenness.items():
                scores[node] += bet / max_bet * 0.2
        except:
            pass
    
    # 排序并返回top_k
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:top_k]


def extract_subgraph_around_node(graph: nx.DiGraph, 
                                center_node: str, 
                                radius: int = 2) -> nx.DiGraph:
    """
    提取以某个节点为中心的子图
    """
    if center_node not in graph:
        return nx.DiGraph()
    
    # 使用BFS找出radius跳内的所有节点
    nodes = {center_node}
    current_layer = {center_node}
    
    for _ in range(radius):
        next_layer = set()
        for node in current_layer:
            # 前驱和后继节点
            next_layer.update(graph.predecessors(node))
            next_layer.update(graph.successors(node))
        nodes.update(next_layer)
        current_layer = next_layer
    
    # 提取子图
    return graph.subgraph(nodes).copy()


def find_paths_between_nodes(graph: nx.DiGraph, 
                           source: str, 
                           target: str, 
                           max_length: int = 5) -> List[List[str]]:
    """
    找出两个节点之间的所有路径（限制最大长度）
    """
    if source not in graph or target not in graph:
        return []
    
    try:
        # 使用all_simple_paths，限制路径长度
        paths = list(nx.all_simple_paths(graph, source, target, cutoff=max_length))
        return paths
    except nx.NetworkXNoPath:
        return []


def detect_communities(graph: nx.DiGraph) -> Dict[str, int]:
    """
    社区检测（使用Louvain算法）
    """
    # 转换为无向图
    undirected = graph.to_undirected()
    
    # 使用python-louvain进行社区检测
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(undirected)
        return partition
    except ImportError:
        logger.warning("未安装python-louvain，使用简单的连通分量作为社区")
        # 使用连通分量作为备选
        communities = {}
        for i, component in enumerate(nx.connected_components(undirected)):
            for node in component:
                communities[node] = i
        return communities


def visualize_graph(graph: nx.DiGraph, 
                   output_path: str = None,
                   node_labels: bool = True,
                   node_colors: Dict[str, str] = None,
                   title: str = "Knowledge Graph") -> None:
    """
    可视化图
    """
    setup_chinese_font()
    
    plt.figure(figsize=(12, 8))
    
    # 布局算法
    if graph.number_of_nodes() < 50:
        pos = nx.spring_layout(graph, k=2, iterations=50)
    else:
        pos = nx.kamada_kawai_layout(graph)
    
    # 节点颜色
    if node_colors:
        node_color_list = [node_colors.get(node, 'lightblue') for node in graph.nodes()]
    else:
        # 根据节点类型着色
        node_color_list = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            if node_data.get('type') == 'product':
                node_color_list.append('lightcoral')
            elif node_data.get('type') == 'technology':
                node_color_list.append('lightgreen')
            elif node_data.get('type') == 'company':
                node_color_list.append('lightyellow')
            else:
                node_color_list.append('lightblue')
    
    # 绘制节点
    nx.draw_networkx_nodes(graph, pos, 
                          node_color=node_color_list,
                          node_size=300,
                          alpha=0.8)
    
    # 绘制边
    nx.draw_networkx_edges(graph, pos,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=10,
                          alpha=0.5)
    
    # 绘制标签
    if node_labels and graph.number_of_nodes() < 50:
        labels = {}
        for node in graph.nodes():
            # 截断过长的标签
            label = str(node)
            if len(label) > 10:
                label = label[:10] + '...'
            labels[node] = label
        
        nx.draw_networkx_labels(graph, pos, labels,
                               font_size=8)
    
    plt.title(title)
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_edge_types(graph: nx.DiGraph) -> Dict[str, int]:
    """
    分析边的类型分布
    """
    edge_types = Counter()
    
    for u, v, data in graph.edges(data=True):
        edge_type = data.get('type', 'unknown')
        edge_types[edge_type] += 1
    
    return dict(edge_types)


def find_triangles(graph: nx.DiGraph) -> List[Tuple[str, str, str]]:
    """
    找出图中的三角形结构
    """
    triangles = []
    
    # 转换为无向图
    undirected = graph.to_undirected()
    
    # 找出所有三角形
    for triangle in nx.enumerate_all_cliques(undirected):
        if len(triangle) == 3:
            triangles.append(tuple(sorted(triangle)))
    
    # 去重
    return list(set(triangles))


def calculate_node_similarity(graph: nx.DiGraph, 
                            node1: str, 
                            node2: str) -> float:
    """
    计算两个节点的相似度（基于共同邻居）
    """
    if node1 not in graph or node2 not in graph:
        return 0.0
    
    # 获取邻居节点
    neighbors1 = set(graph.successors(node1)) | set(graph.predecessors(node1))
    neighbors2 = set(graph.successors(node2)) | set(graph.predecessors(node2))
    
    # Jaccard相似度
    if not neighbors1 and not neighbors2:
        return 0.0
    
    intersection = neighbors1 & neighbors2
    union = neighbors1 | neighbors2
    
    return len(intersection) / len(union)


def merge_duplicate_nodes(graph: nx.DiGraph, 
                         similarity_threshold: float = 0.9) -> nx.DiGraph:
    """
    合并相似的重复节点
    """
    merged_graph = graph.copy()
    
    # 找出相似节点对
    nodes = list(merged_graph.nodes())
    merge_pairs = []
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1, node2 = nodes[i], nodes[j]
            
            # 计算名称相似度
            from difflib import SequenceMatcher
            name_similarity = SequenceMatcher(None, str(node1), str(node2)).ratio()
            
            # 计算结构相似度
            struct_similarity = calculate_node_similarity(merged_graph, node1, node2)
            
            # 综合相似度
            total_similarity = name_similarity * 0.7 + struct_similarity * 0.3
            
            if total_similarity >= similarity_threshold:
                merge_pairs.append((node1, node2, total_similarity))
    
    # 执行合并
    for node1, node2, _ in sorted(merge_pairs, key=lambda x: x[2], reverse=True):
        if node1 in merged_graph and node2 in merged_graph:
            # 合并node2到node1
            for pred in merged_graph.predecessors(node2):
                if pred != node1:
                    merged_graph.add_edge(pred, node1)
            
            for succ in merged_graph.successors(node2):
                if succ != node1:
                    merged_graph.add_edge(node1, succ)
            
            # 合并属性
            attrs1 = merged_graph.nodes[node1]
            attrs2 = merged_graph.nodes[node2]
            for key, value in attrs2.items():
                if key not in attrs1:
                    attrs1[key] = value
            
            # 删除node2
            merged_graph.remove_node(node2)
    
    return merged_graph


def export_graph_to_json(graph: nx.DiGraph, output_path: str) -> None:
    """
    将图导出为JSON格式
    """
    data = {
        'nodes': [],
        'edges': []
    }
    
    # 导出节点
    for node, attrs in graph.nodes(data=True):
        node_data = {'id': str(node)}
        node_data.update(attrs)
        data['nodes'].append(node_data)
    
    # 导出边
    for source, target, attrs in graph.edges(data=True):
        edge_data = {
            'source': str(source),
            'target': str(target)
        }
        edge_data.update(attrs)
        data['edges'].append(edge_data)
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"图已导出到: {output_path}")


def load_graph_from_json(input_path: str) -> nx.DiGraph:
    """
    从JSON文件加载图
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    graph = nx.DiGraph()
    
    # 加载节点
    for node_data in data['nodes']:
        node_id = node_data.pop('id')
        graph.add_node(node_id, **node_data)
    
    # 加载边
    for edge_data in data['edges']:
        source = edge_data.pop('source')
        target = edge_data.pop('target')
        graph.add_edge(source, target, **edge_data)
    
    return graph


def generate_graph_summary(graph: nx.DiGraph) -> str:
    """
    生成图的文本摘要
    """
    metrics = calculate_graph_metrics(graph)
    important_nodes = find_important_nodes(graph, top_k=5)
    edge_types = analyze_edge_types(graph)
    
    summary = f"""
图摘要:
- 节点数: {metrics['num_nodes']}
- 边数: {metrics['num_edges']}
- 密度: {metrics['density']:.4f}
- 平均入度: {metrics['avg_in_degree']:.2f}
- 平均出度: {metrics['avg_out_degree']:.2f}
- 最重要的节点: {', '.join([node for node, _ in important_nodes[:3]])}
- 主要边类型: {', '.join([f"{t}({c})" for t, c in list(edge_types.items())[:3]])}
"""
    
    return summary.strip()


# 导出所有函数
__all__ = [
    'calculate_graph_metrics',
    'find_important_nodes',
    'extract_subgraph_around_node',
    'find_paths_between_nodes',
    'detect_communities',
    'visualize_graph',
    'analyze_edge_types',
    'find_triangles',
    'calculate_node_similarity',
    'merge_duplicate_nodes',
    'export_graph_to_json',
    'load_graph_from_json',
    'generate_graph_summary'
]