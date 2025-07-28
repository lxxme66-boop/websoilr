"""
图处理工具函数
"""

import logging
import networkx as nx
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def visualize_subgraph(subgraph: Dict, output_path: Optional[Path] = None):
    """可视化子图"""
    try:
        from pyvis.network import Network
        
        # 创建pyvis网络
        net = Network(height="600px", width="100%", directed=True)
        
        # 添加节点
        for node in subgraph['nodes']:
            net.add_node(
                node['id'],
                label=node['name'],
                title=f"{node['type']}: {node['name']}",
                color=_get_node_color(node['type'])
            )
        
        # 添加边
        for edge in subgraph['edges']:
            net.add_edge(
                edge['source'],
                edge['target'],
                label=edge['relation'],
                title=edge['relation']
            )
        
        # 保存或显示
        if output_path:
            net.save_graph(str(output_path))
            logger.info(f"子图可视化保存到: {output_path}")
        else:
            return net.generate_html()
            
    except ImportError:
        logger.warning("pyvis未安装，无法进行可视化")
        return None


def _get_node_color(node_type: str) -> str:
    """根据节点类型返回颜色"""
    color_map = {
        "产品": "#FF6B6B",
        "技术": "#4ECDC4",
        "工艺": "#45B7D1",
        "材料": "#96CEB4",
        "设备": "#FECA57",
        "标准": "#DDA0DD",
        "专利": "#98D8C8",
        "公司": "#F7DC6F",
        "人员": "#85C1E2",
        "项目": "#F8B500"
    }
    return color_map.get(node_type, "#95A5A6")


def analyze_graph_structure(graph: nx.Graph) -> Dict:
    """分析图结构特征"""
    analysis = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "is_connected": nx.is_connected(graph.to_undirected()),
        "num_components": nx.number_connected_components(graph.to_undirected())
    }
    
    # 度分布
    degrees = dict(graph.degree())
    analysis["avg_degree"] = sum(degrees.values()) / len(degrees) if degrees else 0
    analysis["max_degree"] = max(degrees.values()) if degrees else 0
    analysis["min_degree"] = min(degrees.values()) if degrees else 0
    
    # 中心性指标
    if graph.number_of_nodes() > 0:
        analysis["avg_clustering"] = nx.average_clustering(graph.to_undirected())
        
        # PageRank（用于重要性采样）
        pagerank = nx.pagerank(graph)
        analysis["top_pagerank_nodes"] = sorted(
            pagerank.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
    
    return analysis


def find_paths(graph: nx.Graph, source: str, target: str, max_length: int = 5) -> List[List[str]]:
    """查找两个节点之间的所有路径"""
    try:
        paths = list(nx.all_simple_paths(
            graph, 
            source=source, 
            target=target, 
            cutoff=max_length
        ))
        return paths
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return []


def extract_subgraph_by_nodes(graph: nx.Graph, nodes: List[str]) -> nx.Graph:
    """根据节点列表提取子图"""
    return graph.subgraph(nodes).copy()


def get_node_neighbors(graph: nx.Graph, node: str, radius: int = 1) -> List[str]:
    """获取节点的邻居（指定跳数内）"""
    if node not in graph:
        return []
    
    neighbors = {node}
    current_level = {node}
    
    for _ in range(radius):
        next_level = set()
        for n in current_level:
            next_level.update(graph.neighbors(n))
        neighbors.update(next_level)
        current_level = next_level
    
    return list(neighbors)


def detect_cycles(graph: nx.Graph) -> List[List[str]]:
    """检测图中的环"""
    try:
        cycles = list(nx.simple_cycles(graph))
        return cycles
    except:
        # 如果是无向图，使用其他方法
        return list(nx.cycle_basis(graph.to_undirected()))


def calculate_graph_similarity(g1: nx.Graph, g2: nx.Graph) -> float:
    """计算两个图的相似度（基于节点和边的Jaccard相似度）"""
    nodes1 = set(g1.nodes())
    nodes2 = set(g2.nodes())
    edges1 = set(g1.edges())
    edges2 = set(g2.edges())
    
    node_similarity = len(nodes1 & nodes2) / len(nodes1 | nodes2) if (nodes1 | nodes2) else 0
    edge_similarity = len(edges1 & edges2) / len(edges1 | edges2) if (edges1 | edges2) else 0
    
    # 加权平均
    return 0.5 * node_similarity + 0.5 * edge_similarity


def convert_to_levi_graph(graph: nx.MultiDiGraph) -> nx.Graph:
    """将多重有向图转换为Levi图（用于处理多关系）"""
    levi = nx.Graph()
    
    # 添加原始节点
    for node, data in graph.nodes(data=True):
        levi.add_node(f"n_{node}", **data, node_type="entity")
    
    # 将边转换为节点
    edge_id = 0
    for u, v, key, data in graph.edges(keys=True, data=True):
        edge_node = f"e_{edge_id}"
        levi.add_node(edge_node, relation=data.get('relation', 'unknown'), node_type="relation")
        levi.add_edge(f"n_{u}", edge_node)
        levi.add_edge(edge_node, f"n_{v}")
        edge_id += 1
    
    return levi