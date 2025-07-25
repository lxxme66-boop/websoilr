"""
图处理工具模块
提供图可视化、度量计算、格式转换等功能
"""

import networkx as nx
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from collections import defaultdict, Counter
import logging

def visualize_graph(graph: nx.Graph, 
                   output_path: str = None,
                   layout: str = "spring",
                   node_size: int = 500,
                   node_color: str = "lightblue",
                   edge_color: str = "gray",
                   with_labels: bool = True,
                   font_size: int = 10,
                   figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    可视化图结构
    
    Args:
        graph: NetworkX图对象
        output_path: 输出路径，如果为None则显示图像
        layout: 布局算法 (spring, circular, random, shell)
        node_size: 节点大小
        node_color: 节点颜色
        edge_color: 边颜色
        with_labels: 是否显示标签
        font_size: 字体大小
        figsize: 图像尺寸
    """
    try:
        plt.figure(figsize=figsize)
        
        # 选择布局算法
        if layout == "spring":
            pos = nx.spring_layout(graph, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        elif layout == "random":
            pos = nx.random_layout(graph)
        elif layout == "shell":
            pos = nx.shell_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        # 绘制图
        nx.draw(graph, pos, 
                node_size=node_size,
                node_color=node_color,
                edge_color=edge_color,
                with_labels=with_labels,
                font_size=font_size,
                font_weight='bold')
        
        # 添加边标签（如果有权重）
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        if edge_labels:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels)
        
        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"图可视化已保存到: {output_path}")
        else:
            plt.show()
            
    except Exception as e:
        logging.error(f"图可视化失败: {str(e)}")
    finally:
        plt.close()

def compute_graph_metrics(graph: nx.Graph) -> Dict[str, Any]:
    """
    计算图的各种度量指标
    
    Args:
        graph: NetworkX图对象
        
    Returns:
        包含各种图度量的字典
    """
    metrics = {}
    
    try:
        # 基本统计
        metrics["num_nodes"] = graph.number_of_nodes()
        metrics["num_edges"] = graph.number_of_edges()
        metrics["density"] = nx.density(graph)
        
        # 连通性
        metrics["is_connected"] = nx.is_connected(graph)
        metrics["num_connected_components"] = nx.number_connected_components(graph)
        
        if metrics["is_connected"]:
            metrics["diameter"] = nx.diameter(graph)
            metrics["radius"] = nx.radius(graph)
            metrics["average_shortest_path_length"] = nx.average_shortest_path_length(graph)
        
        # 度分布
        degrees = [d for n, d in graph.degree()]
        metrics["average_degree"] = np.mean(degrees)
        metrics["max_degree"] = max(degrees)
        metrics["min_degree"] = min(degrees)
        metrics["degree_distribution"] = dict(Counter(degrees))
        
        # 中心性度量
        if graph.number_of_nodes() > 0:
            # 度中心性
            degree_centrality = nx.degree_centrality(graph)
            metrics["top_degree_centrality"] = sorted(
                degree_centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            # 接近中心性（仅对连通图计算）
            if metrics["is_connected"] and graph.number_of_nodes() > 1:
                closeness_centrality = nx.closeness_centrality(graph)
                metrics["top_closeness_centrality"] = sorted(
                    closeness_centrality.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
                
                # 介数中心性
                betweenness_centrality = nx.betweenness_centrality(graph)
                metrics["top_betweenness_centrality"] = sorted(
                    betweenness_centrality.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
        
        # 聚类系数
        if graph.number_of_nodes() > 2:
            metrics["average_clustering"] = nx.average_clustering(graph)
            clustering = nx.clustering(graph)
            metrics["top_clustering"] = sorted(
                clustering.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        
        # 社区检测（简单的连通分量）
        components = list(nx.connected_components(graph))
        metrics["largest_component_size"] = max(len(c) for c in components) if components else 0
        metrics["component_sizes"] = [len(c) for c in components]
        
        logging.info("图度量计算完成")
        
    except Exception as e:
        logging.error(f"图度量计算失败: {str(e)}")
        metrics["error"] = str(e)
    
    return metrics

def export_graph_to_formats(graph: nx.Graph, 
                           output_dir: str,
                           formats: List[str] = ["gexf", "graphml", "json"]) -> Dict[str, str]:
    """
    将图导出为多种格式
    
    Args:
        graph: NetworkX图对象
        output_dir: 输出目录
        formats: 导出格式列表
        
    Returns:
        导出文件路径字典
    """
    import os
    
    exported_files = {}
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        for format_type in formats:
            if format_type == "gexf":
                file_path = os.path.join(output_dir, "knowledge_graph.gexf")
                nx.write_gexf(graph, file_path)
                exported_files["gexf"] = file_path
                
            elif format_type == "graphml":
                file_path = os.path.join(output_dir, "knowledge_graph.graphml")
                nx.write_graphml(graph, file_path)
                exported_files["graphml"] = file_path
                
            elif format_type == "json":
                file_path = os.path.join(output_dir, "knowledge_graph.json")
                graph_data = nx.node_link_data(graph)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, ensure_ascii=False, indent=2)
                exported_files["json"] = file_path
                
            elif format_type == "edgelist":
                file_path = os.path.join(output_dir, "knowledge_graph.edgelist")
                nx.write_edgelist(graph, file_path)
                exported_files["edgelist"] = file_path
                
            elif format_type == "pajek":
                file_path = os.path.join(output_dir, "knowledge_graph.net")
                nx.write_pajek(graph, file_path)
                exported_files["pajek"] = file_path
        
        logging.info(f"图已导出为 {len(exported_files)} 种格式到 {output_dir}")
        
    except Exception as e:
        logging.error(f"图导出失败: {str(e)}")
    
    return exported_files

def find_shortest_paths(graph: nx.Graph, 
                       source: str, 
                       target: str = None,
                       max_paths: int = 5) -> Dict[str, Any]:
    """
    寻找最短路径
    
    Args:
        graph: NetworkX图对象
        source: 源节点
        target: 目标节点，如果为None则计算到所有节点的最短路径
        max_paths: 最大路径数量
        
    Returns:
        包含路径信息的字典
    """
    paths_info = {}
    
    try:
        if target:
            # 单个目标的所有最短路径
            if nx.has_path(graph, source, target):
                all_paths = list(nx.all_shortest_paths(graph, source, target))
                paths_info["paths"] = all_paths[:max_paths]
                paths_info["shortest_distance"] = nx.shortest_path_length(graph, source, target)
                paths_info["num_paths"] = len(all_paths)
            else:
                paths_info["paths"] = []
                paths_info["shortest_distance"] = float('inf')
                paths_info["num_paths"] = 0
        else:
            # 到所有节点的最短路径
            shortest_paths = nx.single_source_shortest_path(graph, source)
            path_lengths = nx.single_source_shortest_path_length(graph, source)
            
            paths_info["all_paths"] = {
                target: path for target, path in shortest_paths.items()
            }
            paths_info["all_distances"] = dict(path_lengths)
            
            # 按距离排序的前N个目标
            sorted_targets = sorted(
                path_lengths.items(), 
                key=lambda x: x[1]
            )[:max_paths]
            paths_info["closest_nodes"] = sorted_targets
            
    except Exception as e:
        logging.error(f"最短路径计算失败: {str(e)}")
        paths_info["error"] = str(e)
    
    return paths_info

def detect_communities(graph: nx.Graph, 
                      algorithm: str = "louvain") -> Dict[str, Any]:
    """
    社区检测
    
    Args:
        graph: NetworkX图对象
        algorithm: 社区检测算法
        
    Returns:
        社区信息字典
    """
    communities_info = {}
    
    try:
        if algorithm == "louvain":
            # 使用贪心模块化优化
            communities = nx.community.greedy_modularity_communities(graph)
            
        elif algorithm == "label_propagation":
            # 标签传播
            communities = nx.community.label_propagation_communities(graph)
            
        elif algorithm == "connected_components":
            # 连通分量
            communities = nx.connected_components(graph)
            
        else:
            # 默认使用贪心模块化
            communities = nx.community.greedy_modularity_communities(graph)
        
        # 转换为列表格式
        community_list = [list(community) for community in communities]
        
        communities_info["communities"] = community_list
        communities_info["num_communities"] = len(community_list)
        communities_info["community_sizes"] = [len(c) for c in community_list]
        communities_info["largest_community_size"] = max(
            communities_info["community_sizes"]
        ) if community_list else 0
        
        # 计算模块化
        if len(community_list) > 1:
            communities_info["modularity"] = nx.community.modularity(
                graph, community_list
            )
        
        logging.info(f"检测到 {len(community_list)} 个社区")
        
    except Exception as e:
        logging.error(f"社区检测失败: {str(e)}")
        communities_info["error"] = str(e)
    
    return communities_info

def analyze_node_importance(graph: nx.Graph, 
                           node: str) -> Dict[str, Any]:
    """
    分析单个节点的重要性
    
    Args:
        graph: NetworkX图对象
        node: 节点ID
        
    Returns:
        节点重要性分析结果
    """
    importance_info = {}
    
    try:
        if node not in graph:
            importance_info["error"] = f"节点 {node} 不存在于图中"
            return importance_info
        
        # 基本信息
        importance_info["degree"] = graph.degree(node)
        importance_info["neighbors"] = list(graph.neighbors(node))
        importance_info["num_neighbors"] = len(list(graph.neighbors(node)))
        
        # 中心性度量
        all_nodes = list(graph.nodes())
        
        # 度中心性
        degree_centrality = nx.degree_centrality(graph)
        importance_info["degree_centrality"] = degree_centrality.get(node, 0)
        importance_info["degree_centrality_rank"] = sorted(
            all_nodes, 
            key=lambda x: degree_centrality.get(x, 0), 
            reverse=True
        ).index(node) + 1
        
        # 如果图连通，计算其他中心性
        if nx.is_connected(graph):
            # 接近中心性
            closeness_centrality = nx.closeness_centrality(graph)
            importance_info["closeness_centrality"] = closeness_centrality.get(node, 0)
            importance_info["closeness_centrality_rank"] = sorted(
                all_nodes, 
                key=lambda x: closeness_centrality.get(x, 0), 
                reverse=True
            ).index(node) + 1
            
            # 介数中心性
            betweenness_centrality = nx.betweenness_centrality(graph)
            importance_info["betweenness_centrality"] = betweenness_centrality.get(node, 0)
            importance_info["betweenness_centrality_rank"] = sorted(
                all_nodes, 
                key=lambda x: betweenness_centrality.get(x, 0), 
                reverse=True
            ).index(node) + 1
        
        # 聚类系数
        clustering = nx.clustering(graph)
        importance_info["clustering_coefficient"] = clustering.get(node, 0)
        
        # 局部重要性（邻居的度的平均值）
        neighbor_degrees = [graph.degree(neighbor) for neighbor in graph.neighbors(node)]
        importance_info["average_neighbor_degree"] = np.mean(neighbor_degrees) if neighbor_degrees else 0
        
        logging.info(f"节点 {node} 重要性分析完成")
        
    except Exception as e:
        logging.error(f"节点重要性分析失败: {str(e)}")
        importance_info["error"] = str(e)
    
    return importance_info

def extract_subgraph_by_nodes(graph: nx.Graph, 
                             nodes: List[str],
                             include_neighbors: bool = False,
                             neighbor_hops: int = 1) -> nx.Graph:
    """
    根据节点列表提取子图
    
    Args:
        graph: 原始图
        nodes: 节点列表
        include_neighbors: 是否包含邻居节点
        neighbor_hops: 邻居跳数
        
    Returns:
        提取的子图
    """
    try:
        # 确保节点存在于图中
        valid_nodes = [node for node in nodes if node in graph]
        
        if not valid_nodes:
            logging.warning("没有有效节点，返回空图")
            return nx.Graph()
        
        # 如果需要包含邻居
        if include_neighbors:
            extended_nodes = set(valid_nodes)
            
            for hop in range(neighbor_hops):
                current_neighbors = set()
                for node in extended_nodes:
                    current_neighbors.update(graph.neighbors(node))
                extended_nodes.update(current_neighbors)
            
            subgraph = graph.subgraph(extended_nodes).copy()
        else:
            subgraph = graph.subgraph(valid_nodes).copy()
        
        logging.info(f"提取子图完成，包含 {subgraph.number_of_nodes()} 个节点，{subgraph.number_of_edges()} 条边")
        
        return subgraph
        
    except Exception as e:
        logging.error(f"子图提取失败: {str(e)}")
        return nx.Graph()