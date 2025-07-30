"""
测试子图格式处理
验证 NetworkX DiGraph 格式是否正确处理
"""

import networkx as nx
from typing import Dict, List

def test_subgraph_format():
    """测试子图格式"""
    # 创建一个示例子图
    subgraph = nx.DiGraph()
    
    # 添加节点
    subgraph.add_node('TCL电视', type='产品', confidence=0.9)
    subgraph.add_node('QLED技术', type='技术', confidence=0.8)
    subgraph.add_node('量子点材料', type='材料', confidence=0.85)
    
    # 添加边（使用 'type' 属性而不是 'relation'）
    subgraph.add_edge('TCL电视', 'QLED技术', type='使用', confidence=0.9)
    subgraph.add_edge('QLED技术', '量子点材料', type='依赖', confidence=0.8)
    
    # 设置子图属性
    subgraph.graph['topology'] = 'chain'
    subgraph.graph['complexity'] = 0.6
    
    # 测试节点访问
    print("节点数量:", subgraph.number_of_nodes())
    print("边数量:", subgraph.number_of_edges())
    
    # 测试节点类型提取
    node_types = set()
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        if 'type' in node_data:
            node_types.add(node_data['type'])
    print("节点类型:", node_types)
    
    # 测试边类型提取
    relation_types = set()
    for u, v, data in subgraph.edges(data=True):
        if 'type' in data:
            relation_types.add(data['type'])
        elif 'relation' in data:
            relation_types.add(data['relation'])
    print("关系类型:", relation_types)
    
    # 测试拓扑属性访问
    print("拓扑类型:", subgraph.graph.get('topology', 'unknown'))
    
    # 测试边的格式化
    print("\n边信息:")
    for u, v, data in subgraph.edges(data=True):
        relation = data.get('type', data.get('relation', 'unknown'))
        print(f"  {u} --[{relation}]--> {v}")
    
    return subgraph

if __name__ == "__main__":
    print("测试子图格式处理...")
    test_subgraph_format()
    print("\n测试完成！")