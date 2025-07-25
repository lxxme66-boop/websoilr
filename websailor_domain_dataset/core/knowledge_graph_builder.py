import networkx as nx
from typing import List, Dict, Any

class KnowledgeGraphBuilder:
    """
    知识图谱构建器：
    支持从结构化/半结构化文本构建networkx图。
    """
    def __init__(self):
        pass

    def build_from_triples(self, triples: List[Dict[str, Any]]) -> nx.Graph:
        """
        通过三元组列表构建知识图谱。
        triples: [{"head": ..., "relation": ..., "tail": ..., "head_attr": ..., "tail_attr": ..., "rel_desc": ...}]
        """
        G = nx.Graph()
        for t in triples:
            G.add_node(t['head'], name=t['head'], attr=t.get('head_attr', ''))
            G.add_node(t['tail'], name=t['tail'], attr=t.get('tail_attr', ''))
            G.add_edge(t['head'], t['tail'], type=t['relation'], desc=t.get('rel_desc', ''))
        return G