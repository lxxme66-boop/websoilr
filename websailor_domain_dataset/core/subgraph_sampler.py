import networkx as nx
import random
from typing import List, Dict, Any

class SubgraphSampler:
    """
    子图采样器：
    从整个知识图中抽取不同拓扑的子图作为问题候选基础。
    每个子图代表了一种“任务场景”，可能包含多个目标、干扰信息、隐含路径。
    """
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def sample_subgraphs(self, num_subgraphs: int = 10, min_size: int = 3, max_size: int = 7) -> List[nx.Graph]:
        """
        随机采样若干子图，保证多样性和覆盖不同拓扑结构。
        """
        subgraphs = []
        nodes = list(self.graph.nodes)
        for _ in range(num_subgraphs):
            size = random.randint(min_size, max_size)
            # 随机选择一个中心节点
            center = random.choice(nodes)
            # 以中心节点为起点，BFS采样size个节点
            sampled_nodes = self._bfs_sample(center, size)
            subgraph = self.graph.subgraph(sampled_nodes).copy()
            subgraphs.append(subgraph)
        return subgraphs

    def _bfs_sample(self, start_node: Any, size: int) -> List[Any]:
        """
        以BFS方式采样size个节点，保证子图连通。
        """
        visited = set()
        queue = [start_node]
        while queue and len(visited) < size:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                neighbors = list(self.graph.neighbors(node))
                random.shuffle(neighbors)
                queue.extend(neighbors)
        return list(visited)[:size]