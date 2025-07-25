from typing import List, Dict, Any
import random
import networkx as nx

class QuestionGenerator:
    """
    问题生成器：
    基于子图中节点与关系，设计QA问题，覆盖多种问题类型。
    """
    def __init__(self, question_templates: List[Dict[str, Any]]):
        self.templates = question_templates

    def generate_questions(self, subgraph, num_questions: int = 3) -> List[Dict[str, Any]]:
        """
        针对子图生成多种类型的问题。
        """
        questions = []
        nodes = list(subgraph.nodes(data=True))
        edges = list(subgraph.edges(data=True))
        for _ in range(num_questions):
            template = random.choice(self.templates)
            # 简单示例：实体属性问答、关系推理、路径推理等
            if template['type'] == 'entity_attr':
                node = random.choice(nodes)
                q = template['template'].format(entity=node[1].get('name', node[0]))
                a = node[1].get('attr', '未知')
            elif template['type'] == 'relation':
                edge = random.choice(edges)
                q = template['template'].format(
                    head=subgraph.nodes[edge[0]].get('name', edge[0]),
                    tail=subgraph.nodes[edge[1]].get('name', edge[1]),
                    relation=edge[2].get('type', '关系')
                )
                a = edge[2].get('desc', '未知')
            elif template['type'] == 'path':
                # 路径推理问题
                if len(nodes) >= 2:
                    n1, n2 = random.sample(nodes, 2)
                    q = template['template'].format(
                        start=n1[1].get('name', n1[0]),
                        end=n2[1].get('name', n2[0])
                    )
                    try:
                        path = nx.shortest_path(subgraph, n1[0], n2[0])
                        a = '->'.join([str(subgraph.nodes[n].get('name', n)) for n in path])
                    except Exception:
                        a = '无路径'
                else:
                    continue
            else:
                continue
            questions.append({'question': q, 'answer': a, 'type': template['type']})
        return questions