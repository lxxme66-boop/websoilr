import networkx as nx
from typing import List, Dict, Any
from .subgraph_sampler import SubgraphSampler
from .question_generator import QuestionGenerator
from .obfuscation_processor import ObfuscationProcessor

class DataSynthesizer:
    """
    数据综合器：
    串联子图采样、问题生成、模糊化处理，输出最终数据集。
    """
    def __init__(self, graph: nx.Graph, question_templates: List[Dict[str, Any]], obfuscation_patterns: List[Dict[str, Any]]):
        self.sampler = SubgraphSampler(graph)
        self.qgen = QuestionGenerator(question_templates)
        self.obfuscator = ObfuscationProcessor(obfuscation_patterns)

    def synthesize(self, num_subgraphs: int = 10, questions_per_subgraph: int = 3, obfuscate: bool = True) -> List[Dict[str, Any]]:
        dataset = []
        subgraphs = self.sampler.sample_subgraphs(num_subgraphs)
        for subgraph in subgraphs:
            qa_pairs = self.qgen.generate_questions(subgraph, questions_per_subgraph)
            for qa in qa_pairs:
                if obfuscate:
                    qa = self.obfuscator.obfuscate(qa)
                dataset.append(qa)
        return dataset