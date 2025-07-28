"""
WebSailor Domain-Specific Dataset Construction Core Modules
核心模块包，实现WebSailor的三大核心思想
"""

from .knowledge_graph_builder import KnowledgeGraphBuilder
from .subgraph_sampler import SubgraphSampler
from .question_generator import QuestionGenerator
from .obfuscation_processor import ObfuscationProcessor
from .trajectory_generator import TrajectoryGenerator
from .data_synthesizer import DataSynthesizer

__all__ = [
    'KnowledgeGraphBuilder',
    'SubgraphSampler',
    'QuestionGenerator',
    'ObfuscationProcessor',
    'TrajectoryGenerator',
    'DataSynthesizer'
]

__version__ = '1.0.0'