"""
问答对质量评测系统

一个高质量的问答对自动评测系统，结合大语言模型和传统NLP方法
"""

__version__ = "1.0.0"
__author__ = "QA Evaluation System"

from .evaluator import QAEvaluator, QAPair, EvaluationResult
from .data_processor import DataProcessor
from .llm_evaluator import LLMEvaluator
from .nlp_metrics import NLPMetrics
from .semantic_analyzer import SemanticAnalyzer
from .scorer import Scorer

__all__ = [
    'QAEvaluator',
    'QAPair',
    'EvaluationResult',
    'DataProcessor',
    'LLMEvaluator',
    'NLPMetrics',
    'SemanticAnalyzer',
    'Scorer'
]