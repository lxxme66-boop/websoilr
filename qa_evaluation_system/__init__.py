"""
QA Evaluation System

A comprehensive system for evaluating the quality of question-answer pairs
using multiple methods including rule-based checks, NLP metrics, and LLM evaluation.
"""

__version__ = "1.0.0"
__author__ = "QA Evaluation Team"

from .qa_evaluator import QAEvaluator
from .llm_evaluator import LLMEvaluator
from .nlp_metrics import NLPMetrics
from .rule_checker import RuleChecker

__all__ = [
    "QAEvaluator",
    "LLMEvaluator",
    "NLPMetrics",
    "RuleChecker"
]