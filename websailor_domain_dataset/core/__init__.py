#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSailor Domain Dataset - Core Module
核心功能模块包

包含以下核心组件：
- KnowledgeGraphBuilder: 知识图谱构建器
- SubgraphSampler: 子图采样器 (WebSailor核心思想1)
- QuestionGenerator: 问题生成器 (WebSailor核心思想2) 
- ObfuscationProcessor: 模糊化处理器 (WebSailor核心思想3)
- TrajectoryGenerator: 推理轨迹生成器
- DataSynthesizer: 数据综合器
"""

__version__ = "1.0.0"
__author__ = "WebSailor Domain Dataset Team"

from .knowledge_graph_builder import KnowledgeGraphBuilder
from .subgraph_sampler import SubgraphSampler
from .question_generator import QuestionGenerator
from .obfuscation_processor import ObfuscationProcessor
from .trajectory_generator import TrajectoryGenerator
from .data_synthesizer import DataSynthesizer

__all__ = [
    "KnowledgeGraphBuilder",
    "SubgraphSampler", 
    "QuestionGenerator",
    "ObfuscationProcessor",
    "TrajectoryGenerator",
    "DataSynthesizer"
]