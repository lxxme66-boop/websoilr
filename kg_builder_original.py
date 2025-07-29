"""
知识图谱构建器
从TCL工业领域文本中提取实体和关系，构建知识图谱
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set
import networkx as nx
from tqdm import tqdm
import jieba
import jieba.posseg as pseg
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.kg_config = config.get('knowledge_graph', {})
        
        # 初始化实体和关系类型
        self.entity_types = self.kg_config.get('entity_types', [])
        self.relation_types = self.kg_config.get('relation_types', [])
        
        # 初始化TCL特定术语
        self.tcl_terms = config.get('tcl_specific', {}).get('technical_terms', {})
        self._init_jieba_dict()
        
        # 加载知识图谱提取模型
        self.kg_extractor_path = config['models']['kg_extractor_model']['path']
        self._load_kg_extractor()
        
        # 知识图谱
        self.graph = nx.MultiDiGraph()
        
    def _init_jieba_dict(self):
        """初始化jieba词典，添加TCL专业术语"""
        logger.info("初始化jieba词典...")
        
        # 添加TCL技术术语
        for category, terms in self.tcl_terms.items():
            for term in terms:
                jieba.add_word(term)
        
        # 添加实体类型作为词
        for entity_type in self.entity_types:
            jieba.add_word(entity_type)