"""
知识图谱构建器 - 改进版
从TCL工业领域文本中提取实体和关系，构建知识图谱
使用大语言模型进行实体和关系抽取
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
import networkx as nx
from tqdm import tqdm
import re
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """知识图谱构建器 - 使用大模型进行实体和关系抽取"""
    
    def __init__(self, config: dict):
        self.config = config
        self.kg_config = config.get('knowledge_graph', {})
        
        # 初始化实体和关系类型
        self.entity_types = self.kg_config.get('entity_types', [])
        self.relation_types = self.kg_config.get('relation_types', [])
        
        # 初始化TCL特定术语
        self.tcl_terms = config.get('tcl_specific', {}).get('technical_terms', {})
        
        # 加载大语言模型
        self.model_path = config['models'].get('llm_model', {}).get('path', 'THUDM/chatglm3-6b')
        self._load_llm_model()
        
        # 知识图谱
        self.graph = nx.MultiDiGraph()
        
        # 配置参数
        self.chunk_size = self.kg_config.get('chunk_size', 1000)
        self.max_chunk_overlap = self.kg_config.get('max_chunk_overlap', 200)
        self.confidence_threshold = self.kg_config.get('extraction_rules', {}).get('confidence_threshold', 0.7)
        
    def _load_llm_model(self):
        """加载大语言模型"""
        logger.info(f"加载大语言模型: {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model.eval()
            logger.info("大语言模型加载成功")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise