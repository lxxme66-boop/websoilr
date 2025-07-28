"""
知识图谱构建器
从文本中提取实体和关系，构建领域知识图谱
"""

import logging
from typing import List, Dict, Tuple, Any, Set
from pathlib import Path
import networkx as nx
import json
import re
from collections import defaultdict
from tqdm import tqdm


class KnowledgeGraphBuilder:
    """
    知识图谱构建器
    负责从领域文本中构建知识图谱
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.entity_types = config.get('entity_types', [])
        self.relation_types = config.get('relation_types', [])
        self.extraction_config = config.get('extraction', {})
        
        # 初始化实体和关系模式
        self._init_patterns()
        
    def _init_patterns(self):
        """初始化实体和关系识别模式"""
        # TCL工业领域的实体模式
        self.entity_patterns = {
            'Product': [
                r'TCL[\s\-]?\w+',
                r'\w+显示器',
                r'\w+电视',
                r'\w+面板',
                r'智能\w+',
            ],
            'Technology': [
                r'QLED技术',
                r'Mini[\s\-]?LED',
                r'量子点\w*',
                r'HDR\d*',
                r'AI\w*技术',
            ],
            'Component': [
                r'\w+芯片',
                r'\w+模组',
                r'\w+背光',
                r'\w+驱动',
            ],
            'Material': [
                r'\w+材料',
                r'\w+基板',
                r'\w+薄膜',
            ],
            'Process': [
                r'\w+工艺',
                r'\w+制程',
                r'\w+流程',
            ],
            'Standard': [
                r'ISO[\s\-]?\d+',
                r'IEC[\s\-]?\d+',
                r'RoHS',
                r'CE认证',
            ],
            'Company': [
                r'TCL',
                r'\w+公司',
                r'\w+集团',
                r'\w+企业',
            ]
        }
        
        # 关系模式
        self.relation_patterns = {
            'manufactures': [
                r'(\w+)生产(\w+)',
                r'(\w+)制造(\w+)',
                r'(\w+)推出(\w+)',
            ],
            'uses_technology': [
                r'(\w+)采用(\w+技术)',
                r'(\w+)使用(\w+)',
                r'(\w+)应用(\w+)',
            ],
            'contains_component': [
                r'(\w+)包含(\w+)',
                r'(\w+)内置(\w+)',
                r'(\w+)配备(\w+)',
            ],
            'developed_by': [
                r'(\w+)由(\w+)开发',
                r'(\w+)研发了(\w+)',
            ],
            'complies_with_standard': [
                r'(\w+)符合(\w+标准)',
                r'(\w+)通过(\w+认证)',
            ]
        }
        
    def build_from_texts(self, input_dir: Path) -> nx.Graph:
        """
        从文本目录构建知识图谱
        """
        self.logger.info(f"Building knowledge graph from texts in {input_dir}")
        
        # 初始化知识图谱
        kg = nx.Graph()
        
        # 读取所有文本文件
        text_files = list(input_dir.glob("*.txt"))
        if not text_files:
            self.logger.warning(f"No text files found in {input_dir}")
            return kg
            
        # 处理每个文本文件
        all_entities = []
        all_relations = []
        
        for text_file in tqdm(text_files, desc="Processing texts"):
            entities, relations = self._process_text_file(text_file)
            all_entities.extend(entities)
            all_relations.extend(relations)
            
        # 构建图
        self._build_graph(kg, all_entities, all_relations)
        
        # 添加推断的关系
        if self.extraction_config.get('use_inference', True):
            self._add_inferred_relations(kg)
            
        self.logger.info(f"Knowledge graph built with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges")
        
        return kg
        
    def _process_text_file(self, text_file: Path) -> Tuple[List[Dict], List[Dict]]:
        """处理单个文本文件"""
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 分句
        sentences = self._split_sentences(text)
        
        entities = []
        relations = []
        
        for sentence in sentences:
            # 提取实体
            sent_entities = self._extract_entities(sentence)
            entities.extend(sent_entities)
            
            # 提取关系
            sent_relations = self._extract_relations(sentence, sent_entities)
            relations.extend(sent_relations)
            
        return entities, relations
        
    def _split_sentences(self, text: str) -> List[str]:
        """文本分句"""
        # 简单的分句规则
        sentences = re.split(r'[。！？\n]+', text)
        return [s.strip() for s in sentences if s.strip()]
        
    def _extract_entities(self, sentence: str) -> List[Dict]:
        """从句子中提取实体"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    entity = {
                        'text': match.group(),
                        'type': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'sentence': sentence
                    }
                    entities.append(entity)
                    
        # 去重
        entities = self._deduplicate_entities(entities)
        
        return entities
        
    def _extract_relations(self, sentence: str, entities: List[Dict]) -> List[Dict]:
        """从句子中提取关系"""
        relations = []
        
        # 基于模式的关系抽取
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    if len(match.groups()) >= 2:
                        source_text = match.group(1)
                        target_text = match.group(2)
                        
                        # 匹配到实体
                        source_entity = self._find_entity(source_text, entities)
                        target_entity = self._find_entity(target_text, entities)
                        
                        if source_entity and target_entity:
                            relation = {
                                'source': source_entity['text'],
                                'target': target_entity['text'],
                                'type': relation_type,
                                'sentence': sentence
                            }
                            relations.append(relation)
                            
        # 基于依存句法的关系抽取（如果配置启用）
        if self.extraction_config.get('use_dependency_parsing', False):
            dep_relations = self._extract_relations_by_dependency(sentence, entities)
            relations.extend(dep_relations)
            
        return relations
        
    def _find_entity(self, text: str, entities: List[Dict]) -> Dict:
        """在实体列表中查找匹配的实体"""
        for entity in entities:
            if text in entity['text'] or entity['text'] in text:
                return entity
        return None
        
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """实体去重"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity['text'], entity['type'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
                
        return unique_entities
        
    def _build_graph(
        self, 
        kg: nx.Graph, 
        entities: List[Dict], 
        relations: List[Dict]
    ):
        """构建知识图谱"""
        # 添加节点
        for entity in entities:
            kg.add_node(
                entity['text'],
                type=entity['type'],
                sentences=[entity['sentence']]
            )
            
        # 合并相同节点的句子
        for node in kg.nodes():
            sentences = []
            for entity in entities:
                if entity['text'] == node:
                    sentences.append(entity['sentence'])
            kg.nodes[node]['sentences'] = list(set(sentences))
            
        # 添加边
        for relation in relations:
            if kg.has_node(relation['source']) and kg.has_node(relation['target']):
                kg.add_edge(
                    relation['source'],
                    relation['target'],
                    relation=relation['type'],
                    sentences=[relation['sentence']]
                )
                
        # 合并相同边的句子
        for source, target in kg.edges():
            sentences = []
            for relation in relations:
                if (relation['source'] == source and relation['target'] == target) or \
                   (relation['source'] == target and relation['target'] == source):
                    sentences.append(relation['sentence'])
            kg[source][target]['sentences'] = list(set(sentences))
            
    def _add_inferred_relations(self, kg: nx.Graph):
        """添加推断的关系"""
        # 推断规则示例
        
        # 规则1：如果A制造B，B包含C，则A间接使用C
        new_edges = []
        for node in kg.nodes():
            # 找出node制造的产品
            manufactured = [
                n for n in kg.neighbors(node)
                if kg[node][n].get('relation') == 'manufactures'
            ]
            
            for product in manufactured:
                # 找出产品包含的组件
                components = [
                    n for n in kg.neighbors(product)
                    if kg[product][n].get('relation') == 'contains_component'
                ]
                
                for component in components:
                    if not kg.has_edge(node, component):
                        new_edges.append((
                            node,
                            component,
                            {'relation': 'uses_indirectly', 'inferred': True}
                        ))
                        
        # 添加推断的边
        for source, target, attrs in new_edges:
            kg.add_edge(source, target, **attrs)
            
    def _extract_relations_by_dependency(
        self, 
        sentence: str, 
        entities: List[Dict]
    ) -> List[Dict]:
        """基于依存句法的关系抽取（简化版本）"""
        # 这里应该使用实际的依存句法分析器
        # 简化实现：基于实体之间的距离和动词
        relations = []
        
        # 查找句子中的动词
        verb_patterns = ['生产', '制造', '使用', '包含', '开发', '符合']
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i >= j:
                    continue
                    
                # 检查两个实体之间是否有动词
                start = min(entity1['end'], entity2['end'])
                end = max(entity1['start'], entity2['start'])
                
                if start < end:
                    between_text = sentence[start:end]
                    
                    for verb in verb_patterns:
                        if verb in between_text:
                            # 推断关系类型
                            relation_type = self._infer_relation_type(verb)
                            if relation_type:
                                relations.append({
                                    'source': entity1['text'],
                                    'target': entity2['text'],
                                    'type': relation_type,
                                    'sentence': sentence
                                })
                                break
                                
        return relations
        
    def _infer_relation_type(self, verb: str) -> str:
        """根据动词推断关系类型"""
        verb_to_relation = {
            '生产': 'manufactures',
            '制造': 'manufactures',
            '使用': 'uses_technology',
            '包含': 'contains_component',
            '开发': 'developed_by',
            '符合': 'complies_with_standard'
        }
        
        return verb_to_relation.get(verb, None)