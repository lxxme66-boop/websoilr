#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Graph Builder - 知识图谱构建器
从TCL工业领域文本中提取实体和关系，构建知识图谱

该模块实现：
1. 实体抽取：从文本中识别TCL工业相关实体
2. 关系抽取：识别实体间的语义关系
3. 图谱构建：将实体和关系组织成知识图谱结构
"""

import logging
import re
import json
from typing import Dict, List, Tuple, Any, Set
from pathlib import Path
from collections import defaultdict, Counter
import networkx as nx

# NLP相关库
try:
    import spacy
    import jieba
    import jieba.posseg as pseg
    from transformers import AutoTokenizer, AutoModel
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError as e:
    logging.warning(f"某些NLP库未安装: {e}")


class Entity:
    """实体类"""
    def __init__(self, name: str, entity_type: str, attributes: Dict[str, Any] = None):
        self.name = name
        self.entity_type = entity_type
        self.attributes = attributes or {}
        self.frequency = 1
        self.contexts = []
    
    def __repr__(self):
        return f"Entity(name='{self.name}', type='{self.entity_type}')"
    
    def to_dict(self):
        return {
            "name": self.name,
            "type": self.entity_type,
            "attributes": self.attributes,
            "frequency": self.frequency,
            "contexts": self.contexts
        }


class Relation:
    """关系类"""
    def __init__(self, head: str, relation: str, tail: str, confidence: float = 1.0):
        self.head = head
        self.relation = relation
        self.tail = tail
        self.confidence = confidence
        self.frequency = 1
        self.contexts = []
    
    def __repr__(self):
        return f"Relation({self.head} --{self.relation}--> {self.tail})"
    
    def to_dict(self):
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "confidence": self.confidence,
            "frequency": self.frequency,
            "contexts": self.contexts
        }


class KnowledgeGraph:
    """知识图谱类"""
    def __init__(self):
        self.entities = {}  # {entity_name: Entity}
        self.relations = []  # List[Relation]
        self.graph = nx.DiGraph()
    
    def add_entity(self, entity: Entity):
        """添加实体"""
        if entity.name in self.entities:
            self.entities[entity.name].frequency += 1
        else:
            self.entities[entity.name] = entity
            self.graph.add_node(entity.name, **entity.to_dict())
    
    def add_relation(self, relation: Relation):
        """添加关系"""
        # 检查是否已存在相同关系
        for existing_rel in self.relations:
            if (existing_rel.head == relation.head and 
                existing_rel.relation == relation.relation and
                existing_rel.tail == relation.tail):
                existing_rel.frequency += 1
                return
        
        self.relations.append(relation)
        self.graph.add_edge(
            relation.head, 
            relation.tail,
            relation=relation.relation,
            confidence=relation.confidence,
            frequency=relation.frequency
        )
    
    def get_neighbors(self, entity_name: str, relation_type: str = None) -> List[str]:
        """获取实体的邻居节点"""
        neighbors = []
        for neighbor in self.graph.neighbors(entity_name):
            edge_data = self.graph.get_edge_data(entity_name, neighbor)
            if relation_type is None or edge_data.get('relation') == relation_type:
                neighbors.append(neighbor)
        return neighbors
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "entities": {name: entity.to_dict() for name, entity in self.entities.items()},
            "relations": [rel.to_dict() for rel in self.relations],
            "statistics": {
                "num_entities": len(self.entities),
                "num_relations": len(self.relations),
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges()
            }
        }


class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化知识图谱构建器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.kg_config = config.get("knowledge_graph", {})
        self.domain_config = config.get("tcl_industry_domain", {})
        
        # 初始化NLP工具
        self._init_nlp_tools()
        
        # 加载领域词典
        self._load_domain_dictionaries()
        
        logging.info("知识图谱构建器初始化完成")
    
    def _init_nlp_tools(self):
        """初始化NLP工具"""
        try:
            # 初始化jieba分词
            jieba.initialize()
            
            # 加载spacy模型（如果可用）
            try:
                self.nlp = spacy.load("zh_core_web_sm")
            except OSError:
                logging.warning("spacy中文模型未安装，将使用基础方法")
                self.nlp = None
            
            # 初始化BERT模型（如果可用）
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
                self.bert_model = AutoModel.from_pretrained("bert-base-chinese")
                self.bert_model.eval()
            except Exception:
                logging.warning("BERT模型加载失败，将使用基础方法")
                self.tokenizer = None
                self.bert_model = None
                
        except Exception as e:
            logging.error(f"NLP工具初始化失败: {e}")
    
    def _load_domain_dictionaries(self):
        """加载TCL工业领域词典"""
        # TCL工业领域实体词典
        self.entity_patterns = {
            "产品": [
                r"TCL.*?电视", r".*?显示器", r".*?屏幕", r".*?面板",
                r"智能.*?终端", r".*?手机", r".*?平板", r".*?音响"
            ],
            "技术": [
                r"QLED.*?技术", r"Mini.*?LED", r"Micro.*?LED", r"OLED.*?技术",
                r"量子点.*?技术", r"HDR.*?技术", r"AI.*?技术", r"5G.*?技术"
            ],
            "公司": [
                r"TCL.*?集团", r"TCL.*?科技", r"华星光电", r"雷鸟.*?科技",
                r".*?有限公司", r".*?股份公司", r".*?集团.*?公司"
            ],
            "专家": [
                r".*?博士", r".*?教授", r".*?工程师", r".*?专家",
                r".*?总裁", r".*?总经理", r".*?CTO", r".*?CEO"
            ],
            "专利": [
                r".*?专利", r"发明.*?专利", r"实用新型.*?专利", r"外观设计.*?专利"
            ],
            "标准": [
                r".*?标准", r"国家.*?标准", r"行业.*?标准", r"企业.*?标准",
                r"ISO.*?标准", r"IEC.*?标准"
            ]
        }
        
        # 关系模式
        self.relation_patterns = {
            "开发": [r"开发", r"研发", r"设计", r"创造", r"制造"],
            "应用": [r"应用", r"使用", r"采用", r"搭载", r"集成"],
            "竞争": [r"竞争", r"对抗", r"抗衡", r"较量"],
            "合作": [r"合作", r"协作", r"联合", r"携手", r"共同"],
            "投资": [r"投资", r"注资", r"融资", r"入股"],
            "收购": [r"收购", r"并购", r"兼并", r"购买"],
            "技术转移": [r"技术转移", r"技术转让", r"技术授权", r"技术引进"],
            "标准制定": [r"制定.*?标准", r"参与.*?标准", r"标准.*?制定"]
        }
    
    def build_from_texts(self, input_dir: str) -> KnowledgeGraph:
        """
        从文本文件构建知识图谱
        
        Args:
            input_dir: 输入文本目录
            
        Returns:
            构建的知识图谱
        """
        logging.info(f"开始从 {input_dir} 构建知识图谱...")
        
        kg = KnowledgeGraph()
        
        # 读取所有文本文件
        input_path = Path(input_dir)
        if not input_path.exists():
            logging.error(f"输入目录不存在: {input_dir}")
            return kg
        
        text_files = list(input_path.glob("*.txt"))
        if not text_files:
            logging.warning(f"在 {input_dir} 中未找到文本文件")
            return kg
        
        # 处理每个文本文件
        for file_path in text_files:
            logging.info(f"处理文件: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 提取实体和关系
            entities = self._extract_entities(text)
            relations = self._extract_relations(text, entities)
            
            # 添加到知识图谱
            for entity in entities:
                kg.add_entity(entity)
            
            for relation in relations:
                kg.add_relation(relation)
        
        # 后处理：合并相似实体、过滤低频项等
        kg = self._post_process_kg(kg)
        
        logging.info(f"知识图谱构建完成: {len(kg.entities)}个实体, {len(kg.relations)}个关系")
        return kg
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            提取的实体列表
        """
        entities = []
        
        # 方法1: 基于规则的实体抽取
        entities.extend(self._rule_based_entity_extraction(text))
        
        # 方法2: 基于spacy的命名实体识别（如果可用）
        if self.nlp:
            entities.extend(self._spacy_entity_extraction(text))
        
        # 方法3: 基于jieba分词的实体识别
        entities.extend(self._jieba_entity_extraction(text))
        
        # 去重和过滤
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _rule_based_entity_extraction(self, text: str) -> List[Entity]:
        """基于规则的实体抽取"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group().strip()
                    if len(entity_name) > 1:  # 过滤单字符
                        entity = Entity(
                            name=entity_name,
                            entity_type=entity_type,
                            attributes={"extraction_method": "rule_based"}
                        )
                        entity.contexts.append(text[max(0, match.start()-50):match.end()+50])
                        entities.append(entity)
        
        return entities
    
    def _spacy_entity_extraction(self, text: str) -> List[Entity]:
        """基于spacy的实体抽取"""
        entities = []
        
        if not self.nlp:
            return entities
        
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                # 映射spacy标签到我们的实体类型
                entity_type = self._map_spacy_label(ent.label_)
                if entity_type:
                    entity = Entity(
                        name=ent.text.strip(),
                        entity_type=entity_type,
                        attributes={
                            "extraction_method": "spacy_ner",
                            "spacy_label": ent.label_,
                            "confidence": 0.8
                        }
                    )
                    entities.append(entity)
        except Exception as e:
            logging.warning(f"spacy实体抽取失败: {e}")
        
        return entities
    
    def _jieba_entity_extraction(self, text: str) -> List[Entity]:
        """基于jieba分词的实体识别"""
        entities = []
        
        try:
            # 使用jieba进行词性标注
            words = pseg.cut(text)
            
            for word, flag in words:
                word = word.strip()
                if len(word) > 1:
                    # 根据词性标注推断实体类型
                    entity_type = self._map_jieba_flag(flag, word)
                    if entity_type:
                        entity = Entity(
                            name=word,
                            entity_type=entity_type,
                            attributes={
                                "extraction_method": "jieba_pos",
                                "pos_tag": flag
                            }
                        )
                        entities.append(entity)
        except Exception as e:
            logging.warning(f"jieba实体抽取失败: {e}")
        
        return entities
    
    def _extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        从文本中提取关系
        
        Args:
            text: 输入文本
            entities: 已提取的实体列表
            
        Returns:
            提取的关系列表
        """
        relations = []
        
        # 创建实体名称到实体的映射
        entity_dict = {entity.name: entity for entity in entities}
        entity_names = list(entity_dict.keys())
        
        # 方法1: 基于模式的关系抽取
        relations.extend(self._pattern_based_relation_extraction(text, entity_names))
        
        # 方法2: 基于依存句法的关系抽取
        if self.nlp:
            relations.extend(self._dependency_relation_extraction(text, entity_names))
        
        # 方法3: 基于窗口的共现关系抽取
        relations.extend(self._cooccurrence_relation_extraction(text, entity_names))
        
        return relations
    
    def _pattern_based_relation_extraction(self, text: str, entity_names: List[str]) -> List[Relation]:
        """基于模式的关系抽取"""
        relations = []
        
        # 为每个实体名创建正则表达式
        entity_patterns = {name: re.escape(name) for name in entity_names}
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                # 构建关系模式：实体1 + 关系词 + 实体2
                for entity1 in entity_names:
                    for entity2 in entity_names:
                        if entity1 != entity2:
                            # 模式：实体1 关系词 实体2
                            relation_pattern = f"{re.escape(entity1)}.{{0,20}}{pattern}.{{0,20}}{re.escape(entity2)}"
                            matches = re.finditer(relation_pattern, text, re.IGNORECASE)
                            
                            for match in matches:
                                relation = Relation(
                                    head=entity1,
                                    relation=relation_type,
                                    tail=entity2,
                                    confidence=0.7
                                )
                                relation.contexts.append(match.group())
                                relations.append(relation)
        
        return relations
    
    def _dependency_relation_extraction(self, text: str, entity_names: List[str]) -> List[Relation]:
        """基于依存句法的关系抽取"""
        relations = []
        
        if not self.nlp:
            return relations
        
        try:
            doc = self.nlp(text)
            
            # 找到实体在句子中的位置
            entity_spans = {}
            for ent_name in entity_names:
                for match in re.finditer(re.escape(ent_name), text):
                    char_start, char_end = match.span()
                    # 找到对应的token
                    for token in doc:
                        if token.idx <= char_start < token.idx + len(token.text):
                            entity_spans[ent_name] = token
                            break
            
            # 基于依存关系提取实体关系
            for ent1_name, token1 in entity_spans.items():
                for ent2_name, token2 in entity_spans.items():
                    if ent1_name != ent2_name:
                        # 检查是否存在依存路径
                        path = self._find_dependency_path(token1, token2)
                        if path and len(path) <= 3:  # 限制路径长度
                            relation_type = self._infer_relation_from_path(path)
                            if relation_type:
                                relation = Relation(
                                    head=ent1_name,
                                    relation=relation_type,
                                    tail=ent2_name,
                                    confidence=0.6
                                )
                                relations.append(relation)
                                
        except Exception as e:
            logging.warning(f"依存句法关系抽取失败: {e}")
        
        return relations
    
    def _cooccurrence_relation_extraction(self, text: str, entity_names: List[str]) -> List[Relation]:
        """基于共现的关系抽取"""
        relations = []
        
        # 设置共现窗口大小
        window_size = 100
        
        for i, entity1 in enumerate(entity_names):
            for j, entity2 in enumerate(entity_names):
                if i >= j:  # 避免重复和自环
                    continue
                
                # 查找两个实体在文本中的位置
                entity1_positions = [m.start() for m in re.finditer(re.escape(entity1), text)]
                entity2_positions = [m.start() for m in re.finditer(re.escape(entity2), text)]
                
                # 检查是否在窗口内共现
                for pos1 in entity1_positions:
                    for pos2 in entity2_positions:
                        if abs(pos1 - pos2) <= window_size:
                            # 创建通用的"相关"关系
                            relation = Relation(
                                head=entity1,
                                relation="相关",
                                tail=entity2,
                                confidence=0.5
                            )
                            
                            # 提取共现上下文
                            start = min(pos1, pos2) - 50
                            end = max(pos1 + len(entity1), pos2 + len(entity2)) + 50
                            context = text[max(0, start):min(len(text), end)]
                            relation.contexts.append(context)
                            
                            relations.append(relation)
                            break  # 只添加一次关系
                    else:
                        continue
                    break
        
        return relations
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """去重实体"""
        entity_dict = {}
        
        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)
            if key in entity_dict:
                # 合并实体信息
                existing = entity_dict[key]
                existing.frequency += entity.frequency
                existing.contexts.extend(entity.contexts)
                # 合并属性
                existing.attributes.update(entity.attributes)
            else:
                entity_dict[key] = entity
        
        return list(entity_dict.values())
    
    def _post_process_kg(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """后处理知识图谱"""
        # 1. 合并相似实体
        if self.kg_config.get("merge_similar_entities", True):
            kg = self._merge_similar_entities(kg)
        
        # 2. 过滤低频实体和关系
        if self.kg_config.get("filter_low_frequency", True):
            kg = self._filter_low_frequency(kg)
        
        return kg
    
    def _merge_similar_entities(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """合并相似实体"""
        similarity_threshold = self.kg_config.get("similarity_threshold", 0.85)
        
        entities_to_merge = []
        entity_names = list(kg.entities.keys())
        
        for i, name1 in enumerate(entity_names):
            for j, name2 in enumerate(entity_names):
                if i >= j:
                    continue
                
                # 计算字符串相似度
                similarity = self._calculate_string_similarity(name1, name2)
                if similarity >= similarity_threshold:
                    entities_to_merge.append((name1, name2, similarity))
        
        # 执行合并
        for name1, name2, similarity in entities_to_merge:
            if name1 in kg.entities and name2 in kg.entities:
                # 保留频率更高的实体名
                entity1 = kg.entities[name1]
                entity2 = kg.entities[name2]
                
                if entity1.frequency >= entity2.frequency:
                    keep_name, remove_name = name1, name2
                else:
                    keep_name, remove_name = name2, name1
                
                # 合并实体信息
                keep_entity = kg.entities[keep_name]
                remove_entity = kg.entities[remove_name]
                
                keep_entity.frequency += remove_entity.frequency
                keep_entity.contexts.extend(remove_entity.contexts)
                keep_entity.attributes.update(remove_entity.attributes)
                
                # 更新关系中的实体引用
                for relation in kg.relations:
                    if relation.head == remove_name:
                        relation.head = keep_name
                    if relation.tail == remove_name:
                        relation.tail = keep_name
                
                # 删除被合并的实体
                del kg.entities[remove_name]
                if kg.graph.has_node(remove_name):
                    kg.graph.remove_node(remove_name)
        
        return kg
    
    def _filter_low_frequency(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """过滤低频实体和关系"""
        min_frequency = self.kg_config.get("min_frequency", 2)
        
        # 过滤低频实体
        entities_to_remove = []
        for name, entity in kg.entities.items():
            if entity.frequency < min_frequency:
                entities_to_remove.append(name)
        
        for name in entities_to_remove:
            del kg.entities[name]
            if kg.graph.has_node(name):
                kg.graph.remove_node(name)
        
        # 过滤涉及已删除实体的关系
        relations_to_remove = []
        for i, relation in enumerate(kg.relations):
            if (relation.head not in kg.entities or 
                relation.tail not in kg.entities or
                relation.frequency < min_frequency):
                relations_to_remove.append(i)
        
        # 从后往前删除，避免索引问题
        for i in reversed(relations_to_remove):
            del kg.relations[i]
        
        # 重建图结构
        kg.graph.clear()
        for entity in kg.entities.values():
            kg.graph.add_node(entity.name, **entity.to_dict())
        
        for relation in kg.relations:
            kg.graph.add_edge(
                relation.head,
                relation.tail,
                relation=relation.relation,
                confidence=relation.confidence,
                frequency=relation.frequency
            )
        
        return kg
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似度"""
        # 简单的编辑距离相似度
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(str1, str2)
        max_len = max(len(str1), len(str2))
        return 1 - (distance / max_len) if max_len > 0 else 1.0
    
    def _map_spacy_label(self, label: str) -> str:
        """映射spacy标签到实体类型"""
        label_mapping = {
            "ORG": "公司",
            "PERSON": "专家", 
            "PRODUCT": "产品",
            "TECH": "技术",
            "GPE": "市场"
        }
        return label_mapping.get(label)
    
    def _map_jieba_flag(self, flag: str, word: str) -> str:
        """根据jieba词性标注推断实体类型"""
        # 根据词性和词汇内容推断实体类型
        if flag in ['nr', 'nrf']:  # 人名
            return "专家"
        elif flag in ['nt', 'nz']:  # 机构名、其他专名
            if any(keyword in word for keyword in ['公司', '集团', '科技', '有限']):
                return "公司"
            elif any(keyword in word for keyword in ['技术', '系统', '方案']):
                return "技术"
            else:
                return "产品"
        elif flag == 'n':  # 一般名词
            if any(keyword in word for keyword in ['技术', '方法', '算法']):
                return "技术"
            elif any(keyword in word for keyword in ['产品', '设备', '终端']):
                return "产品"
        
        return None
    
    def _find_dependency_path(self, token1, token2):
        """查找两个token之间的依存路径"""
        # 简化版本：直接检查是否有直接依存关系
        if token1.head == token2 or token2.head == token1:
            return [token1, token2]
        
        # 检查是否有共同的head
        if token1.head == token2.head:
            return [token1, token1.head, token2]
        
        return None
    
    def _infer_relation_from_path(self, path):
        """从依存路径推断关系类型"""
        # 简化版本：根据路径中的词性和词汇推断关系
        if len(path) >= 2:
            middle_tokens = path[1:-1]
            for token in middle_tokens:
                if token.text in ['开发', '研发', '设计']:
                    return '开发'
                elif token.text in ['使用', '应用', '采用']:
                    return '应用'
                elif token.text in ['合作', '协作']:
                    return '合作'
        
        return '相关'  # 默认关系