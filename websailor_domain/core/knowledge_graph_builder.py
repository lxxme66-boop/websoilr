"""
知识图谱构建器
从输入文本中提取实体和关系，构建领域知识图谱
"""

import json
import logging
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import networkx as nx
import jieba
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
import re

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """实体类"""
    text: str
    type: str
    start: int
    end: int
    confidence: float = 1.0
    
    def __hash__(self):
        return hash((self.text, self.type))
    
    def __eq__(self, other):
        return self.text == other.text and self.type == other.type


@dataclass
class Relation:
    """关系类"""
    head: Entity
    tail: Entity
    type: str
    confidence: float = 1.0
    context: str = ""


class KnowledgeGraphBuilder:
    """
    知识图谱构建器
    实现WebSailor的核心思想：从文本中构建结构化的知识图谱
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # TCL工业垂域实体类型
        self.entity_types = [
            "产品", "技术", "工艺", "材料", "设备",
            "标准", "专利", "公司", "人员", "项目",
            "参数", "指标", "方法", "流程", "组件"
        ]
        
        # TCL工业垂域关系类型
        self.relation_types = [
            "使用", "包含", "生产", "研发", "依赖",
            "改进", "替代", "认证", "合作", "应用于",
            "基于", "优化", "集成", "测试", "制造"
        ]
        
        # 初始化模型
        self._initialize_models()
        
        # 初始化领域词典
        self._initialize_domain_dict()
        
    def _initialize_models(self):
        """初始化NLP模型"""
        logger.info("初始化知识图谱提取模型...")
        
        # 加载embedding模型
        model_path = self.config['models']['kg_extractor_model']['path']
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        # 加载spaCy模型（如果有中文模型）
        try:
            self.nlp = spacy.load("zh_core_web_sm")
        except:
            logger.warning("未找到中文spaCy模型，将使用基础分词")
            self.nlp = None
            
    def _initialize_domain_dict(self):
        """初始化TCL工业领域词典"""
        self.domain_terms = {
            "显示技术": ["LCD", "OLED", "Mini-LED", "Micro-LED", "量子点", "背光", "面板", "像素"],
            "家电": ["压缩机", "变频", "节能", "制冷", "空调", "冰箱", "洗衣机", "智能控制"],
            "半导体": ["芯片", "封装", "制程", "光刻", "晶圆", "蚀刻", "沉积", "掺杂"],
            "制造": ["自动化", "工业4.0", "MES", "数字孪生", "柔性制造", "精益生产", "质量控制"],
            "材料": ["复合材料", "纳米材料", "功能材料", "环保材料", "导电材料", "绝缘材料"]
        }
        
        # 构建领域词典用于分词
        for terms in self.domain_terms.values():
            for term in terms:
                jieba.add_word(term)
                
    def build_from_texts(self, texts: List[str]) -> nx.DiGraph:
        """
        从文本列表构建知识图谱
        
        Args:
            texts: 输入文本列表
            
        Returns:
            nx.DiGraph: 构建的知识图谱
        """
        logger.info(f"开始从{len(texts)}个文本构建知识图谱...")
        
        kg = nx.DiGraph()
        all_entities = []
        all_relations = []
        
        for text in tqdm(texts, desc="提取实体和关系"):
            # 提取实体
            entities = self._extract_entities(text)
            all_entities.extend(entities)
            
            # 提取关系
            relations = self._extract_relations(text, entities)
            all_relations.extend(relations)
            
        # 构建图
        self._build_graph(kg, all_entities, all_relations)
        
        logger.info(f"知识图谱构建完成: {kg.number_of_nodes()}个节点, {kg.number_of_edges()}条边")
        return kg
        
    def _extract_entities(self, text: str) -> List[Entity]:
        """
        从文本中提取实体
        使用规则+模型的混合方法
        """
        entities = []
        
        # 1. 基于规则的实体提取
        entities.extend(self._rule_based_extraction(text))
        
        # 2. 基于模型的实体提取（如果需要）
        if self.nlp:
            entities.extend(self._model_based_extraction(text))
            
        # 3. 去重和合并
        entities = self._merge_entities(entities)
        
        return entities
        
    def _rule_based_extraction(self, text: str) -> List[Entity]:
        """基于规则的实体提取"""
        entities = []
        
        # 产品型号模式 (如: TCL-65Q10G, X11G等)
        product_pattern = r'TCL[-\s]?[A-Z0-9]+[A-Z0-9\-]*'
        for match in re.finditer(product_pattern, text):
            entities.append(Entity(
                text=match.group(),
                type="产品",
                start=match.start(),
                end=match.end(),
                confidence=0.9
            ))
            
        # 技术参数模式 (如: 120Hz, 4K, 8K等)
        tech_pattern = r'\d+[KkGgMmHhz]+\b|\d+×\d+'
        for match in re.finditer(tech_pattern, text):
            entities.append(Entity(
                text=match.group(),
                type="参数",
                start=match.start(),
                end=match.end(),
                confidence=0.8
            ))
            
        # 领域术语匹配
        for domain, terms in self.domain_terms.items():
            for term in terms:
                if term in text:
                    start = text.find(term)
                    entities.append(Entity(
                        text=term,
                        type="技术",
                        start=start,
                        end=start + len(term),
                        confidence=0.85
                    ))
                    
        return entities
        
    def _model_based_extraction(self, text: str) -> List[Entity]:
        """基于模型的实体提取"""
        entities = []
        
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                # 映射到我们的实体类型
                entity_type = self._map_entity_type(ent.label_)
                if entity_type:
                    entities.append(Entity(
                        text=ent.text,
                        type=entity_type,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.7
                    ))
                    
        return entities
        
    def _map_entity_type(self, label: str) -> Optional[str]:
        """映射spaCy实体类型到我们的类型系统"""
        mapping = {
            "ORG": "公司",
            "PERSON": "人员",
            "PRODUCT": "产品",
            "FAC": "设备",
            "LAW": "标准"
        }
        return mapping.get(label, None)
        
    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """合并重复实体"""
        unique_entities = {}
        
        for entity in entities:
            key = (entity.text, entity.type)
            if key not in unique_entities:
                unique_entities[key] = entity
            else:
                # 保留置信度更高的
                if entity.confidence > unique_entities[key].confidence:
                    unique_entities[key] = entity
                    
        return list(unique_entities.values())
        
    def _extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        从文本中提取实体间的关系
        """
        relations = []
        
        # 基于规则的关系提取
        relations.extend(self._rule_based_relation_extraction(text, entities))
        
        # 基于模型的关系提取（使用embedding相似度）
        relations.extend(self._model_based_relation_extraction(text, entities))
        
        return relations
        
    def _rule_based_relation_extraction(self, text: str, entities: List[Entity]) -> List[Relation]:
        """基于规则的关系提取"""
        relations = []
        
        # 定义关系模式
        relation_patterns = {
            "使用": ["使用", "采用", "应用", "配备"],
            "包含": ["包含", "包括", "含有", "集成"],
            "生产": ["生产", "制造", "生产线", "制造商"],
            "研发": ["研发", "开发", "研制", "创新"],
            "基于": ["基于", "基础上", "依托", "建立在"]
        }
        
        # 对每对实体检查是否存在关系
        for i, head in enumerate(entities):
            for j, tail in enumerate(entities):
                if i == j:
                    continue
                    
                # 获取两个实体之间的文本
                start = min(head.end, tail.end)
                end = max(head.start, tail.start)
                
                if start < end and end - start < 50:  # 限制距离
                    context = text[start:end]
                    
                    # 检查是否包含关系词
                    for rel_type, patterns in relation_patterns.items():
                        for pattern in patterns:
                            if pattern in context:
                                relations.append(Relation(
                                    head=head,
                                    tail=tail,
                                    type=rel_type,
                                    confidence=0.8,
                                    context=context
                                ))
                                break
                                
        return relations
        
    def _model_based_relation_extraction(self, text: str, entities: List[Entity]) -> List[Relation]:
        """基于模型的关系提取（使用语义相似度）"""
        relations = []
        
        # 这里简化处理，实际可以使用更复杂的关系抽取模型
        # 基于实体在句子中的共现来推断关系
        sentences = text.split('。')
        
        for sentence in sentences:
            entities_in_sent = [e for e in entities if e.text in sentence]
            
            # 如果句子中有多个实体，可能存在关系
            if len(entities_in_sent) >= 2:
                # 使用embedding计算语义相关性
                embeddings = self._get_embeddings([e.text for e in entities_in_sent])
                
                for i in range(len(entities_in_sent)):
                    for j in range(i+1, len(entities_in_sent)):
                        similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                        
                        if similarity > 0.7:  # 相似度阈值
                            # 推断关系类型
                            rel_type = self._infer_relation_type(
                                entities_in_sent[i], 
                                entities_in_sent[j],
                                sentence
                            )
                            
                            relations.append(Relation(
                                head=entities_in_sent[i],
                                tail=entities_in_sent[j],
                                type=rel_type,
                                confidence=similarity,
                                context=sentence
                            ))
                            
        return relations
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """获取文本的embedding"""
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # 平均池化
            
        return embeddings.cpu().numpy()
        
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
    def _infer_relation_type(self, entity1: Entity, entity2: Entity, context: str) -> str:
        """推断关系类型"""
        # 基于实体类型的启发式规则
        if entity1.type == "产品" and entity2.type == "技术":
            return "使用"
        elif entity1.type == "公司" and entity2.type == "产品":
            return "生产"
        elif entity1.type == "技术" and entity2.type == "技术":
            return "基于"
        else:
            return "相关"
            
    def _build_graph(self, kg: nx.DiGraph, entities: List[Entity], relations: List[Relation]):
        """构建NetworkX图"""
        # 添加节点
        for entity in entities:
            kg.add_node(
                entity.text,
                type=entity.type,
                confidence=entity.confidence
            )
            
        # 添加边
        for relation in relations:
            kg.add_edge(
                relation.head.text,
                relation.tail.text,
                type=relation.type,
                confidence=relation.confidence,
                context=relation.context
            )
            
    def save_graph(self, kg: nx.DiGraph, path: str):
        """保存知识图谱"""
        data = {
            "nodes": [
                {
                    "id": node,
                    "type": kg.nodes[node].get("type", ""),
                    "confidence": kg.nodes[node].get("confidence", 1.0)
                }
                for node in kg.nodes()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "type": kg.edges[u, v].get("type", ""),
                    "confidence": kg.edges[u, v].get("confidence", 1.0)
                }
                for u, v in kg.edges()
            ]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"知识图谱已保存到: {path}")