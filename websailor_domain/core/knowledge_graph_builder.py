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
            
    def _load_kg_extractor(self):
        """加载知识图谱提取模型"""
        logger.info(f"加载知识图谱提取模型: {self.kg_extractor_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.kg_extractor_path)
            self.model = AutoModel.from_pretrained(self.kg_extractor_path)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.warning(f"无法加载指定模型，使用默认BERT模型: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            self.model = AutoModel.from_pretrained('bert-base-chinese')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
    
    def build_from_texts(self, input_dir: str) -> nx.MultiDiGraph:
        """从文本目录构建知识图谱"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.error(f"输入目录不存在: {input_dir}")
            return self.graph
            
        # 读取所有文本文件
        text_files = list(input_path.glob("*.txt"))
        logger.info(f"找到 {len(text_files)} 个文本文件")
        
        all_entities = []
        all_relations = []
        
        # 处理每个文本文件
        for text_file in tqdm(text_files, desc="处理文本文件"):
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # 提取实体和关系
            entities, relations = self._extract_entities_relations(text)
            all_entities.extend(entities)
            all_relations.extend(relations)
        
        # 构建知识图谱
        self._build_graph(all_entities, all_relations)
        
        logger.info(f"知识图谱构建完成: {self.graph.number_of_nodes()} 个节点, "
                   f"{self.graph.number_of_edges()} 条边")
        
        return self.graph
    
    def _extract_entities_relations(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """从文本中提取实体和关系"""
        entities = []
        relations = []
        
        # 分句处理
        sentences = self._split_sentences(text)
        
        for sent in sentences:
            # 提取实体
            sent_entities = self._extract_entities(sent)
            entities.extend(sent_entities)
            
            # 提取关系
            if len(sent_entities) >= 2:
                sent_relations = self._extract_relations(sent, sent_entities)
                relations.extend(sent_relations)
        
        return entities, relations
    
    def _split_sentences(self, text: str) -> List[str]:
        """文本分句"""
        # 简单的分句规则
        sentences = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # 按句号、问号、感叹号分句
            import re
            sents = re.split(r'[。！？]', line)
            sentences.extend([s.strip() for s in sents if s.strip()])
            
        return sentences
    
    def _extract_entities(self, sentence: str) -> List[Dict]:
        """从句子中提取实体"""
        entities = []
        
        # 使用jieba进行词性标注
        words = pseg.cut(sentence)
        
        # 基于规则的实体识别
        current_entity = []
        current_type = None
        
        for word, pos in words:
            # 识别潜在实体
            entity_type = self._identify_entity_type(word, pos)
            
            if entity_type:
                if current_entity and current_type != entity_type:
                    # 保存之前的实体
                    entities.append({
                        'text': ''.join(current_entity),
                        'type': current_type,
                        'start': sentence.find(''.join(current_entity)),
                        'sentence': sentence
                    })
                    current_entity = [word]
                    current_type = entity_type
                else:
                    current_entity.append(word)
                    current_type = entity_type
            else:
                if current_entity:
                    # 保存当前实体
                    entities.append({
                        'text': ''.join(current_entity),
                        'type': current_type,
                        'start': sentence.find(''.join(current_entity)),
                        'sentence': sentence
                    })
                    current_entity = []
                    current_type = None
        
        # 处理最后一个实体
        if current_entity:
            entities.append({
                'text': ''.join(current_entity),
                'type': current_type,
                'start': sentence.find(''.join(current_entity)),
                'sentence': sentence
            })
        
        # 使用深度模型增强实体识别
        enhanced_entities = self._enhance_entity_recognition(sentence, entities)
        
        return enhanced_entities
    
    def _identify_entity_type(self, word: str, pos: str) -> str:
        """识别实体类型"""
        # 检查是否是TCL专业术语
        for category, terms in self.tcl_terms.items():
            if word in terms:
                # 映射到实体类型
                if category in ['display', 'semiconductor']:
                    return '技术'
                elif category == 'appliance':
                    return '产品'
                elif category == 'materials':
                    return '材料'
                elif category == 'manufacturing':
                    return '工艺'
        
        # 基于词性的简单规则
        if pos.startswith('n'):  # 名词
            if any(keyword in word for keyword in ['公司', '集团', '企业']):
                return '公司'
            elif any(keyword in word for keyword in ['技术', '方法', '算法']):
                return '技术'
            elif any(keyword in word for keyword in ['产品', '设备', '系统']):
                return '产品'
            elif any(keyword in word for keyword in ['材料', '原料', '物质']):
                return '材料'
            elif pos == 'nr':  # 人名
                return '人员'
        
        return None
    
    def _enhance_entity_recognition(self, sentence: str, entities: List[Dict]) -> List[Dict]:
        """使用深度模型增强实体识别"""
        # 获取句子的嵌入表示
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors='pt', 
                                  max_length=512, truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_size]
        
        # 这里可以添加更复杂的实体识别逻辑
        # 简化版本：验证已识别的实体
        enhanced_entities = []
        for entity in entities:
            # 计算实体的置信度
            confidence = self._calculate_entity_confidence(entity, embeddings, inputs)
            if confidence > self.kg_config['extraction_rules']['confidence_threshold']:
                entity['confidence'] = confidence
                enhanced_entities.append(entity)
        
        return enhanced_entities
    
    def _calculate_entity_confidence(self, entity: Dict, embeddings: torch.Tensor, 
                                   inputs: Dict) -> float:
        """计算实体置信度"""
        # 简化的置信度计算
        # 实际应用中可以使用更复杂的方法
        text_len = len(entity['text'])
        
        # 基于实体长度的简单置信度
        if text_len < self.kg_config['extraction_rules']['min_entity_length']:
            return 0.0
        elif text_len > self.kg_config['extraction_rules']['max_entity_length']:
            return 0.0
        else:
            # 基于实体类型和长度的启发式置信度
            base_confidence = 0.7
            if entity['type'] in ['技术', '产品', '公司']:
                base_confidence += 0.1
            if 3 <= text_len <= 10:
                base_confidence += 0.1
            return min(base_confidence, 1.0)
    
    def _extract_relations(self, sentence: str, entities: List[Dict]) -> List[Dict]:
        """从句子中提取实体间的关系"""
        relations = []
        
        # 对每对实体尝试提取关系
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                entity1 = entities[i]
                entity2 = entities[j]
                
                # 基于规则的关系抽取
                relation = self._identify_relation(sentence, entity1, entity2)
                if relation:
                    relations.append({
                        'source': entity1['text'],
                        'source_type': entity1['type'],
                        'target': entity2['text'],
                        'target_type': entity2['type'],
                        'relation': relation,
                        'sentence': sentence
                    })
        
        return relations
    
    def _identify_relation(self, sentence: str, entity1: Dict, entity2: Dict) -> str:
        """识别两个实体间的关系"""
        # 关系触发词
        relation_triggers = {
            '使用': ['使用', '采用', '应用', '利用'],
            '包含': ['包含', '包括', '含有', '由...组成'],
            '生产': ['生产', '制造', '生成', '产出'],
            '研发': ['研发', '开发', '研制', '研究'],
            '依赖': ['依赖', '依靠', '基于', '需要'],
            '改进': ['改进', '优化', '提升', '改善'],
            '替代': ['替代', '取代', '代替', '替换'],
            '认证': ['认证', '认可', '批准', '通过'],
            '合作': ['合作', '协作', '联合', '共同'],
            '应用于': ['应用于', '用于', '适用于', '服务于']
        }
        
        # 检查句子中是否包含关系触发词
        for relation, triggers in relation_triggers.items():
            for trigger in triggers:
                if trigger in sentence:
                    # 检查触发词是否在两个实体之间
                    e1_pos = sentence.find(entity1['text'])
                    e2_pos = sentence.find(entity2['text'])
                    trigger_pos = sentence.find(trigger)
                    
                    # 简单的位置判断
                    if min(e1_pos, e2_pos) < trigger_pos < max(e1_pos, e2_pos):
                        return relation
        
        # 基于实体类型的默认关系
        type_pair = (entity1['type'], entity2['type'])
        default_relations = {
            ('公司', '产品'): '生产',
            ('产品', '技术'): '使用',
            ('技术', '材料'): '依赖',
            ('公司', '技术'): '研发',
            ('产品', '材料'): '包含'
        }
        
        return default_relations.get(type_pair, None)
    
    def _build_graph(self, entities: List[Dict], relations: List[Dict]):
        """构建知识图谱"""
        # 添加节点（去重）
        unique_entities = {}
        for entity in entities:
            key = (entity['text'], entity['type'])
            if key not in unique_entities:
                unique_entities[key] = entity
        
        for (text, type_), entity in unique_entities.items():
            self.graph.add_node(text, 
                              type=type_,
                              confidence=entity.get('confidence', 1.0))
        
        # 添加边
        for relation in relations:
            if (relation['source'] in self.graph.nodes() and 
                relation['target'] in self.graph.nodes()):
                self.graph.add_edge(
                    relation['source'],
                    relation['target'],
                    relation=relation['relation'],
                    sentence=relation['sentence']
                )
    
    def save_graph(self, graph: nx.MultiDiGraph, output_path: Path):
        """保存知识图谱"""
        # 转换为可序列化的格式
        graph_data = {
            'nodes': [
                {
                    'id': node,
                    'type': data.get('type', 'unknown'),
                    'confidence': data.get('confidence', 1.0)
                }
                for node, data in graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    'relation': data.get('relation', 'unknown'),
                    'sentence': data.get('sentence', '')
                }
                for source, target, data in graph.edges(data=True)
            ],
            'statistics': {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'node_types': dict(nx.get_node_attributes(graph, 'type').values()),
                'relation_types': dict(nx.get_edge_attributes(graph, 'relation').values())
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"知识图谱已保存到: {output_path}")
    
    def load_graph(self, input_path: Path) -> nx.MultiDiGraph:
        """加载知识图谱"""
        with open(input_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        graph = nx.MultiDiGraph()
        
        # 添加节点
        for node in graph_data['nodes']:
            graph.add_node(node['id'], 
                         type=node['type'],
                         confidence=node['confidence'])
        
        # 添加边
        for edge in graph_data['edges']:
            graph.add_edge(edge['source'], 
                         edge['target'],
                         relation=edge['relation'],
                         sentence=edge['sentence'])
        
        logger.info(f"知识图谱加载完成: {graph.number_of_nodes()} 个节点, "
                   f"{graph.number_of_edges()} 条边")
        
        return graph