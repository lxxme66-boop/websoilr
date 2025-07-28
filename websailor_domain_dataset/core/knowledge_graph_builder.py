"""
知识图谱构建器
从领域文本中构建知识图谱，作为数据生成的基础
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple, Set
from pathlib import Path
import networkx as nx
import jieba
import jieba.posseg as pseg
from collections import defaultdict
import re


class KnowledgeGraphBuilder:
    """
    知识图谱构建器
    从领域文本中抽取实体和关系，构建知识图谱
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 实体抽取配置
        self.entity_config = config.get('entity_extraction', {})
        self.relation_config = config.get('relation_extraction', {})
        self.graph_config = config.get('graph_construction', {})
        
        # 初始化分词器
        self._init_jieba()
        
        # 领域实体类型
        self.domain_entities = self.entity_config.get('domain_entities', [])
        self.relation_types = self.relation_config.get('relation_types', [])
        
    def build_from_texts(self, input_dir: str) -> nx.Graph:
        """
        从文本目录构建知识图谱
        
        Args:
            input_dir: 输入文本目录
            
        Returns:
            构建的知识图谱
        """
        self.logger.info(f"从目录 {input_dir} 构建知识图谱...")
        
        # 读取所有文本
        texts = self._load_texts(input_dir)
        self.logger.info(f"加载了 {len(texts)} 个文本文件")
        
        # 抽取实体和关系
        all_entities = []
        all_relations = []
        
        for text_data in texts:
            entities = self._extract_entities(text_data['content'])
            relations = self._extract_relations(text_data['content'], entities)
            
            all_entities.extend(entities)
            all_relations.extend(relations)
        
        # 构建图
        graph = self._build_graph(all_entities, all_relations)
        
        # 后处理
        if self.graph_config.get('merge_similar_entities', True):
            graph = self._merge_similar_entities(graph)
        
        self.logger.info(f"构建完成：{len(graph.nodes)} 个节点，{len(graph.edges)} 条边")
        return graph
    
    def _load_texts(self, input_dir: str) -> List[Dict[str, Any]]:
        """加载文本文件"""
        texts = []
        input_path = Path(input_dir)
        
        for file_path in input_path.glob('*.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                texts.append({
                    'filename': file_path.name,
                    'content': content
                })
        
        return texts
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """抽取实体"""
        entities = []
        
        if self.entity_config.get('use_ner', True):
            # 使用命名实体识别
            ner_entities = self._ner_extraction(text)
            entities.extend(ner_entities)
        
        if self.entity_config.get('use_dependency_parsing', True):
            # 使用依存句法分析
            dep_entities = self._dependency_extraction(text)
            entities.extend(dep_entities)
        
        # 基于规则的抽取
        rule_entities = self._rule_based_extraction(text)
        entities.extend(rule_entities)
        
        # 去重和过滤
        entities = self._filter_entities(entities)
        
        return entities
    
    def _ner_extraction(self, text: str) -> List[Dict[str, Any]]:
        """基于NER的实体抽取"""
        entities = []
        
        # 使用jieba进行词性标注
        words = pseg.cut(text)
        
        for word, pos in words:
            entity_type = None
            
            # 根据词性判断实体类型
            if pos == 'n':  # 名词
                # 检查是否是领域实体
                for domain_type in self.domain_entities:
                    if domain_type in word or self._is_domain_entity(word, domain_type):
                        entity_type = domain_type
                        break
                
                if not entity_type:
                    entity_type = '通用实体'
            
            elif pos == 'nr':  # 人名
                entity_type = '人员'
            elif pos == 'ns':  # 地名
                entity_type = '地点'
            elif pos == 'nt':  # 机构名
                entity_type = '组织'
            
            if entity_type:
                entities.append({
                    'text': word,
                    'type': entity_type,
                    'method': 'ner',
                    'confidence': 0.8
                })
        
        return entities
    
    def _dependency_extraction(self, text: str) -> List[Dict[str, Any]]:
        """基于依存句法的实体抽取（简化版本）"""
        entities = []
        
        # 这里使用简化的方法，实际应用中应使用专业的依存句法分析工具
        sentences = re.split(r'[。！？]', text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # 寻找主语、宾语等
            words = list(jieba.cut(sentence))
            
            # 简单规则：动词前的名词可能是主语，动词后的名词可能是宾语
            for i, word in enumerate(words):
                if i > 0 and self._is_verb(word):
                    # 检查前面的词
                    if self._is_noun(words[i-1]):
                        entities.append({
                            'text': words[i-1],
                            'type': self._guess_entity_type(words[i-1]),
                            'method': 'dependency',
                            'confidence': 0.7
                        })
                    
                    # 检查后面的词
                    if i < len(words) - 1 and self._is_noun(words[i+1]):
                        entities.append({
                            'text': words[i+1],
                            'type': self._guess_entity_type(words[i+1]),
                            'method': 'dependency',
                            'confidence': 0.7
                        })
        
        return entities
    
    def _rule_based_extraction(self, text: str) -> List[Dict[str, Any]]:
        """基于规则的实体抽取"""
        entities = []
        
        # TCL工业领域的规则模式
        patterns = {
            '产品': [
                r'(\w+(?:电视|空调|冰箱|洗衣机|手机|显示器))',
                r'(\w+(?:系列|型号|产品))',
            ],
            '设备': [
                r'(\w+(?:设备|机器|装置|系统))',
                r'(\w+(?:生产线|流水线|工作站))',
            ],
            '工艺': [
                r'(\w+(?:工艺|流程|技术|方法))',
                r'(\w+(?:加工|处理|制造))',
            ],
            '材料': [
                r'(\w+(?:材料|原料|物料|零件))',
                r'(\w+(?:钢|铁|铝|塑料|玻璃))',
            ],
            '参数': [
                r'(\d+(?:℃|°C|度|mm|cm|m|kg|g|Hz|W|V|A))',
                r'(\w+(?:温度|压力|速度|功率|电压|电流))',
            ],
            '标准': [
                r'((?:ISO|GB|IEC|EN)\d+)',
                r'(\w+(?:标准|规范|要求))',
            ],
            '故障': [
                r'(\w+(?:故障|异常|错误|问题|缺陷))',
                r'(\w+(?:失效|损坏|老化))',
            ],
            '维护': [
                r'(\w+(?:维护|保养|检修|维修))',
                r'(\w+(?:检查|测试|校准))',
            ]
        }
        
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text)
                for match in matches:
                    entities.append({
                        'text': match,
                        'type': entity_type,
                        'method': 'rule',
                        'confidence': 0.9
                    })
        
        return entities
    
    def _extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """抽取关系"""
        relations = []
        
        # 构建实体文本到实体的映射
        entity_map = {e['text']: e for e in entities}
        entity_texts = set(entity_map.keys())
        
        # 关系模式
        relation_patterns = {
            '包含': ['包含', '包括', '含有', '由.*组成'],
            '属于': ['属于', '是.*的一部分', '归属于'],
            '影响': ['影响', '作用于', '决定', '改变'],
            '导致': ['导致', '引起', '造成', '产生'],
            '需要': ['需要', '需求', '要求', '必须'],
            '使用': ['使用', '采用', '应用', '利用'],
            '生产': ['生产', '制造', '加工', '制作'],
            '检测': ['检测', '检查', '测试', '测量'],
            '维护': ['维护', '保养', '维修', '检修'],
            '优化': ['优化', '改进', '提升', '改善'],
            '替代': ['替代', '代替', '取代', '替换'],
            '依赖': ['依赖', '依靠', '基于', '取决于']
        }
        
        # 分句处理
        sentences = re.split(r'[。！？]', text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # 查找句子中的实体
            sentence_entities = []
            for entity_text in entity_texts:
                if entity_text in sentence:
                    sentence_entities.append(entity_text)
            
            # 如果句子中有至少两个实体，尝试抽取关系
            if len(sentence_entities) >= 2:
                for i in range(len(sentence_entities)):
                    for j in range(i + 1, len(sentence_entities)):
                        entity1 = sentence_entities[i]
                        entity2 = sentence_entities[j]
                        
                        # 检查是否存在关系词
                        for relation_type, keywords in relation_patterns.items():
                            for keyword in keywords:
                                pattern = f"{entity1}.*{keyword}.*{entity2}"
                                if re.search(pattern, sentence):
                                    relations.append({
                                        'source': entity1,
                                        'target': entity2,
                                        'relation': relation_type,
                                        'confidence': 0.8,
                                        'context': sentence
                                    })
                                    break
                                
                                # 反向关系
                                pattern = f"{entity2}.*{keyword}.*{entity1}"
                                if re.search(pattern, sentence):
                                    relations.append({
                                        'source': entity2,
                                        'target': entity1,
                                        'relation': relation_type,
                                        'confidence': 0.8,
                                        'context': sentence
                                    })
                                    break
        
        return relations
    
    def _build_graph(self, entities: List[Dict[str, Any]], 
                    relations: List[Dict[str, Any]]) -> nx.Graph:
        """构建知识图谱"""
        graph = nx.Graph()
        
        # 添加节点
        entity_freq = defaultdict(int)
        entity_info = {}
        
        for entity in entities:
            entity_text = entity['text']
            entity_freq[entity_text] += 1
            
            if entity_text not in entity_info:
                entity_info[entity_text] = entity
            else:
                # 更新置信度
                old_conf = entity_info[entity_text].get('confidence', 0)
                new_conf = entity.get('confidence', 0)
                entity_info[entity_text]['confidence'] = max(old_conf, new_conf)
        
        # 过滤低频实体
        min_freq = self.graph_config.get('min_entity_frequency', 2)
        
        for entity_text, freq in entity_freq.items():
            if freq >= min_freq:
                info = entity_info[entity_text]
                graph.add_node(
                    entity_text,
                    type=info['type'],
                    frequency=freq,
                    confidence=info.get('confidence', 0.5)
                )
        
        # 添加边
        for relation in relations:
            source = relation['source']
            target = relation['target']
            
            # 确保节点存在
            if source in graph and target in graph:
                # 如果边已存在，更新权重
                if graph.has_edge(source, target):
                    graph[source][target]['weight'] += 1
                else:
                    graph.add_edge(
                        source, target,
                        relation=relation['relation'],
                        weight=1,
                        confidence=relation.get('confidence', 0.5)
                    )
        
        return graph
    
    def _merge_similar_entities(self, graph: nx.Graph) -> nx.Graph:
        """合并相似实体"""
        threshold = self.graph_config.get('similarity_threshold', 0.85)
        
        # 计算实体相似度并合并
        nodes = list(graph.nodes())
        merged = set()
        
        for i in range(len(nodes)):
            if nodes[i] in merged:
                continue
                
            for j in range(i + 1, len(nodes)):
                if nodes[j] in merged:
                    continue
                    
                similarity = self._calculate_similarity(nodes[i], nodes[j])
                
                if similarity >= threshold:
                    # 合并节点j到节点i
                    self._merge_nodes(graph, nodes[i], nodes[j])
                    merged.add(nodes[j])
        
        # 删除已合并的节点
        graph.remove_nodes_from(merged)
        
        return graph
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简化版本）"""
        # 基于编辑距离的相似度
        from difflib import SequenceMatcher
        
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _merge_nodes(self, graph: nx.Graph, keep_node: str, merge_node: str):
        """合并两个节点"""
        # 将merge_node的所有边转移到keep_node
        for neighbor in graph.neighbors(merge_node):
            if neighbor != keep_node:
                # 获取边属性
                edge_data = graph.get_edge_data(merge_node, neighbor)
                
                if graph.has_edge(keep_node, neighbor):
                    # 更新权重
                    graph[keep_node][neighbor]['weight'] += edge_data.get('weight', 1)
                else:
                    # 添加新边
                    graph.add_edge(keep_node, neighbor, **edge_data)
        
        # 更新节点属性
        merge_freq = graph.nodes[merge_node].get('frequency', 0)
        graph.nodes[keep_node]['frequency'] += merge_freq
    
    def _init_jieba(self):
        """初始化jieba分词器"""
        # 添加领域词典
        domain_words = [
            '液晶显示', 'OLED', 'QLED', '背光模组', '偏光片',
            '彩膜基板', 'TFT', '驱动IC', '玻璃基板',
            '空调压缩机', '制冷剂', '蒸发器', '冷凝器',
            '变频技术', '能效比', 'SMT', 'DIP', 'AOI',
            '回流焊', '波峰焊', '贴片机', '印刷机',
            '质量管理', '六西格玛', 'SPC', 'FMEA'
        ]
        
        for word in domain_words:
            jieba.add_word(word)
    
    def _is_domain_entity(self, word: str, domain_type: str) -> bool:
        """判断是否是领域实体"""
        # 简化的判断逻辑
        domain_keywords = {
            '产品': ['电视', '空调', '冰箱', '洗衣机', '显示器'],
            '设备': ['机器', '设备', '装置', '系统', '生产线'],
            '工艺': ['工艺', '技术', '流程', '方法'],
            '材料': ['材料', '原料', '零件', '组件'],
            '参数': ['温度', '压力', '速度', '功率'],
            '标准': ['标准', '规范', 'ISO', 'GB'],
            '故障': ['故障', '异常', '问题', '缺陷'],
            '维护': ['维护', '保养', '检修', '维修']
        }
        
        keywords = domain_keywords.get(domain_type, [])
        return any(kw in word for kw in keywords)
    
    def _is_verb(self, word: str) -> bool:
        """判断是否是动词"""
        verb_suffixes = ['造', '作', '理', '测', '查', '修', '护', '产']
        return any(word.endswith(suffix) for suffix in verb_suffixes)
    
    def _is_noun(self, word: str) -> bool:
        """判断是否是名词"""
        # 简单判断：长度大于1且不是动词
        return len(word) > 1 and not self._is_verb(word)
    
    def _guess_entity_type(self, word: str) -> str:
        """猜测实体类型"""
        for domain_type in self.domain_entities:
            if self._is_domain_entity(word, domain_type):
                return domain_type
        
        return '通用实体'
    
    def _filter_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤和去重实体"""
        if not entities:
            return []
        
        # 去重：基于实体文本去重
        unique_entities = {}
        for entity in entities:
            text = entity.get('text', '').strip()
            if text and len(text) > 1:  # 过滤掉空白和单字符实体
                if text not in unique_entities:
                    unique_entities[text] = entity
        
        # 过滤低质量实体
        filtered_entities = []
        for entity in unique_entities.values():
            text = entity.get('text', '')
            
            # 跳过纯数字、纯标点或过短的实体
            if (len(text) < 2 or 
                text.isdigit() or 
                all(not c.isalnum() for c in text)):
                continue
            
            # 跳过常见停用词
            stop_words = {'的', '了', '在', '是', '有', '和', '与', '或', '等', '及'}
            if text in stop_words:
                continue
            
            filtered_entities.append(entity)
        
        return filtered_entities