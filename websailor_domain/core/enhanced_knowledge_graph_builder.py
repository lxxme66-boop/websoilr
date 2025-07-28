"""
增强版知识图谱构建器
使用模型管理器实现并行处理，提高构建效率
"""

import logging
from typing import List, Dict, Tuple, Any, Set, Optional
from pathlib import Path
import networkx as nx
import json
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures
from dataclasses import dataclass
import time

from .model_manager import ModelManager, ModelConfig


@dataclass
class ProcessingBatch:
    """处理批次"""
    batch_id: str
    texts: List[str]
    file_paths: List[Path]
    metadata: Dict[str, Any]


class EnhancedKnowledgeGraphBuilder:
    """
    增强版知识图谱构建器
    使用多模型并行处理提高效率
    """
    
    def __init__(self, config: Dict[str, Any], model_manager: Optional[ModelManager] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model_manager = model_manager
        
        # 批处理配置
        self.batch_size = config.get('batch_size', 32)
        self.max_workers = config.get('max_workers', 4)
        
        # 实体和关系类型
        self.entity_types = config.get('entity_types', [])
        self.relation_types = config.get('relation_types', [])
        
        # 统计信息
        self.stats = defaultdict(int)
        
    def build_from_texts_parallel(self, input_dir: Path) -> nx.Graph:
        """
        并行处理文本构建知识图谱
        """
        start_time = time.time()
        self.logger.info(f"Starting parallel knowledge graph construction from {input_dir}")
        
        # 1. 准备文本批次
        batches = self._prepare_batches(input_dir)
        self.logger.info(f"Prepared {len(batches)} batches for processing")
        
        # 2. 并行处理批次
        all_entities = []
        all_relations = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有批次处理任务
            future_to_batch = {
                executor.submit(self._process_batch, batch): batch 
                for batch in batches
            }
            
            # 收集结果
            for future in tqdm(
                concurrent.futures.as_completed(future_to_batch), 
                total=len(batches),
                desc="Processing batches"
            ):
                batch = future_to_batch[future]
                try:
                    entities, relations = future.result()
                    all_entities.extend(entities)
                    all_relations.extend(relations)
                    
                    # 更新统计
                    self.stats['processed_batches'] += 1
                    self.stats['total_entities'] += len(entities)
                    self.stats['total_relations'] += len(relations)
                    
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch.batch_id}: {str(e)}")
                    self.stats['failed_batches'] += 1
                    
        # 3. 构建知识图谱
        kg = self._build_graph_from_results(all_entities, all_relations)
        
        # 4. 后处理和优化
        kg = self._post_process_graph(kg)
        
        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Knowledge graph construction completed in {elapsed_time:.2f}s. "
            f"Nodes: {kg.number_of_nodes()}, Edges: {kg.number_of_edges()}"
        )
        
        return kg
        
    def _prepare_batches(self, input_dir: Path) -> List[ProcessingBatch]:
        """准备处理批次"""
        batches = []
        text_files = list(input_dir.glob("*.txt"))
        
        current_batch_texts = []
        current_batch_files = []
        
        for i, text_file in enumerate(text_files):
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # 分句
            sentences = self._split_sentences(text)
            current_batch_texts.extend(sentences)
            current_batch_files.extend([text_file] * len(sentences))
            
            # 检查是否需要创建新批次
            if len(current_batch_texts) >= self.batch_size:
                batch = ProcessingBatch(
                    batch_id=f"batch_{len(batches)}",
                    texts=current_batch_texts[:self.batch_size],
                    file_paths=current_batch_files[:self.batch_size],
                    metadata={'batch_index': len(batches)}
                )
                batches.append(batch)
                
                # 保留剩余的文本
                current_batch_texts = current_batch_texts[self.batch_size:]
                current_batch_files = current_batch_files[self.batch_size:]
                
        # 处理剩余的文本
        if current_batch_texts:
            batch = ProcessingBatch(
                batch_id=f"batch_{len(batches)}",
                texts=current_batch_texts,
                file_paths=current_batch_files,
                metadata={'batch_index': len(batches), 'is_last': True}
            )
            batches.append(batch)
            
        return batches
        
    def _process_batch(self, batch: ProcessingBatch) -> Tuple[List[Dict], List[Dict]]:
        """处理单个批次"""
        entities = []
        relations = []
        
        # 如果有模型管理器，使用模型进行处理
        if self.model_manager:
            # 并行调用NER和RE模型
            results = self.model_manager.predict_parallel(
                model_names=['ner_bert', 're_bert'],
                inputs={
                    'ner_bert': batch.texts,
                    'default': batch.texts
                }
            )
            
            # 处理NER结果
            if 'ner_bert' in results and results['ner_bert']:
                for text, ner_result in zip(batch.texts, results['ner_bert']):
                    for entity in ner_result:
                        entity['sentence'] = text
                        entities.append(entity)
                        
            # 处理关系抽取结果
            if 're_bert' in results and results['re_bert']:
                # 这里需要根据实际的RE模型输出格式处理
                pass
                
        else:
            # 使用规则方法
            for text in batch.texts:
                text_entities = self._extract_entities_rule_based(text)
                entities.extend(text_entities)
                
                text_relations = self._extract_relations_rule_based(text, text_entities)
                relations.extend(text_relations)
                
        return entities, relations
        
    def _extract_entities_rule_based(self, text: str) -> List[Dict]:
        """基于规则的实体抽取（后备方案）"""
        import re
        
        entities = []
        
        # TCL工业领域的实体模式
        entity_patterns = {
            'Product': [
                r'TCL[\s\-]?\w+',
                r'\w+显示器',
                r'\w+电视',
                r'\w+面板',
            ],
            'Technology': [
                r'QLED技术',
                r'Mini[\s\-]?LED',
                r'量子点\w*',
                r'AI\w*技术',
            ],
            'Component': [
                r'\w+芯片',
                r'\w+模组',
                r'\w+背光',
            ],
        }
        
        for entity_type, patterns in entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity = {
                        'text': match.group(),
                        'type': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'sentence': text
                    }
                    entities.append(entity)
                    
        return entities
        
    def _extract_relations_rule_based(self, text: str, entities: List[Dict]) -> List[Dict]:
        """基于规则的关系抽取（后备方案）"""
        import re
        
        relations = []
        
        relation_patterns = {
            'uses_technology': [
                r'(\w+)采用(\w+技术)',
                r'(\w+)使用(\w+)',
            ],
            'manufactures': [
                r'(\w+)生产(\w+)',
                r'(\w+)制造(\w+)',
            ],
        }
        
        for relation_type, patterns in relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
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
                                'sentence': text
                            }
                            relations.append(relation)
                            
        return relations
        
    def _find_entity(self, text: str, entities: List[Dict]) -> Optional[Dict]:
        """查找匹配的实体"""
        for entity in entities:
            if text in entity['text'] or entity['text'] in text:
                return entity
        return None
        
    def _build_graph_from_results(
        self, 
        entities: List[Dict], 
        relations: List[Dict]
    ) -> nx.Graph:
        """从结果构建图"""
        kg = nx.Graph()
        
        # 添加节点
        entity_map = defaultdict(list)
        for entity in entities:
            entity_map[entity['text']].append(entity)
            
        for entity_text, entity_list in entity_map.items():
            # 合并相同实体的信息
            entity_type = entity_list[0]['type']
            sentences = list(set(e['sentence'] for e in entity_list))
            
            kg.add_node(
                entity_text,
                type=entity_type,
                sentences=sentences,
                frequency=len(entity_list)
            )
            
        # 添加边
        relation_map = defaultdict(list)
        for relation in relations:
            key = (relation['source'], relation['target'], relation['type'])
            relation_map[key].append(relation)
            
        for (source, target, rel_type), rel_list in relation_map.items():
            if kg.has_node(source) and kg.has_node(target):
                sentences = list(set(r['sentence'] for r in rel_list))
                
                kg.add_edge(
                    source,
                    target,
                    relation=rel_type,
                    sentences=sentences,
                    frequency=len(rel_list)
                )
                
        return kg
        
    def _post_process_graph(self, kg: nx.Graph) -> nx.Graph:
        """后处理知识图谱"""
        # 1. 实体消歧
        kg = self._disambiguate_entities(kg)
        
        # 2. 关系推理
        kg = self._infer_relations(kg)
        
        # 3. 图优化
        kg = self._optimize_graph(kg)
        
        return kg
        
    def _disambiguate_entities(self, kg: nx.Graph) -> nx.Graph:
        """实体消歧"""
        # 简单的基于相似度的消歧
        nodes_to_merge = []
        
        nodes = list(kg.nodes())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if self._are_similar_entities(node1, node2, kg):
                    nodes_to_merge.append((node1, node2))
                    
        # 合并相似实体
        for node1, node2 in nodes_to_merge:
            if kg.has_node(node1) and kg.has_node(node2):
                # 合并到node1
                for neighbor in kg.neighbors(node2):
                    if neighbor != node1:
                        edge_data = kg[node2][neighbor]
                        if kg.has_edge(node1, neighbor):
                            # 合并边信息
                            kg[node1][neighbor]['frequency'] += edge_data.get('frequency', 1)
                            kg[node1][neighbor]['sentences'].extend(edge_data.get('sentences', []))
                        else:
                            kg.add_edge(node1, neighbor, **edge_data)
                            
                # 合并节点属性
                kg.nodes[node1]['sentences'].extend(kg.nodes[node2].get('sentences', []))
                kg.nodes[node1]['frequency'] += kg.nodes[node2].get('frequency', 1)
                
                # 删除node2
                kg.remove_node(node2)
                
        return kg
        
    def _are_similar_entities(self, entity1: str, entity2: str, kg: nx.Graph) -> bool:
        """判断两个实体是否相似"""
        # 简单的编辑距离判断
        from difflib import SequenceMatcher
        
        # 类型必须相同
        if kg.nodes[entity1].get('type') != kg.nodes[entity2].get('type'):
            return False
            
        # 名称相似度
        similarity = SequenceMatcher(None, entity1, entity2).ratio()
        if similarity > 0.8:
            return True
            
        # 共享邻居判断
        neighbors1 = set(kg.neighbors(entity1))
        neighbors2 = set(kg.neighbors(entity2))
        
        if len(neighbors1) > 0 and len(neighbors2) > 0:
            jaccard = len(neighbors1 & neighbors2) / len(neighbors1 | neighbors2)
            if jaccard > 0.5:
                return True
                
        return False
        
    def _infer_relations(self, kg: nx.Graph) -> nx.Graph:
        """推理新的关系"""
        new_edges = []
        
        # 规则1：传递性推理
        # 如果 A uses_technology B, B requires_component C, 则 A indirectly_uses C
        for node in kg.nodes():
            for tech in kg.neighbors(node):
                if kg[node][tech].get('relation') == 'uses_technology':
                    for component in kg.neighbors(tech):
                        if kg[tech][component].get('relation') == 'requires_component':
                            if not kg.has_edge(node, component):
                                new_edges.append((
                                    node, 
                                    component, 
                                    {
                                        'relation': 'indirectly_uses',
                                        'inferred': True,
                                        'evidence': f"via {tech}"
                                    }
                                ))
                                
        # 添加推理的边
        for source, target, attrs in new_edges:
            kg.add_edge(source, target, **attrs)
            
        self.stats['inferred_relations'] = len(new_edges)
        
        return kg
        
    def _optimize_graph(self, kg: nx.Graph) -> nx.Graph:
        """优化图结构"""
        # 1. 移除孤立节点（可选）
        if self.config.get('remove_isolated_nodes', False):
            isolated = list(nx.isolates(kg))
            kg.remove_nodes_from(isolated)
            self.stats['removed_isolated_nodes'] = len(isolated)
            
        # 2. 计算节点重要性
        pagerank = nx.pagerank(kg)
        for node in kg.nodes():
            kg.nodes[node]['importance'] = pagerank[node]
            
        # 3. 识别社区
        if kg.number_of_nodes() > 10:
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(kg.to_undirected())
            for i, comm in enumerate(communities):
                for node in comm:
                    kg.nodes[node]['community'] = i
                    
        return kg
        
    def _split_sentences(self, text: str) -> List[str]:
        """分句"""
        import re
        sentences = re.split(r'[。！？\n]+', text)
        return [s.strip() for s in sentences if s.strip()]
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取构建统计信息"""
        return dict(self.stats)