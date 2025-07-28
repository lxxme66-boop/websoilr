# ==================== core/kg_builder.py ====================
import torch
from sentence_transformers import SentenceTransformer, util
import networkx as nx
import re
from typing import List, Dict, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)


class IndustrialKGBuilder:
    def __init__(self, model_manager, config):
        self.config = config
        self.model_manager = model_manager
        self.domain = config.DOMAIN
        self.graph = nx.DiGraph()
        
        # 加载知识图谱提取模型
        self.kg_extractor = model_manager.load_sentence_transformer(
            config.KG_EXTRACTOR_MODEL_PATH, 
            "kg_extractor"
        )
        
        # 定义实体类型映射
        self.entity_types = {
            "材料": ["材料", "化合物", "物质", "薄膜", "靶材", "衬底"],
            "工艺": ["工艺", "方法", "技术", "制备方法", "处理方式"],
            "设备": ["设备", "仪器", "装置", "系统"],
            "参数": ["参数", "性能", "指标", "数值", "温度", "压力", "浓度"],
            "组织": ["组织", "机构", "公司", "大学", "研究所"],
            "人物": ["人物", "研究者", "作者"],
            "概念": ["概念", "理论", "原理", "现象"]
        }
        
    def build_from_documents(self, documents: Dict[str, str]):
        """从多个文档构建知识图谱"""
        logger.info("Building knowledge graph from documents...")
        
        all_entities = []
        all_relations = []
        
        for doc_name, content in documents.items():
            logger.info(f"Processing document: {doc_name}")
            chunks = self._chunk_text(content, self.config.CHUNK_SIZE)
            
            doc_entities = []
            doc_relations = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} from {doc_name}")
                
                # 提取实体和关系
                entities, relations = self._extract_knowledge_robust(chunk, doc_name, i)
                
                if entities:
                    doc_entities.extend(entities)
                    all_entities.extend(entities)
                    
                if relations:
                    doc_relations.extend(relations)
                    all_relations.extend(relations)
                    
                # 每处理10个chunk进行一次图更新
                if (i + 1) % 10 == 0:
                    self._add_to_graph(doc_entities, doc_relations)
                    doc_entities = []
                    doc_relations = []
            
            # 处理剩余的实体和关系
            if doc_entities or doc_relations:
                self._add_to_graph(doc_entities, doc_relations)
        
        # 后处理：合并相似实体，清理无效关系
        self._post_process_graph()
        
        logger.info(f"Knowledge graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return self.graph
    
    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """智能文本分块，保持语义完整性"""
        # 清理文本
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 按句子分割
        sentences = re.split(r'[。！？\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # 如果单个句子超过chunk_size，强制分割
            if sentence_size > chunk_size:
                if current_chunk:
                    chunks.append('。'.join(current_chunk) + '。')
                    current_chunk = []
                    current_size = 0
                
                # 分割长句子
                words = list(sentence)
                for i in range(0, len(words), chunk_size):
                    chunks.append(''.join(words[i:i+chunk_size]))
                continue
            
            # 如果加入当前句子会超过限制，先保存当前chunk
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append('。'.join(current_chunk) + '。')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # 保存最后一个chunk
        if current_chunk:
            chunks.append('。'.join(current_chunk) + '。')
        
        return chunks
    
    def _extract_knowledge_robust(self, text_chunk: str, doc_name: str, chunk_idx: int) -> Tuple[List[Dict], List[Dict]]:
        """增强的知识提取，提高鲁棒性"""
        # 预处理文本
        text_chunk = text_chunk.strip()
        if len(text_chunk) < 20:  # 文本太短，跳过
            return [], []
        
        # 尝试多次提取，使用不同的prompt策略
        entities = self._extract_entities_with_retry(text_chunk, doc_name, chunk_idx)
        
        if not entities:
            return [], []
        
        # 提取关系
        relations = self._extract_relations_with_retry(text_chunk, entities)
        
        return entities, relations
    
    def _extract_entities_with_retry(self, text_chunk: str, doc_name: str, chunk_idx: int, max_retries: int = 2) -> List[Dict]:
        """带重试机制的实体提取"""
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    # 第一次尝试：结构化prompt
                    entities = self._extract_entities_structured(text_chunk, doc_name, chunk_idx)
                else:
                    # 重试：简化prompt
                    entities = self._extract_entities_simple(text_chunk, doc_name, chunk_idx)
                
                if entities:
                    return entities
                    
            except Exception as e:
                logger.debug(f"Entity extraction attempt {attempt + 1} failed: {e}")
                continue
        
        return []
    
    def _extract_entities_structured(self, text_chunk: str, doc_name: str, chunk_idx: int) -> List[Dict]:
        """结构化实体提取"""
        entity_prompt = f"""作为{self.domain}领域专家，请从以下文本中提取关键实体。

文本内容：
{text_chunk[:800]}  # 限制长度避免超出token限制

要求：
1. 每个实体必须包含：id（唯一标识）、name（实体名称）、type（实体类型）
2. 实体类型应从以下类别中选择：{', '.join(self.entity_types.keys())}
3. 只提取文本中明确提到的实体，不要推测
4. 返回格式必须是标准JSON数组

示例输出：
[
  {{"id": "IGZO", "name": "铟镓锌氧化物", "type": "材料", "description": "一种透明导电氧化物"}},
  {{"id": "magnetron_sputtering", "name": "磁控溅射", "type": "工艺", "description": "薄膜制备方法"}}
]

请直接返回JSON数组："""

        response = self.model_manager.generate_text(
            "qa_generator",
            entity_prompt,
            max_new_tokens=400,
            temperature=0.1  # 降低温度提高一致性
        )
        
        entities = self._parse_entities(response, doc_name, chunk_idx)
        return entities
    
    def _extract_entities_simple(self, text_chunk: str, doc_name: str, chunk_idx: int) -> List[Dict]:
        """简化的实体提取（备用方案）"""
        entity_prompt = f"""从以下{self.domain}文本中提取实体名称和类型：

{text_chunk[:600]}

按以下格式列出（每行一个）：
实体名称 | 实体类型

示例：
IGZO | 材料
磁控溅射 | 工艺"""

        response = self.model_manager.generate_text(
            "qa_generator",
            entity_prompt,
            max_new_tokens=300,
            temperature=0.1
        )
        
        # 解析简单格式
        entities = []
        lines = response.strip().split('\n')
        
        for i, line in enumerate(lines):
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    name = parts[0].strip()
                    entity_type = parts[1].strip()
                    
                    if name and entity_type:
                        entity_id = f"{doc_name}_{chunk_idx}_{i}_{name[:20]}"
                        entity_id = re.sub(r'[^\w\-_]', '_', entity_id)
                        
                        entities.append({
                            "id": entity_id,
                            "name": name,
                            "type": self._normalize_entity_type(entity_type),
                            "description": f"来自{doc_name}"
                        })
        
        return entities
    
    def _parse_entities(self, response: str, doc_name: str, chunk_idx: int) -> List[Dict]:
        """解析实体响应，增强容错性"""
        entities = []
        
        # 尝试多种解析策略
        # 策略1：标准JSON数组
        try:
            # 查找JSON数组
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and 'name' in item:
                            entity = self._validate_entity(item, doc_name, chunk_idx)
                            if entity:
                                entities.append(entity)
                    
                    if entities:
                        return entities
        except json.JSONDecodeError:
            pass
        
        # 策略2：逐个JSON对象
        try:
            # 匹配所有的JSON对象
            obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(obj_pattern, response)
            
            for i, match in enumerate(matches):
                try:
                    obj = json.loads(match)
                    entity = self._validate_entity(obj, doc_name, chunk_idx, i)
                    if entity:
                        entities.append(entity)
                except:
                    continue
            
            if entities:
                return entities
        except:
            pass
        
        # 策略3：文本模式匹配
        try:
            # 匹配 "实体名称"、"实体类型" 等模式
            pattern = r'(?:实体名称|name)[：:]\s*([^,，\n]+)'
            names = re.findall(pattern, response)
            
            pattern = r'(?:实体类型|type)[：:]\s*([^,，\n]+)'
            types = re.findall(pattern, response)
            
            for i, (name, entity_type) in enumerate(zip(names, types)):
                if name and entity_type:
                    entity_id = f"{doc_name}_{chunk_idx}_{i}_{name[:20]}"
                    entity_id = re.sub(r'[^\w\-_]', '_', entity_id)
                    
                    entities.append({
                        "id": entity_id,
                        "name": name.strip('"\''),
                        "type": self._normalize_entity_type(entity_type.strip('"\''))
                    })
        except:
            pass
        
        return entities
    
    def _validate_entity(self, entity_dict: Dict, doc_name: str, chunk_idx: int, idx: int = 0) -> Optional[Dict]:
        """验证和规范化实体"""
        # 必须有name
        if 'name' not in entity_dict:
            if 'entity_name' in entity_dict:
                entity_dict['name'] = entity_dict['entity_name']
            elif 'description' in entity_dict:
                entity_dict['name'] = entity_dict['description'][:50]
            else:
                return None
        
        # 生成或验证id
        if 'id' not in entity_dict:
            name_part = re.sub(r'[^\w\-_]', '_', entity_dict['name'][:30])
            entity_dict['id'] = f"{doc_name}_{chunk_idx}_{idx}_{name_part}"
        
        # 规范化type
        if 'type' not in entity_dict:
            entity_dict['type'] = '概念'
        else:
            entity_dict['type'] = self._normalize_entity_type(entity_dict['type'])
        
        # 确保description存在
        if 'description' not in entity_dict:
            entity_dict['description'] = f"{entity_dict['type']}实体"
        
        return entity_dict
    
    def _normalize_entity_type(self, entity_type: str) -> str:
        """规范化实体类型"""
        entity_type = entity_type.strip()
        
        # 检查是否属于预定义类型
        for main_type, subtypes in self.entity_types.items():
            if entity_type in subtypes or entity_type == main_type:
                return main_type
        
        # 模糊匹配
        entity_type_lower = entity_type.lower()
        for main_type, subtypes in self.entity_types.items():
            for subtype in subtypes:
                if subtype in entity_type_lower or entity_type_lower in subtype.lower():
                    return main_type
        
        # 默认类型
        return "概念"
    
    def _extract_relations_with_retry(self, text_chunk: str, entities: List[Dict], max_retries: int = 2) -> List[Dict]:
        """带重试机制的关系提取"""
        if len(entities) < 2:
            return []
        
        for attempt in range(max_retries):
            try:
                relations = self._extract_relations(text_chunk, entities)
                if relations:
                    return relations
            except Exception as e:
                logger.debug(f"Relation extraction attempt {attempt + 1} failed: {e}")
                continue
        
        return []
    
    def _extract_relations(self, text_chunk: str, entities: List[Dict]) -> List[Dict]:
        """提取实体间关系"""
        # 限制实体数量避免prompt过长
        entities_subset = entities[:10] if len(entities) > 10 else entities
        
        entities_desc = "\n".join([
            f"{i+1}. {e['name']} (ID: {e['id']}, 类型: {e['type']})"
            for i, e in enumerate(entities_subset)
        ])
        
        relation_prompt = f"""基于以下实体和文本，识别实体之间的关系。

实体列表：
{entities_desc}

文本内容：
{text_chunk[:600]}

关系类型示例：
- 组成/包含：A包含B、A由B组成
- 制备/生产：A制备B、A生产B
- 影响/改善：A影响B、A改善B
- 应用/用于：A应用于B、A用于B
- 比较/优于：A优于B、A比B更好

请返回JSON格式的关系列表，每个关系包含source（源实体ID）、target（目标实体ID）、relation（关系类型）：
[
  {{"source": "实体ID1", "target": "实体ID2", "relation": "关系类型"}}
]"""

        response = self.model_manager.generate_text(
            "qa_generator",
            relation_prompt,
            max_new_tokens=300,
            temperature=0.1
        )
        
        relations = self._parse_relations(response, entities_subset)
        return relations
    
    def _parse_relations(self, response: str, entities: List[Dict]) -> List[Dict]:
        """解析关系响应"""
        relations = []
        entity_ids = {e['id'] for e in entities}
        
        # 尝试解析JSON
        try:
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                parsed = json.loads(json_match.group())
                
                if isinstance(parsed, list):
                    for item in parsed:
                        if (isinstance(item, dict) and 
                            'source' in item and 
                            'target' in item and 
                            'relation' in item):
                            
                            # 验证实体ID存在
                            if (item['source'] in entity_ids and 
                                item['target'] in entity_ids and
                                item['source'] != item['target']):
                                
                                relations.append({
                                    "source": item['source'],
                                    "target": item['target'],
                                    "relation": item['relation'],
                                    "description": item.get('description', '')
                                })
        except:
            pass
        
        return relations
    
    def _add_to_graph(self, entities: List[Dict], relations: List[Dict]):
        """将实体和关系添加到图中，避免重复"""
        # 添加实体节点
        for entity in entities:
            node_id = entity['id']
            
            # 检查是否已存在相似节点
            existing_node = self._find_similar_node(entity['name'])
            
            if existing_node:
                # 合并到现有节点
                self._merge_nodes(existing_node, entity)
            else:
                # 添加新节点
                self.graph.add_node(
                    node_id,
                    name=entity['name'],
                    type=entity.get('type', ''),
                    description=entity.get('description', ''),
                    aliases=set([entity['name']])
                )
        
        # 添加关系边
        for relation in relations:
            if (relation['source'] in self.graph.nodes and 
                relation['target'] in self.graph.nodes):
                
                # 避免重复边
                if not self.graph.has_edge(relation['source'], relation['target']):
                    self.graph.add_edge(
                        relation['source'],
                        relation['target'],
                        relation=relation['relation'],
                        description=relation.get('description', '')
                    )
    
    def _find_similar_node(self, name: str) -> Optional[str]:
        """查找相似节点"""
        name_lower = name.lower()
        
        for node_id, node_data in self.graph.nodes(data=True):
            node_name = node_data.get('name', '').lower()
            aliases = node_data.get('aliases', set())
            
            # 完全匹配
            if name_lower == node_name:
                return node_id
            
            # 别名匹配
            if name_lower in [a.lower() for a in aliases]:
                return node_id
            
            # 包含关系匹配（避免过度合并）
            if len(name) > 3 and len(node_name) > 3:
                if name_lower in node_name or node_name in name_lower:
                    similarity = len(set(name_lower) & set(node_name)) / len(set(name_lower) | set(node_name))
                    if similarity > 0.8:
                        return node_id
        
        return None
    
    def _merge_nodes(self, existing_id: str, new_entity: Dict):
        """合并节点信息"""
        node_data = self.graph.nodes[existing_id]
        
        # 添加别名
        if 'aliases' not in node_data:
            node_data['aliases'] = set()
        node_data['aliases'].add(new_entity['name'])
        
        # 更新描述（如果更详细）
        if len(new_entity.get('description', '')) > len(node_data.get('description', '')):
            node_data['description'] = new_entity['description']
    
    def _post_process_graph(self):
        """图的后处理：清理和优化"""
        # 移除孤立节点（没有任何连接的节点）
        isolated_nodes = list(nx.isolates(self.graph))
        if isolated_nodes:
            logger.info(f"Removing {len(isolated_nodes)} isolated nodes")
            self.graph.remove_nodes_from(isolated_nodes)
        
        # 合并高度相似的节点
        self._merge_similar_nodes()
        
        # 清理自环边
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
    
    def _merge_similar_nodes(self):
        """合并高度相似的节点"""
        nodes_to_merge = []
        processed = set()
        
        for node1 in self.graph.nodes():
            if node1 in processed:
                continue
                
            for node2 in self.graph.nodes():
                if node1 == node2 or node2 in processed:
                    continue
                
                if self._are_nodes_similar(node1, node2):
                    nodes_to_merge.append((node1, node2))
                    processed.add(node2)
        
        # 执行合并
        for node1, node2 in nodes_to_merge:
            self._merge_two_nodes(node1, node2)
    
    def _are_nodes_similar(self, node1: str, node2: str) -> bool:
        """判断两个节点是否相似"""
        data1 = self.graph.nodes[node1]
        data2 = self.graph.nodes[node2]
        
        name1 = data1.get('name', '').lower()
        name2 = data2.get('name', '').lower()
        
        # 类型必须相同
        if data1.get('type') != data2.get('type'):
            return False
        
        # 名称相似度检查
        if name1 == name2:
            return True
        
        # 编辑距离检查
        if len(name1) > 3 and len(name2) > 3:
            common_chars = len(set(name1) & set(name2))
            total_chars = len(set(name1) | set(name2))
            similarity = common_chars / total_chars if total_chars > 0 else 0
            
            if similarity > 0.85:
                return True
        
        return False
    
    def _merge_two_nodes(self, node1: str, node2: str):
        """合并两个节点"""
        # 将node2的所有边转移到node1
        for pred in self.graph.predecessors(node2):
            if pred != node1:
                edge_data = self.graph.edges[pred, node2]
                self.graph.add_edge(pred, node1, **edge_data)
        
        for succ in self.graph.successors(node2):
            if succ != node1:
                edge_data = self.graph.edges[node2, succ]
                self.graph.add_edge(node1, succ, **edge_data)
        
        # 合并节点属性
        data1 = self.graph.nodes[node1]
        data2 = self.graph.nodes[node2]
        
        if 'aliases' not in data1:
            data1['aliases'] = set()
        data1['aliases'].add(data2.get('name', ''))
        data1['aliases'].update(data2.get('aliases', set()))
        
        # 移除node2
        self.graph.remove_node(node2)