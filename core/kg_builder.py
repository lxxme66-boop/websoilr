# ==================== core/kg_builder.py ====================
import torch
from sentence_transformers import SentenceTransformer, util
import networkx as nx
import re
from typing import List, Dict, Tuple
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
        
    def build_from_documents(self, documents: Dict[str, str]):
        """从多个文档构建知识图谱"""
        logger.info("Building knowledge graph from documents...")
        
        for doc_name, content in documents.items():
            logger.info(f"Processing document: {doc_name}")
            chunks = self._chunk_text(content, self.config.CHUNK_SIZE)
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} from {doc_name}")
                entities, relations = self._extract_knowledge(chunk)
                self._add_to_graph(entities, relations)
        
        logger.info(f"Knowledge graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return self.graph
    
    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """智能文本分块"""
        # 按段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_knowledge(self, text_chunk: str) -> Tuple[List[Dict], List[Dict]]:
        """使用大模型提取实体和关系"""
        # 实体提取 Prompt - 改进版本，强调必需字段
        entity_prompt = f"""你是一个{self.domain}领域的知识工程师。请从以下技术文档中提取关键实体。

文档内容：
{text_chunk}

请按以下JSON格式输出实体列表，每个实体必须包含以下字段：
[
  {{
    "entity_id": "unique_id",  // 必需：唯一标识符，使用英文或拼音，不含空格
    "name": "实体名称",        // 必需：实体的中文或英文名称
    "type": "实体类型",        // 必需：如材料、方法、参数、样品等
    "description": "简短描述"   // 必需：对实体的简要说明
  }},
  ...
]

注意：
1. entity_id 必须是唯一的，使用英文或拼音，不含空格和特殊字符
2. 所有字段都是必需的，不能省略
3. 只返回JSON数组，不要其他内容"""

        # 调用模型生成实体
        entities_text = self.model_manager.generate_text(
            "qa_generator",
            entity_prompt,
            max_new_tokens=512,
            temperature=0.3
        )

        logger.debug(f"[原始实体响应] {entities_text}")

        entities = self._parse_json_response(entities_text, [])

        if not entities:
            logger.warning("未能解析出有效实体")
            return [], []

        # 清洗和规范化实体
        clean_entities = self._clean_entities(entities)

        if not clean_entities:
            logger.warning("实体字段缺失严重，跳过该段文本")
            return [], []

        # 构造实体描述字符串，用于关系抽取 Prompt
        entities_desc = "\n".join([
            f"- {e['name']} ({e['entity_id']}): {e['type']}"
            for e in clean_entities
        ])

        relation_prompt = f"""基于以下实体和文档内容，提取实体间的关系。

实体列表：
{entities_desc}

文档内容：
{text_chunk}

请按以下JSON格式输出关系列表：
[
  {{
    "source": "源实体的entity_id",
    "target": "目标实体的entity_id",
    "relation": "关系类型",
    "description": "关系描述"
  }},
  ...
]

注意：source和target必须使用实体列表中的entity_id
只返回JSON数组，不要其他内容。"""

        relations_text = self.model_manager.generate_text(
            "qa_generator",
            relation_prompt,
            max_new_tokens=512,
            temperature=0.3
        )

        logger.debug(f"[原始关系响应] {relations_text}")

        relations = self._parse_json_response(relations_text, [])

        return clean_entities, relations

    def _clean_entities(self, entities: List[Dict]) -> List[Dict]:
        """清洗和规范化实体数据"""
        clean_entities = []
        
        for e in entities:
            # 处理字段名称的变体
            entity = {}
            
            # 处理 ID 字段（可能是 id, entity_id 等）
            if 'entity_id' in e:
                entity['entity_id'] = e['entity_id']
            elif 'id' in e:
                entity['entity_id'] = e['id']
            else:
                logger.warning(f"[实体缺少ID] {e}")
                # 尝试从 name 生成 ID
                if 'name' in e:
                    entity['entity_id'] = self._generate_entity_id(e['name'])
                else:
                    continue
            
            # 处理 name 字段
            if 'name' in e:
                entity['name'] = e['name']
            else:
                logger.warning(f"[实体缺少name] {e}")
                # 使用 description 或 entity_id 作为 name
                if 'description' in e:
                    entity['name'] = e['description'][:50]  # 限制长度
                else:
                    entity['name'] = entity['entity_id']
            
            # 处理 type 字段
            entity['type'] = e.get('type', '未分类')
            
            # 处理 description 字段
            entity['description'] = e.get('description', entity['name'])
            
            # 验证必需字段
            if all(field in entity for field in ['entity_id', 'name', 'type', 'description']):
                clean_entities.append(entity)
            else:
                logger.warning(f"[实体字段不完整] {entity}")
        
        return clean_entities
    
    def _generate_entity_id(self, name: str) -> str:
        """从名称生成实体ID"""
        # 移除特殊字符，转换为小写
        entity_id = re.sub(r'[^\w\s-]', '', name.lower())
        # 替换空格为下划线
        entity_id = re.sub(r'[\s-]+', '_', entity_id)
        # 限制长度
        return entity_id[:50]

    def _parse_json_response(self, text: str, default_value):
        """解析 JSON 响应，支持多种格式"""
        try:
            # 首先尝试直接解析
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        try:
            # 尝试找到JSON数组
            json_match = re.search(r'\[.*?\]', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except Exception:
            pass
        
        try:
            # 尝试提取多个JSON对象
            matches = re.findall(r'\{[^{}]*\}', text)
            entities = []
            for match in matches:
                try:
                    obj = json.loads(match)
                    entities.append(obj)
                except json.JSONDecodeError:
                    continue
            
            if entities:
                return entities
        except Exception:
            pass
        
        logger.warning(f"Failed to parse JSON response: {text[:200]}...")
        return default_value
    
    def _add_to_graph(self, entities: List[Dict], relations: List[Dict]):
        """将实体和关系添加到图中"""
        # 添加实体节点
        for entity in entities:
            node_id = entity.get('entity_id', entity.get('id'))
            if not node_id:
                logger.warning(f"Skip entity without ID: {entity}")
                continue
                
            self.graph.add_node(
                node_id,
                name=entity.get('name', node_id),
                type=entity.get('type', ''),
                description=entity.get('description', '')
            )
        
        # 添加关系边
        for relation in relations:
            source = relation.get('source')
            target = relation.get('target')
            
            if source and target:
                if source in self.graph.nodes and target in self.graph.nodes:
                    self.graph.add_edge(
                        source,
                        target,
                        relation=relation.get('relation', '相关'),
                        description=relation.get('description', '')
                    )
                else:
                    logger.warning(f"Relation references non-existent nodes: {source} -> {target}")