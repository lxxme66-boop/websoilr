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
            # 如果第一次解析失败，尝试更宽松的解析
            entities = self._parse_entities_fallback(entities_text)
            
        if not entities:
            logger.warning("未能解析出有效实体")
            return [], []

        # 清洗和规范化实体
        clean_entities = self._clean_entities(entities)

        # 如果清洗后没有实体，尝试从文本中直接提取
        if not clean_entities:
            logger.info("尝试使用备用方法提取实体...")
            clean_entities = self._extract_entities_from_text(text_chunk)

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

    def _parse_entities_fallback(self, text: str) -> List[Dict]:
        """备用的实体解析方法，处理格式不规范的输出"""
        entities = []
        
        # 尝试按行解析
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 尝试解析各种可能的格式
            # 格式1: - 实体名称 (类型)
            match = re.match(r'^[-•]\s*(.+?)\s*[\(（](.+?)[\)）]', line)
            if match:
                name, type_info = match.groups()
                entities.append({
                    'name': name.strip(),
                    'type': type_info.strip()
                })
                continue
            
            # 格式2: 实体名称：描述
            match = re.match(r'^(.+?)[：:]\s*(.+)', line)
            if match:
                name, desc = match.groups()
                entities.append({
                    'name': name.strip(),
                    'description': desc.strip()
                })
                continue
            
            # 格式3: 纯文本（可能是实体名称）
            if len(line) > 2 and len(line) < 50 and not any(c in line for c in '{}[]'):
                entities.append({'name': line})
        
        return entities

    def _extract_entities_from_text(self, text: str) -> List[Dict]:
        """从文本中直接提取实体（最后的备用方案）"""
        entities = []
        
        # 使用正则表达式查找可能的实体
        # 查找引号中的内容
        quoted_terms = re.findall(r'["""\'](.*?)["""\'"]', text)
        for term in quoted_terms[:10]:  # 限制数量
            if 2 < len(term) < 30:
                entities.append({
                    'entity_id': self._generate_entity_id(term),
                    'name': term,
                    'type': self._infer_entity_type({'name': term}),
                    'description': f"从文本中提取的术语: {term}"
                })
        
        # 查找专业术语模式（英文缩写、化学式等）
        technical_terms = re.findall(r'\b[A-Z][A-Za-z0-9\-:]+\b', text)
        for term in set(technical_terms[:5]):  # 去重并限制数量
            if len(term) > 2:
                entities.append({
                    'entity_id': term.lower(),
                    'name': term,
                    'type': '技术术语',
                    'description': f"技术术语或缩写: {term}"
                })
        
        return entities
    
    def _clean_entities(self, entities: List[Dict]) -> List[Dict]:
        """清洗和规范化实体数据"""
        clean_entities = []
        entity_counter = 0  # 用于生成唯一ID
        
        for e in entities:
            # 处理字段名称的变体
            entity = {}
            
            # 处理 ID 字段（可能是 id, entity_id 等）
            if 'entity_id' in e:
                entity['entity_id'] = e['entity_id']
            elif 'id' in e:
                entity['entity_id'] = e['id']
            else:
                # 尝试从其他字段生成 ID
                if 'name' in e:
                    entity['entity_id'] = self._generate_entity_id(e['name'])
                elif 'type' in e:
                    # 使用类型和计数器生成ID
                    entity_counter += 1
                    entity['entity_id'] = f"{self._generate_entity_id(e['type'])}_{entity_counter}"
                elif 'description' in e:
                    # 从描述生成ID
                    entity['entity_id'] = self._generate_entity_id(e['description'][:30])
                else:
                    # 最后的后备方案：生成通用ID
                    entity_counter += 1
                    entity['entity_id'] = f"entity_{entity_counter}"
                
                logger.warning(f"[实体缺少ID，已自动生成] 原始数据: {e}, 生成ID: {entity['entity_id']}")
            
            # 处理 name 字段
            if 'name' in e:
                entity['name'] = e['name']
            else:
                # 尝试从其他字段推断 name
                if 'description' in e:
                    # 从描述中提取名称（取前面的部分）
                    entity['name'] = self._extract_name_from_description(e['description'])
                elif 'type' in e:
                    # 使用类型作为名称的一部分
                    entity['name'] = f"{e['type']}_{entity.get('entity_id', 'unknown')}"
                elif 'entity_id' in entity:
                    # 使用ID作为名称
                    entity['name'] = entity['entity_id'].replace('_', ' ').title()
                else:
                    # 最后的后备方案
                    entity['name'] = f"未命名实体{entity_counter}"
                
                logger.warning(f"[实体缺少name，已自动生成] 原始数据: {e}, 生成name: {entity['name']}")
            
            # 处理 type 字段
            if 'type' in e:
                entity['type'] = e['type']
            else:
                # 尝试从名称或描述推断类型
                entity['type'] = self._infer_entity_type(e)
                logger.warning(f"[实体缺少type，已推断] 原始数据: {e}, 推断type: {entity['type']}")
            
            # 处理 description 字段
            if 'description' in e:
                entity['description'] = e['description']
            else:
                # 根据其他字段生成描述
                parts = []
                if 'name' in entity:
                    parts.append(f"名称: {entity['name']}")
                if 'type' in entity:
                    parts.append(f"类型: {entity['type']}")
                
                if parts:
                    entity['description'] = "，".join(parts)
                else:
                    entity['description'] = "暂无描述"
                
                logger.warning(f"[实体缺少description，已生成] 原始数据: {e}, 生成description: {entity['description']}")
            
            # 验证必需字段
            if all(field in entity for field in ['entity_id', 'name', 'type', 'description']):
                clean_entities.append(entity)
                logger.debug(f"[实体清洗成功] {entity}")
            else:
                logger.error(f"[实体字段不完整，跳过] 原始: {e}, 处理后: {entity}")
        
        return clean_entities
    
    def _extract_name_from_description(self, description: str) -> str:
        """从描述中提取名称"""
        # 移除常见的描述性词汇
        desc_words = ['是', '为', '表示', '指', '用于', '描述']
        
        # 尝试提取第一个有意义的短语
        for word in desc_words:
            if word in description:
                parts = description.split(word)
                if len(parts) > 1 and parts[0].strip():
                    return parts[0].strip()[:30]  # 限制长度
        
        # 如果没有找到，返回描述的前30个字符
        return description[:30].strip()
    
    def _infer_entity_type(self, entity_data: Dict) -> str:
        """根据实体数据推断类型"""
        # 定义关键词到类型的映射
        type_keywords = {
            '材料': ['材料', '薄膜', '基板', '衬底', 'substrate', 'film', 'material'],
            '方法': ['方法', '处理', '工艺', '技术', 'method', 'process', 'technique'],
            '参数': ['参数', '数值', '温度', '压力', '时间', 'parameter', 'value', 'temperature'],
            '设备': ['设备', '仪器', '装置', 'equipment', 'device', 'instrument'],
            '样品': ['样品', '试样', 'sample', 'specimen'],
            '结果': ['结果', '性能', '特性', 'result', 'performance', 'property'],
            '化学物质': ['酸', '碱', '溶液', '化合物', 'acid', 'base', 'solution', 'compound']
        }
        
        # 检查所有可用的文本字段
        text_to_check = []
        if 'name' in entity_data:
            text_to_check.append(entity_data['name'].lower())
        if 'description' in entity_data:
            text_to_check.append(entity_data['description'].lower())
        if 'entity_id' in entity_data:
            text_to_check.append(entity_data['entity_id'].lower())
        
        combined_text = ' '.join(text_to_check)
        
        # 检查关键词
        for entity_type, keywords in type_keywords.items():
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    return entity_type
        
        # 默认类型
        return '其他'
    
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