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
        """使用大模型提取实体和关系 - 优化版本"""
        
        # 使用统一的prompt一次性提取实体和关系，提高效率
        unified_prompt = f"""你是一个{self.domain}领域的知识工程师。请仔细阅读以下技术文档，然后提取关键实体和它们之间的关系。

文档内容：
{text_chunk}

请按以下JSON格式输出，确保所有字段都完整：
{{
    "entities": [
        {{
            "entity_id": "unique_id",  // 必需：唯一标识符，使用英文或拼音，不含空格
            "name": "实体名称",        // 必需：实体的中文或英文名称
            "type": "实体类型",        // 必需：如材料、方法、参数、样品等
            "description": "简短描述"   // 必需：对实体的简要说明
        }}
    ],
    "relations": [
        {{
            "source": "源实体的entity_id",
            "target": "目标实体的entity_id", 
            "relation": "关系类型",
            "description": "关系描述"
        }}
    ]
}}

注意：
1. entity_id 必须是唯一的，使用英文或拼音，不含空格和特殊字符
2. 所有字段都是必需的，不能省略
3. relations中的source和target必须使用entities列表中的entity_id
4. 只返回JSON对象，不要其他内容"""

        # 调用模型一次性生成实体和关系
        response_text = self.model_manager.generate_text(
            "qa_generator",
            unified_prompt,
            max_new_tokens=1024,  # 增加token限制以容纳更多内容
            temperature=0.2       # 略微降低温度以提高确定性和速度
        )

        logger.debug(f"[原始响应] {response_text}")

        # 解析响应
        result = self._parse_unified_response(response_text)
        
        if not result:
            logger.warning("未能解析出有效结果")
            return [], []

        entities = result.get('entities', [])
        relations = result.get('relations', [])

        # 清洗和规范化数据
        clean_entities = self._clean_entities(entities)
        clean_relations = self._clean_relations(relations, clean_entities)

        return clean_entities, clean_relations

    def _parse_unified_response(self, text: str) -> Dict:
        """解析统一响应的JSON"""
        # 清理响应文本
        text = text.strip()
        
        # 尝试直接解析
        try:
            result = json.loads(text)
            if isinstance(result, dict) and ('entities' in result or 'relations' in result):
                return result
        except json.JSONDecodeError:
            pass
        
        # 移除可能的markdown标记
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
        text = text.strip()
        
        # 再次尝试解析
        try:
            result = json.loads(text)
            if isinstance(result, dict) and ('entities' in result or 'relations' in result):
                return result
        except json.JSONDecodeError:
            pass
        
        # 尝试提取JSON对象
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict) and ('entities' in result or 'relations' in result):
                    return result
            except json.JSONDecodeError:
                continue
        
        logger.warning(f"Failed to parse unified response: {text[:200]}...")
        return {'entities': [], 'relations': []}

    def _clean_entities(self, entities: List[Dict]) -> List[Dict]:
        """清洗和规范化实体数据"""
        clean_entities = []
        seen_ids = set()
        
        for e in entities:
            if not isinstance(e, dict):
                continue
                
            # 处理字段名称的变体
            entity = {}
            
            # 处理 ID 字段
            if 'entity_id' in e:
                entity['entity_id'] = str(e['entity_id'])
            elif 'id' in e:
                entity['entity_id'] = str(e['id'])
            else:
                if 'name' in e:
                    entity['entity_id'] = self._generate_entity_id(e['name'])
                else:
                    continue
            
            # 确保ID唯一
            original_id = entity['entity_id']
            counter = 1
            while entity['entity_id'] in seen_ids:
                entity['entity_id'] = f"{original_id}_{counter}"
                counter += 1
            seen_ids.add(entity['entity_id'])
            
            # 处理其他字段
            entity['name'] = e.get('name', entity['entity_id'])
            entity['type'] = e.get('type', '未分类')
            entity['description'] = e.get('description', entity['name'])
            
            clean_entities.append(entity)
        
        return clean_entities

    def _clean_relations(self, relations: List[Dict], entities: List[Dict]) -> List[Dict]:
        """清洗和规范化关系数据"""
        clean_relations = []
        entity_ids = {e['entity_id'] for e in entities}
        
        for r in relations:
            if not isinstance(r, dict):
                continue
                
            # 验证必需字段
            if not all(k in r for k in ['source', 'target']):
                continue
            
            # 验证实体ID存在
            if r['source'] not in entity_ids or r['target'] not in entity_ids:
                continue
            
            # 避免自环
            if r['source'] == r['target']:
                continue
            
            relation = {
                'source': r['source'],
                'target': r['target'],
                'relation': r.get('relation', '相关'),
                'description': r.get('description', '')
            }
            
            clean_relations.append(relation)
        
        return clean_relations
    
    def _generate_entity_id(self, name: str) -> str:
        """从名称生成实体ID"""
        # 移除特殊字符，转换为小写
        entity_id = re.sub(r'[^\w\s-]', '', name.lower())
        # 替换空格为下划线
        entity_id = re.sub(r'[\s-]+', '_', entity_id)
        # 限制长度
        return entity_id[:50] if entity_id else "unknown_entity"
    
    def _add_to_graph(self, entities: List[Dict], relations: List[Dict]):
        """将实体和关系添加到图中"""
        # 添加实体节点
        for entity in entities:
            node_id = entity.get('entity_id')
            if not node_id:
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


class IndustrialKGBuilderV2(IndustrialKGBuilder):
    """知识图谱构建器V2 - 不使用思考提示的快速版本"""
    
    def _extract_knowledge(self, text_chunk: str) -> Tuple[List[Dict], List[Dict]]:
        """使用大模型提取实体和关系 - 快速版本（无思考提示）"""
        
        # 移除"仔细阅读"和"思考"的提示词，直接要求提取
        fast_prompt = f"""你是一个{self.domain}领域的知识工程师。从以下技术文档中提取关键实体和关系。

文档内容：
{text_chunk}

直接输出JSON格式：
{{
    "entities": [
        {{
            "entity_id": "unique_id",
            "name": "实体名称",
            "type": "实体类型",
            "description": "简短描述"
        }}
    ],
    "relations": [
        {{
            "source": "源实体entity_id",
            "target": "目标实体entity_id", 
            "relation": "关系类型",
            "description": "关系描述"
        }}
    ]
}}

只返回JSON，不要解释。"""

        # 使用更低的温度和更快的生成
        response_text = self.model_manager.generate_text(
            "qa_generator",
            fast_prompt,
            max_new_tokens=768,   # 减少token限制
            temperature=0.1       # 更低温度，更快生成
        )

        # 使用父类的解析和清洗方法
        result = self._parse_unified_response(response_text)
        
        if not result:
            return [], []

        entities = result.get('entities', [])
        relations = result.get('relations', [])

        clean_entities = self._clean_entities(entities)
        clean_relations = self._clean_relations(relations, clean_entities)

        return clean_entities, clean_relations