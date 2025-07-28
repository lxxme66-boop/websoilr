# ==================== core/kg_builder.py ====================
import torch
from sentence_transformers import SentenceTransformer, util
import networkx as nx
import re
from typing import List, Dict, Tuple, Optional
import json
import logging

try:
    from .prompt_optimizer import PromptOptimizer
except ImportError:
    # 如果无法导入，使用简单的占位符
    class PromptOptimizer:
        def __init__(self, domain):
            self.domain = domain
        
        def optimize_entity_prompt(self, text_chunk):
            return f"""你是一个{self.domain}领域的专业知识工程师。请从文档中提取关键实体。

文档内容：
{text_chunk[:1500]}

请按照以下JSON格式输出，每行一个JSON对象：
{{"id": "实体英文标识", "name": "实体中文名称", "type": "实体类型", "description": "简短描述"}}

只输出JSON对象，不要其他内容。"""
        
        def optimize_relation_prompt(self, entities, text_chunk):
            entities_desc = "\n".join([f"- {e['name']} ({e['id']})" for e in entities[:10]])
            return f"""基于以下实体和文档内容，提取实体间的关系。

实体列表：
{entities_desc}

文档内容：
{text_chunk[:1000]}

请按照以下JSON格式输出关系，每行一个：
{{"source": "源实体ID", "target": "目标实体ID", "relation": "关系类型", "description": "关系描述"}}

只输出JSON对象。"""
        
        def post_process_entities(self, entities):
            return entities

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
        
        # 初始化提示词优化器
        self.prompt_optimizer = PromptOptimizer(self.domain)
        
    def build_from_documents(self, documents: Dict[str, str]):
        """从多个文档构建知识图谱"""
        logger.info("Building knowledge graph from documents...")
        
        total_entities = 0
        total_relations = 0
        
        for doc_name, content in documents.items():
            logger.info(f"Processing document: {doc_name}")
            chunks = self._chunk_text(content, self.config.CHUNK_SIZE)
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} from {doc_name}")
                entities, relations = self._extract_knowledge(chunk)
                
                if entities:
                    total_entities += len(entities)
                    total_relations += len(relations)
                    self._add_to_graph(entities, relations)
                else:
                    logger.debug(f"No entities extracted from chunk {i+1}")
        
        logger.info(f"Knowledge graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        logger.info(f"Total entities extracted: {total_entities}, Total relations: {total_relations}")
        return self.graph
    
    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """智能文本分块，确保句子完整性"""
        # 清理文本
        text = text.strip()
        if not text:
            return []
        
        # 按段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # 如果段落本身超过chunk_size，需要进一步分割
            if len(paragraph) > chunk_size:
                # 按句子分割
                sentences = re.split(r'[。！？.!?]+', paragraph)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if len(current_chunk) + len(sentence) + 1 < chunk_size:
                        current_chunk += sentence + "。"
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + "。"
            else:
                if len(current_chunk) + len(paragraph) + 2 < chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_knowledge(self, text_chunk: str) -> Tuple[List[Dict], List[Dict]]:
        """使用大模型提取实体和关系，增强提示词和错误处理"""
        if not text_chunk or len(text_chunk.strip()) < 10:
            return [], []
        
        # 使用优化后的提示词
        entity_prompt = self.prompt_optimizer.optimize_entity_prompt(text_chunk)

        try:
            # 调用模型生成实体
            entities_text = self.model_manager.generate_text(
                "qa_generator",
                entity_prompt,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9
            )
            
            logger.debug(f"实体提取原始响应长度: {len(entities_text)}")
            
            # 解析实体
            entities = self._parse_entities(entities_text)
            
            if not entities:
                logger.warning("未能提取到有效实体")
                return [], []
            
            # 后处理实体
            entities = self.prompt_optimizer.post_process_entities(entities)
            
            logger.info(f"成功提取 {len(entities)} 个实体")
            
            # 关系提取 - 仅在有实体时进行
            relations = self._extract_relations(entities, text_chunk)
            
            return entities, relations
            
        except Exception as e:
            logger.error(f"知识提取过程出错: {e}")
            return [], []
    
    def _extract_relations(self, entities: List[Dict], text_chunk: str) -> List[Dict]:
        """提取实体间的关系"""
        if len(entities) < 2:
            return []
        
        # 使用优化后的关系提取提示词
        relation_prompt = self.prompt_optimizer.optimize_relation_prompt(entities, text_chunk)

        try:
            relations_text = self.model_manager.generate_text(
                "qa_generator",
                relation_prompt,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9
            )
            
            relations = self._parse_relations(relations_text, entities[:10])
            logger.info(f"成功提取 {len(relations)} 个关系")
            
            return relations
            
        except Exception as e:
            logger.error(f"关系提取出错: {e}")
            return []
    
    def _parse_entities(self, text: str) -> List[Dict]:
        """改进的实体解析方法"""
        entities = []
        
        # 方法1：尝试解析完整的JSON数组
        try:
            # 查找JSON数组
            json_match = re.search(r'\[[\s\S]*\]', text)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    entities.extend(parsed)
        except:
            pass
        
        # 方法2：逐行解析JSON对象
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 尝试提取JSON对象
            json_matches = re.findall(r'\{[^{}]*\}', line)
            for match in json_matches:
                try:
                    obj = json.loads(match)
                    if self._validate_entity(obj):
                        entities.append(obj)
                except:
                    continue
        
        # 方法3：使用更宽松的正则表达式
        if not entities:
            # 匹配包含id和name的模式
            pattern = r'\{[^}]*"id"\s*:\s*"([^"]+)"[^}]*"name"\s*:\s*"([^"]+)"[^}]*\}'
            matches = re.finditer(pattern, text, re.DOTALL)
            
            for match in matches:
                try:
                    # 重新构造完整的JSON对象
                    full_match = match.group(0)
                    obj = json.loads(full_match)
                    if self._validate_entity(obj):
                        entities.append(obj)
                except:
                    # 如果JSON解析失败，手动构造
                    entity = {
                        'id': match.group(1),
                        'name': match.group(2),
                        'type': '未分类',
                        'description': ''
                    }
                    entities.append(entity)
        
        # 去重
        seen_ids = set()
        unique_entities = []
        for entity in entities:
            if entity['id'] not in seen_ids:
                seen_ids.add(entity['id'])
                unique_entities.append(entity)
        
        return unique_entities
    
    def _validate_entity(self, entity: Dict) -> bool:
        """验证实体的完整性"""
        if not isinstance(entity, dict):
            return False
        
        # 必须有id
        if 'id' not in entity or not entity['id']:
            return False
        
        # 补充缺失字段
        if 'name' not in entity:
            entity['name'] = entity.get('description', entity['id'])
        
        if 'type' not in entity:
            entity['type'] = '未分类'
        
        if 'description' not in entity:
            entity['description'] = ''
        
        return True
    
    def _parse_relations(self, text: str, valid_entities: List[Dict]) -> List[Dict]:
        """改进的关系解析方法"""
        relations = []
        valid_ids = {e['id'] for e in valid_entities}
        
        # 方法1：解析JSON数组
        try:
            json_match = re.search(r'\[[\s\S]*\]', text)
            if json_match:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    for rel in parsed:
                        if self._validate_relation(rel, valid_ids):
                            relations.append(rel)
        except:
            pass
        
        # 方法2：逐行解析
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            json_matches = re.findall(r'\{[^{}]*\}', line)
            for match in json_matches:
                try:
                    obj = json.loads(match)
                    if self._validate_relation(obj, valid_ids):
                        relations.append(obj)
                except:
                    continue
        
        # 去重
        seen = set()
        unique_relations = []
        for rel in relations:
            key = (rel['source'], rel['target'], rel['relation'])
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)
        
        return unique_relations
    
    def _validate_relation(self, relation: Dict, valid_ids: set) -> bool:
        """验证关系的有效性"""
        if not isinstance(relation, dict):
            return False
        
        # 检查必需字段
        required = ['source', 'target', 'relation']
        for field in required:
            if field not in relation or not relation[field]:
                return False
        
        # 检查实体ID是否有效
        if relation['source'] not in valid_ids or relation['target'] not in valid_ids:
            return False
        
        # 补充描述字段
        if 'description' not in relation:
            relation['description'] = f"{relation['source']} {relation['relation']} {relation['target']}"
        
        return True
    
    def _add_to_graph(self, entities: List[Dict], relations: List[Dict]):
        """将实体和关系添加到图中"""
        # 添加实体节点
        for entity in entities:
            if entity['id'] not in self.graph:
                self.graph.add_node(
                    entity['id'],
                    name=entity['name'],
                    type=entity.get('type', ''),
                    description=entity.get('description', '')
                )
        
        # 添加关系边
        for relation in relations:
            if (relation['source'] in self.graph.nodes and 
                relation['target'] in self.graph.nodes):
                self.graph.add_edge(
                    relation['source'],
                    relation['target'],
                    relation=relation['relation'],
                    description=relation.get('description', '')
                )
    
    def save_graph(self, filepath: str):
        """保存知识图谱"""
        nx.write_gexf(self.graph, filepath)
        logger.info(f"Knowledge graph saved to {filepath}")
    
    def load_graph(self, filepath: str):
        """加载知识图谱"""
        self.graph = nx.read_gexf(filepath)
        logger.info(f"Knowledge graph loaded from {filepath}")