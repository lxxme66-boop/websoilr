"""
知识图谱构建器 - 改进版
从TCL工业领域文本中提取实体和关系，构建知识图谱
使用大语言模型进行实体和关系抽取
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
import networkx as nx
from tqdm import tqdm
import re
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """知识图谱构建器 - 使用大模型进行实体和关系抽取"""
    
    def __init__(self, config: dict):
        self.config = config
        self.kg_config = config.get('knowledge_graph', {})
        
        # 初始化实体和关系类型
        self.entity_types = self.kg_config.get('entity_types', [])
        self.relation_types = self.kg_config.get('relation_types', [])
        
        # 初始化TCL特定术语
        self.tcl_terms = config.get('tcl_specific', {}).get('technical_terms', {})
        
        # 加载大语言模型
        self.model_path = config['models'].get('llm_model', {}).get('path', 'THUDM/chatglm3-6b')
        self._load_llm_model()
        
        # 知识图谱
        self.graph = nx.MultiDiGraph()
        
        # 配置参数
        self.chunk_size = self.kg_config.get('chunk_size', 1000)
        self.max_chunk_overlap = self.kg_config.get('max_chunk_overlap', 200)
        self.confidence_threshold = self.kg_config.get('extraction_rules', {}).get('confidence_threshold', 0.7)
        
    def _load_llm_model(self):
        """加载大语言模型"""
        logger.info(f"加载大语言模型: {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model.eval()
            logger.info("大语言模型加载成功")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
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
            
            # 将文本分块处理
            chunks = self._chunk_text(text)
            
            for chunk_idx, chunk in enumerate(tqdm(chunks, desc=f"处理 {text_file.name}", leave=False)):
                # 提取实体和关系
                entities, relations = self._extract_entities_relations_with_llm(chunk)
                
                # 添加文件来源信息
                for entity in entities:
                    entity['source_file'] = str(text_file)
                    entity['chunk_idx'] = chunk_idx
                
                for relation in relations:
                    relation['source_file'] = str(text_file)
                    relation['chunk_idx'] = chunk_idx
                
                all_entities.extend(entities)
                all_relations.extend(relations)
        
        # 构建知识图谱
        self._build_graph(all_entities, all_relations)
        
        logger.info(f"知识图谱构建完成: {self.graph.number_of_nodes()} 个节点, "
                   f"{self.graph.number_of_edges()} 条边")
        
        return self.graph
    
    def _chunk_text(self, text: str) -> List[str]:
        """智能文本分块，保持语义完整性"""
        # 按段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # 如果当前段落太长，需要进一步分割
            if len(paragraph) > self.chunk_size:
                # 按句子分割
                sentences = self._split_sentences(paragraph)
                for sent in sentences:
                    if len(current_chunk) + len(sent) < self.chunk_size:
                        current_chunk += sent + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent + " "
            else:
                if len(current_chunk) + len(paragraph) < self.chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """文本分句"""
        # 改进的分句规则，保留句子结束符
        sentences = re.split(r'(?<=[。！？])\s*', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_entities_relations_with_llm(self, text_chunk: str) -> Tuple[List[Dict], List[Dict]]:
        """使用大语言模型提取实体和关系"""
        
        # 构建实体类型和关系类型的说明
        entity_types_desc = "、".join(self.entity_types) if self.entity_types else "技术、产品、材料、工艺、公司、人员"
        relation_types_desc = "、".join(self.relation_types) if self.relation_types else "使用、包含、生产、研发、依赖、改进、替代、认证、合作、应用于"
        
        # 构建TCL专业术语提示
        tcl_terms_hint = self._build_tcl_terms_hint()
        
        # 实体和关系抽取的统一prompt
        extraction_prompt = f"""你是TCL工业领域的知识抽取专家。请从以下文本中抽取实体和它们之间的关系。

TCL专业术语参考：
{tcl_terms_hint}

实体类型包括：{entity_types_desc}
关系类型包括：{relation_types_desc}

文本内容：
{text_chunk}

请按以下JSON格式输出，确保所有字段都完整：
{{
    "entities": [
        {{
            "entity_id": "唯一标识符（英文或拼音，无空格）",
            "text": "实体文本",
            "type": "实体类型",
            "confidence": 0.9,
            "context": "实体所在的句子"
        }}
    ],
    "relations": [
        {{
            "source": "源实体的entity_id",
            "target": "目标实体的entity_id",
            "relation": "关系类型",
            "confidence": 0.9,
            "evidence": "支持这个关系的句子"
        }}
    ]
}}

注意：
1. entity_id必须唯一，使用实体文本的拼音或英文缩写
2. 只抽取文本中明确存在的实体和关系
3. confidence表示置信度，范围0-1
4. 确保relations中的source和target都在entities列表中
"""

        try:
            # 调用模型生成
            response = self._generate_with_llm(extraction_prompt)
            
            # 解析响应
            result = self._parse_llm_response(response)
            
            if result and isinstance(result, dict):
                entities = result.get('entities', [])
                relations = result.get('relations', [])
                
                # 验证和清理数据
                entities = self._validate_entities(entities, text_chunk)
                relations = self._validate_relations(relations, entities)
                
                return entities, relations
            else:
                logger.warning("LLM响应格式错误，返回空结果")
                return [], []
                
        except Exception as e:
            logger.error(f"实体关系抽取失败: {e}")
            return [], []
    
    def _build_tcl_terms_hint(self) -> str:
        """构建TCL专业术语提示"""
        hints = []
        for category, terms in self.tcl_terms.items():
            if terms:
                hints.append(f"{category}: {', '.join(terms[:5])}")  # 只取前5个示例
        return '\n'.join(hints) if hints else "无特定术语"
    
    def _generate_with_llm(self, prompt: str) -> str:
        """使用大语言模型生成响应"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response
    
    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """解析LLM响应，提取JSON数据"""
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取JSON块
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                if 'entities' in data or 'relations' in data:
                    return data
            except json.JSONDecodeError:
                continue
        
        logger.warning(f"无法解析LLM响应: {response[:200]}...")
        return None
    
    def _validate_entities(self, entities: List[Dict], text_chunk: str) -> List[Dict]:
        """验证和清理实体数据"""
        validated_entities = []
        seen_ids = set()
        
        for entity in entities:
            # 确保必要字段存在
            if not all(k in entity for k in ['text', 'type']):
                continue
            
            # 生成或验证entity_id
            if 'entity_id' not in entity or not entity['entity_id']:
                entity['entity_id'] = self._generate_entity_id(entity['text'])
            
            # 确保ID唯一
            original_id = entity['entity_id']
            counter = 1
            while entity['entity_id'] in seen_ids:
                entity['entity_id'] = f"{original_id}_{counter}"
                counter += 1
            seen_ids.add(entity['entity_id'])
            
            # 设置默认值
            entity.setdefault('confidence', 0.8)
            entity.setdefault('context', self._find_context(entity['text'], text_chunk))
            
            # 验证置信度
            if entity['confidence'] >= self.confidence_threshold:
                validated_entities.append(entity)
        
        return validated_entities
    
    def _validate_relations(self, relations: List[Dict], entities: List[Dict]) -> List[Dict]:
        """验证关系数据"""
        validated_relations = []
        entity_ids = {e['entity_id'] for e in entities}
        
        for relation in relations:
            # 确保必要字段存在
            if not all(k in relation for k in ['source', 'target', 'relation']):
                continue
            
            # 验证实体ID存在
            if relation['source'] not in entity_ids or relation['target'] not in entity_ids:
                continue
            
            # 设置默认值
            relation.setdefault('confidence', 0.8)
            relation.setdefault('evidence', '')
            
            # 验证置信度
            if relation['confidence'] >= self.confidence_threshold:
                validated_relations.append(relation)
        
        return validated_relations
    
    def _generate_entity_id(self, text: str) -> str:
        """从文本生成实体ID"""
        # 移除特殊字符，保留字母数字和下划线
        entity_id = re.sub(r'[^\w\u4e00-\u9fa5]', '_', text.lower())
        # 限制长度
        entity_id = entity_id[:30]
        # 确保不以数字开头
        if entity_id and entity_id[0].isdigit():
            entity_id = 'e_' + entity_id
        return entity_id or 'entity'
    
    def _find_context(self, entity_text: str, text_chunk: str) -> str:
        """查找实体所在的上下文句子"""
        sentences = self._split_sentences(text_chunk)
        for sent in sentences:
            if entity_text in sent:
                return sent
        return ""
    
    def _build_graph(self, entities: List[Dict], relations: List[Dict]):
        """构建知识图谱"""
        # 实体去重和合并
        unique_entities = {}
        for entity in entities:
            key = entity['entity_id']
            if key not in unique_entities:
                unique_entities[key] = entity
            else:
                # 合并相同实体的信息
                existing = unique_entities[key]
                # 更新置信度（取最大值）
                existing['confidence'] = max(existing.get('confidence', 0), entity.get('confidence', 0))
                # 合并来源信息
                if 'source_file' in entity:
                    if 'source_files' not in existing:
                        existing['source_files'] = set()
                    existing['source_files'].add(entity['source_file'])
        
        # 添加节点
        for entity_id, entity in unique_entities.items():
            node_attrs = {
                'name': entity['text'],
                'type': entity['type'],
                'confidence': entity.get('confidence', 1.0),
                'context': entity.get('context', '')
            }
            
            # 添加来源信息
            if 'source_files' in entity:
                node_attrs['source_files'] = list(entity['source_files'])
            elif 'source_file' in entity:
                node_attrs['source_files'] = [entity['source_file']]
            
            self.graph.add_node(entity_id, **node_attrs)
        
        # 添加边（关系）
        for relation in relations:
            if (relation['source'] in self.graph.nodes() and 
                relation['target'] in self.graph.nodes()):
                self.graph.add_edge(
                    relation['source'],
                    relation['target'],
                    relation=relation['relation'],
                    confidence=relation.get('confidence', 1.0),
                    evidence=relation.get('evidence', ''),
                    source_file=relation.get('source_file', '')
                )
    
    def save_graph(self, graph: nx.MultiDiGraph, output_path: Path):
        """保存知识图谱"""
        # 转换为可序列化的格式
        nodes_data = []
        for node, data in graph.nodes(data=True):
            node_data = {'id': node}
            node_data.update(data)
            nodes_data.append(node_data)
        
        edges_data = []
        for source, target, data in graph.edges(data=True):
            edge_data = {
                'source': source,
                'target': target
            }
            edge_data.update(data)
            edges_data.append(edge_data)
        
        # 统计信息
        node_types_count = defaultdict(int)
        for node, data in graph.nodes(data=True):
            node_types_count[data.get('type', 'unknown')] += 1
        
        relation_types_count = defaultdict(int)
        for _, _, data in graph.edges(data=True):
            relation_types_count[data.get('relation', 'unknown')] += 1
        
        graph_data = {
            'nodes': nodes_data,
            'edges': edges_data,
            'statistics': {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'node_types': dict(node_types_count),
                'relation_types': dict(relation_types_count),
                'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
            }
        }
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
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
            node_id = node.pop('id')
            graph.add_node(node_id, **node)
        
        # 添加边
        for edge in graph_data['edges']:
            source = edge.pop('source')
            target = edge.pop('target')
            graph.add_edge(source, target, **edge)
        
        logger.info(f"知识图谱加载完成: {graph.number_of_nodes()} 个节点, "
                   f"{graph.number_of_edges()} 条边")
        
        return graph


# 使用示例
if __name__ == "__main__":
    # 配置示例
    config = {
        'knowledge_graph': {
            'entity_types': ['技术', '产品', '材料', '工艺', '公司', '人员'],
            'relation_types': ['使用', '包含', '生产', '研发', '依赖', '改进', '替代', '认证', '合作', '应用于'],
            'chunk_size': 1000,
            'max_chunk_overlap': 200,
            'extraction_rules': {
                'confidence_threshold': 0.7
            }
        },
        'tcl_specific': {
            'technical_terms': {
                'display': ['OLED', 'LCD', 'Mini-LED', 'QLED'],
                'semiconductor': ['芯片', '半导体', '集成电路'],
                'appliance': ['空调', '冰箱', '洗衣机'],
                'materials': ['面板', '背光', '驱动IC'],
                'manufacturing': ['贴片', '封装', '测试']
            }
        },
        'models': {
            'llm_model': {
                'path': 'THUDM/chatglm3-6b'  # 可以替换为其他模型
            }
        }
    }
    
    # 创建知识图谱构建器
    builder = KnowledgeGraphBuilder(config)
    
    # 从文本构建知识图谱
    kg = builder.build_from_texts('./data/texts')
    
    # 保存知识图谱
    builder.save_graph(kg, Path('./output/tcl_knowledge_graph.json'))