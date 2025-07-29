"""
知识图谱构建器 - 改进版
使用大语言模型从TCL工业领域文本中提取实体和关系，构建知识图谱
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
    """知识图谱构建器 - 基于大模型的改进版"""
    
    def __init__(self, config: dict):
        self.config = config
        self.kg_config = config.get('knowledge_graph', {})
        
        # 初始化实体和关系类型
        self.entity_types = self.kg_config.get('entity_types', [])
        self.relation_types = self.kg_config.get('relation_types', [])
        
        # TCL特定配置
        self.tcl_terms = config.get('tcl_specific', {}).get('technical_terms', {})
        self.domain = config.get('domain', 'TCL工业')
        
        # 加载大语言模型
        self.model_path = config['models'].get('llm_model', {}).get('path', 
                                                config['models']['kg_extractor_model']['path'])
        self._load_llm_model()
        
        # 知识图谱
        self.graph = nx.MultiDiGraph()
        
        # 配置参数
        self.chunk_size = self.kg_config.get('chunk_size', 1000)
        self.max_tokens = self.kg_config.get('max_tokens', 1024)
        self.temperature = self.kg_config.get('temperature', 0.3)
        
    def _load_llm_model(self):
        """加载大语言模型"""
        logger.info(f"加载大语言模型: {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
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
                
            # 分块处理长文本
            chunks = self._chunk_text(text)
            
            for chunk in chunks:
                # 提取实体和关系
                entities, relations = self._extract_entities_relations_with_llm(chunk)
                all_entities.extend(entities)
                all_relations.extend(relations)
        
        # 构建知识图谱
        self._build_graph(all_entities, all_relations)
        
        logger.info(f"知识图谱构建完成: {self.graph.number_of_nodes()} 个节点, "
                   f"{self.graph.number_of_edges()} 条边")
        
        return self.graph
    
    def _chunk_text(self, text: str) -> List[str]:
        """智能文本分块"""
        # 按段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # 检查是否需要分块
            if len(current_chunk) + len(paragraph) < self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 如果没有分块，按句子分割
        if not chunks and text:
            sentences = self._split_sentences(text)
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) < self.chunk_size:
                    current_chunk += sent + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent + " "
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """文本分句"""
        sentences = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # 按句号、问号、感叹号分句
            sents = re.split(r'[。！？]', line)
            sentences.extend([s.strip() for s in sents if s.strip()])
        return sentences
    
    def _extract_entities_relations_with_llm(self, text_chunk: str) -> Tuple[List[Dict], List[Dict]]:
        """使用大语言模型提取实体和关系"""
        # 构建实体提取prompt
        entity_prompt = self._build_entity_extraction_prompt(text_chunk)
        
        # 调用模型提取实体
        entities_response = self._generate_text(entity_prompt)
        entities = self._parse_entities_response(entities_response, text_chunk)
        
        if not entities:
            logger.warning("未能提取到有效实体")
            return [], []
        
        # 构建关系提取prompt
        relation_prompt = self._build_relation_extraction_prompt(text_chunk, entities)
        
        # 调用模型提取关系
        relations_response = self._generate_text(relation_prompt)
        relations = self._parse_relations_response(relations_response, entities)
        
        return entities, relations
    
    def _build_entity_extraction_prompt(self, text: str) -> str:
        """构建实体提取prompt"""
        # 构建实体类型说明
        entity_types_desc = "\n".join([f"- {et}" for et in self.entity_types]) if self.entity_types else "- 技术\n- 产品\n- 材料\n- 工艺\n- 公司\n- 人员\n- 参数"
        
        # 构建TCL术语提示
        tcl_hints = ""
        if self.tcl_terms:
            tcl_hints = "\n\n重点关注以下TCL专业术语类别：\n"
            for category, terms in self.tcl_terms.items():
                tcl_hints += f"- {category}: {', '.join(terms[:5])}...\n"
        
        prompt = f"""你是一个{self.domain}领域的知识工程师。请从以下技术文档中提取所有重要实体。

文档内容：
{text}

实体类型包括但不限于：
{entity_types_desc}
{tcl_hints}

请按以下JSON格式输出实体列表：
[
  {{
    "text": "实体文本",
    "type": "实体类型",
    "start": 在原文中的起始位置（可选）,
    "confidence": 置信度（0-1之间的数值）
  }},
  ...
]

要求：
1. 提取所有重要的技术实体、产品、材料、公司等
2. 实体文本要完整准确，保持原文中的表述
3. 类型要准确，如果不确定可以使用"其他"
4. 置信度反映你对该实体识别的确信程度
5. 只返回JSON数组，不要其他内容

JSON输出："""
        
        return prompt
    
    def _build_relation_extraction_prompt(self, text: str, entities: List[Dict]) -> str:
        """构建关系提取prompt"""
        # 构建实体列表描述
        entities_desc = "\n".join([
            f"{i+1}. {e['text']} ({e['type']})"
            for i, e in enumerate(entities)
        ])
        
        # 构建关系类型说明
        relation_types_desc = "\n".join([f"- {rt}" for rt in self.relation_types]) if self.relation_types else "- 使用\n- 包含\n- 生产\n- 研发\n- 依赖\n- 改进\n- 替代\n- 认证\n- 合作\n- 应用于"
        
        prompt = f"""基于以下实体和文档内容，提取实体间的关系。

文档内容：
{text}

已识别的实体：
{entities_desc}

关系类型包括但不限于：
{relation_types_desc}

请按以下JSON格式输出关系列表：
[
  {{
    "source": "源实体文本",
    "source_type": "源实体类型",
    "target": "目标实体文本", 
    "target_type": "目标实体类型",
    "relation": "关系类型",
    "sentence": "支持该关系的原文句子（可选）"
  }},
  ...
]

要求：
1. source和target必须使用实体列表中的准确文本
2. 只提取有明确依据的关系
3. 关系类型要准确描述实体间的关联
4. 如果可能，提供支持该关系的原文句子
5. 只返回JSON数组，不要其他内容

JSON输出："""
        
        return prompt
    
    def _generate_text(self, prompt: str) -> str:
        """调用大语言模型生成文本"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                  max_length=2048).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                           skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"模型生成失败: {e}")
            return "[]"
    
    def _parse_entities_response(self, response: str, original_text: str) -> List[Dict]:
        """解析实体提取响应"""
        entities = self._parse_json_response(response, [])
        
        # 规范化和验证实体
        valid_entities = []
        for entity in entities:
            # 确保必要字段存在
            if 'text' not in entity or not entity['text']:
                continue
                
            # 规范化实体格式
            normalized_entity = {
                'text': entity['text'].strip(),
                'type': entity.get('type', '其他'),
                'start': entity.get('start', original_text.find(entity['text'])),
                'confidence': float(entity.get('confidence', 0.8)),
                'sentence': original_text  # 保存原始文本块
            }
            
            # 验证实体长度
            min_len = self.kg_config.get('extraction_rules', {}).get('min_entity_length', 1)
            max_len = self.kg_config.get('extraction_rules', {}).get('max_entity_length', 50)
            
            if min_len <= len(normalized_entity['text']) <= max_len:
                valid_entities.append(normalized_entity)
            else:
                logger.debug(f"实体长度不符合要求，跳过: {normalized_entity['text']}")
        
        return valid_entities
    
    def _parse_relations_response(self, response: str, entities: List[Dict]) -> List[Dict]:
        """解析关系提取响应"""
        relations = self._parse_json_response(response, [])
        
        # 创建实体文本到实体对象的映射
        entity_map = {e['text']: e for e in entities}
        
        # 规范化和验证关系
        valid_relations = []
        for relation in relations:
            # 确保必要字段存在
            if not all(key in relation for key in ['source', 'target', 'relation']):
                continue
            
            # 验证实体是否存在
            source_text = relation['source'].strip()
            target_text = relation['target'].strip()
            
            if source_text not in entity_map or target_text not in entity_map:
                logger.debug(f"关系引用了不存在的实体: {source_text} -> {target_text}")
                continue
            
            # 规范化关系格式
            normalized_relation = {
                'source': source_text,
                'source_type': relation.get('source_type', entity_map[source_text]['type']),
                'target': target_text,
                'target_type': relation.get('target_type', entity_map[target_text]['type']),
                'relation': relation['relation'],
                'sentence': relation.get('sentence', '')
            }
            
            valid_relations.append(normalized_relation)
        
        return valid_relations
    
    def _parse_json_response(self, text: str, default_value):
        """解析JSON响应"""
        try:
            # 直接解析
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取JSON数组
        try:
            json_match = re.search(r'\[.*?\]', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        
        # 尝试提取多个JSON对象
        try:
            matches = re.findall(r'\{[^{}]*\}', text)
            objects = []
            for match in matches:
                try:
                    obj = json.loads(match)
                    objects.append(obj)
                except json.JSONDecodeError:
                    continue
            if objects:
                return objects
        except Exception:
            pass
        
        logger.warning(f"无法解析JSON响应: {text[:200]}...")
        return default_value
    
    def _build_graph(self, entities: List[Dict], relations: List[Dict]):
        """构建知识图谱"""
        # 添加节点（去重）
        unique_entities = {}
        for entity in entities:
            key = (entity['text'], entity['type'])
            if key not in unique_entities or entity['confidence'] > unique_entities[key]['confidence']:
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
                    sentence=relation.get('sentence', '')
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
                'node_types': self._count_types(nx.get_node_attributes(graph, 'type')),
                'relation_types': self._count_types(nx.get_edge_attributes(graph, 'relation'))
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"知识图谱已保存到: {output_path}")
    
    def _count_types(self, type_dict: dict) -> dict:
        """统计类型分布"""
        type_counts = defaultdict(int)
        for type_value in type_dict.values():
            type_counts[type_value] += 1
        return dict(type_counts)
    
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


# 使用示例
if __name__ == "__main__":
    # 配置示例
    config = {
        'domain': 'TCL工业',
        'knowledge_graph': {
            'entity_types': ['技术', '产品', '材料', '工艺', '公司', '人员', '参数'],
            'relation_types': ['使用', '包含', '生产', '研发', '依赖', '改进', '替代'],
            'chunk_size': 1000,
            'max_tokens': 1024,
            'temperature': 0.3,
            'extraction_rules': {
                'min_entity_length': 2,
                'max_entity_length': 50,
                'confidence_threshold': 0.5
            }
        },
        'models': {
            'llm_model': {
                'path': 'Qwen/Qwen2.5-7B-Instruct'  # 或其他支持的中文大模型
            },
            'kg_extractor_model': {
                'path': 'Qwen/Qwen2.5-7B-Instruct'  # 兼容原配置
            }
        },
        'tcl_specific': {
            'technical_terms': {
                'display': ['液晶显示', 'OLED', '量子点'],
                'semiconductor': ['芯片', '半导体', '集成电路'],
                'appliance': ['空调', '冰箱', '洗衣机'],
                'materials': ['面板', '基板', '偏光片'],
                'manufacturing': ['蒸镀', '光刻', '封装']
            }
        }
    }
    
    # 创建知识图谱构建器
    builder = KnowledgeGraphBuilder(config)
    
    # 构建知识图谱
    kg = builder.build_from_texts("./data/texts")
    
    # 保存知识图谱
    builder.save_graph(kg, Path("./output/knowledge_graph.json"))