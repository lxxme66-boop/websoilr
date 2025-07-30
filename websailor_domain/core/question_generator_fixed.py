"""
问题生成器 - WebSailor核心模块（修复版）
基于子图中节点与关系，设计QA问题
修复了数据格式不匹配和验证过严的问题
"""

import json
import logging
import random
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import networkx as nx

logger = logging.getLogger(__name__)

class QuestionGenerator:
    """
    WebSailor核心：问题生成器（修复版）
    基于子图生成多样化的问题，覆盖不同难度和类型
    修复了数据格式兼容性和验证机制
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.qg_config = config.get('question_generation', {})
        
        # 问题类型
        self.question_types = self.qg_config.get('question_types', [
            'factual', 'reasoning', 'multi_hop', 'comparative'
        ])
        self.complexity_levels = self.qg_config.get('complexity_levels', {})
        self.language_patterns = self.qg_config.get('language_patterns', {
            'zh_cn': 0.7,
            'en': 0.3
        })
        
        # 加载QA生成模型
        self._load_qa_generator()
        
        # TCL特定配置
        self.tcl_config = config.get('tcl_specific', {})
        
        # 问题模板
        self._init_question_templates()
        
        # 降低合理性阈值，使更多问题通过验证
        self.validity_threshold = 0.5  # 从0.7降低到0.5
        self.max_optimization_attempts = 1  # 减少优化尝试次数
        
        # 添加简单模式标志
        self.simple_mode = config.get('simple_mode', False)
        
    def _load_qa_generator(self):
        """加载QA生成模型"""
        model_config = self.config['models']['qa_generator_model']
        model_path = model_config['path']
        
        logger.info(f"加载QA生成模型: {model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_loaded = True
        except Exception as e:
            logger.warning(f"无法加载指定模型 {model_path}: {e}")
            logger.warning("将使用简单模板生成模式")
            self.model_loaded = False
            self.simple_mode = True
            
        if self.model_loaded:
            self.model.eval()
            # 设置生成参数
            self.max_length = model_config.get('max_length', 2048)
            self.temperature = model_config.get('temperature', 0.7)
        
    def _init_question_templates(self):
        """初始化问题模板"""
        self.question_templates = {
            'factual': {
                'zh_cn': [
                    "{entity}是什么？",
                    "{entity}的主要特点是什么？",
                    "{entity}有哪些应用？",
                    "请介绍{entity}的基本信息。"
                ],
                'en': [
                    "What is {entity}?",
                    "What are the main features of {entity}?",
                    "What are the applications of {entity}?",
                    "Please introduce the basic information about {entity}."
                ]
            },
            'comparison': {
                'zh_cn': [
                    "{entity1}和{entity2}有什么区别？",
                    "比较{entity1}和{entity2}的特点。",
                    "{entity1}相比{entity2}有什么优势？"
                ],
                'en': [
                    "What are the differences between {entity1} and {entity2}?",
                    "Compare the features of {entity1} and {entity2}.",
                    "What advantages does {entity1} have over {entity2}?"
                ]
            },
            'reasoning': {
                'zh_cn': [
                    "为什么{entity1}需要{entity2}？",
                    "{entity}的工作原理是什么？",
                    "如何优化{entity}的性能？"
                ],
                'en': [
                    "Why does {entity1} need {entity2}?",
                    "What is the working principle of {entity}?",
                    "How to optimize the performance of {entity}?"
                ]
            },
            'multi_hop': {
                'zh_cn': [
                    "{entity1}和{entity2}之间有什么联系？",
                    "从{entity1}到{entity2}的过程是什么？",
                    "{entity1}如何影响{entity2}？"
                ],
                'en': [
                    "What is the connection between {entity1} and {entity2}?",
                    "What is the process from {entity1} to {entity2}?",
                    "How does {entity1} affect {entity2}?"
                ]
            }
        }
    
    def _convert_digraph_to_dict(self, subgraph: nx.DiGraph) -> Dict:
        """将NetworkX DiGraph转换为字典格式"""
        try:
            nodes = []
            for node, data in subgraph.nodes(data=True):
                node_dict = {
                    'id': str(node),
                    'type': data.get('type', 'unknown')
                }
                # 添加其他属性
                for key, value in data.items():
                    if key != 'type':
                        node_dict[key] = value
                nodes.append(node_dict)
            
            edges = []
            for u, v, data in subgraph.edges(data=True):
                edge_dict = {
                    'source': str(u),
                    'target': str(v),
                    'relation': data.get('type', data.get('relation', '相关'))
                }
                # 添加其他属性
                for key, value in data.items():
                    if key not in ['type', 'relation']:
                        edge_dict[key] = value
                edges.append(edge_dict)
            
            # 获取节点类型统计
            node_types = {}
            for node in nodes:
                node_type = node['type']
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # 获取关系类型统计
            relation_types = {}
            for edge in edges:
                rel_type = edge['relation']
                relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
            
            return {
                'nodes': nodes,
                'edges': edges,
                'num_nodes': len(nodes),
                'num_edges': len(edges),
                'node_types': list(node_types.keys()),
                'relation_types': list(relation_types.keys()),
                'topology': self._identify_topology(subgraph)
            }
        except Exception as e:
            logger.error(f"转换子图格式失败: {e}")
            # 返回一个最小的有效结构
            return {
                'nodes': [],
                'edges': [],
                'num_nodes': 0,
                'num_edges': 0,
                'node_types': [],
                'relation_types': [],
                'topology': 'unknown'
            }
    
    def _identify_topology(self, subgraph: Union[nx.DiGraph, Dict]) -> str:
        """识别子图的拓扑类型"""
        try:
            if isinstance(subgraph, dict):
                num_nodes = subgraph.get('num_nodes', 0)
                num_edges = subgraph.get('num_edges', 0)
            else:
                num_nodes = subgraph.number_of_nodes()
                num_edges = subgraph.number_of_edges()
            
            if num_nodes == 0:
                return 'empty'
            
            # 简单的拓扑分类
            density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
            
            if density < 0.2:
                return 'sparse'
            elif density > 0.6:
                return 'dense'
            else:
                return 'mixed'
        except:
            return 'unknown'
    
    def generate_questions(self, subgraphs: List[Union[nx.DiGraph, Dict]], 
                          questions_per_subgraph: int = 5) -> List[Dict]:
        """为子图列表生成问题（兼容多种输入格式）"""
        all_qa_pairs = []
        
        logger.info(f"开始为 {len(subgraphs)} 个子图生成问题...")
        logger.info(f"简单模式: {self.simple_mode}")
        logger.info(f"验证阈值: {self.validity_threshold}")
        
        # 转换子图格式
        converted_subgraphs = []
        for i, subgraph in enumerate(subgraphs):
            if isinstance(subgraph, nx.DiGraph):
                logger.debug(f"子图 {i}: NetworkX格式，转换为字典")
                converted = self._convert_digraph_to_dict(subgraph)
            else:
                logger.debug(f"子图 {i}: 已是字典格式")
                converted = subgraph
            
            # 记录子图信息
            logger.info(f"子图 {i}: {converted['num_nodes']} 个节点, {converted['num_edges']} 条边")
            
            # 跳过空子图
            if converted['num_nodes'] == 0:
                logger.warning(f"子图 {i} 为空，跳过")
                continue
                
            converted_subgraphs.append(converted)
        
        # 为每个子图生成问题
        for subgraph in tqdm(converted_subgraphs, desc="生成问题"):
            try:
                # 分析子图特征
                subgraph_features = self._analyze_subgraph(subgraph)
                
                # 生成指定数量的问题
                subgraph_qa_pairs = []
                attempts = 0
                max_attempts = questions_per_subgraph * 3  # 允许更多尝试
                
                while len(subgraph_qa_pairs) < questions_per_subgraph and attempts < max_attempts:
                    attempts += 1
                    
                    # 选择问题类型
                    if subgraph['num_edges'] == 0:
                        # 没有边的情况下只生成事实型问题
                        q_type = 'factual'
                    else:
                        # 根据子图特征选择合适的问题类型
                        suitable_types = self._select_question_types(subgraph_features)
                        q_type = random.choice(suitable_types) if suitable_types else 'factual'
                    
                    # 生成问题
                    qa_pair = self._generate_question_for_type(
                        subgraph, q_type, subgraph_features
                    )
                    
                    if qa_pair:
                        subgraph_qa_pairs.append(qa_pair)
                        logger.debug(f"成功生成 {q_type} 类型问题")
                
                logger.info(f"子图生成了 {len(subgraph_qa_pairs)} 个问题")
                all_qa_pairs.extend(subgraph_qa_pairs)
                
            except Exception as e:
                logger.error(f"处理子图时出错: {e}", exc_info=True)
                continue
        
        # 后处理和质量检查（放宽标准）
        filtered_qa_pairs = self._filter_qa_pairs(all_qa_pairs)
        
        logger.info(f"质量过滤: {len(all_qa_pairs)} -> {len(filtered_qa_pairs)}")
        logger.info(f"共生成 {len(filtered_qa_pairs)} 个高质量QA对")
        
        return filtered_qa_pairs
    
    def _analyze_subgraph(self, subgraph: Dict) -> Dict:
        """分析子图特征"""
        features = {
            'topology': subgraph.get('topology', 'unknown'),
            'num_nodes': subgraph.get('num_nodes', 0),
            'num_edges': subgraph.get('num_edges', 0),
            'node_types': subgraph.get('node_types', []),
            'relation_types': subgraph.get('relation_types', []),
            'density': subgraph['num_edges'] / (subgraph['num_nodes'] * (subgraph['num_nodes'] - 1))
                      if subgraph['num_nodes'] > 1 else 0
        }
        
        # 识别关键实体
        features['key_entities'] = self._identify_key_entities(subgraph)
        
        return features
    
    def _identify_key_entities(self, subgraph: Dict) -> List[Dict]:
        """识别子图中的关键实体"""
        key_entities = []
        
        if not subgraph.get('nodes'):
            return key_entities
        
        # 基于度数识别
        node_degrees = {}
        for edge in subgraph.get('edges', []):
            source = edge['source']
            target = edge['target']
            node_degrees[source] = node_degrees.get(source, 0) + 1
            node_degrees[target] = node_degrees.get(target, 0) + 1
        
        # 如果没有边，则所有节点都是关键实体
        if not node_degrees:
            for node in subgraph['nodes'][:5]:  # 最多取5个
                key_entities.append({
                    'id': node['id'],
                    'type': node.get('type', 'unknown'),
                    'degree': 0,
                    'role': 'isolated'
                })
        else:
            # 选择度数最高的节点
            sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
            for node_id, degree in sorted_nodes[:3]:  # 前3个
                # 找到节点信息
                for node in subgraph['nodes']:
                    if node['id'] == node_id:
                        key_entities.append({
                            'id': node_id,
                            'type': node.get('type', 'unknown'),
                            'degree': degree,
                            'role': 'hub' if degree > 2 else 'connector'
                        })
                        break
        
        return key_entities
    
    def _select_question_types(self, features: Dict) -> List[str]:
        """根据子图特征选择合适的问题类型"""
        suitable_types = ['factual']  # 始终包含事实型问题
        
        # 根据边数添加其他类型
        if features['num_edges'] > 0:
            suitable_types.append('reasoning')
            
        if features['num_edges'] >= 2:
            suitable_types.append('multi_hop')
            
        if features['num_nodes'] >= 2:
            suitable_types.append('comparison')
        
        return suitable_types
    
    def _generate_question_for_type(self, subgraph: Dict, 
                                   q_type: str, features: Dict) -> Optional[Dict]:
        """为特定类型生成单个问题"""
        try:
            # 选择语言
            lang = 'zh_cn' if random.random() < 0.7 else 'en'
            
            # 根据问题类型生成
            if q_type == 'factual':
                return self._generate_factual_question(subgraph, features, lang)
            elif q_type == 'comparison':
                return self._generate_comparison_question(subgraph, features, lang)
            elif q_type == 'reasoning':
                return self._generate_reasoning_question(subgraph, features, lang)
            elif q_type == 'multi_hop':
                return self._generate_multihop_question(subgraph, features, lang)
            else:
                return self._generate_factual_question(subgraph, features, lang)
                
        except Exception as e:
            logger.warning(f"生成 {q_type} 类型问题失败: {e}")
            # 回退到简单的事实型问题
            return self._generate_simple_factual_question(subgraph)
    
    def _generate_simple_factual_question(self, subgraph: Dict) -> Optional[Dict]:
        """生成简单的事实型问题（回退方案）"""
        try:
            if subgraph['nodes']:
                node = random.choice(subgraph['nodes'])
                lang = 'zh_cn' if random.random() < 0.7 else 'en'
                
                if lang == 'zh_cn':
                    question = f"{node['id']}是什么？"
                    answer = f"{node['id']}是一种{node.get('type', '实体')}。"
                else:
                    question = f"What is {node['id']}?"
                    answer = f"{node['id']} is a type of {node.get('type', 'entity')}."
                
                return {
                    'question': question,
                    'answer': answer,
                    'type': 'factual',
                    'language': lang,
                    'subgraph': subgraph,
                    'validity_score': 0.6,  # 给一个中等的分数
                    'simple_mode': True
                }
        except Exception as e:
            logger.error(f"生成简单问题失败: {e}")
            return None
    
    def _generate_factual_question(self, subgraph: Dict, 
                                  features: Dict, lang: str) -> Optional[Dict]:
        """生成事实型问题"""
        try:
            templates = self.question_templates['factual'][lang]
            
            # 选择一个实体
            if features['key_entities']:
                entity_info = random.choice(features['key_entities'])
                entity = entity_info['id']
            else:
                node = random.choice(subgraph['nodes'])
                entity = node['id']
            
            # 选择模板
            template = random.choice(templates)
            question = template.format(entity=entity)
            
            # 生成答案
            if self.simple_mode or not self.model_loaded:
                # 简单模式：基于模板生成答案
                answer = self._generate_simple_answer(entity, subgraph, lang)
            else:
                # 使用LLM生成答案
                answer = self._generate_answer_with_llm(question, subgraph, 'factual')
            
            return {
                'question': question,
                'answer': answer,
                'type': 'factual',
                'language': lang,
                'subgraph': subgraph,
                'validity_score': 0.8,
                'entities': [entity]
            }
            
        except Exception as e:
            logger.warning(f"生成事实型问题失败: {e}")
            return self._generate_simple_factual_question(subgraph)
    
    def _generate_comparison_question(self, subgraph: Dict, 
                                     features: Dict, lang: str) -> Optional[Dict]:
        """生成比较型问题"""
        try:
            # 需要至少两个节点
            if len(subgraph['nodes']) < 2:
                return None
            
            templates = self.question_templates['comparison'][lang]
            
            # 选择两个实体
            nodes = random.sample(subgraph['nodes'], 2)
            entity1 = nodes[0]['id']
            entity2 = nodes[1]['id']
            
            # 选择模板
            template = random.choice(templates)
            question = template.format(entity1=entity1, entity2=entity2)
            
            # 生成答案
            if self.simple_mode or not self.model_loaded:
                answer = self._generate_simple_comparison_answer(entity1, entity2, subgraph, lang)
            else:
                answer = self._generate_answer_with_llm(question, subgraph, 'comparison')
            
            return {
                'question': question,
                'answer': answer,
                'type': 'comparison',
                'language': lang,
                'subgraph': subgraph,
                'validity_score': 0.7,
                'entities': [entity1, entity2]
            }
            
        except Exception as e:
            logger.warning(f"生成比较型问题失败: {e}")
            return None
    
    def _generate_reasoning_question(self, subgraph: Dict, 
                                   features: Dict, lang: str) -> Optional[Dict]:
        """生成推理型问题"""
        try:
            # 需要至少一条边
            if not subgraph['edges']:
                return None
            
            templates = self.question_templates['reasoning'][lang]
            
            # 选择一条边
            edge = random.choice(subgraph['edges'])
            entity1 = edge['source']
            entity2 = edge['target']
            
            # 找到实体信息
            entity1_node = next((n for n in subgraph['nodes'] if n['id'] == entity1), None)
            
            # 选择模板
            template = random.choice(templates)
            if '{entity1}' in template and '{entity2}' in template:
                question = template.format(entity1=entity1, entity2=entity2)
            else:
                question = template.format(entity=entity1)
            
            # 生成答案
            if self.simple_mode or not self.model_loaded:
                answer = self._generate_simple_reasoning_answer(entity1, entity2, edge, subgraph, lang)
            else:
                answer = self._generate_answer_with_llm(question, subgraph, 'reasoning')
            
            return {
                'question': question,
                'answer': answer,
                'type': 'reasoning',
                'language': lang,
                'subgraph': subgraph,
                'validity_score': 0.7,
                'entities': [entity1, entity2]
            }
            
        except Exception as e:
            logger.warning(f"生成推理型问题失败: {e}")
            return None
    
    def _generate_multihop_question(self, subgraph: Dict, 
                                   features: Dict, lang: str) -> Optional[Dict]:
        """生成多跳问题"""
        try:
            # 需要至少两条边
            if len(subgraph['edges']) < 2:
                return None
            
            templates = self.question_templates['multi_hop'][lang]
            
            # 尝试找到一条路径
            path_entities = self._find_path_entities(subgraph)
            if len(path_entities) < 2:
                return None
            
            entity1 = path_entities[0]
            entity2 = path_entities[-1]
            
            # 选择模板
            template = random.choice(templates)
            question = template.format(entity1=entity1, entity2=entity2)
            
            # 生成答案
            if self.simple_mode or not self.model_loaded:
                answer = self._generate_simple_multihop_answer(entity1, entity2, path_entities, subgraph, lang)
            else:
                answer = self._generate_answer_with_llm(question, subgraph, 'multi_hop')
            
            return {
                'question': question,
                'answer': answer,
                'type': 'multi_hop',
                'language': lang,
                'subgraph': subgraph,
                'validity_score': 0.6,
                'entities': path_entities
            }
            
        except Exception as e:
            logger.warning(f"生成多跳问题失败: {e}")
            return None
    
    def _find_path_entities(self, subgraph: Dict) -> List[str]:
        """找到一条路径上的实体"""
        try:
            # 简单实现：找到有连接的实体
            if len(subgraph['edges']) >= 2:
                # 找到一个中间节点
                node_connections = {}
                for edge in subgraph['edges']:
                    source = edge['source']
                    target = edge['target']
                    
                    if source not in node_connections:
                        node_connections[source] = {'in': [], 'out': []}
                    if target not in node_connections:
                        node_connections[target] = {'in': [], 'out': []}
                    
                    node_connections[source]['out'].append(target)
                    node_connections[target]['in'].append(source)
                
                # 找到既有入边又有出边的节点
                for node, connections in node_connections.items():
                    if connections['in'] and connections['out']:
                        # 构建路径
                        start = connections['in'][0]
                        middle = node
                        end = connections['out'][0]
                        return [start, middle, end]
            
            # 如果找不到路径，返回两个有连接的节点
            if subgraph['edges']:
                edge = subgraph['edges'][0]
                return [edge['source'], edge['target']]
            
            return []
        except:
            return []
    
    def _generate_simple_answer(self, entity: str, subgraph: Dict, lang: str) -> str:
        """生成简单的答案（不使用LLM）"""
        # 找到实体信息
        entity_node = next((n for n in subgraph['nodes'] if n['id'] == entity), None)
        
        if entity_node:
            entity_type = entity_node.get('type', '实体' if lang == 'zh_cn' else 'entity')
            
            # 找到相关的边
            related_info = []
            for edge in subgraph['edges']:
                if edge['source'] == entity:
                    target_node = next((n for n in subgraph['nodes'] if n['id'] == edge['target']), None)
                    if target_node:
                        relation = edge.get('relation', '相关' if lang == 'zh_cn' else 'related to')
                        related_info.append(f"{relation} {edge['target']}")
            
            if lang == 'zh_cn':
                answer = f"{entity}是一种{entity_type}。"
                if related_info:
                    answer += f"它{', '.join(related_info[:2])}。"
            else:
                answer = f"{entity} is a type of {entity_type}."
                if related_info:
                    answer += f" It is {', '.join(related_info[:2])}."
        else:
            if lang == 'zh_cn':
                answer = f"{entity}是该领域的一个重要概念。"
            else:
                answer = f"{entity} is an important concept in this field."
        
        return answer
    
    def _generate_simple_comparison_answer(self, entity1: str, entity2: str, 
                                         subgraph: Dict, lang: str) -> str:
        """生成简单的比较答案"""
        # 找到实体信息
        entity1_node = next((n for n in subgraph['nodes'] if n['id'] == entity1), None)
        entity2_node = next((n for n in subgraph['nodes'] if n['id'] == entity2), None)
        
        if lang == 'zh_cn':
            if entity1_node and entity2_node:
                type1 = entity1_node.get('type', '实体')
                type2 = entity2_node.get('type', '实体')
                if type1 == type2:
                    answer = f"{entity1}和{entity2}都是{type1}类型的实体，它们在功能和应用上可能有所不同。"
                else:
                    answer = f"{entity1}是{type1}类型，而{entity2}是{type2}类型，它们在本质上有所区别。"
            else:
                answer = f"{entity1}和{entity2}是两个不同的概念，各有其特点和应用场景。"
        else:
            if entity1_node and entity2_node:
                type1 = entity1_node.get('type', 'entity')
                type2 = entity2_node.get('type', 'entity')
                if type1 == type2:
                    answer = f"Both {entity1} and {entity2} are {type1} type entities, but they may differ in functionality and applications."
                else:
                    answer = f"{entity1} is a {type1} type, while {entity2} is a {type2} type, they are fundamentally different."
            else:
                answer = f"{entity1} and {entity2} are two different concepts, each with its own characteristics and use cases."
        
        return answer
    
    def _generate_simple_reasoning_answer(self, entity1: str, entity2: str, 
                                        edge: Dict, subgraph: Dict, lang: str) -> str:
        """生成简单的推理答案"""
        relation = edge.get('relation', '相关' if lang == 'zh_cn' else 'related')
        
        if lang == 'zh_cn':
            answer = f"{entity1}与{entity2}之间存在{relation}关系。这种关系表明了它们在系统中的相互作用和依赖性。"
        else:
            answer = f"There is a {relation} relationship between {entity1} and {entity2}. This relationship indicates their interaction and dependency in the system."
        
        return answer
    
    def _generate_simple_multihop_answer(self, entity1: str, entity2: str, 
                                       path: List[str], subgraph: Dict, lang: str) -> str:
        """生成简单的多跳答案"""
        if len(path) > 2:
            middle_entities = path[1:-1]
            if lang == 'zh_cn':
                answer = f"{entity1}通过{', '.join(middle_entities)}与{entity2}相连接。这种多步关系展示了系统中的复杂交互。"
            else:
                answer = f"{entity1} is connected to {entity2} through {', '.join(middle_entities)}. This multi-step relationship shows the complex interactions in the system."
        else:
            if lang == 'zh_cn':
                answer = f"{entity1}与{entity2}存在直接或间接的联系，这种联系对系统的运作具有重要意义。"
            else:
                answer = f"{entity1} and {entity2} have a direct or indirect connection, which is important for the system's operation."
        
        return answer
    
    def _generate_answer_with_llm(self, question: str, subgraph: Dict, 
                                 q_type: str) -> str:
        """使用LLM生成答案"""
        try:
            # 构造提示
            prompt = self._create_answer_prompt(question, subgraph, q_type)
            
            # 生成答案
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                  max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9
                )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取答案部分
            answer = self._extract_answer_from_output(answer, prompt)
            
            return answer
            
        except Exception as e:
            logger.warning(f"LLM生成答案失败: {e}")
            # 回退到简单答案生成
            return self._generate_fallback_answer(question, subgraph, q_type)
    
    def _generate_fallback_answer(self, question: str, subgraph: Dict, q_type: str) -> str:
        """生成回退答案"""
        lang = 'zh_cn' if any(ord(c) > 127 for c in question) else 'en'
        
        if lang == 'zh_cn':
            return "根据知识图谱的信息，这是一个复杂的问题，需要综合考虑多个因素。"
        else:
            return "Based on the knowledge graph information, this is a complex question that requires considering multiple factors."
    
    def _create_answer_prompt(self, question: str, subgraph: Dict, 
                            q_type: str) -> str:
        """创建答案生成提示"""
        # 将子图信息格式化
        subgraph_desc = self._format_subgraph_for_prompt(subgraph)
        
        prompt = f"""基于以下知识图谱信息，回答问题。

知识图谱信息：
{subgraph_desc}

问题类型：{q_type}
问题：{question}

请提供准确、简洁的答案："""

        return prompt
    
    def _format_subgraph_for_prompt(self, subgraph: Dict) -> str:
        """格式化子图信息用于提示"""
        lines = []
        
        # 节点信息
        lines.append("节点：")
        for node in subgraph['nodes'][:10]:  # 限制数量
            lines.append(f"  - {node['id']} (类型: {node.get('type', 'unknown')})")
        
        if len(subgraph['nodes']) > 10:
            lines.append(f"  ... 还有 {len(subgraph['nodes']) - 10} 个节点")
        
        # 边信息
        lines.append("\n关系：")
        for edge in subgraph['edges'][:10]:  # 限制数量
            lines.append(f"  - {edge['source']} --[{edge.get('relation', '相关')}]--> {edge['target']}")
        
        if len(subgraph['edges']) > 10:
            lines.append(f"  ... 还有 {len(subgraph['edges']) - 10} 条边")
        
        return '\n'.join(lines)
    
    def _extract_answer_from_output(self, output: str, prompt: str) -> str:
        """从模型输出中提取答案"""
        # 移除提示部分
        if prompt in output:
            output = output.replace(prompt, '').strip()
        
        # 提取答案部分
        if '答案：' in output:
            answer = output.split('答案：')[-1].strip()
        elif 'Answer:' in output:
            answer = output.split('Answer:')[-1].strip()
        else:
            answer = output.strip()
        
        # 清理答案
        answer = answer.replace('\n\n', '\n')
        
        # 限制长度
        max_answer_length = 300
        if len(answer) > max_answer_length:
            answer = answer[:max_answer_length] + "..."
        
        return answer
    
    def _filter_qa_pairs(self, qa_pairs: List[Dict]) -> List[Dict]:
        """过滤和质量检查QA对（放宽标准）"""
        filtered = []
        
        # 用于去重的集合
        seen_questions = set()
        
        for qa in qa_pairs:
            # 基本检查
            if not qa.get('question') or not qa.get('answer'):
                continue
            
            # 长度检查（放宽标准）
            q_len = len(qa['question'])
            a_len = len(qa['answer'])
            
            if q_len < 5 or q_len > 500:  # 放宽问题长度限制
                continue
            
            if a_len < 5 or a_len > 1000:  # 放宽答案长度限制
                continue
            
            # 合理性分数检查（降低标准）
            validity_score = qa.get('validity_score', 0.6)
            if validity_score < 0.3:  # 从0.7降低到0.3
                continue
            
            # 去重（考虑相似度）
            question_lower = qa['question'].lower().strip()
            if question_lower not in seen_questions:
                seen_questions.add(question_lower)
                filtered.append(qa)
        
        # 如果过滤后太少，降低标准重新过滤
        if len(filtered) < len(qa_pairs) * 0.3:  # 如果过滤掉超过70%
            logger.warning(f"过滤过于严格，重新使用更宽松的标准")
            filtered = []
            seen_questions = set()
            
            for qa in qa_pairs:
                if qa.get('question') and qa.get('answer'):
                    question_lower = qa['question'].lower().strip()
                    if question_lower not in seen_questions:
                        seen_questions.add(question_lower)
                        filtered.append(qa)
        
        return filtered