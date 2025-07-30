"""
问题生成器 - 修复版
支持NetworkX图和字典格式的子图
实现WebSailor的核心思想：基于子图中节点与关系,设计 QA 问题
"""

import json
import logging
import random
from typing import List, Dict, Tuple, Optional, Union, Any
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    问题生成器
    WebSailor核心组件：基于子图生成多种类型的问题
    支持NetworkX图和字典格式的输入
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 问题类型
        self.question_types = config.get('data_settings', {}).get(
            'question_types',
            ['factual', 'reasoning', 'multi_hop', 'comparative', 'causal']
        )
        
        # 初始化模型
        self._initialize_model()
        
        # 加载问题模板
        self._load_question_templates()
        
        # 合理性阈值
        self.validity_threshold = 0.7
        
    def _initialize_model(self):
        """初始化问题生成模型"""
        logger.info("初始化问题生成模型...")
        
        model_config = self.config['models']['qa_generator_model']
        model_path = model_config['path']
        
        try:
            logger.info(f"加载QA生成模型: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model.eval()
            
            # 设置生成参数
            self.generation_config = {
                'max_length': model_config.get('max_length', 4096),
                'temperature': model_config.get('temperature', 0.8),
                'top_p': model_config.get('top_p', 0.9),
                'do_sample': True,
                'pad_token_id': self.tokenizer.pad_token_id
            }
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
        
    def _load_question_templates(self):
        """加载问题模板"""
        # TCL工业垂域问题模板
        self.question_templates = {
            'factual': {
                'zh': [
                    "{entity}的{attribute}是什么？",
                    "请说明{entity}的主要特征。",
                    "{entity}包含哪些{component}？",
                    "什么是{entity}？请详细解释。",
                    "{entity}的技术参数是什么？",
                    "{entity}使用了什么{relation_type}？",
                    "哪个{entity_type}与{entity}有{relation}关系？",
                    "{entity1}和{entity2}之间的关系是什么？"
                ],
                'en': [
                    "What is the {attribute} of {entity}?",
                    "Please describe the main features of {entity}.",
                    "What {component} does {entity} contain?",
                    "What is {entity}? Please explain in detail.",
                    "What are the technical parameters of {entity}?",
                    "What {relation_type} does {entity} use?",
                    "Which {entity_type} has {relation} relationship with {entity}?",
                    "What is the relationship between {entity1} and {entity2}?"
                ]
            },
            'reasoning': {
                'zh': [
                    "如果{condition}，那么{entity}会如何变化？",
                    "为什么{entity1}需要{entity2}？",
                    "基于{evidence}，可以得出什么结论？",
                    "{entity}的工作原理是什么？",
                    "如何优化{entity}的{attribute}？",
                    "如果{condition}，那么{entity}会如何影响{target}？",
                    "基于{evidence}，可以推断出{entity}的什么特性？",
                    "为什么{entity}需要{requirement}？"
                ],
                'en': [
                    "If {condition}, how would {entity} change?",
                    "Why does {entity1} need {entity2}?",
                    "Based on {evidence}, what can be concluded?",
                    "What is the working principle of {entity}?",
                    "How to optimize the {attribute} of {entity}?",
                    "If {condition}, how would {entity} affect {target}?",
                    "Based on {evidence}, what characteristics of {entity} can be inferred?",
                    "Why does {entity} need {requirement}?"
                ]
            },
            'multi_hop': {
                'zh': [
                    "{entity1}通过什么与{entity2}相关联？",
                    "从{start}到{end}的完整流程是什么？",
                    "{entity1}、{entity2}和{entity3}之间有什么关系？",
                    "请追踪{entity}的完整生产链路。",
                    "解释{process}中各个步骤的作用。",
                    "{entity1}通过什么中间{entity_type}与{entity2}产生联系？",
                    "从{start}到{end}的{path_type}路径是什么？"
                ],
                'en': [
                    "How is {entity1} related to {entity2}?",
                    "What is the complete process from {start} to {end}?",
                    "What is the relationship between {entity1}, {entity2}, and {entity3}?",
                    "Please trace the complete production chain of {entity}.",
                    "Explain the role of each step in {process}.",
                    "Through which intermediate {entity_type} are {entity1} and {entity2} connected?",
                    "What is the {path_type} path from {start} to {end}?"
                ]
            },
            'comparative': {
                'zh': [
                    "{entity1}和{entity2}的主要区别是什么？",
                    "比较{entity1}和{entity2}的{attribute}。",
                    "哪个更适合{scenario}：{entity1}还是{entity2}？",
                    "{entity1}相比{entity2}有什么优势？",
                    "在{aspect}方面，{entity1}和{entity2}如何比较？"
                ],
                'en': [
                    "What are the main differences between {entity1} and {entity2}?",
                    "Compare the {attribute} of {entity1} and {entity2}.",
                    "Which is more suitable for {scenario}: {entity1} or {entity2}?",
                    "What advantages does {entity1} have over {entity2}?",
                    "How do {entity1} and {entity2} compare in terms of {aspect}?"
                ]
            },
            'causal': {
                'zh': [
                    "什么导致了{effect}？",
                    "{cause}会产生什么影响？",
                    "为什么{entity}会出现{phenomenon}？",
                    "{action}的结果是什么？",
                    "解释{entity1}如何影响{entity2}。"
                ],
                'en': [
                    "What causes {effect}?",
                    "What impact does {cause} have?",
                    "Why does {entity} exhibit {phenomenon}?",
                    "What is the result of {action}?",
                    "Explain how {entity1} affects {entity2}."
                ]
            }
        }
    
    def _convert_dict_to_networkx(self, subgraph_dict: Dict) -> nx.DiGraph:
        """将字典格式的子图转换为NetworkX图"""
        G = nx.DiGraph()
        
        # 添加节点
        for node in subgraph_dict.get('nodes', []):
            G.add_node(node['id'], **node)
        
        # 添加边
        for edge in subgraph_dict.get('edges', []):
            G.add_edge(edge['source'], edge['target'], **edge)
        
        # 添加图属性
        G.graph['topology'] = subgraph_dict.get('topology', 'unknown')
        G.graph['complexity'] = subgraph_dict.get('complexity', 0.5)
        
        return G
    
    def _convert_networkx_to_dict(self, G: nx.DiGraph) -> Dict:
        """将NetworkX图转换为字典格式"""
        nodes = []
        for node_id, node_data in G.nodes(data=True):
            node_dict = {'id': node_id}
            node_dict.update(node_data)
            nodes.append(node_dict)
        
        edges = []
        for source, target, edge_data in G.edges(data=True):
            edge_dict = {
                'source': source,
                'target': target,
                'relation': edge_data.get('type', edge_data.get('relation', '相关'))
            }
            edge_dict.update(edge_data)
            edges.append(edge_dict)
        
        # 统计节点类型
        node_types = {}
        for node in nodes:
            node_type = node.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # 统计关系类型
        relation_types = {}
        for edge in edges:
            rel_type = edge.get('relation', 'unknown')
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
        
        return {
            'nodes': nodes,
            'edges': edges,
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'node_types': node_types,
            'relation_types': relation_types,
            'topology': G.graph.get('topology', 'unknown'),
            'complexity': G.graph.get('complexity', 0.5)
        }
        
    def generate_questions(self, subgraphs: List[Any], 
                          questions_per_subgraph: int = 5) -> List[Dict]:
        """
        为子图生成问题
        支持NetworkX图和字典格式的输入
        
        Args:
            subgraphs: 子图列表（NetworkX图或字典格式）
            questions_per_subgraph: 每个子图生成的问题数
            
        Returns:
            List[Dict]: 生成的问题列表
        """
        logger.info(f"开始为{len(subgraphs)}个子图生成问题...")
        
        if not subgraphs:
            logger.warning("没有提供子图")
            return []
        
        # 检查并转换子图格式
        processed_subgraphs = []
        dict_subgraphs = []
        
        for i, subgraph in enumerate(subgraphs):
            try:
                if isinstance(subgraph, nx.DiGraph):
                    # NetworkX图转换为字典格式
                    dict_subgraph = self._convert_networkx_to_dict(subgraph)
                    processed_subgraphs.append(subgraph)
                    dict_subgraphs.append(dict_subgraph)
                elif isinstance(subgraph, dict):
                    # 字典格式转换为NetworkX图
                    nx_subgraph = self._convert_dict_to_networkx(subgraph)
                    processed_subgraphs.append(nx_subgraph)
                    dict_subgraphs.append(subgraph)
                else:
                    logger.warning(f"跳过不支持的子图类型: {type(subgraph)}")
                    continue
            except Exception as e:
                logger.error(f"处理第{i}个子图时出错: {e}")
                continue
        
        all_questions = []
        
        for nx_graph, dict_graph in tqdm(zip(processed_subgraphs, dict_subgraphs), 
                                        total=len(processed_subgraphs), desc="生成问题"):
            try:
                # 分析子图特征
                graph_features = self._analyze_subgraph(nx_graph, dict_graph)
                
                # 根据子图特征选择合适的问题类型
                suitable_types = self._select_question_types(graph_features)
                
                # 生成问题
                questions = []
                attempts = 0
                max_attempts = questions_per_subgraph * 3  # 允许更多尝试
                
                while len(questions) < questions_per_subgraph and attempts < max_attempts:
                    attempts += 1
                    q_type = random.choice(suitable_types)
                    question = self._generate_question(nx_graph, dict_graph, q_type, graph_features)
                    
                    if question:
                        questions.append(question)
                        
                all_questions.extend(questions)
                
            except Exception as e:
                logger.error(f"为子图生成问题时出错: {e}", exc_info=True)
                continue
            
        # 质量过滤
        filtered_questions = self._filter_questions(all_questions)
        
        logger.info(f"问题生成完成，共生成{len(filtered_questions)}个问题")
        return filtered_questions
        
    def _analyze_subgraph(self, nx_graph: nx.DiGraph, dict_graph: Dict) -> Dict:
        """
        分析子图特征
        结合NetworkX图和字典格式的优势
        """
        features = {
            'topology': nx_graph.graph.get('topology', dict_graph.get('topology', 'unknown')),
            'complexity': nx_graph.graph.get('complexity', dict_graph.get('complexity', 0.5)),
            'num_nodes': nx_graph.number_of_nodes(),
            'num_edges': nx_graph.number_of_edges(),
            'density': nx.density(nx_graph) if nx_graph.number_of_nodes() > 0 else 0,
            'has_cycle': len(list(nx.simple_cycles(nx_graph))) > 0 if nx_graph.number_of_nodes() > 0 else False,
            'max_path_length': self._get_max_path_length(nx_graph),
            'node_types': dict_graph.get('node_types', self._get_node_types(nx_graph)),
            'edge_types': dict_graph.get('relation_types', self._get_edge_types(nx_graph)),
            'key_entities': self._identify_key_entities(nx_graph, dict_graph)
        }
        
        return features
        
    def _get_max_path_length(self, subgraph: nx.DiGraph) -> int:
        """获取最长路径长度"""
        try:
            if subgraph.number_of_nodes() == 0:
                return 0
            # 转为无向图计算
            undirected = subgraph.to_undirected()
            if nx.is_connected(undirected):
                lengths = dict(nx.all_pairs_shortest_path_length(undirected))
                max_length = 0
                for source in lengths:
                    for target, length in lengths[source].items():
                        max_length = max(max_length, length)
                return max_length
        except:
            pass
        return 1
        
    def _get_node_types(self, subgraph: nx.DiGraph) -> Dict[str, int]:
        """统计节点类型"""
        type_counts = {}
        for node, data in subgraph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts
        
    def _get_edge_types(self, subgraph: nx.DiGraph) -> Dict[str, int]:
        """统计边类型"""
        type_counts = {}
        for u, v, data in subgraph.edges(data=True):
            edge_type = data.get('type', data.get('relation', 'unknown'))
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        return type_counts
        
    def _identify_key_entities(self, nx_graph: nx.DiGraph, dict_graph: Dict) -> List[str]:
        """
        识别关键实体
        基于度中心性和类型重要性
        """
        if nx_graph.number_of_nodes() == 0:
            return []
            
        # 计算度中心性
        centrality = nx.degree_centrality(nx_graph)
        
        # 根据节点类型调整重要性
        importance_weights = {
            '产品': 1.5,
            '技术': 1.3,
            '公司': 1.2,
            '材料': 1.1,
            '工艺': 1.1
        }
        
        # 计算综合重要性
        node_importance = {}
        for node, cent in centrality.items():
            node_type = nx_graph.nodes[node].get('type', 'unknown')
            weight = importance_weights.get(node_type, 1.0)
            node_importance[node] = cent * weight
            
        # 选择最重要的节点
        sorted_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)
        key_entities = [node for node, _ in sorted_nodes[:min(5, len(sorted_nodes))]]
        
        return key_entities
        
    def _select_question_types(self, features: Dict) -> List[str]:
        """
        根据子图特征选择合适的问题类型
        WebSailor思想：不同拓扑结构适合不同类型的问题
        """
        suitable_types = []
        
        # 基于拓扑类型
        topology = features['topology']
        if topology == 'chain':
            # 链式结构适合多跳推理
            suitable_types.extend(['multi_hop', 'reasoning', 'causal'])
        elif topology == 'star':
            # 星型结构适合比较和聚合
            suitable_types.extend(['comparative', 'factual'])
        elif topology == 'tree':
            # 树形结构适合层次推理
            suitable_types.extend(['reasoning', 'multi_hop'])
        elif topology == 'cycle':
            # 环形结构适合因果分析
            suitable_types.extend(['causal', 'reasoning'])
        else:
            # 默认所有类型
            suitable_types = list(self.question_types)
            
        # 基于复杂度调整
        if features['complexity'] > 0.7:
            # 高复杂度适合推理类问题
            if 'reasoning' not in suitable_types:
                suitable_types.append('reasoning')
            if 'multi_hop' not in suitable_types:
                suitable_types.append('multi_hop')
        elif features['complexity'] < 0.3:
            # 低复杂度适合事实类问题
            if 'factual' not in suitable_types:
                suitable_types.append('factual')
                
        # 确保至少有一种类型
        if not suitable_types:
            suitable_types = ['factual']
            
        return suitable_types
        
    def _generate_question(self, nx_graph: nx.DiGraph, dict_graph: Dict,
                          question_type: str, features: Dict) -> Optional[Dict]:
        """
        生成单个问题
        """
        try:
            # 根据问题类型选择生成策略
            if question_type == 'factual':
                return self._generate_factual_question(nx_graph, dict_graph, features)
            elif question_type == 'reasoning':
                return self._generate_reasoning_question(nx_graph, dict_graph, features)
            elif question_type == 'multi_hop':
                return self._generate_multihop_question(nx_graph, dict_graph, features)
            elif question_type == 'comparative':
                return self._generate_comparative_question(nx_graph, dict_graph, features)
            elif question_type == 'causal':
                return self._generate_causal_question(nx_graph, dict_graph, features)
            else:
                return self._generate_factual_question(nx_graph, dict_graph, features)
                
        except Exception as e:
            logger.error(f"问题生成失败 - 类型: {question_type}, 错误: {str(e)}", exc_info=True)
            return None
            
    def _generate_factual_question(self, nx_graph: nx.DiGraph, dict_graph: Dict, 
                                  features: Dict) -> Dict:
        """生成事实类问题"""
        # 选择一个实体
        if features['key_entities']:
            entity = random.choice(features['key_entities'])
        else:
            nodes = list(nx_graph.nodes())
            if not nodes:
                return None
            entity = random.choice(nodes)
            
        # 获取实体信息
        entity_data = nx_graph.nodes[entity]
        entity_type = entity_data.get('type', 'unknown')
        
        # 获取相关属性（从连接的边推断）
        attributes = []
        for _, target, data in nx_graph.out_edges(entity, data=True):
            relation = data.get('type', data.get('relation', ''))
            if relation in ['包含', '具有', '使用']:
                attributes.append(target)
                
        # 选择语言
        lang = random.choice(['zh', 'en'])
        
        # 选择模板
        template = random.choice(self.question_templates['factual'][lang])
        
        # 填充模板
        if attributes:
            attribute = random.choice(attributes)
            question_text = template.format(
                entity=entity,
                entity1=entity,
                entity2=attribute,
                attribute=attribute,
                component=entity_type,
                relation_type="技术" if lang == 'zh' else "technology",
                entity_type=entity_type,
                relation="使用" if lang == 'zh' else "uses"
            )
        else:
            question_text = template.format(
                entity=entity,
                entity1=entity,
                entity2=entity,
                attribute="特性" if lang == 'zh' else "characteristics",
                component="组件" if lang == 'zh' else "component",
                relation_type="技术" if lang == 'zh' else "technology",
                entity_type=entity_type,
                relation="相关" if lang == 'zh' else "related"
            )
            
        # 生成答案
        answer = self._generate_answer(dict_graph, entity, question_text, 'factual')
        
        return {
            'type': 'factual',
            'subgraph_id': id(nx_graph),
            'question': question_text,
            'answer': answer,
            'entities': [entity],
            'language': lang,
            'difficulty': features['complexity']
        }
        
    def _generate_reasoning_question(self, nx_graph: nx.DiGraph, dict_graph: Dict,
                                   features: Dict) -> Dict:
        """生成推理类问题"""
        # 选择两个相关实体
        if len(features['key_entities']) >= 2:
            entity1, entity2 = random.sample(features['key_entities'], 2)
        else:
            nodes = list(nx_graph.nodes())
            if len(nodes) >= 2:
                entity1, entity2 = random.sample(nodes, 2)
            else:
                return self._generate_factual_question(nx_graph, dict_graph, features)
                
        # 找到它们之间的关系
        try:
            path = nx.shortest_path(nx_graph, entity1, entity2)
            relations = []
            for i in range(len(path) - 1):
                edge_data = nx_graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    relations.append(edge_data.get('type', edge_data.get('relation', '关联')))
        except:
            relations = ['相关']
            
        # 选择语言和模板
        lang = random.choice(['zh', 'en'])
        template = random.choice(self.question_templates['reasoning'][lang])
        
        # 生成问题
        question_text = template.format(
            entity=entity1,
            entity1=entity1,
            entity2=entity2,
            condition=f"{entity1}发生变化" if lang == 'zh' else f"{entity1} changes",
            evidence=f"{entity1}与{entity2}的关系" if lang == 'zh' else f"relationship between {entity1} and {entity2}",
            attribute="性能" if lang == 'zh' else "performance",
            target=entity2,
            requirement="高质量标准" if lang == 'zh' else "high quality standards"
        )
        
        # 生成答案
        answer = self._generate_answer(dict_graph, [entity1, entity2], question_text, 'reasoning')
        
        return {
            'type': 'reasoning',
            'subgraph_id': id(nx_graph),
            'question': question_text,
            'answer': answer,
            'entities': [entity1, entity2],
            'relations': relations,
            'language': lang,
            'difficulty': features['complexity'] * 1.2
        }
        
    def _generate_multihop_question(self, nx_graph: nx.DiGraph, dict_graph: Dict,
                                   features: Dict) -> Dict:
        """生成多跳问题"""
        # 找到最长路径
        max_path = []
        max_length = 0
        
        nodes = list(nx_graph.nodes())
        if len(nodes) < 3:
            return self._generate_reasoning_question(nx_graph, dict_graph, features)
            
        for i in range(min(10, len(nodes))):
            source = random.choice(nodes)
            for j in range(min(10, len(nodes))):
                target = random.choice(nodes)
                if source != target:
                    try:
                        path = nx.shortest_path(nx_graph, source, target)
                        if len(path) > max_length and len(path) >= 3:
                            max_path = path
                            max_length = len(path)
                    except:
                        continue
                        
        if len(max_path) < 3:
            # 如果没有足够长的路径，降级到推理问题
            return self._generate_reasoning_question(nx_graph, dict_graph, features)
            
        # 选择路径中的关键节点
        start = max_path[0]
        end = max_path[-1]
        intermediate = max_path[len(max_path)//2]
        
        # 选择语言和模板
        lang = random.choice(['zh', 'en'])
        template = random.choice(self.question_templates['multi_hop'][lang])
        
        # 生成问题
        question_text = template.format(
            entity1=start,
            entity2=intermediate,
            entity3=end,
            start=start,
            end=end,
            entity=start,
            process=f"从{start}到{end}" if lang == 'zh' else f"from {start} to {end}",
            entity_type="组件" if lang == 'zh' else "component",
            path_type="技术演进" if lang == 'zh' else "technology evolution"
        )
        
        # 生成答案（包含完整路径）
        answer = self._generate_answer(dict_graph, max_path, question_text, 'multi_hop')
        
        return {
            'type': 'multi_hop',
            'subgraph_id': id(nx_graph),
            'question': question_text,
            'answer': answer,
            'entities': max_path,
            'path_length': len(max_path),
            'language': lang,
            'difficulty': features['complexity'] * 1.5
        }
        
    def _generate_comparative_question(self, nx_graph: nx.DiGraph, dict_graph: Dict,
                                      features: Dict) -> Dict:
        """生成比较类问题"""
        # 找到相同类型的实体进行比较
        node_types = features['node_types']
        
        comparable_entities = []
        for node_type, count in node_types.items():
            if count >= 2:
                # 找到这种类型的所有节点
                nodes_of_type = [
                    n for n, d in nx_graph.nodes(data=True) 
                    if d.get('type') == node_type
                ]
                if len(nodes_of_type) >= 2:
                    comparable_entities.extend(nodes_of_type)
                    
        if len(comparable_entities) < 2:
            # 如果没有可比较的实体，随机选择
            nodes = list(nx_graph.nodes())
            if len(nodes) >= 2:
                entity1, entity2 = random.sample(nodes, 2)
            else:
                return self._generate_factual_question(nx_graph, dict_graph, features)
        else:
            entity1, entity2 = random.sample(comparable_entities, 2)
            
        # 找到比较维度
        entity1_attrs = set(nx_graph.successors(entity1))
        entity2_attrs = set(nx_graph.successors(entity2))
        common_attrs = entity1_attrs & entity2_attrs
        diff_attrs = (entity1_attrs | entity2_attrs) - common_attrs
        
        # 选择语言和模板
        lang = random.choice(['zh', 'en'])
        template = random.choice(self.question_templates['comparative'][lang])
        
        # 生成问题
        attribute = random.choice(['性能', '特性', '应用'] if lang == 'zh' else ['performance', 'features', 'applications'])
        aspect = random.choice(['技术指标', '成本效益', '可靠性'] if lang == 'zh' else ['technical specs', 'cost-effectiveness', 'reliability'])
        
        question_text = template.format(
            entity1=entity1,
            entity2=entity2,
            attribute=attribute,
            scenario="工业应用" if lang == 'zh' else "industrial applications",
            aspect=aspect
        )
        
        # 生成答案
        answer = self._generate_answer(dict_graph, [entity1, entity2], question_text, 'comparative')
        
        return {
            'type': 'comparative',
            'subgraph_id': id(nx_graph),
            'question': question_text,
            'answer': answer,
            'entities': [entity1, entity2],
            'common_attributes': list(common_attrs),
            'diff_attributes': list(diff_attrs),
            'language': lang,
            'difficulty': features['complexity'] * 1.1
        }
        
    def _generate_causal_question(self, nx_graph: nx.DiGraph, dict_graph: Dict,
                                 features: Dict) -> Dict:
        """生成因果类问题"""
        # 寻找因果关系
        causal_relations = ['导致', '影响', '产生', '引起', '基于']
        
        causal_edges = []
        for u, v, data in nx_graph.edges(data=True):
            if data.get('type', data.get('relation', '')) in causal_relations:
                causal_edges.append((u, v, data.get('type', data.get('relation', '影响'))))
                
        if causal_edges:
            cause, effect, relation = random.choice(causal_edges)
        else:
            # 如果没有明确的因果关系，基于拓扑推断
            nodes = list(nx_graph.nodes())
            if len(nodes) >= 2:
                # 选择入度小出度大的作为原因
                out_degrees = dict(nx_graph.out_degree())
                in_degrees = dict(nx_graph.in_degree())
                
                cause_candidates = [
                    n for n in nodes 
                    if out_degrees.get(n, 0) > in_degrees.get(n, 0)
                ]
                effect_candidates = [
                    n for n in nodes 
                    if in_degrees.get(n, 0) > out_degrees.get(n, 0)
                ]
                
                if cause_candidates and effect_candidates:
                    cause = random.choice(cause_candidates)
                    effect = random.choice(effect_candidates)
                else:
                    cause, effect = random.sample(nodes, 2)
            else:
                return self._generate_factual_question(nx_graph, dict_graph, features)
                
        # 选择语言和模板
        lang = random.choice(['zh', 'en'])
        template = random.choice(self.question_templates['causal'][lang])
        
        # 生成问题
        question_text = template.format(
            cause=cause,
            effect=effect,
            entity=cause,
            entity1=cause,
            entity2=effect,
            phenomenon=f"{effect}的变化" if lang == 'zh' else f"changes in {effect}",
            action=f"改变{cause}" if lang == 'zh' else f"changing {cause}"
        )
        
        # 生成答案
        answer = self._generate_answer(dict_graph, [cause, effect], question_text, 'causal')
        
        return {
            'type': 'causal',
            'subgraph_id': id(nx_graph),
            'question': question_text,
            'answer': answer,
            'entities': [cause, effect],
            'causal_chain': self._find_causal_chain(nx_graph, cause, effect),
            'language': lang,
            'difficulty': features['complexity'] * 1.3
        }
        
    def _find_causal_chain(self, subgraph: nx.DiGraph, cause: str, effect: str) -> List[str]:
        """找到因果链"""
        try:
            path = nx.shortest_path(subgraph, cause, effect)
            chain = []
            for i in range(len(path) - 1):
                edge_data = subgraph.get_edge_data(path[i], path[i+1])
                chain.append({
                    'from': path[i],
                    'to': path[i+1],
                    'relation': edge_data.get('type', edge_data.get('relation', '影响')) if edge_data else '影响'
                })
            return chain
        except:
            return [{'from': cause, 'to': effect, 'relation': '影响'}]
            
    def _generate_answer(self, dict_graph: Dict, 
                        entities: Union[str, List[str]], 
                        question: str, 
                        q_type: str) -> str:
        """
        使用LLM生成答案
        """
        # 确保entities是列表
        if isinstance(entities, str):
            entities = [entities]
            
        # 构建上下文
        context = self._build_context(dict_graph, entities)
        
        # 构建提示
        prompt = f"""基于以下知识图谱信息，回答问题。

知识图谱信息：
{context}

问题类型：{q_type}
问题：{question}

请提供准确、完整的答案。答案应该：
1. 直接回答问题
2. 基于提供的知识图谱信息
3. 结构清晰，逻辑严谨
4. 使用专业术语

答案："""

        try:
            # 生成答案
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                  max_length=self.generation_config['max_length']).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=self.generation_config['temperature'],
                    top_p=self.generation_config['top_p'],
                    do_sample=self.generation_config['do_sample'],
                    pad_token_id=self.generation_config['pad_token_id']
                )
                
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的答案部分
            if "答案：" in answer:
                answer = answer.split("答案：")[-1].strip()
            else:
                # 移除prompt部分
                answer = answer[len(prompt):].strip()
                
            # 清理答案
            answer = answer.replace('\n\n', '\n').strip()
            
            # 如果答案为空或太短，生成默认答案
            if not answer or len(answer) < 10:
                answer = self._generate_default_answer(dict_graph, entities, q_type)
                
            return answer
            
        except Exception as e:
            logger.error(f"生成答案时出错: {e}")
            return self._generate_default_answer(dict_graph, entities, q_type)
        
    def _build_context(self, dict_graph: Dict, entities: List[str]) -> str:
        """构建知识图谱上下文"""
        context_parts = []
        
        # 添加实体信息
        context_parts.append("实体信息：")
        entity_set = set(entities)
        for node in dict_graph['nodes']:
            if node['id'] in entity_set:
                context_parts.append(f"- {node['id']} (类型: {node.get('type', 'unknown')})")
                
        # 添加关系信息
        context_parts.append("\n关系信息：")
        
        # 获取相关的边
        relevant_edges = []
        for edge in dict_graph['edges']:
            if edge['source'] in entity_set or edge['target'] in entity_set:
                relevant_edges.append(edge)
                
        # 格式化边信息
        for edge in relevant_edges:
            relation = edge.get('relation', edge.get('type', '相关'))
            context_parts.append(f"- {edge['source']} --[{relation}]--> {edge['target']}")
            
        return "\n".join(context_parts)
        
    def _generate_default_answer(self, dict_graph: Dict, entities: List[str], q_type: str) -> str:
        """生成默认答案"""
        if q_type == 'factual':
            return f"根据知识图谱信息，{entities[0]}是一个重要的实体，在系统中发挥着关键作用。"
        elif q_type == 'reasoning':
            return f"基于知识图谱中的关系分析，{entities[0]}和{entities[-1] if len(entities) > 1 else '其他组件'}之间存在密切联系，相互影响。"
        elif q_type == 'multi_hop':
            return f"从{entities[0]}到{entities[-1]}的路径显示了它们之间的间接关系，通过中间节点形成了完整的连接。"
        elif q_type == 'comparative':
            return f"{entities[0]}和{entities[1] if len(entities) > 1 else '其他实体'}各有特点，在不同场景下有各自的优势。"
        elif q_type == 'causal':
            return f"{entities[0]}的变化会对{entities[-1] if len(entities) > 1 else '系统'}产生影响，这种因果关系在实际应用中需要特别关注。"
        else:
            return "根据知识图谱信息，这个问题涉及的实体之间存在复杂的关系网络。"
            
    def _filter_questions(self, questions: List[Dict]) -> List[Dict]:
        """过滤低质量问题"""
        filtered = []
        seen_questions = set()
        
        for q in questions:
            # 检查问题长度
            if len(q['question']) < 10 or len(q['question']) > 500:
                continue
                
            # 检查答案长度
            if len(q['answer']) < 20:
                continue
                
            # 去重
            q_lower = q['question'].lower().strip()
            if q_lower in seen_questions:
                continue
            seen_questions.add(q_lower)
            
            filtered.append(q)
            
        return filtered
        
    def save_questions(self, questions: List[Dict], output_path: str):
        """保存生成的问题"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
            
        logger.info(f"已保存{len(questions)}个问题到: {output_path}")