"""
问题生成器
实现WebSailor的核心思想：基于子图中节点与关系,设计 QA 问题
覆盖多种问题类型
"""

import json
import logging
import random
from typing import List, Dict, Tuple, Optional
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    问题生成器
    WebSailor核心组件：基于子图生成多种类型的问题
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 问题类型
        self.question_types = config['data_settings'].get(
            'question_types',
            ['factual', 'reasoning', 'multi_hop', 'comparative', 'causal']
        )
        
        # 初始化模型
        self._initialize_model()
        
        # 加载问题模板
        self._load_question_templates()
        
    def _initialize_model(self):
        """初始化问题生成模型"""
        logger.info("初始化问题生成模型...")
        
        model_config = self.config['models']['qa_generator_model']
        model_path = model_config['path']
        
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
                    "{entity}的技术参数是什么？"
                ],
                'en': [
                    "What is the {attribute} of {entity}?",
                    "Please describe the main features of {entity}.",
                    "What {component} does {entity} contain?",
                    "What is {entity}? Please explain in detail.",
                    "What are the technical parameters of {entity}?"
                ]
            },
            'reasoning': {
                'zh': [
                    "如果{condition}，那么{entity}会如何变化？",
                    "为什么{entity1}需要{entity2}？",
                    "基于{evidence}，可以得出什么结论？",
                    "{entity}的工作原理是什么？",
                    "如何优化{entity}的{attribute}？"
                ],
                'en': [
                    "If {condition}, how would {entity} change?",
                    "Why does {entity1} need {entity2}?",
                    "Based on {evidence}, what can be concluded?",
                    "What is the working principle of {entity}?",
                    "How to optimize the {attribute} of {entity}?"
                ]
            },
            'multi_hop': {
                'zh': [
                    "{entity1}通过什么与{entity2}相关联？",
                    "从{start}到{end}的完整流程是什么？",
                    "{entity1}、{entity2}和{entity3}之间有什么关系？",
                    "请追踪{entity}的完整生产链路。",
                    "解释{process}中各个步骤的作用。"
                ],
                'en': [
                    "How is {entity1} related to {entity2}?",
                    "What is the complete process from {start} to {end}?",
                    "What is the relationship between {entity1}, {entity2}, and {entity3}?",
                    "Please trace the complete production chain of {entity}.",
                    "Explain the role of each step in {process}."
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
        
    def generate_questions(self, subgraphs: List[nx.DiGraph], 
                          questions_per_subgraph: int = 5) -> List[Dict]:
        """
        为子图生成问题
        
        Args:
            subgraphs: 子图列表
            questions_per_subgraph: 每个子图生成的问题数
            
        Returns:
            List[Dict]: 生成的问题列表
        """
        logger.info(f"开始为{len(subgraphs)}个子图生成问题...")
        
        all_questions = []
        
        for subgraph in tqdm(subgraphs, desc="生成问题"):
            # 分析子图特征
            graph_features = self._analyze_subgraph(subgraph)
            
            # 根据子图特征选择合适的问题类型
            suitable_types = self._select_question_types(graph_features)
            
            # 生成问题
            questions = []
            for _ in range(questions_per_subgraph):
                q_type = random.choice(suitable_types)
                question = self._generate_question(subgraph, q_type, graph_features)
                
                if question:
                    questions.append(question)
                    
            all_questions.extend(questions)
            
        logger.info(f"问题生成完成，共生成{len(all_questions)}个问题")
        return all_questions
        
    def _analyze_subgraph(self, subgraph: nx.DiGraph) -> Dict:
        """
        分析子图特征
        用于选择合适的问题类型
        """
        features = {
            'topology': subgraph.graph.get('topology', 'unknown'),
            'complexity': subgraph.graph.get('complexity', 0.5),
            'num_nodes': subgraph.number_of_nodes(),
            'num_edges': subgraph.number_of_edges(),
            'density': nx.density(subgraph),
            'has_cycle': len(list(nx.simple_cycles(subgraph))) > 0,
            'max_path_length': self._get_max_path_length(subgraph),
            'node_types': self._get_node_types(subgraph),
            'edge_types': self._get_edge_types(subgraph),
            'key_entities': self._identify_key_entities(subgraph)
        }
        
        return features
        
    def _get_max_path_length(self, subgraph: nx.DiGraph) -> int:
        """获取最长路径长度"""
        try:
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
            edge_type = data.get('type', 'unknown')
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        return type_counts
        
    def _identify_key_entities(self, subgraph: nx.DiGraph) -> List[str]:
        """
        识别关键实体
        基于度中心性和类型重要性
        """
        # 计算度中心性
        centrality = nx.degree_centrality(subgraph)
        
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
            node_type = subgraph.nodes[node].get('type', 'unknown')
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
        
    def _generate_question(self, subgraph: nx.DiGraph, 
                          question_type: str, 
                          features: Dict) -> Optional[Dict]:
        """
        生成单个问题
        """
        try:
            # 根据问题类型选择生成策略
            if question_type == 'factual':
                return self._generate_factual_question(subgraph, features)
            elif question_type == 'reasoning':
                return self._generate_reasoning_question(subgraph, features)
            elif question_type == 'multi_hop':
                return self._generate_multihop_question(subgraph, features)
            elif question_type == 'comparative':
                return self._generate_comparative_question(subgraph, features)
            elif question_type == 'causal':
                return self._generate_causal_question(subgraph, features)
            else:
                return self._generate_factual_question(subgraph, features)
                
        except Exception as e:
            logger.warning(f"问题生成失败: {e}")
            return None
            
    def _generate_factual_question(self, subgraph: nx.DiGraph, features: Dict) -> Dict:
        """生成事实类问题"""
        # 选择一个实体
        if features['key_entities']:
            entity = random.choice(features['key_entities'])
        else:
            entity = random.choice(list(subgraph.nodes()))
            
        # 获取实体信息
        entity_data = subgraph.nodes[entity]
        entity_type = entity_data.get('type', 'unknown')
        
        # 获取相关属性（从连接的边推断）
        attributes = []
        for _, target, data in subgraph.out_edges(entity, data=True):
            relation = data.get('type', '')
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
                attribute=attribute,
                component=entity_type
            )
        else:
            question_text = template.format(
                entity=entity,
                attribute="特性",
                component="组件"
            )
            
        # 生成答案
        answer = self._generate_answer(subgraph, entity, question_text, 'factual')
        
        return {
            'type': 'factual',
            'subgraph_id': id(subgraph),
            'question': question_text,
            'answer': answer,
            'entities': [entity],
            'language': lang,
            'difficulty': features['complexity']
        }
        
    def _generate_reasoning_question(self, subgraph: nx.DiGraph, features: Dict) -> Dict:
        """生成推理类问题"""
        # 选择两个相关实体
        if len(features['key_entities']) >= 2:
            entity1, entity2 = random.sample(features['key_entities'], 2)
        else:
            nodes = list(subgraph.nodes())
            if len(nodes) >= 2:
                entity1, entity2 = random.sample(nodes, 2)
            else:
                return self._generate_factual_question(subgraph, features)
                
        # 找到它们之间的关系
        try:
            path = nx.shortest_path(subgraph, entity1, entity2)
            relations = []
            for i in range(len(path) - 1):
                edge_data = subgraph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    relations.append(edge_data.get('type', '关联'))
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
            condition=f"{entity1}发生变化",
            evidence=f"{entity1}与{entity2}的关系"
        )
        
        # 生成答案
        answer = self._generate_answer(subgraph, [entity1, entity2], question_text, 'reasoning')
        
        return {
            'type': 'reasoning',
            'subgraph_id': id(subgraph),
            'question': question_text,
            'answer': answer,
            'entities': [entity1, entity2],
            'relations': relations,
            'language': lang,
            'difficulty': features['complexity'] * 1.2
        }
        
    def _generate_multihop_question(self, subgraph: nx.DiGraph, features: Dict) -> Dict:
        """生成多跳问题"""
        # 找到最长路径
        max_path = []
        max_length = 0
        
        nodes = list(subgraph.nodes())
        for i in range(min(10, len(nodes))):
            source = random.choice(nodes)
            for j in range(min(10, len(nodes))):
                target = random.choice(nodes)
                if source != target:
                    try:
                        path = nx.shortest_path(subgraph, source, target)
                        if len(path) > max_length and len(path) >= 3:
                            max_path = path
                            max_length = len(path)
                    except:
                        continue
                        
        if len(max_path) < 3:
            # 如果没有足够长的路径，降级到推理问题
            return self._generate_reasoning_question(subgraph, features)
            
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
            process=f"从{start}到{end}"
        )
        
        # 生成答案（包含完整路径）
        answer = self._generate_answer(subgraph, max_path, question_text, 'multi_hop')
        
        return {
            'type': 'multi_hop',
            'subgraph_id': id(subgraph),
            'question': question_text,
            'answer': answer,
            'entities': max_path,
            'path_length': len(max_path),
            'language': lang,
            'difficulty': features['complexity'] * 1.5
        }
        
    def _generate_comparative_question(self, subgraph: nx.DiGraph, features: Dict) -> Dict:
        """生成比较类问题"""
        # 找到相同类型的实体进行比较
        node_types = features['node_types']
        
        comparable_entities = []
        for node_type, count in node_types.items():
            if count >= 2:
                # 找到这种类型的所有节点
                nodes_of_type = [
                    n for n, d in subgraph.nodes(data=True) 
                    if d.get('type') == node_type
                ]
                if len(nodes_of_type) >= 2:
                    comparable_entities.extend(nodes_of_type)
                    
        if len(comparable_entities) < 2:
            # 如果没有可比较的实体，随机选择
            nodes = list(subgraph.nodes())
            if len(nodes) >= 2:
                entity1, entity2 = random.sample(nodes, 2)
            else:
                return self._generate_factual_question(subgraph, features)
        else:
            entity1, entity2 = random.sample(comparable_entities, 2)
            
        # 找到比较维度
        entity1_attrs = set(subgraph.successors(entity1))
        entity2_attrs = set(subgraph.successors(entity2))
        common_attrs = entity1_attrs & entity2_attrs
        diff_attrs = (entity1_attrs | entity2_attrs) - common_attrs
        
        # 选择语言和模板
        lang = random.choice(['zh', 'en'])
        template = random.choice(self.question_templates['comparative'][lang])
        
        # 生成问题
        attribute = random.choice(['性能', '特性', '应用']) if not common_attrs else random.choice(list(common_attrs))
        question_text = template.format(
            entity1=entity1,
            entity2=entity2,
            attribute=attribute,
            scenario="工业应用",
            aspect="技术指标"
        )
        
        # 生成答案
        answer = self._generate_answer(subgraph, [entity1, entity2], question_text, 'comparative')
        
        return {
            'type': 'comparative',
            'subgraph_id': id(subgraph),
            'question': question_text,
            'answer': answer,
            'entities': [entity1, entity2],
            'common_attributes': list(common_attrs),
            'diff_attributes': list(diff_attrs),
            'language': lang,
            'difficulty': features['complexity'] * 1.1
        }
        
    def _generate_causal_question(self, subgraph: nx.DiGraph, features: Dict) -> Dict:
        """生成因果类问题"""
        # 寻找因果关系
        causal_relations = ['导致', '影响', '产生', '引起', '基于']
        
        causal_edges = []
        for u, v, data in subgraph.edges(data=True):
            if data.get('type') in causal_relations:
                causal_edges.append((u, v, data.get('type')))
                
        if causal_edges:
            cause, effect, relation = random.choice(causal_edges)
        else:
            # 如果没有明确的因果关系，基于拓扑推断
            nodes = list(subgraph.nodes())
            if len(nodes) >= 2:
                # 选择入度小出度大的作为原因
                out_degrees = dict(subgraph.out_degree())
                in_degrees = dict(subgraph.in_degree())
                
                cause_candidates = [
                    n for n in nodes 
                    if out_degrees[n] > in_degrees[n]
                ]
                effect_candidates = [
                    n for n in nodes 
                    if in_degrees[n] > out_degrees[n]
                ]
                
                if cause_candidates and effect_candidates:
                    cause = random.choice(cause_candidates)
                    effect = random.choice(effect_candidates)
                else:
                    cause, effect = random.sample(nodes, 2)
            else:
                return self._generate_factual_question(subgraph, features)
                
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
            phenomenon=f"{effect}的变化",
            action=f"改变{cause}"
        )
        
        # 生成答案
        answer = self._generate_answer(subgraph, [cause, effect], question_text, 'causal')
        
        return {
            'type': 'causal',
            'subgraph_id': id(subgraph),
            'question': question_text,
            'answer': answer,
            'entities': [cause, effect],
            'causal_chain': self._find_causal_chain(subgraph, cause, effect),
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
                    'relation': edge_data.get('type', '影响') if edge_data else '影响'
                })
            return chain
        except:
            return [{'from': cause, 'to': effect, 'relation': '影响'}]
            
    def _generate_answer(self, subgraph: nx.DiGraph, 
                        entities: list, 
                        question: str, 
                        q_type: str) -> str:
        """
        使用LLM生成答案
        """
        # 构建上下文
        context = self._build_context(subgraph, entities)
        
        # 构建提示
        prompt = f"""基于以下知识图谱信息，回答问题。

知识图谱信息：
{context}

问题类型：{q_type}
问题：{question}

请提供准确、完整的答案："""

        # 生成答案
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
            
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的答案部分
        if "请提供准确、完整的答案：" in answer:
            answer = answer.split("请提供准确、完整的答案：")[-1].strip()
            
        return answer
        
    def _build_context(self, subgraph: nx.DiGraph, entities: list) -> str:
        """构建知识图谱上下文"""
        context_parts = []
        
        # 添加实体信息
        context_parts.append("实体信息：")
        for entity in entities:
            if entity in subgraph:
                node_data = subgraph.nodes[entity]
                context_parts.append(f"- {entity} (类型: {node_data.get('type', 'unknown')})")
                
        # 添加关系信息
        context_parts.append("\n关系信息：")
        
        # 获取相关的边
        relevant_edges = []
        for entity in entities:
            # 出边
            for _, target, data in subgraph.out_edges(entity, data=True):
                relevant_edges.append((entity, target, data))
            # 入边
            for source, _, data in subgraph.in_edges(entity, data=True):
                relevant_edges.append((source, entity, data))
                
        # 去重并格式化
        seen = set()
        for source, target, data in relevant_edges:
            edge_key = (source, target)
            if edge_key not in seen:
                seen.add(edge_key)
                relation = data.get('type', '相关')
                context_parts.append(f"- {source} --[{relation}]--> {target}")
                
        return "\n".join(context_parts)
        
    def save_questions(self, questions: List[Dict], output_path: str):
        """保存生成的问题"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
            
        logger.info(f"已保存{len(questions)}个问题到: {output_path}")