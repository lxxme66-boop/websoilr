"""
问题生成器 - WebSailor核心模块（优化版）
基于子图中节点与关系，设计QA问题
覆盖多种问题类型：事实型、比较型、推理型、多跳型等
增加了问题合理性验证和优化机制
"""

import json
import logging
import random
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import networkx as nx

logger = logging.getLogger(__name__)

class QuestionGenerator:
    """
    WebSailor核心：问题生成器（优化版）
    基于子图生成多样化的问题，覆盖不同难度和类型
    包含问题合理性验证和优化机制
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.qg_config = config.get('question_generation', {})
        
        # 问题类型
        self.question_types = self.qg_config.get('question_types', [])
        self.complexity_levels = self.qg_config.get('complexity_levels', {})
        self.language_patterns = self.qg_config.get('language_patterns', {})
        
        # 加载QA生成模型
        self._load_qa_generator()
        
        # TCL特定配置
        self.tcl_config = config.get('tcl_specific', {})
        
        # 问题模板
        self._init_question_templates()
        
        # 合理性阈值
        self.validity_threshold = 0.7
        self.max_optimization_attempts = 2
        
    def _load_qa_generator(self):
        """加载QA生成模型"""
        model_config = self.config['models']['qa_generator_model']
        model_path = model_config['path']
        
        logger.info(f"加载QA生成模型: {model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            logger.warning(f"无法加载指定模型，使用默认模型: {e}")
            # 使用较小的默认模型
            self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                "THUDM/chatglm-6b",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.model.eval()
        
        # 设置生成参数
        self.max_length = model_config.get('max_length', 4096)
        self.temperature = model_config.get('temperature', 0.8)
        
    def _init_question_templates(self):
        """初始化问题模板"""
        self.question_templates = {
            'factual': {
                'zh_cn': [
                    "{entity}使用了什么{relation_type}？",
                    "{entity}的{attribute}是什么？",
                    "哪个{entity_type}与{entity}有{relation}关系？",
                    "{entity1}和{entity2}之间的关系是什么？"
                ],
                'en': [
                    "What {relation_type} does {entity} use?",
                    "What is the {attribute} of {entity}?",
                    "Which {entity_type} has {relation} relationship with {entity}?",
                    "What is the relationship between {entity1} and {entity2}?"
                ]
            },
            'comparison': {
                'zh_cn': [
                    "{entity1}和{entity2}在{aspect}方面有什么区别？",
                    "比较{entity1}和{entity2}的{attribute}。",
                    "{entity_list}中哪个{criterion}最{superlative}？",
                    "从{aspect}角度，{entity1}相比{entity2}有什么优势？"
                ],
                'en': [
                    "What are the differences between {entity1} and {entity2} in terms of {aspect}?",
                    "Compare the {attribute} of {entity1} and {entity2}.",
                    "Which of {entity_list} is the {superlative} in terms of {criterion}?",
                    "What advantages does {entity1} have over {entity2} from {aspect} perspective?"
                ]
            },
            'reasoning': {
                'zh_cn': [
                    "如果{condition}，那么{entity}会如何影响{target}？",
                    "基于{evidence}，可以推断出{entity}的什么特性？",
                    "为什么{entity}需要{requirement}？",
                    "{entity}的{attribute}如何影响其{performance}？"
                ],
                'en': [
                    "If {condition}, how would {entity} affect {target}?",
                    "Based on {evidence}, what characteristics of {entity} can be inferred?",
                    "Why does {entity} need {requirement}?",
                    "How does the {attribute} of {entity} affect its {performance}?"
                ]
            },
            'multi_hop': {
                'zh_cn': [
                    "{entity1}通过什么中间{entity_type}与{entity2}产生联系？",
                    "从{start}到{end}的{path_type}路径是什么？",
                    "{entity}的{relation1}的{relation2}是什么？",
                    "哪些{entity_type}同时与{entity1}和{entity2}有{relation}关系？"
                ],
                'en': [
                    "Through which intermediate {entity_type} are {entity1} and {entity2} connected?",
                    "What is the {path_type} path from {start} to {end}?",
                    "What is the {relation2} of {entity}'s {relation1}?",
                    "Which {entity_type} have {relation} relationships with both {entity1} and {entity2}?"
                ]
            }
        }
        
    def generate_questions(self, subgraphs: List[nx.DiGraph]) -> List[Dict]:
        """为子图列表生成问题"""
        all_qa_pairs = []
        
        for subgraph in tqdm(subgraphs, desc="生成问题"):
            # 分析子图特征
            subgraph_features = self._analyze_subgraph(subgraph)
            
            # 根据子图特征选择合适的问题类型
            suitable_types = self._select_question_types(subgraph_features)
            
            # 为每种问题类型生成问题
            for q_type in suitable_types:
                qa_pairs = self._generate_questions_for_type(
                    subgraph, q_type, subgraph_features
                )
                all_qa_pairs.extend(qa_pairs)
        
        # 后处理和质量检查
        filtered_qa_pairs = self._filter_qa_pairs(all_qa_pairs)
        
        logger.info(f"共生成 {len(filtered_qa_pairs)} 个高质量QA对")
        
        return filtered_qa_pairs
    
    def _analyze_subgraph(self, subgraph: nx.DiGraph) -> Dict:
        """分析子图特征"""
        num_nodes = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()
        
        # 获取节点类型
        node_types = set()
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            if 'type' in node_data:
                node_types.add(node_data['type'])
        
        # 获取关系类型
        relation_types = set()
        for u, v, data in subgraph.edges(data=True):
            if 'relation' in data:
                relation_types.add(data['relation'])
        
        features = {
            'topology': subgraph.graph.get('topology', 'unknown'),
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'node_types': list(node_types),
            'relation_types': list(relation_types),
            'has_path': 'path' in subgraph.graph,
            'has_center': 'center' in subgraph.graph,
            'has_cycle': len(list(nx.simple_cycles(subgraph))) > 0 if subgraph.is_directed() else False,
            'density': nx.density(subgraph)
        }
        
        # 识别关键实体
        features['key_entities'] = self._identify_key_entities(subgraph)
        
        # 识别路径模式
        features['path_patterns'] = self._identify_path_patterns(subgraph)
        
        return features
    
    def _identify_key_entities(self, subgraph: nx.DiGraph) -> List[Dict]:
        """识别子图中的关键实体"""
        key_entities = []
        
        # 基于度数识别
        node_degrees = dict(subgraph.degree())
        
        # 选择度数最高的节点
        if node_degrees:
            sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
            for node_id, degree in sorted_nodes[:3]:  # 前3个
                node_data = subgraph.nodes[node_id]
                key_entities.append({
                    'id': node_id,
                    'type': node_data.get('type', 'unknown'),
                    'degree': degree,
                    'role': 'hub' if degree > 3 else 'connector'
                })
        
        # 如果有特殊标记的节点
        if 'center' in subgraph.graph:
            center_id = subgraph.graph['center']
            if center_id in subgraph.nodes:
                node_data = subgraph.nodes[center_id]
                key_entities.append({
                    'id': center_id,
                    'type': node_data.get('type', 'unknown'),
                    'role': 'center'
                })
        
        return key_entities
    
    def _identify_path_patterns(self, subgraph: nx.DiGraph) -> List[Dict]:
        """识别子图中的路径模式"""
        patterns = []
        
        # 链式路径
        if 'path' in subgraph.graph:
            path_nodes = subgraph.graph['path']
            patterns.append({
                'type': 'chain',
                'length': len(path_nodes),
                'nodes': path_nodes
            })
        
        # 多跳路径（通过边连接）
        if subgraph.number_of_edges() >= 2:
            # 简化版：找到一些2跳路径
            edges = list(subgraph.edges(data=True))
            for i, (u1, v1, data1) in enumerate(edges):
                for u2, v2, data2 in edges[i+1:]:
                    if v1 == u2:
                        patterns.append({
                            'type': 'two_hop',
                            'path': [u1, v1, v2],
                            'relations': [data1.get('relation', ''), data2.get('relation', '')]
                        })
        
        return patterns
    
    def _select_question_types(self, features: Dict) -> List[str]:
        """根据子图特征选择合适的问题类型"""
        suitable_types = []
        
        # 基于拓扑选择
        topology = features['topology']
        
        if topology == 'chain' or features['has_path']:
            suitable_types.extend(['multi_hop', 'temporal', 'causal'])
        
        if topology == 'star' or features.get('has_center'):
            suitable_types.extend(['comparison', 'factual'])
        
        if topology == 'tree':
            suitable_types.extend(['reasoning', 'multi_hop'])
        
        if topology == 'cycle' or features.get('has_cycle'):
            suitable_types.extend(['causal', 'reasoning'])
        
        # 基于节点数量
        if features['num_nodes'] >= 5:
            suitable_types.append('comparison')
        
        # 基于密度
        if features['density'] > 0.3:
            suitable_types.append('reasoning')
        
        # 添加基础类型
        suitable_types.extend(['factual'])
        
        # 根据复杂度分布随机选择
        complexity_weights = self.complexity_levels
        if features['num_nodes'] > 8:
            # 复杂子图，增加复杂问题的权重
            suitable_types.extend(['counterfactual', 'multi_hop'])
        
        # 去重
        suitable_types = list(set(suitable_types))
        
        # 限制数量
        max_types = min(3, len(suitable_types))
        return random.sample(suitable_types, max_types)
    
    def _generate_questions_for_type(self, subgraph: nx.DiGraph, 
                                   q_type: str, features: Dict) -> List[Dict]:
        """为特定类型生成问题"""
        qa_pairs = []
        
        # 选择语言
        lang_weights = list(self.language_patterns.values())
        languages = list(self.language_patterns.keys())
        selected_lang = random.choices(languages, weights=lang_weights)[0]
        
        # 根据问题类型生成
        if q_type == 'factual':
            qa_pairs.extend(self._generate_factual_questions(
                subgraph, features, selected_lang
            ))
        elif q_type == 'comparison':
            qa_pairs.extend(self._generate_comparison_questions(
                subgraph, features, selected_lang
            ))
        elif q_type == 'reasoning':
            qa_pairs.extend(self._generate_reasoning_questions(
                subgraph, features, selected_lang
            ))
        elif q_type == 'multi_hop':
            qa_pairs.extend(self._generate_multihop_questions(
                subgraph, features, selected_lang
            ))
        
        return qa_pairs
    
    def _generate_factual_questions(self, subgraph: nx.DiGraph, 
                                  features: Dict, lang: str) -> List[Dict]:
        """生成事实型问题"""
        qa_pairs = []
        templates = self.question_templates['factual'][lang]
        
        # 基于边生成问题
        edges = list(subgraph.edges(data=True))
        for u, v, edge_data in edges[:5]:  # 限制数量
            # 获取源节点和目标节点数据
            source_node = {'id': u, **subgraph.nodes[u]}
            target_node = {'id': v, **subgraph.nodes[v]}
            edge = {
                'source': u,
                'target': v,
                'relation': edge_data.get('relation', '')
            }
            
            # 智能选择模板（基于实体和关系类型）
            template = self._select_best_template(templates, source_node, target_node, edge, lang)
            
            # 填充模板
            question = template.format(
                entity=edge['source'],
                entity1=edge['source'],
                entity2=edge['target'],
                relation=edge['relation'],
                relation_type=self._get_relation_type_name(edge['relation'], lang),
                entity_type=self._get_entity_type_name(target_node.get('type', 'unknown'), lang),
                attribute=self._get_attribute_name(source_node.get('type', 'unknown'), lang)
            )
            
            # 验证问题合理性
            is_valid, validity_score, suggestion = self._validate_question(question, subgraph, 'factual')
            
            if not is_valid:
                # 尝试优化问题
                optimized_question = self._optimize_question(question, subgraph, suggestion)
                if optimized_question:
                    question = optimized_question
                else:
                    continue  # 跳过不合理的问题
            
            # 生成答案
            answer = self._generate_answer_with_llm(question, subgraph, 'factual')
            
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'type': 'factual',
                'language': lang,
                'subgraph': subgraph,
                'validity_score': validity_score,
                'evidence': {
                    'nodes': [source_node, target_node],
                    'edges': [edge]
                }
            })
        
        return qa_pairs
    
    def _generate_comparison_questions(self, subgraph: nx.DiGraph, 
                                     features: Dict, lang: str) -> List[Dict]:
        """生成比较型问题"""
        qa_pairs = []
        templates = self.question_templates['comparison'][lang]
        
        # 找到相同类型的实体进行比较
        entity_groups = {}
        for node_id in subgraph.nodes():
            node_data = subgraph.nodes[node_id]
            entity_type = node_data.get('type', 'unknown')
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append({'id': node_id, **node_data})
        
        # 为每组生成比较问题
        for entity_type, entities in entity_groups.items():
            if len(entities) >= 2:
                # 随机选择两个实体
                entity1, entity2 = random.sample(entities, 2)
                
                # 验证这两个实体是否适合比较
                if not self._can_compare_entities(entity1, entity2, subgraph):
                    continue
                
                template = random.choice(templates)
                
                # 选择比较维度
                aspect = self._select_comparison_aspect(entity_type, lang)
                
                question = template.format(
                    entity1=entity1['id'],
                    entity2=entity2['id'],
                    aspect=aspect,
                    attribute=self._get_attribute_name(entity_type, lang),
                    entity_list=', '.join([e['id'] for e in entities]),
                    criterion=aspect,
                    superlative=self._get_superlative(lang)
                )
                
                # 验证问题合理性
                is_valid, validity_score, suggestion = self._validate_question(question, subgraph, 'comparison')
                
                if not is_valid:
                    optimized_question = self._optimize_question(question, subgraph, suggestion)
                    if optimized_question:
                        question = optimized_question
                    else:
                        continue
                
                # 使用LLM生成答案
                answer = self._generate_answer_with_llm(question, subgraph, 'comparison')
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'type': 'comparison',
                    'language': lang,
                    'subgraph': subgraph,
                    'validity_score': validity_score,
                    'evidence': {
                        'nodes': entities,
                        'comparison_aspect': aspect
                    }
                })
        
        return qa_pairs
    
    def _generate_multihop_questions(self, subgraph: nx.DiGraph, 
                                   features: Dict, lang: str) -> List[Dict]:
        """生成多跳问题"""
        qa_pairs = []
        templates = self.question_templates['multi_hop'][lang]
        
        # 使用已识别的路径模式
        for pattern in features['path_patterns']:
            if pattern['type'] == 'two_hop' and len(pattern['path']) >= 3:
                start = pattern['path'][0]
                middle = pattern['path'][1]
                end = pattern['path'][2]
                
                # 获取节点信息
                start_node = {'id': start, **subgraph.nodes[start]}
                middle_node = {'id': middle, **subgraph.nodes[middle]}
                end_node = {'id': end, **subgraph.nodes[end]}
                
                # 验证路径的语义合理性
                if not self._validate_path_semantics(start_node, middle_node, end_node, pattern['relations']):
                    continue
                
                template = random.choice(templates)
                
                relations = pattern['relations']
                # 选最长字符串作为代表relation
                relation = max(relations, key=len) if relations else ''
                
                question = template.format(
                    entity1=start,
                    entity2=end,
                    entity_type=self._get_entity_type_name(middle_node.get('type', 'unknown'), lang),
                    start=start,
                    end=end,
                    path_type=self._get_path_type_name(lang),
                    entity=start,
                    relation1=relations[0] if len(relations) > 0 else '',
                    relation2=relations[1] if len(relations) > 1 else '',
                    relation=relation
                )
                
                # 验证问题合理性
                is_valid, validity_score, suggestion = self._validate_question(question, subgraph, 'multi_hop')
                
                if not is_valid:
                    optimized_question = self._optimize_question(question, subgraph, suggestion)
                    if optimized_question:
                        question = optimized_question
                    else:
                        continue
                
                # 生成答案
                answer = self._generate_answer_with_llm(question, subgraph, 'multi_hop')
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'type': 'multi_hop',
                    'language': lang,
                    'subgraph': subgraph,
                    'validity_score': validity_score,
                    'evidence': {
                        'path': pattern['path'],
                        'relations': pattern['relations']
                    }
                })
        
        return qa_pairs
    
    def _generate_reasoning_questions(self, subgraph: nx.DiGraph, 
                                    features: Dict, lang: str) -> List[Dict]:
        """生成推理型问题"""
        qa_pairs = []
        templates = self.question_templates['reasoning'][lang]
        
        # 选择关键实体
        if features['key_entities']:
            key_entity = random.choice(features['key_entities'])
            
            # 找到相关的边和节点作为证据
            related_edges = []
            for u, v, data in subgraph.edges(data=True):
                if u == key_entity['id'] or v == key_entity['id']:
                    related_edges.append({
                        'source': u,
                        'target': v,
                        'relation': data.get('relation', '')
                    })
            
            if related_edges:
                # 验证是否有足够的信息进行推理
                if not self._has_sufficient_reasoning_context(key_entity, related_edges, subgraph):
                    return qa_pairs
                
                template = random.choice(templates)
                
                # 构造条件和推理目标
                condition = self._create_condition(related_edges, lang)
                target = self._select_reasoning_target(subgraph, key_entity, lang)
                
                question = template.format(
                    entity=key_entity['id'],
                    condition=condition,
                    target=target,
                    evidence=self._summarize_evidence(related_edges, lang),
                    requirement=self._select_requirement(key_entity['type'], lang),
                    attribute=self._get_attribute_name(key_entity['type'], lang),
                    performance=self._get_performance_metric(key_entity['type'], lang)
                )
                
                # 验证问题合理性
                is_valid, validity_score, suggestion = self._validate_question(question, subgraph, 'reasoning')
                
                if not is_valid:
                    optimized_question = self._optimize_question(question, subgraph, suggestion)
                    if optimized_question:
                        question = optimized_question
                    else:
                        return qa_pairs
                
                answer = self._generate_answer_with_llm(question, subgraph, 'reasoning')
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'type': 'reasoning',
                    'language': lang,
                    'subgraph': subgraph,
                    'validity_score': validity_score,
                    'evidence': {
                        'key_entity': key_entity,
                        'related_edges': related_edges
                    }
                })
        
        return qa_pairs
    
    def _validate_question(self, question: str, subgraph: nx.DiGraph, q_type: str) -> Tuple[bool, float, str]:
        """
        验证问题的合理性
        返回：(是否合理, 合理性分数, 改进建议)
        """
        validation_prompt = f"""请评估以下问题的合理性：

请评估以下问题的合理性：

问题：{question}
问题类型：{q_type}

基于的知识图谱信息：
{self._format_subgraph_for_prompt(subgraph)}

请从以下几个方面进行评分（总分为1分）：

1. 语义合理性（0-0.3分）：问题在现实应用中是否有意义，涉及的实体和关系是否合理关联，避免简单拼凑无关实体。
2. 信息完整性（0-0.2分）：问题是否包含足够的上下文信息，使问题清晰明确。
3. 答案可得性（0-0.3分）：基于给定的知识图谱信息，是否可以找到问题的合理答案。
4. 语言流畅性（0-0.2分）：问题的表述是否自然、通顺，没有语法或表达上的障碍。

请给出以下输出：

1. 总分（0-1之间的小数）
2. 是否合理（合理 / 不合理）
3. 改进建议（若不合理，请具体说明如何改进）

格式示例：
总分：0.8
是否合理：不合理
改进建议：问题中的实体关系较为松散，缺乏实际联系。建议明确实体间的因果或逻辑关系，确保问题有实际意义且可根据知识图谱信息回答。

---

示例1（合理问题）：
问题：在某新型电子纸显示屏出现不规则的色斑，且在不同温度下表现各异，初步判断可能是与ZnO TFT的柔性有关，但具体原因不明。请分析可能的原因并提出解决方案。该电子纸显示屏在不同温度下的色斑现象，疑似与ZnO TFT的柔性特性及全尺寸应用有关。请详细分析ZnO TFT的柔性是如何影响设备性能，并提出改进方案。
原因：该问题语义合理，实体关系紧密，符合现实工程背景。

示例2（不合理问题）：
问题：在某型号TCL电视机中，用户反映屏幕出现时有时无的闪烁现象，并伴有音频输出不稳定。经初步检测，电源模块工作正常，但当温度升高时，GaAs半导体材料的AlₓGa₁₋ₓAs合金层的压力单位kbar值异常波动，导致电子迁移率降低，进而影响了图像信号处理电路的工作稳定性。请分析可能的原因并提出解决方案？解决方案需考虑温度对合金层压力的影响及如何优化材料稳定性。
原因：实体间关联较弱，问题更像是拼凑多种专业名词，缺乏针对性和实际工程背景支持。
"""

        # 使用LLM进行验证
        inputs = self.tokenizer(validation_prompt, return_tensors="pt", truncation=True, 
                              max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,  # 降低温度以获得更稳定的评估
                do_sample=True,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 解析响应
        validity_score = 0.8  # 默认分数
        is_valid = True
        suggestion = ""
        
        try:
            # 提取分数
            if "总分：" in response:
                score_str = response.split("总分：")[1].split("\n")[0].strip()
                validity_score = float(score_str)
            
            # 提取是否合理
            if "是否合理：" in response:
                validity_str = response.split("是否合理：")[1].split("\n")[0].strip()
                is_valid = "合理" in validity_str and "不合理" not in validity_str
            
            # 提取改进建议
            if "改进建议：" in response:
                suggestion = response.split("改进建议：")[1].strip()
        except:
            # 解析失败，使用默认值
            pass
        
        # 基于分数阈值判断
        if validity_score < self.validity_threshold:
            is_valid = False
            if not suggestion:
                suggestion = "问题可能不够合理，建议重新构造或调整实体关系的组合"
        
        return is_valid, validity_score, suggestion
    
    def _optimize_question(self, question: str, subgraph: nx.DiGraph, suggestion: str) -> Optional[str]:
        """
        基于建议优化问题
        """
        optimization_prompt = f"""请根据以下信息优化问题：

原始问题：{question}
改进建议：{suggestion}

基于的知识图谱信息：
{self._format_subgraph_for_prompt(subgraph)}

请生成一个更合理、更自然的问题。保持问题的核心意图不变，但要：
1. 确保实体和关系的组合在语义上合理
2. 使问题表述更加自然流畅
3. 确保基于给定的知识图谱可以回答
4. 避免生硬的模板痕迹
5.问题和心不变
6.符合专业术语和标准流程

优化后的问题："""

        # 使用LLM优化
        inputs = self.tokenizer(optimization_prompt, return_tensors="pt", truncation=True, 
                              max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取优化后的问题
        if "优化后的问题：" in response:
            optimized = response.split("优化后的问题：")[1].strip()
            # 清理可能的多余内容
            optimized = optimized.split("\n")[0].strip()
            
            # 再次验证优化后的问题
            is_valid, score, _ = self._validate_question(optimized, subgraph, 'optimized')
            if is_valid and score > self.validity_threshold:
                return optimized
        
        return None
    
    def _select_best_template(self, templates: List[str], source_node: Dict, 
                            target_node: Dict, edge: Dict, lang: str) -> str:
        """基于实体和关系类型智能选择最合适的模板"""
        # 简化实现：基于实体类型和关系类型的组合选择
        # 实际可以使用更复杂的规则或学习的方法
        
        # 如果是技术类实体使用技术相关的关系，选择特定模板
        if source_node.get('type', '') in ['技术', '工艺'] and edge['relation'] in ['使用', '应用于']:
            # 优先选择包含relation_type的模板
            for template in templates:
                if 'relation_type' in template:
                    return template
        
        # 如果是比较类的场景，选择包含entity1和entity2的模板
        if 'entity1' in templates[0] and 'entity2' in templates[0]:
            for template in templates:
                if 'entity1' in template and 'entity2' in template:
                    return template
        
        # 默认随机选择
        return random.choice(templates)
    
    def _can_compare_entities(self, entity1: Dict, entity2: Dict, subgraph: nx.DiGraph) -> bool:
        """判断两个实体是否适合进行比较"""
        # 检查是否有共同的关系或属性
        entity1_relations = set()
        entity2_relations = set()
        
        for u, v, data in subgraph.edges(data=True):
            if u == entity1['id']:
                entity1_relations.add(data.get('relation', ''))
            if u == entity2['id']:
                entity2_relations.add(data.get('relation', ''))
        
        # 如果有共同的关系类型，则适合比较
        common_relations = entity1_relations.intersection(entity2_relations)
        return len(common_relations) > 0
    
    def _validate_path_semantics(self, start_node: Dict, middle_node: Dict, 
                               end_node: Dict, relations: List[str]) -> bool:
        """验证路径的语义合理性"""
        # 简化实现：检查关系类型是否能形成合理的传递
        # 例如：A使用B，B包含C => A通过B间接使用C（合理）
        
        # 定义一些合理的关系传递模式
        valid_patterns = [
            (['使用', '包含'], True),
            (['生产', '使用'], True),
            (['研发', '应用于'], True),
            (['依赖', '依赖'], True),  # 传递依赖
            (['改进', '替代'], False),  # 改进和替代不能传递
        ]
        
        if len(relations) >= 2:
            for pattern, is_valid in valid_patterns:
                if relations[0] in pattern and relations[1] in pattern:
                    return is_valid
        
        # 默认认为合理
        return True
    
    def _has_sufficient_reasoning_context(self, entity: Dict, edges: List[Dict], 
                                        subgraph: nx.DiGraph) -> bool:
        """判断是否有足够的上下文进行推理"""
        # 至少需要2条相关边才能进行有意义的推理
        return len(edges) >= 2
    
    def _generate_answer_with_llm(self, question: str, subgraph: nx.DiGraph, 
                                 q_type: str) -> str:
        """使用LLM生成答案"""
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
    
    def _create_answer_prompt(self, question: str, subgraph: nx.DiGraph, 
                            q_type: str) -> str:
        """创建答案生成提示"""
        # 将子图信息格式化
        subgraph_desc = self._format_subgraph_for_prompt(subgraph)
        
        prompt = f"""基于以下知识图谱信息，回答问题。

知识图谱信息：
{subgraph_desc}

问题类型：
{q_type}

问题：
{question}

请完成以下任务：

1. 判断问题合理性：
   - 该问题是否合理且能基于给定知识图谱回答？
   - 如果不合理或无法回答，请说明原因，并尝试推测问题的真实意图。

2. 回答问题：
   - 如果合理，基于知识图谱信息和你的专业知识，给出准确且详细的答案。
   - 如需推理，请简要说明推理过程。

3. 答案要求：
   - 重点突出，避免冗余信息。
   - 结构清晰，条理分明。
   - 使用专业术语和标准表达。

示例参考：

【正确示例】
问题：在使用SMT工艺的EPD设备中，用户反映屏幕刷新率异常低下，尽管Epson技术已经优化了substrate resin的测试条件。请分析可能的原因并提供解决策略。
答案：在进行故障分析时，首先需要确认TFT性能是否正常，包括开关特性和响应时间。其次，检查substrate resin是否完全固化且厚度均匀。然后，核实SMT工艺参数，如焊膏粘度和印刷压力。最后，验证屏幕刷新率的计算方法是否准确。具体步骤：1) 测试TFT性能；2) 检查substrate resin固化情况；3) 核实SMT工艺参数；4) 验证刷新率计算方法。根据检查结果调整相关参数，最终提升屏幕刷新率，改善用户体验。

【错误示例】
问题：某款采用a-Si TFT和EPD的电子纸显示器出现不稳定的屏幕刷新率，且部分区域显示异常，可能与solvent-free材料和ZnO TFT的应用有关，请分析潜在原因并提出解决方案。
答案：该问题可能源于solvent-free材料和ZnO TFT的兼容性问题。具体步骤包括检查材料兼容性、测试ZnO TFT电性能、分析信号传输稳定性及驱动电路异常。解决方案涵盖材料兼容性优化、工艺改进及电路调整。虽然内容相关，但表述较笼统，缺乏具体推理与细节说明，回答不够精确。

分析：
【正确示例】  
问题表述具体，答案结构清晰，包含详细推理过程和具体操作步骤，逻辑严谨，专业术语使用恰当，便于理解和执行。
【错误示例】  
答案过于笼统，缺少详细推理和因果分析，表述简单，缺乏系统性和专业诊断步骤，导致答案不够精准和权威。


仔细阅读分析后请开始回答：

答案：


"""

        return prompt
    
    def _format_subgraph_for_prompt(self, subgraph: nx.DiGraph) -> str:
        """格式化子图信息用于提示"""
        lines = []
        
        # 节点信息
        lines.append("节点：")
        for node_id in subgraph.nodes():
            node_data = subgraph.nodes[node_id]
            node_type = node_data.get('type', 'unknown')
            lines.append(f"  - {node_id} (类型: {node_type})")
        
        # 边信息
        lines.append("\n关系：")
        for u, v, data in subgraph.edges(data=True):
            relation = data.get('relation', 'unknown')
            lines.append(f"  - {u} --[{relation}]--> {v}")
        
        # 特殊信息
        if 'topology' in subgraph.graph:
            lines.append(f"\n拓扑类型：{subgraph.graph['topology']}")
        
        return '\n'.join(lines)
    
    def _extract_answer_from_output(self, output: str, prompt: str) -> str:
        """从模型输出中提取答案"""
        # 移除提示部分
        if prompt in output:
            output = output.replace(prompt, '').strip()
        
        # 提取答案部分
        if '答案：' in output:
            answer = output.split('答案：')[-1].strip()
        else:
            answer = output.strip()
        
        # 清理答案
        answer = answer.replace('\n\n', '\n')
        
        # 限制长度
        max_answer_length = 500
        if len(answer) > max_answer_length:
            answer = answer[:max_answer_length] + "..."
        
        return answer
    
    def _filter_qa_pairs(self, qa_pairs: List[Dict]) -> List[Dict]:
        """过滤和质量检查QA对"""
        filtered = []
        
        quality_config = self.config['dataset_synthesis']['quality_checks']
        min_q_len = quality_config['min_question_length']
        max_q_len = quality_config['max_question_length']
        
        # 用于去重的集合
        seen_questions = set()
        
        for qa in qa_pairs:
            # 长度检查
            q_len = len(qa['question'])
            if not (min_q_len <= q_len <= max_q_len):
                continue
            
            # 答案验证
            if quality_config['answer_validation']:
                if not qa['answer'] or len(qa['answer']) < 10:
                    continue
            
            # 合理性分数检查
            if 'validity_score' in qa and qa['validity_score'] < self.validity_threshold:
                continue
            
            # 去重（考虑相似度）
            question_lower = qa['question'].lower().strip()
            if question_lower not in seen_questions:
                seen_questions.add(question_lower)
                filtered.append(qa)
        
        # 确保类型分布均衡
        filtered = self._balance_question_types(filtered)
        
        logger.info(f"质量过滤: {len(qa_pairs)} -> {len(filtered)}")
        
        return filtered
    
    def _balance_question_types(self, qa_pairs: List[Dict]) -> List[Dict]:
        """平衡不同类型问题的分布"""
        # 按类型分组
        type_groups = {}
        for qa in qa_pairs:
            q_type = qa['type']
            if q_type not in type_groups:
                type_groups[q_type] = []
            type_groups[q_type].append(qa)
        
        # 计算每种类型的目标数量
        total_target = len(qa_pairs)
        num_types = len(type_groups)
        if num_types == 0:
            return qa_pairs
        
        target_per_type = total_target // num_types
        
        # 平衡选择
        balanced = []
        for q_type, type_qa_pairs in type_groups.items():
            # 按合理性分数排序，选择最好的
            type_qa_pairs.sort(key=lambda x: x.get('validity_score', 0.8), reverse=True)
            selected = type_qa_pairs[:target_per_type]
            balanced.extend(selected)
        
        return balanced
    
    # 辅助方法保持不变
    def _get_relation_type_name(self, relation: str, lang: str) -> str:
        """获取关系类型的自然语言名称"""
        relation_names = {
            'zh_cn': {
                '使用': '技术',
                '包含': '组件',
                '生产': '产品',
                '研发': '技术',
                '依赖': '依赖项',
                '改进': '改进方案',
                '替代': '替代品',
                '认证': '认证',
                '合作': '合作伙伴',
                '应用于': '应用'
            },
            'en': {
                '使用': 'technology',
                '包含': 'component',
                '生产': 'product',
                '研发': 'technology',
                '依赖': 'dependency',
                '改进': 'improvement',
                '替代': 'alternative',
                '认证': 'certification',
                '合作': 'partner',
                '应用于': 'application'
            }
        }
        
        return relation_names.get(lang, {}).get(relation, relation)
    
    def _get_entity_type_name(self, entity_type: str, lang: str) -> str:
        """获取实体类型的自然语言名称"""
        type_names = {
            'zh_cn': {
                '产品': '产品',
                '技术': '技术',
                '工艺': '工艺',
                '材料': '材料',
                '设备': '设备',
                '标准': '标准',
                '专利': '专利',
                '公司': '公司',
                '人员': '人员',
                '项目': '项目'
            },
            'en': {
                '产品': 'product',
                '技术': 'technology',
                '工艺': 'process',
                '材料': 'material',
                '设备': 'equipment',
                '标准': 'standard',
                '专利': 'patent',
                '公司': 'company',
                '人员': 'personnel',
                '项目': 'project'
            }
        }
        
        return type_names.get(lang, {}).get(entity_type, entity_type)
    
    def _get_attribute_name(self, entity_type: str, lang: str) -> str:
        """获取实体属性名称"""
        attributes = {
            'zh_cn': {
                '产品': '性能参数',
                '技术': '技术指标',
                '工艺': '工艺参数',
                '材料': '材料特性',
                '设备': '设备规格',
                '公司': '核心竞争力'
            },
            'en': {
                '产品': 'performance parameters',
                '技术': 'technical specifications',
                '工艺': 'process parameters',
                '材料': 'material properties',
                '设备': 'equipment specifications',
                '公司': 'core competencies'
            }
        }
        
        return attributes.get(lang, {}).get(entity_type, '特性' if lang == 'zh_cn' else 'characteristics')
    
    def _select_comparison_aspect(self, entity_type: str, lang: str) -> str:
        """选择比较维度"""
        aspects = {
            'zh_cn': {
                '产品': ['性能', '成本', '可靠性', '能效'],
                '技术': ['先进性', '成熟度', '应用范围', '技术壁垒'],
                '材料': ['强度', '耐久性', '成本', '环保性'],
                '工艺': ['效率', '精度', '成本', '稳定性']
            },
            'en': {
                '产品': ['performance', 'cost', 'reliability', 'energy efficiency'],
                '技术': ['advancement', 'maturity', 'application scope', 'technical barriers'],
                '材料': ['strength', 'durability', 'cost', 'environmental friendliness'],
                '工艺': ['efficiency', 'precision', 'cost', 'stability']
            }
        }
        
        type_aspects = aspects.get(lang, {}).get(entity_type, 
                                                ['特性'] if lang == 'zh_cn' else ['characteristics'])
        return random.choice(type_aspects)
    
    def _get_superlative(self, lang: str) -> str:
        """获取最高级词汇"""
        superlatives = {
            'zh_cn': ['好', '优秀', '先进', '高效', '稳定'],
            'en': ['best', 'excellent', 'advanced', 'efficient', 'stable']
        }
        
        return random.choice(superlatives.get(lang, ['best']))
    
    def _create_condition(self, edges: List[Dict], lang: str) -> str:
        """创建条件描述"""
        if not edges:
            return "在当前条件下" if lang == 'zh_cn' else "under current conditions"
        
        edge = edges[0]
        if lang == 'zh_cn':
            return f"{edge['source']}{edge['relation']}{edge['target']}"
        else:
            return f"{edge['source']} {edge['relation']} {edge['target']}"
    
    def _select_reasoning_target(self, subgraph: nx.DiGraph, entity: Dict, lang: str) -> str:
        """选择推理目标"""
        # 找到与实体相关的其他节点
        related_nodes = []
        for u, v in subgraph.edges():
            if u == entity['id']:
                related_nodes.append(v)
            elif v == entity['id']:
                related_nodes.append(u)
        
        if related_nodes:
            return random.choice(related_nodes)
        else:
            return "系统性能" if lang == 'zh_cn' else "system performance"
    
    def _summarize_evidence(self, edges: List[Dict], lang: str) -> str:
        """总结证据"""
        if not edges:
            return ""
        
        if lang == 'zh_cn':
            evidence_parts = []
            for edge in edges[:3]:  # 最多3条
                evidence_parts.append(f"{edge['source']}{edge['relation']}{edge['target']}")
            return "、".join(evidence_parts)
        else:
            evidence_parts = []
            for edge in edges[:3]:
                evidence_parts.append(f"{edge['source']} {edge['relation']} {edge['target']}")
            return ", ".join(evidence_parts)
    
    def _select_requirement(self, entity_type: str, lang: str) -> str:
        """选择需求"""
        requirements = {
            'zh_cn': {
                '产品': '高质量标准',
                '技术': '技术创新',
                '材料': '特殊性能',
                '工艺': '精确控制'
            },
            'en': {
                '产品': 'high quality standards',
                '技术': 'technological innovation',
                '材料': 'special properties',
                '工艺': 'precise control'
            }
        }
        
        return requirements.get(lang, {}).get(entity_type, 
                                            '特定要求' if lang == 'zh_cn' else 'specific requirements')
    
    def _get_performance_metric(self, entity_type: str, lang: str) -> str:
        """获取性能指标"""
        metrics = {
            'zh_cn': {
                '产品': '整体性能',
                '技术': '技术效果',
                '材料': '使用性能',
                '工艺': '生产效率'
            },
            'en': {
                '产品': 'overall performance',
                '技术': 'technical effectiveness',
                '材料': 'usage performance',
                '工艺': 'production efficiency'
            }
        }
        
        return metrics.get(lang, {}).get(entity_type, 
                                       '性能' if lang == 'zh_cn' else 'performance')
    
    def _get_path_type_name(self, lang: str) -> str:
        """获取路径类型名称"""
        path_types = {
            'zh_cn': ['技术演进', '供应链', '研发', '生产'],
            'en': ['technology evolution', 'supply chain', 'R&D', 'production']
        }
        
        return random.choice(path_types.get(lang, ['connection']))