"""
问题生成器 - WebSailor核心模块
基于子图中节点与关系，设计QA问题
覆盖多种问题类型：事实型、比较型、推理型、多跳型等
"""

import json
import logging
import random
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    WebSailor核心：问题生成器
    基于子图生成多样化的问题，覆盖不同难度和类型
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
        except Exception as e:
            logger.warning(f"无法加载指定模型，使用默认模型: {e}")
            # 使用较小的默认模型
            self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b")
            self.model = AutoModelForCausalLM.from_pretrained(
                "THUDM/chatglm-6b",
                torch_dtype=torch.float16,
                device_map="auto"
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
            },
            'counterfactual': {
                'zh_cn': [
                    "如果{entity}不使用{current_tech}，可能会采用什么替代方案？",
                    "假设没有{constraint}，{entity}会如何改进？",
                    "如果{entity1}替换为{entity2}，会产生什么影响？",
                    "在{alternative_scenario}的情况下，{entity}的表现会如何？"
                ],
                'en': [
                    "If {entity} doesn't use {current_tech}, what alternatives might be adopted?",
                    "Assuming no {constraint}, how would {entity} be improved?",
                    "What would be the impact if {entity1} is replaced by {entity2}?",
                    "How would {entity} perform under {alternative_scenario}?"
                ]
            },
            'temporal': {
                'zh_cn': [
                    "{entity}的{attribute}是如何随时间演变的？",
                    "在{time_period}期间，{entity}发生了什么变化？",
                    "{entity}的发展历程中有哪些关键节点？",
                    "从{tech1}到{tech2}的技术演进经历了哪些阶段？"
                ],
                'en': [
                    "How has the {attribute} of {entity} evolved over time?",
                    "What changes occurred to {entity} during {time_period}?",
                    "What are the key milestones in the development of {entity}?",
                    "What stages did the evolution from {tech1} to {tech2} go through?"
                ]
            },
            'causal': {
                'zh_cn': [
                    "是什么原因导致了{entity}采用{technology}？",
                    "{event}对{entity}产生了什么影响？",
                    "{entity}的{problem}是由什么引起的？",
                    "哪些因素促进了{entity}的{improvement}？"
                ],
                'en': [
                    "What caused {entity} to adopt {technology}?",
                    "What impact did {event} have on {entity}?",
                    "What caused the {problem} of {entity}?",
                    "What factors contributed to the {improvement} of {entity}?"
                ]
            }
        }
        
    def generate_questions(self, subgraphs: List[Dict]) -> List[Dict]:
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
    
    def _analyze_subgraph(self, subgraph: Dict) -> Dict:
        """分析子图特征"""
        features = {
            'topology': subgraph.get('topology', 'unknown'),
            'num_nodes': subgraph['num_nodes'],
            'num_edges': subgraph['num_edges'],
            'node_types': subgraph['node_types'],
            'relation_types': subgraph['relation_types'],
            'has_path': 'path' in subgraph,
            'has_center': 'center' in subgraph,
            'has_cycle': 'cycle' in subgraph,
            'density': subgraph['num_edges'] / (subgraph['num_nodes'] * (subgraph['num_nodes'] - 1))
            if subgraph['num_nodes'] > 1 else 0
        }
        
        # 识别关键实体
        features['key_entities'] = self._identify_key_entities(subgraph)
        
        # 识别路径模式
        features['path_patterns'] = self._identify_path_patterns(subgraph)
        
        return features
    
    def _identify_key_entities(self, subgraph: Dict) -> List[Dict]:
        """识别子图中的关键实体"""
        key_entities = []
        
        # 基于度数识别
        node_degrees = {}
        for edge in subgraph['edges']:
            node_degrees[edge['source']] = node_degrees.get(edge['source'], 0) + 1
            node_degrees[edge['target']] = node_degrees.get(edge['target'], 0) + 1
        
        # 选择度数最高的节点
        if node_degrees:
            sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
            for node_id, degree in sorted_nodes[:3]:  # 前3个
                # 找到节点信息
                for node in subgraph['nodes']:
                    if node['id'] == node_id:
                        key_entities.append({
                            'id': node_id,
                            'type': node['type'],
                            'degree': degree,
                            'role': 'hub' if degree > 3 else 'connector'
                        })
                        break
        
        # 如果有特殊标记的节点
        if 'center' in subgraph:
            for node in subgraph['nodes']:
                if node['id'] == subgraph['center']:
                    key_entities.append({
                        'id': node['id'],
                        'type': node['type'],
                        'role': 'center'
                    })
        
        return key_entities
    
    def _identify_path_patterns(self, subgraph: Dict) -> List[Dict]:
        """识别子图中的路径模式"""
        patterns = []
        
        # 链式路径
        if 'path' in subgraph:
            patterns.append({
                'type': 'chain',
                'length': len(subgraph['path']),
                'nodes': subgraph['path']
            })
        
        # 多跳路径（通过边连接）
        if subgraph['num_edges'] >= 2:
            # 简化版：找到一些2跳路径
            for i, edge1 in enumerate(subgraph['edges']):
                for edge2 in subgraph['edges'][i+1:]:
                    if edge1['target'] == edge2['source']:
                        patterns.append({
                            'type': 'two_hop',
                            'path': [edge1['source'], edge1['target'], edge2['target']],
                            'relations': [edge1['relation'], edge2['relation']]
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
    
    def _generate_questions_for_type(self, subgraph: Dict, 
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
        elif q_type == 'counterfactual':
            qa_pairs.extend(self._generate_counterfactual_questions(
                subgraph, features, selected_lang
            ))
        elif q_type == 'temporal':
            qa_pairs.extend(self._generate_temporal_questions(
                subgraph, features, selected_lang
            ))
        elif q_type == 'causal':
            qa_pairs.extend(self._generate_causal_questions(
                subgraph, features, selected_lang
            ))
        
        return qa_pairs
    
    def _generate_factual_questions(self, subgraph: Dict, 
                                  features: Dict, lang: str) -> List[Dict]:
        """生成事实型问题"""
        qa_pairs = []
        templates = self.question_templates['factual'][lang]
        
        # 基于边生成问题
        for edge in subgraph['edges'][:5]:  # 限制数量
            # 找到源节点和目标节点
            source_node = next(n for n in subgraph['nodes'] if n['id'] == edge['source'])
            target_node = next(n for n in subgraph['nodes'] if n['id'] == edge['target'])
            
            # 选择模板
            template = random.choice(templates)
            
            # 填充模板
            question = template.format(
                entity=edge['source'],
                entity1=edge['source'],
                entity2=edge['target'],
                relation=edge['relation'],
                relation_type=self._get_relation_type_name(edge['relation'], lang),
                entity_type=self._get_entity_type_name(target_node['type'], lang),
                attribute=self._get_attribute_name(source_node['type'], lang)
            )
            
            # 生成答案
            answer = self._generate_answer_for_factual(edge, source_node, target_node, lang)
            
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'type': 'factual',
                'language': lang,
                'subgraph': subgraph,
                'evidence': {
                    'nodes': [source_node, target_node],
                    'edges': [edge]
                }
            })
        
        return qa_pairs
    
    def _generate_comparison_questions(self, subgraph: Dict, 
                                     features: Dict, lang: str) -> List[Dict]:
        """生成比较型问题"""
        qa_pairs = []
        templates = self.question_templates['comparison'][lang]
        
        # 找到相同类型的实体进行比较
        entity_groups = {}
        for node in subgraph['nodes']:
            entity_type = node['type']
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(node)
        
        # 为每组生成比较问题
        for entity_type, entities in entity_groups.items():
            if len(entities) >= 2:
                # 随机选择两个实体
                entity1, entity2 = random.sample(entities, 2)
                
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
                
                # 使用LLM生成答案
                answer = self._generate_answer_with_llm(question, subgraph, 'comparison')
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'type': 'comparison',
                    'language': lang,
                    'subgraph': subgraph,
                    'evidence': {
                        'nodes': entities,
                        'comparison_aspect': aspect
                    }
                })
        
        return qa_pairs
    
    def _generate_multihop_questions(self, subgraph: Dict, 
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
                
                # 找到节点信息
                start_node = next(n for n in subgraph['nodes'] if n['id'] == start)
                middle_node = next(n for n in subgraph['nodes'] if n['id'] == middle)
                end_node = next(n for n in subgraph['nodes'] if n['id'] == end)
                
                template = random.choice(templates)
                
                question = template.format(
                    entity1=start,
                    entity2=end,
                    entity_type=self._get_entity_type_name(middle_node['type'], lang),
                    start=start,
                    end=end,
                    path_type=self._get_path_type_name(lang),
                    entity=start,
                    relation1=pattern['relations'][0],
                    relation2=pattern['relations'][1]
                )
                
                # 生成答案
                answer = self._generate_answer_with_llm(question, subgraph, 'multi_hop')
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'type': 'multi_hop',
                    'language': lang,
                    'subgraph': subgraph,
                    'evidence': {
                        'path': pattern['path'],
                        'relations': pattern['relations']
                    }
                })
        
        return qa_pairs
    
    def _generate_reasoning_questions(self, subgraph: Dict, 
                                    features: Dict, lang: str) -> List[Dict]:
        """生成推理型问题"""
        qa_pairs = []
        templates = self.question_templates['reasoning'][lang]
        
        # 选择关键实体
        if features['key_entities']:
            key_entity = random.choice(features['key_entities'])
            
            # 找到相关的边和节点作为证据
            related_edges = [e for e in subgraph['edges'] 
                           if e['source'] == key_entity['id'] or 
                              e['target'] == key_entity['id']]
            
            if related_edges:
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
                
                answer = self._generate_answer_with_llm(question, subgraph, 'reasoning')
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'type': 'reasoning',
                    'language': lang,
                    'subgraph': subgraph,
                    'evidence': {
                        'key_entity': key_entity,
                        'related_edges': related_edges
                    }
                })
        
        return qa_pairs
    
    def _generate_answer_with_llm(self, question: str, subgraph: Dict, 
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

请基于知识图谱中的信息，给出准确、详细的答案。如果需要推理，请说明推理过程。

答案："""
        
        return prompt
    
    def _format_subgraph_for_prompt(self, subgraph: Dict) -> str:
        """格式化子图信息用于提示"""
        lines = []
        
        # 节点信息
        lines.append("节点：")
        for node in subgraph['nodes']:
            lines.append(f"  - {node['id']} (类型: {node['type']})")
        
        # 边信息
        lines.append("\n关系：")
        for edge in subgraph['edges']:
            lines.append(f"  - {edge['source']} --[{edge['relation']}]--> {edge['target']}")
        
        # 特殊信息
        if 'topology' in subgraph:
            lines.append(f"\n拓扑类型：{subgraph['topology']}")
        
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
        
        for qa in qa_pairs:
            # 长度检查
            q_len = len(qa['question'])
            if not (min_q_len <= q_len <= max_q_len):
                continue
            
            # 答案验证
            if quality_config['answer_validation']:
                if not qa['answer'] or len(qa['answer']) < 10:
                    continue
            
            # 去重（简化版）
            if qa['question'] not in [q['question'] for q in filtered]:
                filtered.append(qa)
        
        logger.info(f"质量过滤: {len(qa_pairs)} -> {len(filtered)}")
        
        return filtered
    
    # 辅助方法
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
    
    def _select_reasoning_target(self, subgraph: Dict, entity: Dict, lang: str) -> str:
        """选择推理目标"""
        # 找到与实体相关的其他节点
        related_nodes = []
        for edge in subgraph['edges']:
            if edge['source'] == entity['id']:
                related_nodes.append(edge['target'])
            elif edge['target'] == entity['id']:
                related_nodes.append(edge['source'])
        
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
    
    def _generate_answer_for_factual(self, edge: Dict, source_node: Dict, 
                                   target_node: Dict, lang: str) -> str:
        """生成事实型问题的答案"""
        if lang == 'zh_cn':
            return f"{source_node['id']}{edge['relation']}{target_node['id']}。"
        else:
            return f"{source_node['id']} {edge['relation']} {target_node['id']}."
    
    def _generate_counterfactual_questions(self, subgraph: Dict, 
                                         features: Dict, lang: str) -> List[Dict]:
        """生成反事实问题"""
        # 简化实现
        return []
    
    def _generate_temporal_questions(self, subgraph: Dict, 
                                   features: Dict, lang: str) -> List[Dict]:
        """生成时序问题"""
        # 简化实现
        return []
    
    def _generate_causal_questions(self, subgraph: Dict, 
                                 features: Dict, lang: str) -> List[Dict]:
        """生成因果问题"""
        # 简化实现
        return []