"""
模糊化处理器 - WebSailor核心模块
模糊描述中间实体或关系，添加冗余或干扰信息
使问题信息密度高但精确信息少，增加推理难度
"""

import json
import logging
import random
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class ObfuscationProcessor:
    """
    WebSailor核心：模糊化处理器
    - 模糊描述中间实体或关系（例如"这位领导人"代指子图中多个可能节点）
    - 添加冗余或干扰信息，使问题信息密度高但精确信息少
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.obf_config = config.get('obfuscation', {})
        
        # 模糊化策略
        self.strategies = self.obf_config.get('strategies', [])
        self.obfuscation_rate = self.obf_config.get('obfuscation_rate', 0.6)
        
        # 模糊化模式
        self.entity_patterns = self.obf_config['patterns']['entity_patterns']
        self.relation_patterns = self.obf_config['patterns']['relation_patterns']
        
        # TCL特定配置
        self.tcl_config = config.get('tcl_specific', {})
        
        # 初始化模糊化词典
        self._init_obfuscation_dict()
        
    def _init_obfuscation_dict(self):
        """初始化模糊化词典"""
        # 实体类型的模糊表达
        self.entity_obfuscations = {
            '产品': ['这种产品', '某个产品', '相关产品', '该产品', '此类产品'],
            '技术': ['这项技术', '某种技术', '相关技术', '该技术', '此类技术'],
            '工艺': ['这种工艺', '某个工艺', '相关工艺', '该工艺', '此类工艺'],
            '材料': ['这种材料', '某种材料', '相关材料', '该材料', '此类材料'],
            '设备': ['这种设备', '某个设备', '相关设备', '该设备', '此类设备'],
            '公司': ['这家公司', '某个企业', '相关企业', '该公司', '此类企业'],
            '人员': ['这位专家', '某位人员', '相关人员', '该人员', '此类专家'],
            '标准': ['这项标准', '某个标准', '相关标准', '该标准', '此类标准'],
            '专利': ['这项专利', '某个专利', '相关专利', '该专利', '此类专利'],
            '项目': ['这个项目', '某个项目', '相关项目', '该项目', '此类项目']
        }
        
        # 关系的模糊表达
        self.relation_obfuscations = {
            '使用': ['采用了', '应用了', '利用了', '涉及到', '与...有关'],
            '包含': ['包括', '含有', '涉及', '关联到', '相关于'],
            '生产': ['制造', '产出', '创造', '形成', '产生'],
            '研发': ['开发', '研究', '探索', '创新', '发展'],
            '依赖': ['需要', '基于', '依靠', '关联于', '相关于'],
            '改进': ['优化', '提升', '改善', '增强', '发展'],
            '替代': ['取代', '代替', '更换', '转换', '改变'],
            '认证': ['认可', '批准', '验证', '确认', '评定'],
            '合作': ['协作', '联合', '共同', '协同', '联系'],
            '应用于': ['用于', '适用于', '服务于', '面向', '针对']
        }
        
        # 干扰词汇库
        self.distractor_words = {
            'zh_cn': {
                'connectors': ['并且', '同时', '另外', '此外', '而且', '以及'],
                'qualifiers': ['可能', '也许', '大概', '似乎', '据说', '一般来说'],
                'fillers': ['在某种程度上', '从某个角度看', '在特定条件下', '根据相关资料'],
                'vague_refs': ['前面提到的', '之前说的', '相关的', '类似的', '对应的']
            },
            'en': {
                'connectors': ['and', 'also', 'moreover', 'furthermore', 'additionally'],
                'qualifiers': ['possibly', 'perhaps', 'probably', 'seemingly', 'reportedly'],
                'fillers': ['to some extent', 'from a certain perspective', 'under specific conditions'],
                'vague_refs': ['the aforementioned', 'the related', 'the similar', 'the corresponding']
            }
        }
        
    def process_qa_pairs(self, qa_pairs: List[Dict]) -> List[Dict]:
        """处理QA对，应用模糊化"""
        obfuscated_pairs = []
        
        for qa in qa_pairs:
            # 决定是否应用模糊化
            if random.random() < self.obfuscation_rate:
                obfuscated_qa = self._obfuscate_qa_pair(qa)
                obfuscated_pairs.append(obfuscated_qa)
            else:
                # 保留一些清晰的问题作为对比
                obfuscated_pairs.append(qa)
        
        logger.info(f"模糊化处理完成: {len(obfuscated_pairs)} 个QA对")
        
        return obfuscated_pairs
    
    def _obfuscate_qa_pair(self, qa: Dict) -> Dict:
        """对单个QA对应用模糊化"""
        # 复制原始QA对
        obf_qa = qa.copy()
        
        # 选择模糊化策略
        strategies = random.sample(
            self.strategies, 
            k=min(len(self.strategies), random.randint(1, 3))
        )
        
        # 应用各种模糊化策略
        for strategy in strategies:
            if strategy == 'entity_replacement':
                obf_qa = self._apply_entity_replacement(obf_qa)
            elif strategy == 'relation_paraphrase':
                obf_qa = self._apply_relation_paraphrase(obf_qa)
            elif strategy == 'redundancy_injection':
                obf_qa = self._apply_redundancy_injection(obf_qa)
            elif strategy == 'ambiguity_introduction':
                obf_qa = self._apply_ambiguity_introduction(obf_qa)
            elif strategy == 'context_mixing':
                obf_qa = self._apply_context_mixing(obf_qa)
        
        # 记录应用的策略
        obf_qa['obfuscation_strategies'] = strategies
        obf_qa['is_obfuscated'] = True
        
        return obf_qa
    
    def _apply_entity_replacement(self, qa: Dict) -> Dict:
        """应用实体替换策略"""
        question = qa['question']
        evidence = qa.get('evidence', {})
        
        # 获取问题中的实体
        entities = evidence.get('nodes', [])
        
        for entity in entities:
            entity_id = entity['id']
            entity_type = entity['type']
            
            # 检查实体是否在问题中
            if entity_id in question:
                # 选择模糊表达
                if entity_type in self.entity_obfuscations:
                    obfuscations = self.entity_obfuscations[entity_type]
                    obf_expression = random.choice(obfuscations)
                    
                    # 替换实体
                    # 保留第一次出现，后续使用模糊表达
                    first_occurrence = question.find(entity_id)
                    if question.count(entity_id) > 1:
                        # 替换除第一次外的所有出现
                        parts = question.split(entity_id)
                        new_question = parts[0] + entity_id
                        for i in range(1, len(parts)):
                            if i == len(parts) - 1:
                                new_question += parts[i]
                            else:
                                new_question += obf_expression + parts[i]
                        question = new_question
                    else:
                        # 如果只出现一次，有50%概率替换
                        if random.random() < 0.5:
                            question = question.replace(entity_id, obf_expression)
        
        qa['question'] = question
        qa['obfuscation_details'] = qa.get('obfuscation_details', [])
        qa['obfuscation_details'].append({
            'strategy': 'entity_replacement',
            'description': '实体被模糊化表达替换'
        })
        
        return qa
    
    def _apply_relation_paraphrase(self, qa: Dict) -> Dict:
        """应用关系改写策略"""
        question = qa['question']
        evidence = qa.get('evidence', {})
        
        # 获取问题中的关系
        edges = evidence.get('edges', [])
        
        for edge in edges:
            relation = edge['relation']
            
            # 检查关系词是否在问题中
            if relation in question and relation in self.relation_obfuscations:
                obfuscations = self.relation_obfuscations[relation]
                obf_relation = random.choice(obfuscations)
                
                # 替换关系词
                question = question.replace(relation, obf_relation)
        
        qa['question'] = question
        qa['obfuscation_details'] = qa.get('obfuscation_details', [])
        qa['obfuscation_details'].append({
            'strategy': 'relation_paraphrase',
            'description': '关系词被改写为更模糊的表达'
        })
        
        return qa
    
    def _apply_redundancy_injection(self, qa: Dict) -> Dict:
        """注入冗余信息"""
        question = qa['question']
        lang = qa.get('language', 'zh_cn')
        
        # 获取干扰词汇
        distractors = self.distractor_words.get(lang, self.distractor_words['zh_cn'])
        
        # 添加连接词
        if random.random() < 0.7:
            connector = random.choice(distractors['connectors'])
            # 在句子中间插入连接词
            parts = self._split_question_smartly(question)
            if len(parts) > 1:
                insert_pos = random.randint(1, len(parts) - 1)
                parts.insert(insert_pos, connector)
                question = ' '.join(parts)
        
        # 添加限定词
        if random.random() < 0.6:
            qualifier = random.choice(distractors['qualifiers'])
            question = qualifier + '，' + question
        
        # 添加填充词
        if random.random() < 0.5:
            filler = random.choice(distractors['fillers'])
            question = question.rstrip('？?') + '，' + filler + '？'
        
        qa['question'] = question
        qa['obfuscation_details'] = qa.get('obfuscation_details', [])
        qa['obfuscation_details'].append({
            'strategy': 'redundancy_injection',
            'description': '注入了冗余词汇和短语'
        })
        
        return qa
    
    def _apply_ambiguity_introduction(self, qa: Dict) -> Dict:
        """引入歧义"""
        question = qa['question']
        lang = qa.get('language', 'zh_cn')
        subgraph = qa.get('subgraph', {})
        
        # 策略1：添加模糊指代
        vague_refs = self.distractor_words[lang]['vague_refs']
        
        # 找到问题中的具体实体
        entities_in_question = self._extract_entities_from_question(question, subgraph)
        
        if entities_in_question and random.random() < 0.7:
            # 选择一个实体进行模糊化
            entity_to_obfuscate = random.choice(entities_in_question)
            vague_ref = random.choice(vague_refs)
            
            # 创建模糊指代
            if lang == 'zh_cn':
                vague_expression = f"{vague_ref}{self._get_entity_type_name_zh(entity_to_obfuscate['type'])}"
            else:
                vague_expression = f"{vague_ref} {self._get_entity_type_name_en(entity_to_obfuscate['type'])}"
            
            # 替换实体
            question = question.replace(entity_to_obfuscate['id'], vague_expression, 1)
        
        # 策略2：添加多个可能的指代对象
        if random.random() < 0.5:
            question = self._add_multiple_referents(question, subgraph, lang)
        
        qa['question'] = question
        qa['obfuscation_details'] = qa.get('obfuscation_details', [])
        qa['obfuscation_details'].append({
            'strategy': 'ambiguity_introduction',
            'description': '引入了模糊指代和歧义'
        })
        
        return qa
    
    def _apply_context_mixing(self, qa: Dict) -> Dict:
        """混合上下文信息"""
        question = qa['question']
        subgraph = qa.get('subgraph', {})
        lang = qa.get('language', 'zh_cn')
        
        # 从子图中提取额外的上下文信息
        extra_context = self._extract_extra_context(subgraph, qa['evidence'])
        
        if extra_context:
            # 将额外上下文编织到问题中
            context_phrase = self._create_context_phrase(extra_context, lang)
            
            # 在问题开头或中间插入上下文
            if random.random() < 0.5:
                question = context_phrase + '，' + question
            else:
                # 在问题中间插入
                parts = self._split_question_smartly(question)
                if len(parts) > 1:
                    insert_pos = random.randint(1, len(parts) - 1)
                    parts.insert(insert_pos, '，' + context_phrase + '，')
                    question = ''.join(parts)
        
        qa['question'] = question
        qa['obfuscation_details'] = qa.get('obfuscation_details', [])
        qa['obfuscation_details'].append({
            'strategy': 'context_mixing',
            'description': '混入了额外的上下文信息'
        })
        
        return qa
    
    def _split_question_smartly(self, question: str) -> List[str]:
        """智能分割问题"""
        # 基于标点和关键词分割
        # 保留简单实现
        parts = []
        
        # 尝试按逗号分割
        if '，' in question:
            parts = question.split('，')
        elif ',' in question:
            parts = question.split(',')
        else:
            # 按空格分割（英文）或按字符长度分割（中文）
            if any(c.isalpha() and ord(c) < 128 for c in question):
                # 英文
                words = question.split()
                if len(words) > 4:
                    mid = len(words) // 2
                    parts = [' '.join(words[:mid]), ' '.join(words[mid:])]
                else:
                    parts = [question]
            else:
                # 中文
                if len(question) > 10:
                    mid = len(question) // 2
                    parts = [question[:mid], question[mid:]]
                else:
                    parts = [question]
        
        return parts
    
    def _extract_entities_from_question(self, question: str, 
                                      subgraph: Dict) -> List[Dict]:
        """从问题中提取实体"""
        entities = []
        
        for node in subgraph.get('nodes', []):
            if node['id'] in question:
                entities.append(node)
        
        return entities
    
    def _get_entity_type_name_zh(self, entity_type: str) -> str:
        """获取实体类型的中文名称"""
        type_names = {
            '产品': '产品',
            '技术': '技术',
            '工艺': '工艺',
            '材料': '材料',
            '设备': '设备',
            '公司': '企业',
            '人员': '人员',
            '标准': '标准',
            '专利': '专利',
            '项目': '项目'
        }
        return type_names.get(entity_type, '对象')
    
    def _get_entity_type_name_en(self, entity_type: str) -> str:
        """获取实体类型的英文名称"""
        type_names = {
            '产品': 'product',
            '技术': 'technology',
            '工艺': 'process',
            '材料': 'material',
            '设备': 'equipment',
            '公司': 'company',
            '人员': 'person',
            '标准': 'standard',
            '专利': 'patent',
            '项目': 'project'
        }
        return type_names.get(entity_type, 'entity')
    
    def _add_multiple_referents(self, question: str, subgraph: Dict, 
                              lang: str) -> str:
        """添加多个可能的指代对象"""
        # 找到子图中相同类型的实体
        entity_groups = defaultdict(list)
        for node in subgraph.get('nodes', []):
            entity_groups[node['type']].append(node)
        
        # 找到有多个实例的类型
        multi_instance_types = {k: v for k, v in entity_groups.items() if len(v) > 1}
        
        if multi_instance_types:
            # 随机选择一个类型
            selected_type = random.choice(list(multi_instance_types.keys()))
            entities = multi_instance_types[selected_type]
            
            # 创建包含多个实体的短语
            if lang == 'zh_cn':
                entity_list = '、'.join([e['id'] for e in entities[:3]])
                phrase = f"在{entity_list}等{self._get_entity_type_name_zh(selected_type)}中"
            else:
                entity_list = ', '.join([e['id'] for e in entities[:3]])
                phrase = f"among {self._get_entity_type_name_en(selected_type)}s like {entity_list}"
            
            # 在问题开头添加这个短语
            question = phrase + '，' + question
        
        return question
    
    def _extract_extra_context(self, subgraph: Dict, evidence: Dict) -> Dict:
        """提取额外的上下文信息"""
        extra_context = {
            'nodes': [],
            'edges': []
        }
        
        # 获取证据中已使用的节点和边
        used_nodes = {n['id'] for n in evidence.get('nodes', [])}
        used_edges = {(e['source'], e['target']) for e in evidence.get('edges', [])}
        
        # 找到未使用的节点和边
        for node in subgraph.get('nodes', []):
            if node['id'] not in used_nodes:
                extra_context['nodes'].append(node)
                if len(extra_context['nodes']) >= 2:
                    break
        
        for edge in subgraph.get('edges', []):
            edge_tuple = (edge['source'], edge['target'])
            if edge_tuple not in used_edges:
                extra_context['edges'].append(edge)
                if len(extra_context['edges']) >= 1:
                    break
        
        return extra_context if (extra_context['nodes'] or extra_context['edges']) else None
    
    def _create_context_phrase(self, context: Dict, lang: str) -> str:
        """创建上下文短语"""
        phrases = []
        
        if context['nodes']:
            node = context['nodes'][0]
            if lang == 'zh_cn':
                phrases.append(f"考虑到{node['id']}的存在")
            else:
                phrases.append(f"considering the presence of {node['id']}")
        
        if context['edges']:
            edge = context['edges'][0]
            if lang == 'zh_cn':
                phrases.append(f"鉴于{edge['source']}与{edge['target']}的关系")
            else:
                phrases.append(f"given the relationship between {edge['source']} and {edge['target']}")
        
        return random.choice(phrases) if phrases else ""