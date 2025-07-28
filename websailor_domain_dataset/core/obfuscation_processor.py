"""
模糊化处理器 - WebSailor核心思想实现
模糊描述中间实体或关系(例如"这位领导人"代指子图中多个可能节点)
添加冗余或干扰信息,使问题信息密度高但精确信息少
"""

import logging
import random
import re
from typing import Dict, List, Tuple, Any, Optional, Set
import networkx as nx
from collections import defaultdict

logger = logging.getLogger(__name__)

class ObfuscationProcessor:
    """
    模糊化处理器 - WebSailor的核心创新组件
    
    核心思想：
    1. 模糊描述中间实体或关系：
       - 使用指代词替换具体实体名称
       - 使用上位词替换下位词
       - 使用模糊的描述性短语
    2. 添加冗余或干扰信息：
       - 插入与问题相关但非必要的信息
       - 添加干扰性的实体和关系
       - 增加问题的信息密度但降低精确信息
    3. 创造不确定性：
       - 让模型需要在多个候选答案中选择
       - 增加推理的复杂度和不确定性
    """
    
    def __init__(self, config: Dict[str, Any], patterns: Dict[str, Any]):
        self.config = config
        self.patterns = patterns
        self.obfuscation_ratio = config['obfuscation_ratio']
        self.noise_ratio = config['noise_ratio']
        self.max_noise_entities = config['max_noise_entities']
        self.ambiguity_level = config['ambiguity_level']
        
        # TCL工业领域的模糊化模式
        self.tcl_obfuscation_patterns = {
            # 实体模糊化模式：具体实体 -> 模糊描述
            'entity_obfuscation': {
                '产品': ['这个产品', '该设备', '这种装置', '相关产品', '此类设备'],
                '技术': ['这项技术', '该方法', '这种工艺', '相关技术', '此类方案'],
                '材料': ['这种材料', '该物质', '这类原料', '相关材料', '此种成分'],
                '设备': ['这台设备', '该机器', '这种装置', '相关设备', '此类机械'],
                '工艺': ['这个工艺', '该流程', '这种方法', '相关工艺', '此类过程'],
                '质量指标': ['这项指标', '该参数', '这个标准', '相关指标', '此类要求'],
                '性能参数': ['这个参数', '该指标', '这种性能', '相关参数', '此类数据'],
                '应用场景': ['这种应用', '该场景', '这类用途', '相关应用', '此种情况'],
                '问题': ['这个问题', '该难题', '这种情况', '相关问题', '此类挑战'],
                '解决方案': ['这个方案', '该解决办法', '这种方法', '相关方案', '此类策略']
            },
            
            # 关系模糊化模式：具体关系 -> 模糊描述
            'relation_obfuscation': {
                '包含': ['涉及', '相关', '关联', '涵盖', '包括'],
                '依赖': ['需要', '要求', '基于', '依靠', '依赖于'],
                '影响': ['作用于', '影响到', '对...有影响', '产生作用', '起作用'],
                '改进': ['优化', '提升', '改善', '完善', '增强'],
                '应用于': ['用于', '适用于', '运用在', '应用在', '使用于'],
                '导致': ['引起', '造成', '产生', '带来', '导致'],
                '解决': ['处理', '应对', '解决', '处置', '应付'],
                '优化': ['改进', '提升', '完善', '优化', '改善'],
                '测试': ['检测', '验证', '测试', '检验', '评估'],
                '生产': ['制造', '生产', '加工', '制作', '产出']
            },
            
            # 数值模糊化模式
            'value_obfuscation': {
                'exact_numbers': ['大约', '约', '接近', '大概', '左右'],
                'ranges': ['在...之间', '从...到...', '范围为', '介于', '处于'],
                'comparatives': ['较高', '较低', '更大', '更小', '相当']
            },
            
            # 时间模糊化模式
            'temporal_obfuscation': {
                'specific_time': ['最近', '近期', '之前', '当时', '那时'],
                'duration': ['一段时间', '较长时间', '短时间内', '持续', '经过']
            }
        }
        
        # 干扰信息生成模式
        self.noise_patterns = {
            'irrelevant_entities': [
                '另外还有{entity}也参与其中',
                '同时{entity}也发挥了作用',
                '此外{entity}也是重要因素',
                '值得注意的是{entity}同样重要',
                '不可忽视{entity}的影响'
            ],
            'redundant_relations': [
                '需要说明的是',
                '值得一提的是',
                '另外需要考虑',
                '同时还要注意',
                '此外还应该关注'
            ],
            'background_noise': [
                '在当前的工业环境下',
                '考虑到技术发展趋势',
                '基于市场需求变化',
                '结合行业标准要求',
                '根据实际应用情况'
            ]
        }
        
        # 指代词系统
        self.pronoun_system = {
            'entity_pronouns': ['它', '这个', '该', '此', '其'],
            'group_pronouns': ['它们', '这些', '那些', '此类', '这类'],
            'possessive_pronouns': ['其', '它的', '这个的', '该的', '此的']
        }
    
    def obfuscate_question(self, question: Dict[str, Any], subgraph: nx.Graph, 
                          scenario_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        对问题进行模糊化处理
        
        Args:
            question: 原始问题
            subgraph: 子图结构
            scenario_features: 场景特征
            
        Returns:
            Dict: 模糊化后的问题
        """
        logger.debug(f"开始模糊化问题: {question['question']}")
        
        obfuscated_question = question.copy()
        original_text = question['question']
        
        # 1. 实体模糊化
        obfuscated_text, entity_mappings = self._obfuscate_entities(
            original_text, subgraph, scenario_features
        )
        
        # 2. 关系模糊化
        obfuscated_text, relation_mappings = self._obfuscate_relations(
            obfuscated_text, subgraph
        )
        
        # 3. 添加干扰信息
        obfuscated_text, noise_info = self._add_noise_information(
            obfuscated_text, subgraph, scenario_features
        )
        
        # 4. 增加指代歧义
        obfuscated_text, pronoun_mappings = self._add_pronoun_ambiguity(
            obfuscated_text, entity_mappings
        )
        
        # 5. 数值和时间模糊化
        obfuscated_text = self._obfuscate_values_and_time(obfuscated_text)
        
        # 更新问题内容
        obfuscated_question.update({
            'question': obfuscated_text,
            'original_question': original_text,
            'obfuscation_info': {
                'entity_mappings': entity_mappings,
                'relation_mappings': relation_mappings,
                'noise_info': noise_info,
                'pronoun_mappings': pronoun_mappings,
                'obfuscation_level': self._calculate_obfuscation_level(
                    original_text, obfuscated_text
                )
            },
            'is_obfuscated': True
        })
        
        logger.debug(f"模糊化完成: {obfuscated_text}")
        return obfuscated_question
    
    def _obfuscate_entities(self, text: str, subgraph: nx.Graph, 
                           scenario_features: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        """实体模糊化 - 将具体实体替换为模糊描述"""
        entity_mappings = {}
        obfuscated_text = text
        
        # 获取模糊实体候选
        ambiguous_entities = scenario_features.get('ambiguous_entities', [])
        
        # 对每个实体进行模糊化处理
        for node in subgraph.nodes():
            if node in text and random.random() < self.obfuscation_ratio:
                node_type = subgraph.nodes[node].get('type', '实体')
                
                # 选择模糊化策略
                if node in ambiguous_entities:
                    # 对于模糊实体，使用更强的模糊化
                    obfuscation_candidates = self.tcl_obfuscation_patterns['entity_obfuscation'].get(
                        node_type, ['这个实体', '该对象', '相关项目']
                    )
                    obfuscated_form = random.choice(obfuscation_candidates)
                    
                    # 添加额外的模糊描述
                    if random.random() < 0.5:
                        descriptors = [
                            f'具有特定{random.choice(["功能", "特性", "属性"])}的',
                            f'在{random.choice(["系统", "流程", "环境"])}中的',
                            f'与{random.choice(["质量", "性能", "效率"])}相关的'
                        ]
                        obfuscated_form = random.choice(descriptors) + obfuscated_form
                else:
                    # 常规模糊化
                    obfuscation_candidates = self.tcl_obfuscation_patterns['entity_obfuscation'].get(
                        node_type, ['这个', '该', '相关的']
                    )
                    obfuscated_form = random.choice(obfuscation_candidates)
                
                # 执行替换
                obfuscated_text = obfuscated_text.replace(node, obfuscated_form)
                entity_mappings[node] = obfuscated_form
        
        return obfuscated_text, entity_mappings
    
    def _obfuscate_relations(self, text: str, subgraph: nx.Graph) -> Tuple[str, Dict[str, str]]:
        """关系模糊化 - 将具体关系替换为模糊描述"""
        relation_mappings = {}
        obfuscated_text = text
        
        # 提取文本中的关系词
        for _, _, edge_data in subgraph.edges(data=True):
            relation = edge_data.get('relation', '')
            if relation and relation in text and random.random() < self.obfuscation_ratio:
                
                # 获取模糊化候选
                obfuscation_candidates = self.tcl_obfuscation_patterns['relation_obfuscation'].get(
                    relation, ['相关', '关联', '连接']
                )
                obfuscated_relation = random.choice(obfuscation_candidates)
                
                # 添加模糊化修饰
                if random.random() < 0.3:
                    modifiers = ['某种程度上', '在一定条件下', '通过特定方式', '以某种形式']
                    obfuscated_relation = random.choice(modifiers) + obfuscated_relation
                
                obfuscated_text = obfuscated_text.replace(relation, obfuscated_relation)
                relation_mappings[relation] = obfuscated_relation
        
        return obfuscated_text, relation_mappings
    
    def _add_noise_information(self, text: str, subgraph: nx.Graph, 
                              scenario_features: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """添加干扰信息 - 增加冗余和干扰内容"""
        noise_info = {
            'added_entities': [],
            'added_relations': [],
            'background_noise': []
        }
        
        if random.random() > self.noise_ratio:
            return text, noise_info
        
        obfuscated_text = text
        
        # 1. 添加无关实体
        interference_nodes = scenario_features.get('interference_nodes', [])
        if interference_nodes:
            num_noise_entities = min(
                random.randint(1, self.max_noise_entities),
                len(interference_nodes)
            )
            
            selected_noise = random.sample(interference_nodes, num_noise_entities)
            for noise_entity in selected_noise:
                noise_template = random.choice(self.noise_patterns['irrelevant_entities'])
                noise_text = noise_template.format(entity=noise_entity)
                
                # 插入到文本中的随机位置
                sentences = obfuscated_text.split('。')
                if len(sentences) > 1:
                    insert_pos = random.randint(0, len(sentences) - 1)
                    sentences.insert(insert_pos, noise_text)
                    obfuscated_text = '。'.join(sentences)
                else:
                    obfuscated_text = obfuscated_text + '，' + noise_text
                
                noise_info['added_entities'].append(noise_entity)
        
        # 2. 添加冗余关系描述
        if random.random() < 0.4:
            redundant_phrase = random.choice(self.noise_patterns['redundant_relations'])
            # 在问题开头添加冗余信息
            obfuscated_text = redundant_phrase + '，' + obfuscated_text
            noise_info['added_relations'].append(redundant_phrase)
        
        # 3. 添加背景噪声
        if random.random() < 0.3:
            background_phrase = random.choice(self.noise_patterns['background_noise'])
            obfuscated_text = background_phrase + '，' + obfuscated_text
            noise_info['background_noise'].append(background_phrase)
        
        return obfuscated_text, noise_info
    
    def _add_pronoun_ambiguity(self, text: str, entity_mappings: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
        """增加指代歧义 - 使用指代词创造歧义"""
        pronoun_mappings = {}
        obfuscated_text = text
        
        # 对已经模糊化的实体进一步使用指代词
        for original_entity, obfuscated_entity in entity_mappings.items():
            if random.random() < 0.3:  # 30%概率进一步模糊化
                
                # 选择合适的指代词
                if '这些' in obfuscated_entity or '那些' in obfuscated_entity:
                    pronoun = random.choice(self.pronoun_system['group_pronouns'])
                else:
                    pronoun = random.choice(self.pronoun_system['entity_pronouns'])
                
                # 替换模糊化实体为指代词
                obfuscated_text = obfuscated_text.replace(obfuscated_entity, pronoun)
                pronoun_mappings[obfuscated_entity] = pronoun
        
        # 在文本中随机位置添加指代歧义
        if random.random() < 0.2:
            ambiguous_phrases = [
                '其中', '其中之一', '其中某个', '某个', '某些'
            ]
            ambiguous_phrase = random.choice(ambiguous_phrases)
            
            # 在适当位置插入模糊指代
            sentences = obfuscated_text.split('，')
            if len(sentences) > 1:
                insert_pos = random.randint(0, len(sentences) - 1)
                sentences[insert_pos] = ambiguous_phrase + sentences[insert_pos]
                obfuscated_text = '，'.join(sentences)
        
        return obfuscated_text, pronoun_mappings
    
    def _obfuscate_values_and_time(self, text: str) -> str:
        """数值和时间模糊化"""
        obfuscated_text = text
        
        # 数值模糊化
        number_pattern = r'\d+\.?\d*'
        numbers = re.findall(number_pattern, text)
        
        for number in numbers:
            if random.random() < 0.4:
                modifier = random.choice(self.tcl_obfuscation_patterns['value_obfuscation']['exact_numbers'])
                obfuscated_text = obfuscated_text.replace(number, f"{modifier}{number}")
        
        # 时间表达式模糊化
        time_patterns = [
            r'\d{4}年', r'\d{1,2}月', r'\d{1,2}日',
            r'今天', r'昨天', r'明天', r'现在', r'当前'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if random.random() < 0.3:
                    fuzzy_time = random.choice(
                        self.tcl_obfuscation_patterns['temporal_obfuscation']['specific_time']
                    )
                    obfuscated_text = obfuscated_text.replace(match, fuzzy_time)
        
        return obfuscated_text
    
    def _calculate_obfuscation_level(self, original_text: str, obfuscated_text: str) -> float:
        """计算模糊化程度"""
        if not original_text:
            return 0.0
        
        # 计算文本变化程度
        original_words = set(original_text.split())
        obfuscated_words = set(obfuscated_text.split())
        
        # 计算词汇重叠率
        common_words = original_words.intersection(obfuscated_words)
        overlap_ratio = len(common_words) / len(original_words) if original_words else 0
        
        # 模糊化程度 = 1 - 重叠率
        obfuscation_level = 1 - overlap_ratio
        
        # 考虑文本长度变化（添加噪声会增加长度）
        length_ratio = len(obfuscated_text) / len(original_text) if original_text else 1
        if length_ratio > 1.2:  # 如果文本长度增加超过20%
            obfuscation_level += 0.1
        
        return min(obfuscation_level, 1.0)
    
    def batch_obfuscate_questions(self, questions: List[Dict[str, Any]], 
                                 subgraph: nx.Graph, scenario_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """批量模糊化问题"""
        obfuscated_questions = []
        
        for question in questions:
            # 决定是否对此问题进行模糊化
            if random.random() < self.obfuscation_ratio:
                obfuscated_question = self.obfuscate_question(question, subgraph, scenario_features)
                obfuscated_questions.append(obfuscated_question)
            else:
                # 保留原始问题，但标记为未模糊化
                question['is_obfuscated'] = False
                obfuscated_questions.append(question)
        
        logger.info(f"批量模糊化完成，处理了 {len(questions)} 个问题")
        return obfuscated_questions
    
    def create_uncertainty_variants(self, question: Dict[str, Any], 
                                   subgraph: nx.Graph, num_variants: int = 3) -> List[Dict[str, Any]]:
        """
        创建不确定性变体 - 为同一个问题创建多个模糊化版本
        这是WebSailor的核心思想：增加模型推理的不确定性
        """
        variants = []
        original_question = question.copy()
        
        for i in range(num_variants):
            # 为每个变体使用不同的随机种子
            random.seed(random.randint(1, 10000))
            
            # 调整模糊化参数
            original_obfuscation_ratio = self.obfuscation_ratio
            original_noise_ratio = self.noise_ratio
            
            # 为不同变体使用不同的模糊化强度
            self.obfuscation_ratio = 0.3 + i * 0.2  # 递增模糊化程度
            self.noise_ratio = 0.2 + i * 0.15
            
            # 创建变体
            variant = self.obfuscate_question(
                original_question, subgraph, 
                original_question.get('scenario_features', {})
            )
            
            variant['variant_id'] = i
            variant['uncertainty_level'] = self.obfuscation_ratio
            variants.append(variant)
            
            # 恢复原始参数
            self.obfuscation_ratio = original_obfuscation_ratio
            self.noise_ratio = original_noise_ratio
        
        return variants