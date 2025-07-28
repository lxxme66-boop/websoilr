"""
模糊化处理器 - WebSailor核心思想实现
模糊描述中间实体或关系（例如"这位领导人"代指子图中多个可能节点）
添加冗余或干扰信息，使问题信息密度高但精确信息少
"""

import random
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
import networkx as nx
import re
from collections import defaultdict


@dataclass
class ObfuscatedQA:
    """模糊化后的问答对"""
    original_qa: Any  # 原始QA对
    obfuscated_question: str  # 模糊化后的问题
    obfuscated_answer: str  # 模糊化后的答案
    obfuscation_map: Dict[str, str]  # 实体模糊化映射
    added_distractors: List[str]  # 添加的干扰信息
    difficulty_increase: float  # 难度增加程度
    metadata: Dict[str, Any] = field(default_factory=dict)


class ObfuscationProcessor:
    """
    WebSailor模糊化处理器
    核心思想：通过实体抽象、信息分散、冗余注入等方式增加问题难度
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.strategies = config.get('strategies', {})
        self.maintain_difficulty_balance = config.get('maintain_difficulty_balance', True)
        self.validation_checks = config.get('validation_checks', True)
        
        # 初始化模糊化模式
        self._init_obfuscation_patterns()
        
    def process_qa_pairs(self, qa_pairs: List[Any], 
                        knowledge_graph: nx.Graph) -> List[ObfuscatedQA]:
        """
        对问答对进行模糊化处理
        
        Args:
            qa_pairs: 原始问答对列表
            knowledge_graph: 知识图谱
            
        Returns:
            模糊化后的问答对列表
        """
        self.logger.info(f"开始对{len(qa_pairs)}个问答对进行模糊化处理...")
        
        obfuscated_qa_list = []
        
        for qa in qa_pairs:
            # 根据问题类型和复杂度选择模糊化策略
            obfuscated = self._obfuscate_qa(qa, knowledge_graph)
            
            if obfuscated:
                obfuscated_qa_list.append(obfuscated)
        
        # 验证模糊化结果
        if self.validation_checks:
            obfuscated_qa_list = self._validate_obfuscation(obfuscated_qa_list)
        
        self.logger.info(f"完成模糊化处理，生成{len(obfuscated_qa_list)}个模糊化问答对")
        return obfuscated_qa_list
    
    def _obfuscate_qa(self, qa: Any, knowledge_graph: nx.Graph) -> Optional[ObfuscatedQA]:
        """对单个问答对进行模糊化"""
        obfuscation_map = {}
        added_distractors = []
        
        # 复制原始问题和答案
        obfuscated_question = qa.question
        obfuscated_answer = qa.answer
        
        # 应用各种模糊化策略
        if self.strategies.get('entity_abstraction', {}).get('enabled', True):
            obfuscated_question, obfuscated_answer, entity_map = \
                self._apply_entity_abstraction(
                    obfuscated_question, obfuscated_answer, 
                    qa.entities, knowledge_graph
                )
            obfuscation_map.update(entity_map)
        
        if self.strategies.get('information_scattering', {}).get('enabled', True):
            obfuscated_question, scatter_info = \
                self._apply_information_scattering(
                    obfuscated_question, qa.source_subgraph
                )
            added_distractors.extend(scatter_info)
        
        if self.strategies.get('redundancy_injection', {}).get('enabled', True):
            obfuscated_question, redundant_info = \
                self._apply_redundancy_injection(
                    obfuscated_question, knowledge_graph, qa.entities
                )
            added_distractors.extend(redundant_info)
        
        if self.strategies.get('ambiguity_introduction', {}).get('enabled', True):
            obfuscated_question, obfuscated_answer = \
                self._apply_ambiguity_introduction(
                    obfuscated_question, obfuscated_answer,
                    qa.entities, knowledge_graph
                )
        
        # 计算难度增加程度
        difficulty_increase = self._calculate_difficulty_increase(
            qa.question, obfuscated_question,
            len(obfuscation_map), len(added_distractors)
        )
        
        # 如果需要平衡难度
        if self.maintain_difficulty_balance:
            if difficulty_increase > 2.0:  # 难度增加过多
                # 减少一些模糊化
                obfuscated_question, obfuscated_answer, obfuscation_map = \
                    self._reduce_obfuscation(
                        qa, obfuscated_question, obfuscated_answer, 
                        obfuscation_map
                    )
        
        return ObfuscatedQA(
            original_qa=qa,
            obfuscated_question=obfuscated_question,
            obfuscated_answer=obfuscated_answer,
            obfuscation_map=obfuscation_map,
            added_distractors=added_distractors,
            difficulty_increase=difficulty_increase,
            metadata={
                'strategies_applied': list(self.strategies.keys()),
                'original_complexity': qa.complexity
            }
        )
    
    def _apply_entity_abstraction(self, question: str, answer: str,
                                 entities: List[str], 
                                 knowledge_graph: nx.Graph) -> Tuple[str, str, Dict[str, str]]:
        """应用实体抽象策略"""
        entity_map = {}
        patterns = self.strategies['entity_abstraction'].get('patterns', [])
        level = self.strategies['entity_abstraction'].get('level', 'moderate')
        
        # 对每个实体进行抽象
        for entity in entities:
            if entity not in knowledge_graph:
                continue
                
            # 获取实体类型
            entity_type = knowledge_graph.nodes[entity].get('type', '实体')
            
            # 根据抽象级别选择模式
            if level == 'high':
                # 高度抽象
                abstract_patterns = [
                    f"某个{entity_type}",
                    f"这个{entity_type}",
                    f"相关的{entity_type}",
                    f"上述提到的{entity_type}"
                ]
            elif level == 'moderate':
                # 中度抽象
                abstract_patterns = patterns or [
                    f"这个{entity_type}",
                    f"某种{entity_type}",
                    f"相关的{entity_type}"
                ]
            else:
                # 轻度抽象
                abstract_patterns = [f"该{entity_type}"]
            
            # 选择抽象表达
            abstract_expr = random.choice(abstract_patterns)
            
            # 考虑上下文，避免过度模糊
            if entities.count(entity) == 1 and len(entities) > 2:
                # 如果实体唯一且有多个实体，可以更抽象
                entity_map[entity] = abstract_expr
                
                # 替换问题和答案中的实体
                question = self._smart_replace(question, entity, abstract_expr)
                answer = self._smart_replace(answer, entity, abstract_expr)
            else:
                # 需要保持一定的可区分性
                neighbors = list(knowledge_graph.neighbors(entity))
                if neighbors:
                    # 使用关系来限定
                    relation = knowledge_graph.get_edge_data(
                        entity, neighbors[0], {}
                    ).get('relation', '相关')
                    
                    specific_abstract = f"与{neighbors[0]}{relation}的{entity_type}"
                    entity_map[entity] = specific_abstract
                    
                    question = self._smart_replace(question, entity, specific_abstract)
                    answer = self._smart_replace(answer, entity, specific_abstract)
        
        return question, answer, entity_map
    
    def _apply_information_scattering(self, question: str,
                                    subgraph: Any) -> Tuple[str, List[str]]:
        """应用信息分散策略"""
        scatter_range = self.strategies['information_scattering'].get(
            'scatter_range', [2, 5]
        )
        preserve_coherence = self.strategies['information_scattering'].get(
            'preserve_coherence', True
        )
        
        scatter_info = []
        
        # 从子图中提取额外信息
        graph = subgraph.graph
        num_scatter = random.randint(scatter_range[0], scatter_range[1])
        
        # 收集可以分散的信息
        available_info = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            
            # 收集属性信息
            for attr, value in node_data.items():
                if attr not in ['id', 'type'] and value:
                    info = f"{node}的{attr}是{value}"
                    available_info.append(info)
            
            # 收集关系信息
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor, {})
                relation = edge_data.get('relation', '相关')
                info = f"{node}{relation}{neighbor}"
                available_info.append(info)
        
        # 随机选择要分散的信息
        if available_info:
            num_to_scatter = min(num_scatter, len(available_info))
            scattered = random.sample(available_info, num_to_scatter)
            scatter_info.extend(scattered)
            
            # 将信息插入到问题中
            if preserve_coherence:
                # 保持连贯性的插入
                question = self._insert_scattered_info_coherently(
                    question, scattered
                )
            else:
                # 随机插入
                for info in scattered:
                    insert_pos = random.randint(0, len(question))
                    question = (
                        question[:insert_pos] + 
                        f"（顺便提一下，{info}）" + 
                        question[insert_pos:]
                    )
        
        return question, scatter_info
    
    def _apply_redundancy_injection(self, question: str,
                                  knowledge_graph: nx.Graph,
                                  entities: List[str]) -> Tuple[str, List[str]]:
        """应用冗余信息注入策略"""
        redundancy_rate = self.strategies['redundancy_injection'].get(
            'redundancy_rate', 0.2
        )
        redundancy_types = self.strategies['redundancy_injection'].get(
            'types', ['similar_facts', 'related_context', 'background_info']
        )
        
        redundant_info = []
        
        # 根据冗余率确定要添加的信息量
        num_redundant = int(len(entities) * redundancy_rate) + 1
        
        for _ in range(num_redundant):
            redundancy_type = random.choice(redundancy_types)
            
            if redundancy_type == 'similar_facts':
                # 添加相似事实
                info = self._generate_similar_facts(entities, knowledge_graph)
            elif redundancy_type == 'related_context':
                # 添加相关上下文
                info = self._generate_related_context(entities, knowledge_graph)
            elif redundancy_type == 'background_info':
                # 添加背景信息
                info = self._generate_background_info(entities, knowledge_graph)
            else:
                continue
                
            if info:
                redundant_info.append(info)
                
                # 将冗余信息添加到问题中
                question = self._add_redundant_info(question, info)
        
        return question, redundant_info
    
    def _apply_ambiguity_introduction(self, question: str, answer: str,
                                    entities: List[str],
                                    knowledge_graph: nx.Graph) -> Tuple[str, str]:
        """应用歧义引入策略"""
        ambiguity_level = self.strategies['ambiguity_introduction'].get(
            'ambiguity_level', 'controlled'
        )
        preserve_solvability = self.strategies['ambiguity_introduction'].get(
            'preserve_solvability', True
        )
        
        # 引入代词歧义
        if ambiguity_level in ['controlled', 'high']:
            question = self._introduce_pronoun_ambiguity(question, entities)
        
        # 引入时间歧义
        if ambiguity_level == 'high':
            question = self._introduce_temporal_ambiguity(question)
        
        # 引入空间歧义
        if random.random() < 0.3:
            question = self._introduce_spatial_ambiguity(question)
        
        # 如果需要保持可解性，在答案中提供消歧信息
        if preserve_solvability:
            answer = self._add_disambiguation_info(answer, entities, knowledge_graph)
        
        return question, answer
    
    def _smart_replace(self, text: str, old: str, new: str) -> str:
        """智能替换，避免破坏词语"""
        # 使用正则表达式进行全词匹配替换
        pattern = r'\b' + re.escape(old) + r'\b'
        return re.sub(pattern, new, text)
    
    def _insert_scattered_info_coherently(self, question: str,
                                        scattered_info: List[str]) -> str:
        """连贯地插入分散信息"""
        # 寻找合适的插入点（句子边界）
        sentences = re.split(r'[。？！]', question)
        
        if len(sentences) > 1:
            # 在句子之间插入
            for info in scattered_info:
                insert_idx = random.randint(0, len(sentences) - 1)
                sentences.insert(
                    insert_idx,
                    f"值得注意的是，{info}"
                )
            
            # 重新组合
            question = '。'.join(sentences)
            if not question.endswith('？'):
                question += '？'
        else:
            # 在问题前添加背景信息
            prefix = '。'.join([f"已知{info}" for info in scattered_info])
            question = prefix + '。' + question
        
        return question
    
    def _generate_similar_facts(self, entities: List[str],
                              knowledge_graph: nx.Graph) -> Optional[str]:
        """生成相似事实"""
        if not entities:
            return None
            
        entity = random.choice(entities)
        
        if entity not in knowledge_graph:
            return None
            
        # 寻找相似实体
        entity_type = knowledge_graph.nodes[entity].get('type', '')
        similar_entities = [
            n for n in knowledge_graph.nodes()
            if n != entity and 
            knowledge_graph.nodes[n].get('type', '') == entity_type
        ]
        
        if similar_entities:
            similar = random.choice(similar_entities)
            # 生成相似事实
            attrs = knowledge_graph.nodes[similar]
            if attrs:
                attr_key = random.choice([k for k in attrs.keys() 
                                        if k not in ['id', 'type']])
                return f"类似地，{similar}的{attr_key}是{attrs[attr_key]}"
        
        return None
    
    def _generate_related_context(self, entities: List[str],
                                knowledge_graph: nx.Graph) -> Optional[str]:
        """生成相关上下文"""
        if not entities:
            return None
            
        entity = random.choice(entities)
        
        if entity not in knowledge_graph:
            return None
            
        # 获取二跳邻居
        two_hop_neighbors = set()
        for neighbor in knowledge_graph.neighbors(entity):
            two_hop_neighbors.update(knowledge_graph.neighbors(neighbor))
        
        two_hop_neighbors.discard(entity)
        
        if two_hop_neighbors:
            context_node = random.choice(list(two_hop_neighbors))
            return f"在更广泛的背景下，{context_node}也是相关的因素"
        
        return None
    
    def _generate_background_info(self, entities: List[str],
                                knowledge_graph: nx.Graph) -> Optional[str]:
        """生成背景信息"""
        # 生成领域相关的背景信息
        backgrounds = [
            "在工业4.0的背景下",
            "考虑到当前的技术发展",
            "从历史经验来看",
            "根据行业标准",
            "在实际应用中"
        ]
        
        return random.choice(backgrounds) + "，这是一个重要的考虑因素"
    
    def _add_redundant_info(self, question: str, info: str) -> str:
        """添加冗余信息到问题中"""
        # 使用不同的连接词
        connectors = [
            "另外，",
            "此外，",
            "需要注意的是，",
            "补充说明，",
            "顺便提及，"
        ]
        
        connector = random.choice(connectors)
        
        # 随机决定添加位置
        if random.random() < 0.5:
            # 添加到开头
            return f"{connector}{info}。{question}"
        else:
            # 添加到中间
            sentences = question.split('。')
            if len(sentences) > 1:
                insert_idx = random.randint(0, len(sentences) - 1)
                sentences.insert(insert_idx, f"{connector}{info}")
                return '。'.join(sentences)
            else:
                return f"{question[:-1]}。{connector}{info}？"
    
    def _introduce_pronoun_ambiguity(self, question: str,
                                   entities: List[str]) -> str:
        """引入代词歧义"""
        pronouns = {
            '它': ['设备', '系统', '产品', '材料'],
            '他们': ['工人', '技术人员', '管理者'],
            '这个': ['过程', '方法', '标准', '参数'],
            '那些': ['因素', '条件', '要求', '指标']
        }
        
        for entity in entities:
            # 随机决定是否替换为代词
            if random.random() < 0.3:
                # 根据实体类型选择合适的代词
                entity_type = entity.split('_')[0] if '_' in entity else entity
                
                suitable_pronouns = []
                for pronoun, types in pronouns.items():
                    if any(t in entity_type for t in types):
                        suitable_pronouns.append(pronoun)
                
                if suitable_pronouns:
                    pronoun = random.choice(suitable_pronouns)
                    # 只替换第二次及以后出现的实体
                    first_occurrence = question.find(entity)
                    if first_occurrence != -1:
                        second_occurrence = question.find(
                            entity, first_occurrence + len(entity)
                        )
                        if second_occurrence != -1:
                            question = (
                                question[:second_occurrence] + 
                                pronoun + 
                                question[second_occurrence + len(entity):]
                            )
        
        return question
    
    def _introduce_temporal_ambiguity(self, question: str) -> str:
        """引入时间歧义"""
        temporal_terms = [
            "最近", "之前", "早期", "后来", "当时",
            "目前", "现在", "将来", "过去", "曾经"
        ]
        
        # 随机添加时间词
        if random.random() < 0.4:
            term = random.choice(temporal_terms)
            question = f"在{term}的情况下，{question}"
        
        return question
    
    def _introduce_spatial_ambiguity(self, question: str) -> str:
        """引入空间歧义"""
        spatial_terms = [
            "这里", "那里", "附近", "周围", "内部",
            "外部", "上方", "下方", "旁边", "中间"
        ]
        
        # 随机添加空间词
        if random.random() < 0.3:
            term = random.choice(spatial_terms)
            question = question.replace(
                "在", f"在{term}的", 1
            )
        
        return question
    
    def _add_disambiguation_info(self, answer: str, entities: List[str],
                               knowledge_graph: nx.Graph) -> str:
        """在答案中添加消歧信息"""
        disambiguation_prefix = "具体来说，"
        
        # 为每个实体添加明确的说明
        for entity in entities[:2]:  # 只为前两个实体消歧
            if entity in knowledge_graph:
                entity_type = knowledge_graph.nodes[entity].get('type', '')
                if entity_type:
                    disambiguation = f"{entity}（即{entity_type}）"
                    answer = answer.replace(entity, disambiguation, 1)
        
        return disambiguation_prefix + answer
    
    def _calculate_difficulty_increase(self, original_question: str,
                                     obfuscated_question: str,
                                     num_abstractions: int,
                                     num_distractors: int) -> float:
        """计算难度增加程度"""
        # 基于多个因素计算难度增加
        
        # 1. 长度增加
        length_increase = len(obfuscated_question) / max(len(original_question), 1)
        
        # 2. 抽象化程度
        abstraction_factor = 1 + (num_abstractions * 0.2)
        
        # 3. 干扰信息量
        distractor_factor = 1 + (num_distractors * 0.15)
        
        # 4. 复杂度估计（基于句子数量）
        original_sentences = len(re.split(r'[。？！]', original_question))
        obfuscated_sentences = len(re.split(r'[。？！]', obfuscated_question))
        complexity_factor = obfuscated_sentences / max(original_sentences, 1)
        
        # 综合计算
        difficulty_increase = (
            length_increase * 0.2 +
            abstraction_factor * 0.3 +
            distractor_factor * 0.3 +
            complexity_factor * 0.2
        )
        
        return round(difficulty_increase, 2)
    
    def _reduce_obfuscation(self, original_qa: Any,
                          obfuscated_question: str,
                          obfuscated_answer: str,
                          obfuscation_map: Dict[str, str]) -> Tuple[str, str, Dict[str, str]]:
        """减少模糊化程度"""
        # 恢复一些实体
        entities_to_restore = random.sample(
            list(obfuscation_map.keys()),
            min(len(obfuscation_map) // 2, len(obfuscation_map))
        )
        
        for entity in entities_to_restore:
            abstract = obfuscation_map[entity]
            obfuscated_question = obfuscated_question.replace(abstract, entity)
            obfuscated_answer = obfuscated_answer.replace(abstract, entity)
            del obfuscation_map[entity]
        
        # 移除一些括号内的干扰信息
        obfuscated_question = re.sub(r'（[^）]+）', '', obfuscated_question)
        
        return obfuscated_question, obfuscated_answer, obfuscation_map
    
    def _validate_obfuscation(self, 
                            obfuscated_qa_list: List[ObfuscatedQA]) -> List[ObfuscatedQA]:
        """验证模糊化结果"""
        validated = []
        
        for oqa in obfuscated_qa_list:
            # 检查问题是否仍然可理解
            if len(oqa.obfuscated_question) < 10:
                continue
                
            # 检查是否保留了关键信息
            if not any(entity in oqa.obfuscated_answer 
                      for entity in oqa.original_qa.entities):
                # 如果答案中没有任何原始实体，可能过度模糊
                continue
                
            # 检查难度是否在合理范围内
            if oqa.difficulty_increase > 3.0:
                self.logger.warning(f"难度增加过大: {oqa.difficulty_increase}")
                continue
                
            validated.append(oqa)
        
        return validated
    
    def _init_obfuscation_patterns(self):
        """初始化模糊化模式"""
        # 可以从配置或外部文件加载更多模式
        self.abstraction_patterns = {
            '设备': ['这台设备', '相关设备', '该装置', '此类机器'],
            '材料': ['这种材料', '相关物质', '该原料', '此类材质'],
            '工艺': ['这个工艺', '相关流程', '该过程', '此类方法'],
            '参数': ['这个参数', '相关指标', '该数值', '此类标准'],
            '人员': ['相关人员', '负责人', '操作者', '技术人员']
        }
        
        self.ambiguity_patterns = {
            'temporal': ['之前', '之后', '期间', '当时', '现在'],
            'spatial': ['这里', '那里', '附近', '内部', '外部'],
            'referential': ['它', '他们', '这个', '那些', '这些']
        }