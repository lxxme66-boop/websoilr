"""
模糊化处理器 - WebSailor核心模块
模糊描述中间实体或关系（例如"这位领导人"代指子图中多个可能节点）
添加冗余或干扰信息，使问题信息密度高但精确信息少
"""

import random
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import networkx as nx
from collections import defaultdict
import re

from .question_generator import QAPair


@dataclass
class ObfuscatedQAPair(QAPair):
    """模糊化后的问答对"""
    original_question: str = ""
    obfuscation_metadata: Dict[str, Any] = field(default_factory=dict)
    ambiguity_score: float = 0.0


class ObfuscationProcessor:
    """
    WebSailor模糊化处理器
    负责对问答对进行模糊化处理，增加推理难度
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.strategies = config.get('strategies', [])
        self.preserve_answerability = config.get('preserve_answerability', True)
        self.validation_config = config.get('validation', {})
        
        # 初始化模糊化规则
        self._init_obfuscation_rules()
        
    def _init_obfuscation_rules(self):
        """初始化模糊化规则"""
        self.entity_substitutions = {
            # TCL工业领域特定的模糊化规则
            'display': ['显示设备', '这款显示产品', '某显示装置'],
            'semiconductor': ['半导体组件', '这种芯片', '某半导体产品'],
            'manufacturer': ['制造商', '某知名企业', '这家公司'],
            'technology': ['技术', '这项技术', '某种技术方案'],
            'component': ['组件', '这个部件', '某关键组件'],
            'standard': ['标准', '某项标准', '相关规范'],
            'process': ['工艺', '这道工序', '某生产流程']
        }
        
        self.relation_ambiguities = {
            'manufactures': ['生产', '制造', '出品', '推出'],
            'uses_technology': ['采用', '使用', '应用', '集成'],
            'contains_component': ['包含', '内置', '配备', '搭载'],
            'developed_by': ['由...开发', '...研发', '...创新', '...设计'],
            'complies_with_standard': ['符合', '遵循', '满足', '达到']
        }
        
    def process_qa_pairs(
        self, 
        qa_pairs: List[QAPair], 
        knowledge_graph: nx.Graph
    ) -> List[ObfuscatedQAPair]:
        """
        对问答对进行模糊化处理
        这是WebSailor的核心功能之一
        """
        obfuscated_pairs = []
        
        for qa_pair in qa_pairs:
            # 根据配置的策略进行模糊化
            obfuscated = self._apply_obfuscation_strategies(qa_pair, knowledge_graph)
            
            # 验证模糊化后的质量
            if self._validate_obfuscation(obfuscated):
                obfuscated_pairs.append(obfuscated)
            else:
                # 如果模糊化失败，保留原始版本
                self.logger.debug(f"Obfuscation validation failed for: {qa_pair.question}")
                obfuscated_pairs.append(self._convert_to_obfuscated(qa_pair))
                
        self.logger.info(f"Obfuscated {len(obfuscated_pairs)} QA pairs")
        return obfuscated_pairs
        
    def _apply_obfuscation_strategies(
        self, 
        qa_pair: QAPair, 
        knowledge_graph: nx.Graph
    ) -> ObfuscatedQAPair:
        """应用多种模糊化策略"""
        obfuscated = self._convert_to_obfuscated(qa_pair)
        obfuscated.original_question = qa_pair.question
        
        # 记录应用的策略
        applied_strategies = []
        
        for strategy in self.strategies:
            if random.random() < strategy.get('probability', 0.5):
                strategy_name = strategy['name']
                
                if strategy_name == 'entity_substitution':
                    obfuscated = self._apply_entity_substitution(
                        obfuscated, strategy, knowledge_graph
                    )
                elif strategy_name == 'information_injection':
                    obfuscated = self._apply_information_injection(
                        obfuscated, strategy, knowledge_graph
                    )
                elif strategy_name == 'relation_ambiguity':
                    obfuscated = self._apply_relation_ambiguity(
                        obfuscated, strategy
                    )
                elif strategy_name == 'context_expansion':
                    obfuscated = self._apply_context_expansion(
                        obfuscated, strategy, knowledge_graph
                    )
                    
                applied_strategies.append(strategy_name)
                
        obfuscated.obfuscation_metadata['applied_strategies'] = applied_strategies
        obfuscated.ambiguity_score = self._calculate_ambiguity_score(obfuscated)
        
        return obfuscated
        
    def _apply_entity_substitution(
        self, 
        qa_pair: ObfuscatedQAPair, 
        strategy: Dict, 
        knowledge_graph: nx.Graph
    ) -> ObfuscatedQAPair:
        """
        实体替换策略
        用模糊描述代替具体实体名称
        """
        question = qa_pair.question
        substitutions_made = []
        
        # 识别问题中的实体
        entities_in_question = self._extract_entities_from_question(
            question, qa_pair.subgraph.graph
        )
        
        for entity in entities_in_question:
            # 确定实体类型
            entity_type = self._get_entity_type(entity, knowledge_graph)
            
            if entity_type in self.entity_substitutions:
                # 选择一个模糊描述
                substitution = random.choice(self.entity_substitutions[entity_type])
                
                # 执行替换
                question = question.replace(entity, substitution)
                substitutions_made.append({
                    'original': entity,
                    'substitution': substitution,
                    'type': entity_type
                })
                
        qa_pair.question = question
        qa_pair.obfuscation_metadata['entity_substitutions'] = substitutions_made
        
        return qa_pair
        
    def _apply_information_injection(
        self, 
        qa_pair: ObfuscatedQAPair, 
        strategy: Dict, 
        knowledge_graph: nx.Graph
    ) -> ObfuscatedQAPair:
        """
        信息注入策略
        添加相关但不必要的干扰信息
        """
        injection_types = strategy.get('types', [])
        max_injections = strategy.get('max_injections_per_question', 2)
        
        injections_made = []
        
        for _ in range(random.randint(1, max_injections)):
            injection_type = random.choice(injection_types)
            
            if injection_type == 'related_but_irrelevant':
                injection = self._inject_related_info(qa_pair, knowledge_graph)
            elif injection_type == 'similar_entities':
                injection = self._inject_similar_entities(qa_pair, knowledge_graph)
            elif injection_type == 'historical_context':
                injection = self._inject_historical_context(qa_pair)
            else:
                continue
                
            if injection:
                qa_pair.question = self._integrate_injection(
                    qa_pair.question, injection
                )
                injections_made.append({
                    'type': injection_type,
                    'content': injection
                })
                
        qa_pair.obfuscation_metadata['information_injections'] = injections_made
        
        return qa_pair
        
    def _apply_relation_ambiguity(
        self, 
        qa_pair: ObfuscatedQAPair, 
        strategy: Dict
    ) -> ObfuscatedQAPair:
        """
        关系模糊化策略
        使关系描述更加模糊
        """
        question = qa_pair.question
        ambiguities_applied = []
        
        # 识别问题中的关系
        for original_relation, ambiguous_forms in self.relation_ambiguities.items():
            if original_relation in question:
                # 选择一个模糊形式
                ambiguous_form = random.choice(ambiguous_forms)
                question = question.replace(original_relation, ambiguous_form)
                
                ambiguities_applied.append({
                    'original': original_relation,
                    'ambiguous': ambiguous_form
                })
                
        # 应用其他模糊化策略
        ambiguity_strategies = strategy.get('strategies', [])
        
        if 'indirect_reference' in ambiguity_strategies:
            question = self._make_indirect_reference(question)
            
        if 'temporal_ambiguity' in ambiguity_strategies:
            question = self._add_temporal_ambiguity(question)
            
        if 'spatial_ambiguity' in ambiguity_strategies:
            question = self._add_spatial_ambiguity(question)
            
        qa_pair.question = question
        qa_pair.obfuscation_metadata['relation_ambiguities'] = ambiguities_applied
        
        return qa_pair
        
    def _apply_context_expansion(
        self, 
        qa_pair: ObfuscatedQAPair, 
        strategy: Dict, 
        knowledge_graph: nx.Graph
    ) -> ObfuscatedQAPair:
        """
        上下文扩展策略
        添加背景信息增加复杂度
        """
        min_sentences = strategy.get('min_sentences', 2)
        max_sentences = strategy.get('max_sentences', 5)
        relevance_threshold = strategy.get('relevance_threshold', 0.6)
        
        # 生成背景句子
        num_sentences = random.randint(min_sentences, max_sentences)
        background_sentences = []
        
        # 从子图中提取相关信息
        subgraph = qa_pair.subgraph.graph
        
        for _ in range(num_sentences):
            # 随机选择一个节点
            if subgraph.nodes():
                node = random.choice(list(subgraph.nodes()))
                
                # 生成关于该节点的背景信息
                background = self._generate_background_sentence(
                    node, subgraph, knowledge_graph
                )
                
                if background:
                    background_sentences.append(background)
                    
        # 将背景信息整合到问题中
        if background_sentences:
            context = ' '.join(background_sentences)
            qa_pair.question = f"{context} 基于以上信息，{qa_pair.question}"
            
        qa_pair.obfuscation_metadata['context_expansion'] = {
            'num_sentences': len(background_sentences),
            'sentences': background_sentences
        }
        
        return qa_pair
        
    # 辅助方法
    def _extract_entities_from_question(
        self, 
        question: str, 
        subgraph: nx.Graph
    ) -> List[str]:
        """从问题中提取实体"""
        entities = []
        
        # 简单实现：检查子图中的节点是否出现在问题中
        for node in subgraph.nodes():
            if node in question:
                entities.append(node)
                
        return entities
        
    def _get_entity_type(self, entity: str, knowledge_graph: nx.Graph) -> str:
        """获取实体类型"""
        # 简单实现：基于实体名称或属性推断类型
        entity_lower = entity.lower()
        
        if any(keyword in entity_lower for keyword in ['显示', 'display', '屏幕']):
            return 'display'
        elif any(keyword in entity_lower for keyword in ['芯片', 'chip', '半导体']):
            return 'semiconductor'
        elif any(keyword in entity_lower for keyword in ['公司', 'company', 'tcl']):
            return 'manufacturer'
        elif any(keyword in entity_lower for keyword in ['技术', 'technology']):
            return 'technology'
        elif any(keyword in entity_lower for keyword in ['组件', 'component', '部件']):
            return 'component'
        elif any(keyword in entity_lower for keyword in ['标准', 'standard', 'iso']):
            return 'standard'
        elif any(keyword in entity_lower for keyword in ['工艺', 'process', '流程']):
            return 'process'
        else:
            return 'unknown'
            
    def _inject_related_info(
        self, 
        qa_pair: ObfuscatedQAPair, 
        knowledge_graph: nx.Graph
    ) -> str:
        """注入相关但不必要的信息"""
        # 从子图中选择一个与问题相关但不影响答案的信息
        subgraph = qa_pair.subgraph.graph
        
        # 找出不在证据路径中的边
        evidence_edges = set(
            (s, t) for s, _, t in qa_pair.evidence_path
        )
        
        irrelevant_edges = [
            (s, t, d) for s, t, d in subgraph.edges(data=True)
            if (s, t) not in evidence_edges and (t, s) not in evidence_edges
        ]
        
        if irrelevant_edges:
            edge = random.choice(irrelevant_edges)
            source, target, data = edge
            relation = data.get('relation', 'related_to')
            
            return f"{source}{relation}{target}，"
            
        return ""
        
    def _inject_similar_entities(
        self, 
        qa_pair: ObfuscatedQAPair, 
        knowledge_graph: nx.Graph
    ) -> str:
        """注入相似实体信息"""
        # 找出与问题中实体相似的其他实体
        entities_in_question = self._extract_entities_from_question(
            qa_pair.question, qa_pair.subgraph.graph
        )
        
        if entities_in_question:
            entity = random.choice(entities_in_question)
            entity_type = self._get_entity_type(entity, knowledge_graph)
            
            # 生成相似实体的描述
            similar_descriptions = {
                'display': '类似的显示产品还包括OLED和QLED技术，',
                'semiconductor': '其他半导体制造商也在开发类似技术，',
                'manufacturer': '行业内还有多家竞争对手，',
                'technology': '相关技术还包括多种变体，'
            }
            
            return similar_descriptions.get(entity_type, '')
            
        return ""
        
    def _inject_historical_context(self, qa_pair: ObfuscatedQAPair) -> str:
        """注入历史背景信息"""
        contexts = [
            "在过去几年的发展中，",
            "根据行业发展趋势，",
            "从技术演进的角度看，",
            "考虑到市场竞争格局，"
        ]
        
        return random.choice(contexts)
        
    def _integrate_injection(self, question: str, injection: str) -> str:
        """将注入的信息整合到问题中"""
        if not injection:
            return question
            
        # 随机选择注入位置
        positions = ['beginning', 'middle', 'end']
        position = random.choice(positions)
        
        if position == 'beginning':
            return f"{injection}{question}"
        elif position == 'end':
            return f"{question} {injection}"
        else:
            # 在句子中间插入
            words = question.split()
            if len(words) > 2:
                insert_pos = random.randint(1, len(words) - 1)
                words.insert(insert_pos, injection)
                return ' '.join(words)
            else:
                return f"{injection}{question}"
                
    def _make_indirect_reference(self, question: str) -> str:
        """使引用更加间接"""
        indirect_patterns = [
            (r'(\w+)的', r'与\1相关的'),
            (r'是什么', r'可能是什么'),
            (r'哪些', r'哪些可能的'),
            (r'多少', r'大约多少')
        ]
        
        for pattern, replacement in indirect_patterns:
            question = re.sub(pattern, replacement, question)
            
        return question
        
    def _add_temporal_ambiguity(self, question: str) -> str:
        """添加时间模糊性"""
        temporal_phrases = [
            "在最近的发展中，",
            "根据当前情况，",
            "考虑到时间因素，"
        ]
        
        if random.random() < 0.5:
            phrase = random.choice(temporal_phrases)
            return f"{phrase}{question}"
            
        return question
        
    def _add_spatial_ambiguity(self, question: str) -> str:
        """添加空间模糊性"""
        spatial_phrases = [
            "在某些地区，",
            "从全球范围看，",
            "在特定市场中，"
        ]
        
        if random.random() < 0.5:
            phrase = random.choice(spatial_phrases)
            return f"{phrase}{question}"
            
        return question
        
    def _generate_background_sentence(
        self, 
        node: str, 
        subgraph: nx.Graph, 
        knowledge_graph: nx.Graph
    ) -> str:
        """生成关于节点的背景句子"""
        # 获取节点的邻居和关系
        neighbors = list(subgraph.neighbors(node))
        
        if neighbors:
            neighbor = random.choice(neighbors)
            if subgraph.has_edge(node, neighbor):
                edge_data = subgraph[node][neighbor]
                relation = edge_data.get('relation', 'related_to')
                
                templates = [
                    f"{node}{relation}{neighbor}。",
                    f"值得注意的是，{node}与{neighbor}存在{relation}关系。",
                    f"在行业中，{node}通常{relation}{neighbor}。"
                ]
                
                return random.choice(templates)
                
        return ""
        
    def _calculate_ambiguity_score(self, qa_pair: ObfuscatedQAPair) -> float:
        """计算模糊度分数"""
        score = 0.0
        
        # 基于应用的策略计算
        strategies = qa_pair.obfuscation_metadata.get('applied_strategies', [])
        strategy_weights = {
            'entity_substitution': 0.2,
            'information_injection': 0.3,
            'relation_ambiguity': 0.25,
            'context_expansion': 0.25
        }
        
        for strategy in strategies:
            score += strategy_weights.get(strategy, 0.1)
            
        # 基于具体的模糊化数量
        num_substitutions = len(
            qa_pair.obfuscation_metadata.get('entity_substitutions', [])
        )
        num_injections = len(
            qa_pair.obfuscation_metadata.get('information_injections', [])
        )
        
        score += num_substitutions * 0.1
        score += num_injections * 0.15
        
        # 归一化到0-1范围
        return min(score, 1.0)
        
    def _validate_obfuscation(self, qa_pair: ObfuscatedQAPair) -> bool:
        """验证模糊化的质量"""
        if not self.validation_config:
            return True
            
        # 检查模糊度级别
        if self.validation_config.get('check_ambiguity_level', True):
            max_ambiguity = self.validation_config.get('max_ambiguity_score', 0.8)
            if qa_pair.ambiguity_score > max_ambiguity:
                return False
                
        # 确保问题仍然可以回答
        if self.validation_config.get('ensure_unique_answer', True):
            # 简单检查：问题不应该完全失去原意
            if len(qa_pair.question) < 10:
                return False
                
            # 检查是否还包含关键信息
            if not any(word in qa_pair.question for word in ['什么', '哪些', '多少', '？']):
                return False
                
        return True
        
    def _convert_to_obfuscated(self, qa_pair: QAPair) -> ObfuscatedQAPair:
        """将普通QAPair转换为ObfuscatedQAPair"""
        return ObfuscatedQAPair(
            question=qa_pair.question,
            answer=qa_pair.answer,
            question_type=qa_pair.question_type,
            subgraph=qa_pair.subgraph,
            evidence_path=qa_pair.evidence_path,
            difficulty=qa_pair.difficulty,
            metadata=qa_pair.metadata,
            original_question=qa_pair.question,
            obfuscation_metadata={},
            ambiguity_score=0.0
        )