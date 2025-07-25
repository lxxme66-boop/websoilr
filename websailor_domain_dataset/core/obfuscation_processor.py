#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Obfuscation Processor - 模糊化处理器 (WebSailor核心思想3)
模糊描述中间实体或关系，添加冗余或干扰信息，使问题信息密度高但精确信息少

WebSailor核心思想3：模糊化处理
- 模糊描述中间实体或关系(例如"这位领导人"代指子图中多个可能节点)
- 添加冗余或干扰信息,使问题信息密度高但精确信息少
- 增加推理难度，模拟真实世界的不确定性

该模块实现：
1. 实体匿名化：用模糊描述替代具体实体名
2. 关系模糊化：模糊描述实体间的具体关系
3. 噪声注入：添加干扰信息和冗余描述
4. 信息隐藏：隐藏部分关键信息，增加推理难度
"""

import logging
import random
import re
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

from .question_generator import QuestionAnswer


class ObfuscatedQuestionAnswer:
    """模糊化问答对类"""
    
    def __init__(self, original_qa: QuestionAnswer, obfuscated_question: str, 
                 obfuscated_answer: str, obfuscation_type: str, 
                 obfuscation_metadata: Dict[str, Any] = None):
        """
        初始化模糊化问答对
        
        Args:
            original_qa: 原始问答对
            obfuscated_question: 模糊化后的问题
            obfuscated_answer: 模糊化后的答案
            obfuscation_type: 模糊化类型
            obfuscation_metadata: 模糊化元数据
        """
        self.original_qa = original_qa
        self.obfuscated_question = obfuscated_question
        self.obfuscated_answer = obfuscated_answer
        self.obfuscation_type = obfuscation_type
        self.obfuscation_metadata = obfuscation_metadata or {}
        
        # 继承原始问答对的基本信息
        self.question_type = original_qa.question_type
        self.subgraph = original_qa.subgraph
        
        # 计算模糊化特征
        self._compute_obfuscation_features()
    
    def _compute_obfuscation_features(self):
        """计算模糊化特征"""
        self.obfuscation_features = {
            "obfuscation_ratio": self._calculate_obfuscation_ratio(),
            "noise_level": self._calculate_noise_level(),
            "ambiguity_score": self._calculate_ambiguity_score(),
            "information_density": self._calculate_information_density(),
            "reasoning_difficulty": self._calculate_reasoning_difficulty()
        }
    
    def _calculate_obfuscation_ratio(self) -> float:
        """计算模糊化比例"""
        original_entities = set(re.findall(r'\b\w+\b', self.original_qa.question))
        obfuscated_entities = set(re.findall(r'\b\w+\b', self.obfuscated_question))
        
        if not original_entities:
            return 0.0
        
        # 计算被替换的实体比例
        preserved_entities = original_entities.intersection(obfuscated_entities)
        return 1.0 - (len(preserved_entities) / len(original_entities))
    
    def _calculate_noise_level(self) -> float:
        """计算噪声水平"""
        original_length = len(self.original_qa.question)
        obfuscated_length = len(self.obfuscated_question)
        
        if original_length == 0:
            return 0.0
        
        # 长度增加比例作为噪声水平的指标
        return max(0.0, (obfuscated_length - original_length) / original_length)
    
    def _calculate_ambiguity_score(self) -> float:
        """计算歧义分数"""
        # 统计模糊指代词的数量
        ambiguous_patterns = [
            r'这\w*', r'该\w*', r'某\w*', r'相关\w*', r'有关\w*',
            r'类似\w*', r'此类\w*', r'上述\w*', r'以上\w*'
        ]
        
        ambiguous_count = 0
        for pattern in ambiguous_patterns:
            ambiguous_count += len(re.findall(pattern, self.obfuscated_question))
        
        # 归一化到0-1范围
        return min(1.0, ambiguous_count / 5.0)
    
    def _calculate_information_density(self) -> float:
        """计算信息密度"""
        # 信息密度 = 有效信息量 / 总文本长度
        effective_info = len([word for word in self.obfuscated_question.split() 
                            if not any(pattern in word for pattern in ['这', '该', '某', '相关'])])
        total_length = len(self.obfuscated_question.split())
        
        return effective_info / total_length if total_length > 0 else 0.0
    
    def _calculate_reasoning_difficulty(self) -> float:
        """计算推理难度"""
        # 基于多个因素计算推理难度
        base_difficulty = self.original_qa.features.get("complexity_score", 1.0)
        
        # 模糊化增加的难度
        obfuscation_difficulty = (
            self.obfuscation_features.get("obfuscation_ratio", 0) * 0.5 +
            self.obfuscation_features.get("ambiguity_score", 0) * 0.3 +
            self.obfuscation_features.get("noise_level", 0) * 0.2
        )
        
        return base_difficulty * (1 + obfuscation_difficulty)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = self.original_qa.to_dict()
        result.update({
            "obfuscated_question": self.obfuscated_question,
            "obfuscated_answer": self.obfuscated_answer,
            "obfuscation_type": self.obfuscation_type,
            "obfuscation_features": self.obfuscation_features,
            "obfuscation_metadata": self.obfuscation_metadata
        })
        return result


class ObfuscationProcessor:
    """模糊化处理器 - WebSailor核心思想3"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模糊化处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.obfuscation_config = config.get("obfuscation_processing", {})
        self.domain_config = config.get("tcl_industry_domain", {})
        
        # 加载模糊化策略
        self.obfuscation_strategies = self._load_obfuscation_strategies()
        
        # 加载领域特定信息
        self.entity_types = self.domain_config.get("entity_types", [])
        self.relation_types = self.domain_config.get("relation_types", [])
        
        logging.info("模糊化处理器初始化完成")
    
    def _load_obfuscation_strategies(self) -> Dict[str, Dict[str, Any]]:
        """加载模糊化策略配置"""
        strategies = {}
        
        for strategy_config in self.obfuscation_config.get("obfuscation_strategies", []):
            name = strategy_config["name"]
            strategies[name] = {
                "description": strategy_config["description"],
                "weight": strategy_config["weight"],
                "parameters": strategy_config["parameters"],
                "processor": self._get_processor_function(name)
            }
        
        return strategies
    
    def _get_processor_function(self, strategy_name: str):
        """获取对应的模糊化处理函数"""
        processor_mapping = {
            "entity_anonymization": self._apply_entity_anonymization,
            "relation_obfuscation": self._apply_relation_obfuscation,
            "noise_injection": self._apply_noise_injection,
            "information_hiding": self._apply_information_hiding
        }
        
        return processor_mapping.get(strategy_name, self._apply_entity_anonymization)
    
    def process_qa_pairs(self, qa_pairs: List[QuestionAnswer], 
                        obfuscation_strategies: List[str] = None,
                        obfuscation_ratio: float = 0.7) -> List[ObfuscatedQuestionAnswer]:
        """
        处理问答对进行模糊化 - WebSailor核心思想3的实现
        
        Args:
            qa_pairs: 原始问答对列表
            obfuscation_strategies: 要应用的模糊化策略
            obfuscation_ratio: 模糊化比例
            
        Returns:
            模糊化后的问答对列表
        """
        logging.info(f"开始模糊化处理 {len(qa_pairs)} 个问答对...")
        
        obfuscated_qa_pairs = []
        
        for qa_pair in qa_pairs:
            try:
                # 决定是否对当前问答对进行模糊化
                if random.random() < obfuscation_ratio:
                    obfuscated_qa = self._obfuscate_single_qa(qa_pair, obfuscation_strategies)
                    if obfuscated_qa:
                        obfuscated_qa_pairs.append(obfuscated_qa)
                    else:
                        # 如果模糊化失败，保留原始问答对
                        obfuscated_qa_pairs.append(self._create_unobfuscated_qa(qa_pair))
                else:
                    # 不进行模糊化，但仍包装为ObfuscatedQuestionAnswer
                    obfuscated_qa_pairs.append(self._create_unobfuscated_qa(qa_pair))
                    
            except Exception as e:
                logging.warning(f"模糊化问答对时出错: {e}")
                # 出错时保留原始问答对
                obfuscated_qa_pairs.append(self._create_unobfuscated_qa(qa_pair))
        
        logging.info(f"模糊化处理完成，共处理 {len(obfuscated_qa_pairs)} 个问答对")
        return obfuscated_qa_pairs
    
    def _obfuscate_single_qa(self, qa_pair: QuestionAnswer, 
                           strategies: List[str] = None) -> Optional[ObfuscatedQuestionAnswer]:
        """对单个问答对进行模糊化"""
        if not strategies:
            strategies = list(self.obfuscation_strategies.keys())
        
        # 随机选择一个或多个策略
        selected_strategies = self._select_obfuscation_strategies(strategies)
        
        obfuscated_question = qa_pair.question
        obfuscated_answer = qa_pair.answer
        combined_metadata = {}
        
        # 依次应用选中的策略
        for strategy_name in selected_strategies:
            if strategy_name in self.obfuscation_strategies:
                strategy = self.obfuscation_strategies[strategy_name]
                processor_func = strategy["processor"]
                parameters = strategy["parameters"]
                
                try:
                    result = processor_func(
                        obfuscated_question, 
                        obfuscated_answer, 
                        qa_pair, 
                        parameters
                    )
                    
                    if result:
                        obfuscated_question, obfuscated_answer, metadata = result
                        combined_metadata.update(metadata)
                        
                except Exception as e:
                    logging.warning(f"应用策略 {strategy_name} 时出错: {e}")
        
        # 创建模糊化问答对
        obfuscation_type = "+".join(selected_strategies)
        
        return ObfuscatedQuestionAnswer(
            original_qa=qa_pair,
            obfuscated_question=obfuscated_question,
            obfuscated_answer=obfuscated_answer,
            obfuscation_type=obfuscation_type,
            obfuscation_metadata=combined_metadata
        )
    
    def _select_obfuscation_strategies(self, available_strategies: List[str]) -> List[str]:
        """选择要应用的模糊化策略"""
        # 根据权重选择策略
        strategy_weights = {}
        for strategy in available_strategies:
            if strategy in self.obfuscation_strategies:
                strategy_weights[strategy] = self.obfuscation_strategies[strategy]["weight"]
        
        # 随机选择1-2个策略
        num_strategies = random.randint(1, min(2, len(strategy_weights)))
        
        # 按权重选择
        selected = []
        remaining_strategies = list(strategy_weights.keys())
        
        for _ in range(num_strategies):
            if not remaining_strategies:
                break
            
            # 计算权重
            weights = [strategy_weights[s] for s in remaining_strategies]
            total_weight = sum(weights)
            
            if total_weight == 0:
                selected_strategy = random.choice(remaining_strategies)
            else:
                # 按权重随机选择
                rand_val = random.uniform(0, total_weight)
                cumulative_weight = 0
                selected_strategy = remaining_strategies[0]
                
                for i, weight in enumerate(weights):
                    cumulative_weight += weight
                    if rand_val <= cumulative_weight:
                        selected_strategy = remaining_strategies[i]
                        break
            
            selected.append(selected_strategy)
            remaining_strategies.remove(selected_strategy)
        
        return selected
    
    def _apply_entity_anonymization(self, question: str, answer: str, 
                                   qa_pair: QuestionAnswer, 
                                   parameters: Dict[str, Any]) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """
        应用实体匿名化策略
        
        用模糊描述替代具体实体名
        """
        anonymization_patterns = parameters.get("anonymization_patterns", [])
        preserve_type_info = parameters.get("preserve_type_info", True)
        
        # 获取子图中的实体信息
        entities_info = {}
        for node in qa_pair.subgraph.nodes:
            node_data = qa_pair.subgraph.graph.nodes.get(node, {})
            entity_type = node_data.get("type", "实体")
            entities_info[node] = entity_type
        
        obfuscated_question = question
        obfuscated_answer = answer
        anonymized_entities = {}
        
        # 随机选择一些实体进行匿名化
        entities_to_anonymize = random.sample(
            list(entities_info.keys()), 
            min(random.randint(1, 3), len(entities_info))
        )
        
        for entity in entities_to_anonymize:
            entity_type = entities_info[entity]
            
            # 选择合适的匿名化模式
            if preserve_type_info:
                anonymous_ref = self._get_type_specific_anonymization(entity_type, anonymization_patterns)
            else:
                anonymous_ref = random.choice(anonymization_patterns)
            
            # 替换实体名
            obfuscated_question = obfuscated_question.replace(entity, anonymous_ref)
            obfuscated_answer = obfuscated_answer.replace(entity, anonymous_ref)
            
            anonymized_entities[entity] = anonymous_ref
        
        metadata = {
            "anonymized_entities": anonymized_entities,
            "preserve_type_info": preserve_type_info
        }
        
        return obfuscated_question, obfuscated_answer, metadata
    
    def _apply_relation_obfuscation(self, question: str, answer: str, 
                                   qa_pair: QuestionAnswer, 
                                   parameters: Dict[str, Any]) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """
        应用关系模糊化策略
        
        模糊描述实体间的具体关系
        """
        obfuscation_patterns = parameters.get("obfuscation_patterns", [])
        
        obfuscated_question = question
        obfuscated_answer = answer
        obfuscated_relations = {}
        
        # 识别问题中的关系词
        relation_words = []
        for relation_type in self.relation_types:
            if relation_type in question:
                relation_words.append(relation_type)
        
        # 随机选择一些关系进行模糊化
        relations_to_obfuscate = random.sample(
            relation_words, 
            min(random.randint(1, 2), len(relation_words))
        )
        
        for relation in relations_to_obfuscate:
            # 选择模糊化模式
            obfuscated_relation = random.choice(obfuscation_patterns)
            
            # 替换关系词
            obfuscated_question = obfuscated_question.replace(relation, obfuscated_relation)
            obfuscated_answer = obfuscated_answer.replace(relation, obfuscated_relation)
            
            obfuscated_relations[relation] = obfuscated_relation
        
        metadata = {
            "obfuscated_relations": obfuscated_relations
        }
        
        return obfuscated_question, obfuscated_answer, metadata
    
    def _apply_noise_injection(self, question: str, answer: str, 
                              qa_pair: QuestionAnswer, 
                              parameters: Dict[str, Any]) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """
        应用噪声注入策略
        
        添加干扰信息和冗余描述
        """
        noise_types = parameters.get("noise_types", [])
        noise_ratio = parameters.get("noise_ratio", 0.3)
        
        obfuscated_question = question
        obfuscated_answer = answer
        injected_noise = []
        
        # 随机选择噪声类型
        selected_noise_types = random.sample(
            noise_types, 
            min(random.randint(1, 2), len(noise_types))
        )
        
        for noise_type in selected_noise_types:
            if noise_type == "irrelevant_entities":
                # 添加无关实体
                noise_text = self._generate_irrelevant_entities(qa_pair)
                if noise_text:
                    obfuscated_question = self._insert_noise_text(obfuscated_question, noise_text)
                    injected_noise.append(f"irrelevant_entities: {noise_text}")
            
            elif noise_type == "redundant_descriptions":
                # 添加冗余描述
                noise_text = self._generate_redundant_descriptions(qa_pair)
                if noise_text:
                    obfuscated_question = self._insert_noise_text(obfuscated_question, noise_text)
                    injected_noise.append(f"redundant_descriptions: {noise_text}")
            
            elif noise_type == "ambiguous_references":
                # 添加歧义引用
                noise_text = self._generate_ambiguous_references(qa_pair)
                if noise_text:
                    obfuscated_question = self._insert_noise_text(obfuscated_question, noise_text)
                    injected_noise.append(f"ambiguous_references: {noise_text}")
            
            elif noise_type == "temporal_confusion":
                # 添加时间混淆
                noise_text = self._generate_temporal_confusion()
                if noise_text:
                    obfuscated_question = self._insert_noise_text(obfuscated_question, noise_text)
                    injected_noise.append(f"temporal_confusion: {noise_text}")
        
        metadata = {
            "injected_noise": injected_noise,
            "noise_ratio": noise_ratio
        }
        
        return obfuscated_question, obfuscated_answer, metadata
    
    def _apply_information_hiding(self, question: str, answer: str, 
                                 qa_pair: QuestionAnswer, 
                                 parameters: Dict[str, Any]) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """
        应用信息隐藏策略
        
        隐藏部分关键信息，增加推理难度
        """
        hiding_strategies = parameters.get("hiding_strategies", [])
        
        obfuscated_question = question
        obfuscated_answer = answer
        hidden_information = []
        
        # 随机选择隐藏策略
        selected_strategies = random.sample(
            hiding_strategies, 
            min(random.randint(1, 2), len(hiding_strategies))
        )
        
        for strategy in selected_strategies:
            if strategy == "implicit_relations":
                # 隐式关系：移除明确的关系词
                obfuscated_question, hidden_info = self._hide_explicit_relations(obfuscated_question)
                if hidden_info:
                    hidden_information.extend(hidden_info)
            
            elif strategy == "partial_attributes":
                # 部分属性：只给出部分属性信息
                obfuscated_question, hidden_info = self._hide_partial_attributes(obfuscated_question, qa_pair)
                if hidden_info:
                    hidden_information.extend(hidden_info)
            
            elif strategy == "contextual_clues":
                # 上下文线索：用上下文暗示代替直接描述
                obfuscated_question, hidden_info = self._add_contextual_clues(obfuscated_question, qa_pair)
                if hidden_info:
                    hidden_information.extend(hidden_info)
            
            elif strategy == "indirect_references":
                # 间接引用：用间接方式引用实体
                obfuscated_question, hidden_info = self._add_indirect_references(obfuscated_question, qa_pair)
                if hidden_info:
                    hidden_information.extend(hidden_info)
        
        metadata = {
            "hidden_information": hidden_information,
            "hiding_strategies": selected_strategies
        }
        
        return obfuscated_question, obfuscated_answer, metadata
    
    def _get_type_specific_anonymization(self, entity_type: str, patterns: List[str]) -> str:
        """根据实体类型获取特定的匿名化模式"""
        type_mapping = {
            "产品": ["该产品", "这款设备", "此类产品"],
            "技术": ["这项技术", "该技术方案", "相关技术"],
            "公司": ["这家公司", "该企业", "相关厂商"],
            "专家": ["这位专家", "该专业人士", "相关人员"],
            "专利": ["这项专利", "相关专利", "该知识产权"],
            "标准": ["相关标准", "该规范", "这项标准"]
        }
        
        specific_patterns = type_mapping.get(entity_type, patterns)
        return random.choice(specific_patterns) if specific_patterns else random.choice(patterns)
    
    def _generate_irrelevant_entities(self, qa_pair: QuestionAnswer) -> Optional[str]:
        """生成无关实体"""
        # 从子图外选择一些实体作为干扰
        irrelevant_entities = ["某知名品牌", "相关供应商", "行业领导者", "技术先驱"]
        selected = random.choice(irrelevant_entities)
        return f"除了{selected}之外，"
    
    def _generate_redundant_descriptions(self, qa_pair: QuestionAnswer) -> Optional[str]:
        """生成冗余描述"""
        redundant_phrases = [
            "在当前市场环境下，",
            "考虑到行业发展趋势，",
            "从技术角度来看，",
            "在相关领域中，",
            "根据最新信息，"
        ]
        return random.choice(redundant_phrases)
    
    def _generate_ambiguous_references(self, qa_pair: QuestionAnswer) -> Optional[str]:
        """生成歧义引用"""
        ambiguous_phrases = [
            "如前所述的",
            "上述提到的",
            "相关的",
            "类似的",
            "对应的"
        ]
        return random.choice(ambiguous_phrases)
    
    def _generate_temporal_confusion(self) -> Optional[str]:
        """生成时间混淆"""
        temporal_phrases = [
            "在过去几年中，",
            "最近，",
            "不久前，",
            "在某个时期，",
            "历史上，"
        ]
        return random.choice(temporal_phrases)
    
    def _insert_noise_text(self, original_text: str, noise_text: str) -> str:
        """在原文中插入噪声文本"""
        # 随机选择插入位置
        insertion_positions = [0, len(original_text) // 2, len(original_text)]
        position = random.choice(insertion_positions)
        
        if position == 0:
            return noise_text + original_text
        elif position == len(original_text):
            return original_text + " " + noise_text
        else:
            return original_text[:position] + " " + noise_text + " " + original_text[position:]
    
    def _hide_explicit_relations(self, question: str) -> Tuple[str, List[str]]:
        """隐藏明确的关系"""
        explicit_relations = ["开发", "应用", "合作", "竞争", "投资", "收购"]
        hidden_relations = []
        
        obfuscated_question = question
        for relation in explicit_relations:
            if relation in question:
                # 用更模糊的词替换
                vague_replacements = ["涉及", "关联", "相关", "连接"]
                replacement = random.choice(vague_replacements)
                obfuscated_question = obfuscated_question.replace(relation, replacement)
                hidden_relations.append(f"hidden_relation: {relation} -> {replacement}")
        
        return obfuscated_question, hidden_relations
    
    def _hide_partial_attributes(self, question: str, qa_pair: QuestionAnswer) -> Tuple[str, List[str]]:
        """隐藏部分属性"""
        # 简单实现：用"某些特性"替换具体属性
        attribute_words = ["特性", "功能", "性能", "优势", "特点"]
        hidden_attributes = []
        
        obfuscated_question = question
        for attr in attribute_words:
            if attr in question:
                vague_attr = "某些" + attr
                obfuscated_question = obfuscated_question.replace(attr, vague_attr)
                hidden_attributes.append(f"partial_attribute: {attr} -> {vague_attr}")
        
        return obfuscated_question, hidden_attributes
    
    def _add_contextual_clues(self, question: str, qa_pair: QuestionAnswer) -> Tuple[str, List[str]]:
        """添加上下文线索"""
        contextual_clues = []
        
        # 添加一些上下文暗示
        context_phrases = [
            "在相关背景下，",
            "考虑到具体情况，",
            "在这种环境中，"
        ]
        
        selected_phrase = random.choice(context_phrases)
        obfuscated_question = selected_phrase + question
        contextual_clues.append(f"contextual_clue: {selected_phrase}")
        
        return obfuscated_question, contextual_clues
    
    def _add_indirect_references(self, question: str, qa_pair: QuestionAnswer) -> Tuple[str, List[str]]:
        """添加间接引用"""
        indirect_refs = []
        
        # 用间接方式引用实体
        entities = qa_pair.subgraph.nodes
        if entities:
            entity = random.choice(entities)
            if entity in question:
                indirect_ref = "与此相关的实体"
                question = question.replace(entity, indirect_ref, 1)  # 只替换第一个
                indirect_refs.append(f"indirect_reference: {entity} -> {indirect_ref}")
        
        return question, indirect_refs
    
    def _create_unobfuscated_qa(self, qa_pair: QuestionAnswer) -> ObfuscatedQuestionAnswer:
        """为未模糊化的问答对创建包装"""
        return ObfuscatedQuestionAnswer(
            original_qa=qa_pair,
            obfuscated_question=qa_pair.question,
            obfuscated_answer=qa_pair.answer,
            obfuscation_type="none",
            obfuscation_metadata={"obfuscated": False}
        )