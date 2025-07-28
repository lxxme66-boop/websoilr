"""
数据综合器
将所有生成的组件综合成最终数据集
"""

import json
import logging
import hashlib
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass, asdict
import networkx as nx
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DatasetEntry:
    """数据集条目"""
    entry_id: str
    question: str
    answer: str
    question_type: str
    difficulty: str
    obfuscated_question: str
    obfuscated_answer: str
    trajectory: Dict[str, Any]
    source_subgraph: Dict[str, Any]
    metadata: Dict[str, Any]


class DataSynthesizer:
    """
    数据综合器
    整合所有生成的组件，进行质量控制和数据增强
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_formats = config.get('output_formats', {})
        self.quality_control = config.get('quality_control', {})
        self.augmentation = config.get('augmentation', {})
        
        # 质量控制参数
        self.min_answer_length = self.quality_control.get('min_answer_length', 10)
        self.max_answer_length = self.quality_control.get('max_answer_length', 500)
        self.check_factual_consistency = self.quality_control.get('check_factual_consistency', True)
        self.remove_duplicates = self.quality_control.get('remove_duplicates', True)
        self.similarity_threshold = self.quality_control.get('similarity_threshold', 0.9)
        
        # 数据增强参数
        self.paraphrase_questions = self.augmentation.get('paraphrase_questions', True)
        self.paraphrase_count = self.augmentation.get('paraphrase_count', 2)
        self.generate_negative_examples = self.augmentation.get('generate_negative_examples', True)
        self.negative_ratio = self.augmentation.get('negative_ratio', 0.2)
        
    def synthesize(self, qa_pairs: List[Any], trajectories: List[Any],
                   knowledge_graph: nx.Graph, subgraphs: List[Any]) -> Dict[str, Any]:
        """
        综合所有组件生成最终数据集
        
        Args:
            qa_pairs: 问答对列表（可能已模糊化）
            trajectories: 推理轨迹列表
            knowledge_graph: 知识图谱
            subgraphs: 子图列表
            
        Returns:
            最终数据集
        """
        self.logger.info("开始综合数据集...")
        
        # 1. 组合数据
        combined_data = self._combine_data(qa_pairs, trajectories, subgraphs)
        
        # 2. 质量控制
        if self.quality_control:
            combined_data = self._apply_quality_control(combined_data, knowledge_graph)
        
        # 3. 数据增强
        if self.augmentation:
            combined_data = self._apply_augmentation(combined_data, knowledge_graph)
        
        # 4. 格式化输出
        final_dataset = self._format_output(combined_data, knowledge_graph)
        
        self.logger.info(f"数据集综合完成，共{len(final_dataset['qa_pairs'])}个条目")
        
        return final_dataset
    
    def _combine_data(self, qa_pairs: List[Any], trajectories: List[Any],
                      subgraphs: List[Any]) -> List[DatasetEntry]:
        """组合所有数据组件"""
        combined_data = []
        
        # 创建轨迹映射
        trajectory_map = {}
        for traj in trajectories:
            qa_id = id(traj.qa_pair)
            if qa_id not in trajectory_map:
                trajectory_map[qa_id] = []
            trajectory_map[qa_id].append(traj)
        
        # 创建子图映射
        subgraph_map = {sg.subgraph_id: sg for sg in subgraphs}
        
        # 组合数据
        for qa in qa_pairs:
            qa_id = id(qa)
            
            # 获取对应的轨迹
            qa_trajectories = trajectory_map.get(qa_id, [])
            
            # 选择最佳轨迹（置信度最高的正确轨迹）
            best_trajectory = None
            for traj in qa_trajectories:
                if traj.is_correct:
                    if best_trajectory is None or traj.confidence_score > best_trajectory.confidence_score:
                        best_trajectory = traj
            
            if not best_trajectory and qa_trajectories:
                best_trajectory = qa_trajectories[0]
            
            # 获取源子图
            source_subgraph = None
            if hasattr(qa, 'subgraph_id'):
                source_subgraph = subgraph_map.get(qa.subgraph_id)
            
            # 创建数据集条目
            entry = DatasetEntry(
                entry_id=self._generate_entry_id(qa),
                question=getattr(qa, 'question', ''),
                answer=getattr(qa, 'answer', ''),
                question_type=getattr(qa, 'question_type', 'unknown'),
                difficulty=getattr(qa, 'difficulty', 'medium'),
                obfuscated_question=getattr(qa, 'obfuscated_question', qa.question),
                obfuscated_answer=getattr(qa, 'obfuscated_answer', qa.answer),
                trajectory=self._serialize_trajectory(best_trajectory) if best_trajectory else {},
                source_subgraph=self._serialize_subgraph(source_subgraph) if source_subgraph else {},
                metadata={
                    'entities': getattr(qa, 'entities', []),
                    'relations': getattr(qa, 'relations', []),
                    'obfuscation_applied': hasattr(qa, 'obfuscation_strategies'),
                    'alternative_trajectories': len(qa_trajectories) - 1 if best_trajectory else len(qa_trajectories)
                }
            )
            
            combined_data.append(entry)
        
        return combined_data
    
    def _apply_quality_control(self, data: List[DatasetEntry],
                              knowledge_graph: nx.Graph) -> List[DatasetEntry]:
        """应用质量控制"""
        self.logger.info("应用质量控制...")
        
        # 1. 长度过滤
        data = self._filter_by_length(data)
        
        # 2. 去重
        if self.remove_duplicates:
            data = self._remove_duplicates(data)
        
        # 3. 事实一致性检查
        if self.check_factual_consistency:
            data = self._check_factual_consistency(data, knowledge_graph)
        
        # 4. 答案完整性检查
        data = self._check_answer_completeness(data)
        
        return data
    
    def _filter_by_length(self, data: List[DatasetEntry]) -> List[DatasetEntry]:
        """根据答案长度过滤"""
        filtered_data = []
        
        for entry in data:
            answer_length = len(entry.obfuscated_answer)
            
            if self.min_answer_length <= answer_length <= self.max_answer_length:
                filtered_data.append(entry)
            else:
                self.logger.debug(f"过滤掉答案长度为{answer_length}的条目")
        
        self.logger.info(f"长度过滤后剩余{len(filtered_data)}个条目")
        return filtered_data
    
    def _remove_duplicates(self, data: List[DatasetEntry]) -> List[DatasetEntry]:
        """去除重复的问答对"""
        if not data:
            return data
        
        # 使用TF-IDF计算问题相似度
        questions = [entry.obfuscated_question for entry in data]
        
        # 向量化
        vectorizer = TfidfVectorizer(max_features=1000)
        try:
            question_vectors = vectorizer.fit_transform(questions)
            
            # 计算相似度矩阵
            similarity_matrix = cosine_similarity(question_vectors)
            
            # 标记重复项
            to_remove = set()
            for i in range(len(data)):
                if i in to_remove:
                    continue
                    
                for j in range(i + 1, len(data)):
                    if similarity_matrix[i][j] > self.similarity_threshold:
                        # 保留更复杂的那个
                        if len(data[i].obfuscated_answer) < len(data[j].obfuscated_answer):
                            to_remove.add(i)
                        else:
                            to_remove.add(j)
            
            # 过滤重复项
            filtered_data = [entry for i, entry in enumerate(data) if i not in to_remove]
            
            self.logger.info(f"去重后剩余{len(filtered_data)}个条目")
            return filtered_data
            
        except Exception as e:
            self.logger.warning(f"去重过程出错: {e}，返回原始数据")
            return data
    
    def _check_factual_consistency(self, data: List[DatasetEntry],
                                  knowledge_graph: nx.Graph) -> List[DatasetEntry]:
        """检查事实一致性"""
        filtered_data = []
        
        for entry in data:
            # 检查实体是否存在于知识图谱中
            entities = entry.metadata.get('entities', [])
            
            if entities:
                # 至少一半的实体应该在知识图谱中
                existing_entities = [e for e in entities if e in knowledge_graph]
                
                if len(existing_entities) >= len(entities) * 0.5:
                    filtered_data.append(entry)
                else:
                    self.logger.debug(f"过滤掉实体不一致的条目: {entry.entry_id}")
            else:
                # 没有实体信息的保留
                filtered_data.append(entry)
        
        self.logger.info(f"事实一致性检查后剩余{len(filtered_data)}个条目")
        return filtered_data
    
    def _check_answer_completeness(self, data: List[DatasetEntry]) -> List[DatasetEntry]:
        """检查答案完整性"""
        filtered_data = []
        
        for entry in data:
            # 检查答案是否包含必要的信息
            answer = entry.obfuscated_answer
            
            # 基本完整性检查
            if answer and not answer.isspace():
                # 检查是否有实质内容
                if len(answer.split()) >= 3:  # 至少3个词
                    filtered_data.append(entry)
                else:
                    self.logger.debug(f"过滤掉答案过短的条目: {entry.entry_id}")
            else:
                self.logger.debug(f"过滤掉空答案的条目: {entry.entry_id}")
        
        return filtered_data
    
    def _apply_augmentation(self, data: List[DatasetEntry],
                           knowledge_graph: nx.Graph) -> List[DatasetEntry]:
        """应用数据增强"""
        self.logger.info("应用数据增强...")
        
        augmented_data = list(data)  # 复制原始数据
        
        # 1. 问题改写
        if self.paraphrase_questions:
            paraphrased = self._paraphrase_questions(data)
            augmented_data.extend(paraphrased)
        
        # 2. 生成负例
        if self.generate_negative_examples:
            negatives = self._generate_negative_examples(data, knowledge_graph)
            augmented_data.extend(negatives)
        
        self.logger.info(f"数据增强后共{len(augmented_data)}个条目")
        return augmented_data
    
    def _paraphrase_questions(self, data: List[DatasetEntry]) -> List[DatasetEntry]:
        """改写问题"""
        paraphrased_entries = []
        
        paraphrase_templates = [
            ("什么是", "请解释"),
            ("如何", "怎样"),
            ("为什么", "什么原因导致"),
            ("哪些", "有哪些"),
            ("是什么", "指的是什么"),
            ("有什么", "包含什么")
        ]
        
        for entry in data:
            for i in range(self.paraphrase_count):
                # 创建改写版本
                paraphrased_question = entry.obfuscated_question
                
                # 应用模板替换
                for old_phrase, new_phrase in paraphrase_templates:
                    if old_phrase in paraphrased_question:
                        paraphrased_question = paraphrased_question.replace(
                            old_phrase, new_phrase, 1
                        )
                        break
                
                # 如果问题确实被改写了
                if paraphrased_question != entry.obfuscated_question:
                    new_entry = DatasetEntry(
                        entry_id=f"{entry.entry_id}_para_{i}",
                        question=entry.question,
                        answer=entry.answer,
                        question_type=entry.question_type,
                        difficulty=entry.difficulty,
                        obfuscated_question=paraphrased_question,
                        obfuscated_answer=entry.obfuscated_answer,
                        trajectory=entry.trajectory,
                        source_subgraph=entry.source_subgraph,
                        metadata={**entry.metadata, 'is_paraphrase': True}
                    )
                    paraphrased_entries.append(new_entry)
        
        return paraphrased_entries
    
    def _generate_negative_examples(self, data: List[DatasetEntry],
                                   knowledge_graph: nx.Graph) -> List[DatasetEntry]:
        """生成负例（错误答案）"""
        negative_entries = []
        
        # 计算需要生成的负例数量
        num_negatives = int(len(data) * self.negative_ratio)
        
        # 随机选择条目生成负例
        import random
        selected_entries = random.sample(data, min(num_negatives, len(data)))
        
        for entry in selected_entries:
            # 生成错误答案
            wrong_answer = self._generate_wrong_answer(entry, knowledge_graph)
            
            if wrong_answer:
                new_entry = DatasetEntry(
                    entry_id=f"{entry.entry_id}_neg",
                    question=entry.question,
                    answer=wrong_answer,  # 错误答案
                    question_type=entry.question_type,
                    difficulty=entry.difficulty,
                    obfuscated_question=entry.obfuscated_question,
                    obfuscated_answer=wrong_answer,  # 错误答案
                    trajectory={},  # 负例不需要推理轨迹
                    source_subgraph=entry.source_subgraph,
                    metadata={**entry.metadata, 'is_negative': True}
                )
                negative_entries.append(new_entry)
        
        return negative_entries
    
    def _generate_wrong_answer(self, entry: DatasetEntry,
                              knowledge_graph: nx.Graph) -> str:
        """生成错误答案"""
        entities = entry.metadata.get('entities', [])
        
        if entities and entities[0] in knowledge_graph:
            # 找一个相似但错误的实体
            target_entity = entities[0]
            target_type = knowledge_graph.nodes[target_entity].get('type', '')
            
            # 找同类型的其他实体
            similar_entities = [
                n for n in knowledge_graph.nodes()
                if n != target_entity and
                knowledge_graph.nodes[n].get('type', '') == target_type
            ]
            
            if similar_entities:
                wrong_entity = random.choice(similar_entities)
                # 用错误实体替换答案中的正确实体
                wrong_answer = entry.answer.replace(target_entity, wrong_entity)
                
                if wrong_answer != entry.answer:
                    return wrong_answer
        
        # 生成一般性的错误答案
        wrong_answers = [
            "这个问题无法根据现有信息回答。",
            "没有足够的数据支持这个结论。",
            "这种情况不适用于当前场景。",
            "相关信息存在矛盾，无法确定。"
        ]
        
        return random.choice(wrong_answers)
    
    def _format_output(self, data: List[DatasetEntry],
                      knowledge_graph: nx.Graph) -> Dict[str, Any]:
        """格式化输出"""
        # QA对格式
        qa_pairs = []
        for entry in data:
            qa_format = {
                'id': entry.entry_id,
                'question': entry.obfuscated_question,
                'answer': entry.obfuscated_answer,
                'type': entry.question_type,
                'difficulty': entry.difficulty
            }
            
            # 根据配置添加元数据
            if self.output_formats['qa_pairs'].get('include_metadata'):
                qa_format['metadata'] = entry.metadata
            
            if self.output_formats['qa_pairs'].get('include_source_subgraph'):
                qa_format['source_subgraph'] = entry.source_subgraph
            
            qa_pairs.append(qa_format)
        
        # 轨迹格式
        trajectories = []
        for entry in data:
            if entry.trajectory:
                traj_format = entry.trajectory
                
                if self.output_formats['trajectories'].get('include_reasoning_type'):
                    traj_format['qa_id'] = entry.entry_id
                
                trajectories.append(traj_format)
        
        # 知识图谱格式
        kg_format = self._format_knowledge_graph(knowledge_graph)
        
        return {
            'qa_pairs': qa_pairs,
            'trajectories': trajectories,
            'knowledge_graph': kg_format
        }
    
    def _serialize_trajectory(self, trajectory: Any) -> Dict[str, Any]:
        """序列化轨迹对象"""
        if not trajectory:
            return {}
        
        return {
            'trajectory_id': trajectory.trajectory_id,
            'reasoning_pattern': trajectory.reasoning_pattern,
            'steps': [
                {
                    'step_id': step.step_id,
                    'content': step.content,
                    'type': step.step_type,
                    'confidence': step.confidence,
                    'entities': step.entities_involved,
                    'relations': step.relations_used
                }
                for step in trajectory.steps
            ],
            'is_correct': trajectory.is_correct,
            'confidence_score': trajectory.confidence_score,
            'alternative_paths_count': len(trajectory.alternative_paths)
        }
    
    def _serialize_subgraph(self, subgraph: Any) -> Dict[str, Any]:
        """序列化子图对象"""
        if not subgraph:
            return {}
        
        return {
            'subgraph_id': subgraph.subgraph_id,
            'topology_type': subgraph.topology_type,
            'nodes': subgraph.nodes,
            'edges': [(u, v, data) for u, v, data in subgraph.graph.edges(data=True)],
            'node_count': len(subgraph.nodes),
            'edge_count': subgraph.graph.number_of_edges()
        }
    
    def _format_knowledge_graph(self, knowledge_graph: nx.Graph) -> Dict[str, Any]:
        """格式化知识图谱"""
        kg_data = {
            'nodes': [],
            'edges': []
        }
        
        # 节点信息
        for node, attrs in knowledge_graph.nodes(data=True):
            node_data = {
                'id': node,
                'attributes': attrs
            }
            kg_data['nodes'].append(node_data)
        
        # 边信息
        for u, v, attrs in knowledge_graph.edges(data=True):
            edge_data = {
                'source': u,
                'target': v,
                'attributes': attrs
            }
            kg_data['edges'].append(edge_data)
        
        # 统计信息
        if self.output_formats['knowledge_graph'].get('include_statistics'):
            kg_data['statistics'] = {
                'total_nodes': knowledge_graph.number_of_nodes(),
                'total_edges': knowledge_graph.number_of_edges(),
                'density': nx.density(knowledge_graph),
                'is_connected': nx.is_connected(knowledge_graph.to_undirected()),
                'node_types': self._count_node_types(knowledge_graph),
                'edge_types': self._count_edge_types(knowledge_graph)
            }
        
        return kg_data
    
    def _count_node_types(self, graph: nx.Graph) -> Dict[str, int]:
        """统计节点类型"""
        type_counts = defaultdict(int)
        
        for node, attrs in graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            type_counts[node_type] += 1
        
        return dict(type_counts)
    
    def _count_edge_types(self, graph: nx.Graph) -> Dict[str, int]:
        """统计边类型"""
        type_counts = defaultdict(int)
        
        for u, v, attrs in graph.edges(data=True):
            edge_type = attrs.get('relation', 'unknown')
            type_counts[edge_type] += 1
        
        return dict(type_counts)
    
    def _generate_entry_id(self, qa: Any) -> str:
        """生成条目ID"""
        # 使用问题内容的哈希值作为ID
        content = f"{getattr(qa, 'question', '')}_{getattr(qa, 'answer', '')}"
        hash_obj = hashlib.md5(content.encode('utf-8'))
        return f"entry_{hash_obj.hexdigest()[:12]}"