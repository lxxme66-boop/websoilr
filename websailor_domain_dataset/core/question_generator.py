"""
问题生成器 - WebSailor核心思想实现
基于子图中节点与关系，设计QA问题
覆盖多种问题类型，从简单事实到复杂推理
"""

import random
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class QAPair:
    """问答对数据结构"""
    question: str
    answer: str
    question_type: str  # factual, reasoning, multi_hop, comparative
    complexity: str  # simple, medium, complex
    source_subgraph: Any  # 来源子图
    entities: List[str]  # 涉及的实体
    relations: List[str]  # 涉及的关系
    metadata: Dict[str, Any]  # 其他元数据


class QuestionGenerator:
    """
    WebSailor问题生成器
    核心思想：基于子图的结构和语义信息生成多样化的问题
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.question_types = config.get('question_types', {})
        self.complexity_distribution = config.get('complexity_distribution', {})
        self.ensure_answerability = config.get('ensure_answerability', True)
        self.generate_distractors = config.get('generate_distractors', True)
        
        # 添加合理性阈值
        self.validity_threshold = config.get('validity_threshold', 0.7)
        
        # 初始化质量检查模型
        self.qa_validator_model = None
        self.qa_validator_tokenizer = None
        
        # 加载问题模板
        self._load_templates()
        
    def generate_questions(self, subgraphs: List[Any]) -> List[QAPair]:
        """
        为给定的子图生成问题
        
        Args:
            subgraphs: 子图列表
            
        Returns:
            生成的问答对列表
        """
        self.logger.info(f"开始为{len(subgraphs)}个子图生成问题...")
        
        all_qa_pairs = []
        
        for subgraph in subgraphs:
            # 根据子图的拓扑类型和任务场景选择合适的问题生成策略
            qa_pairs = self._generate_for_subgraph(subgraph)
            all_qa_pairs.extend(qa_pairs)
        
        # 质量过滤
        if self.ensure_answerability:
            all_qa_pairs = self._filter_answerable(all_qa_pairs)
        
        # 生成干扰项
        if self.generate_distractors:
            all_qa_pairs = self._add_distractors(all_qa_pairs)
        
        self.logger.info(f"总共生成了{len(all_qa_pairs)}个问答对")
        return all_qa_pairs
    
    def _generate_for_subgraph(self, subgraph: Any) -> List[QAPair]:
        """为单个子图生成问题"""
        qa_pairs = []
        
        # 根据问题类型权重分配问题数量
        total_questions = self._determine_question_count(subgraph)
        type_weights = [self.question_types[qt]['weight'] 
                       for qt in self.question_types]
        type_counts = self._distribute_counts(total_questions, type_weights)
        
        # 为每种类型生成问题
        for (question_type, type_config), count in zip(
            self.question_types.items(), type_counts
        ):
            for _ in range(count):
                qa_pair = self._generate_question_by_type(
                    subgraph, question_type, type_config
                )
                if qa_pair:
                    qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def _generate_question_by_type(self, subgraph: Any, 
                                  question_type: str,
                                  type_config: Dict[str, Any]) -> Optional[QAPair]:
        """根据类型生成问题"""
        if question_type == 'factual':
            return self._generate_factual_question(subgraph, type_config)
        elif question_type == 'reasoning':
            return self._generate_reasoning_question(subgraph, type_config)
        elif question_type == 'multi_hop':
            return self._generate_multi_hop_question(subgraph, type_config)
        elif question_type == 'comparative':
            return self._generate_comparative_question(subgraph, type_config)
        else:
            self.logger.warning(f"未知的问题类型: {question_type}")
            return None
    
    def _generate_factual_question(self, subgraph: Any,
                                  type_config: Dict[str, Any]) -> Optional[QAPair]:
        """生成事实性问题"""
        graph = subgraph.graph
        templates = type_config.get('templates', [])
        
        if not templates or len(graph.nodes) == 0:
            return None
        
        # 随机选择一个节点
        node = random.choice(list(graph.nodes()))
        node_data = graph.nodes[node]
        
        # 选择一个属性
        attributes = [k for k in node_data.keys() 
                     if k not in ['id', 'type'] and node_data[k]]
        
        if not attributes:
            # 如果没有属性，尝试使用关系
            edges = list(graph.edges(node, data=True))
            if not edges:
                return None
                
            # 使用边关系生成问题
            edge = random.choice(edges)
            source, target, edge_data = edge
            relation = edge_data.get('relation', '相关')
            
            template = random.choice(templates)
            question = template.format(
                entity1=source,
                entity2=target,
                relation=relation
            )
            answer = f"{source}和{target}之间的关系是{relation}。"
            
        else:
            # 使用属性生成问题
            attribute = random.choice(attributes)
            value = node_data[attribute]
            
            template = random.choice(templates)
            question = template.format(
                entity=node,
                attribute=attribute,
                value=value
            )
            answer = f"{node}的{attribute}是{value}。"
        
        return QAPair(
            question=question,
            answer=answer,
            question_type='factual',
            complexity='simple',
            source_subgraph=subgraph,
            entities=[node],
            relations=[],
            metadata={'template_used': template}
        )
    
    def _generate_reasoning_question(self, subgraph: Any,
                                   type_config: Dict[str, Any]) -> Optional[QAPair]:
        """生成推理性问题"""
        graph = subgraph.graph
        templates = type_config.get('templates', [])
        
        # 寻找因果关系链
        causal_chains = self._find_causal_chains(graph)
        
        if not causal_chains:
            return None
        
        # 选择一条因果链
        chain = random.choice(causal_chains)
        
        if len(chain) < 2:
            return None
        
        # 生成问题
        template = random.choice(templates)
        
        if '如果' in template:
            # 条件推理问题
            condition = chain[0]
            result = chain[-1]
            
            question = template.format(
                condition=f"{condition}发生",
                entity=result
            )
            
            # 构建推理答案
            reasoning_steps = []
            for i in range(len(chain) - 1):
                edge_data = graph.get_edge_data(chain[i], chain[i+1], {})
                relation = edge_data.get('relation', '导致')
                reasoning_steps.append(f"{chain[i]}{relation}{chain[i+1]}")
            
            answer = f"如果{condition}发生，通过以下推理链：" + \
                    "→".join(reasoning_steps) + \
                    f"，最终{result}会受到影响。"
            
        elif '为什么' in template:
            # 解释性问题
            cause = chain[0]
            effect = chain[-1]
            
            question = template.format(
                entity1=cause,
                entity2=effect
            )
            
            answer = self._build_causal_explanation(graph, chain)
            
        else:
            # 其他推理问题
            entities = random.sample(chain, min(2, len(chain)))
            question = template.format(
                entity1=entities[0],
                entity2=entities[-1] if len(entities) > 1 else entities[0]
            )
            answer = self._build_reasoning_answer(graph, chain)
        
        return QAPair(
            question=question,
            answer=answer,
            question_type='reasoning',
            complexity='medium',
            source_subgraph=subgraph,
            entities=chain,
            relations=self._extract_relations(graph, chain),
            metadata={'chain_length': len(chain)}
        )
    
    def _generate_multi_hop_question(self, subgraph: Any,
                                   type_config: Dict[str, Any]) -> Optional[QAPair]:
        """生成多跳问题"""
        graph = subgraph.graph
        templates = type_config.get('templates', [])
        
        # 寻找多跳路径
        paths = self._find_multi_hop_paths(graph, min_hops=2, max_hops=4)
        
        if not paths:
            return None
        
        # 选择一条路径
        path = random.choice(paths)
        
        if len(path) < 3:
            return None
        
        # 生成问题
        template = random.choice(templates)
        
        start = path[0]
        end = path[-1]
        intermediate = path[1:-1]
        
        # 根据模板类型生成问题
        if '通过什么' in template:
            question = template.format(
                entity1=start,
                entity3=end
            )
            
            # 构建答案，说明中间节点
            answer = f"{start}通过"
            for i, node in enumerate(intermediate):
                if i > 0:
                    answer += "，然后通过"
                answer += node
            answer += f"最终影响{end}。"
            
        elif '完整流程' in template:
            question = template.format(
                start=start,
                end=end
            )
            
            # 构建完整流程答案
            answer = self._build_process_answer(graph, path)
            
        else:
            # 其他多跳问题
            question = template.format(
                entity1=start,
                entity2=intermediate[0] if intermediate else end,
                entity3=end
            )
            answer = self._build_multi_hop_answer(graph, path)
        
        return QAPair(
            question=question,
            answer=answer,
            question_type='multi_hop',
            complexity='complex',
            source_subgraph=subgraph,
            entities=path,
            relations=self._extract_relations(graph, path),
            metadata={'hop_count': len(path) - 1}
        )
    
    def _generate_comparative_question(self, subgraph: Any,
                                     type_config: Dict[str, Any]) -> Optional[QAPair]:
        """生成比较性问题"""
        graph = subgraph.graph
        templates = type_config.get('templates', [])
        
        # 寻找可比较的实体对
        comparable_pairs = self._find_comparable_entities(graph)
        
        if not comparable_pairs:
            return None
        
        # 选择一对实体
        entity1, entity2, comparison_aspect = random.choice(comparable_pairs)
        
        # 生成问题
        template = random.choice(templates)
        
        question = template.format(
            entity1=entity1,
            entity2=entity2,
            aspect=comparison_aspect
        )
        
        # 生成比较答案
        answer = self._build_comparative_answer(
            graph, entity1, entity2, comparison_aspect
        )
        
        return QAPair(
            question=question,
            answer=answer,
            question_type='comparative',
            complexity='medium',
            source_subgraph=subgraph,
            entities=[entity1, entity2],
            relations=[],
            metadata={'comparison_aspect': comparison_aspect}
        )
    
    def _find_causal_chains(self, graph: nx.Graph) -> List[List[str]]:
        """寻找因果关系链"""
        chains = []
        
        # 寻找包含因果关系的边
        causal_edges = []
        for u, v, data in graph.edges(data=True):
            relation = data.get('relation', '')
            if any(kw in relation for kw in ['导致', '引起', '造成', '产生']):
                causal_edges.append((u, v))
        
        # 构建因果链
        for start, _ in causal_edges:
            chain = self._build_chain_from_node(graph, start, causal_edges)
            if len(chain) >= 2:
                chains.append(chain)
        
        return chains
    
    def _build_chain_from_node(self, graph: nx.Graph, start: str,
                             causal_edges: List[Tuple[str, str]]) -> List[str]:
        """从节点构建因果链"""
        chain = [start]
        current = start
        
        while True:
            next_nodes = [v for u, v in causal_edges if u == current]
            if not next_nodes:
                break
            
            next_node = next_nodes[0]  # 简化：选择第一个
            if next_node in chain:  # 避免循环
                break
                
            chain.append(next_node)
            current = next_node
        
        return chain
    
    def _find_multi_hop_paths(self, graph: nx.Graph, 
                            min_hops: int, max_hops: int) -> List[List[str]]:
        """寻找多跳路径"""
        paths = []
        nodes = list(graph.nodes())
        
        # 对每对节点寻找路径
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                try:
                    # 寻找最短路径
                    path = nx.shortest_path(graph, nodes[i], nodes[j])
                    
                    if min_hops <= len(path) - 1 <= max_hops:
                        paths.append(path)
                        
                except nx.NetworkXNoPath:
                    continue
        
        return paths
    
    def _find_comparable_entities(self, graph: nx.Graph) -> List[Tuple[str, str, str]]:
        """寻找可比较的实体对"""
        comparable_pairs = []
        
        # 按类型分组节点
        type_groups = defaultdict(list)
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'unknown')
            type_groups[node_type].append(node)
        
        # 在同类型节点中寻找可比较的对
        for node_type, nodes in type_groups.items():
            if len(nodes) < 2:
                continue
                
            # 随机选择一些对
            for _ in range(min(3, len(nodes) // 2)):
                if len(nodes) >= 2:
                    pair = random.sample(nodes, 2)
                    
                    # 确定比较方面
                    aspect = self._determine_comparison_aspect(
                        graph, pair[0], pair[1]
                    )
                    
                    if aspect:
                        comparable_pairs.append((pair[0], pair[1], aspect))
        
        return comparable_pairs
    
    def _determine_comparison_aspect(self, graph: nx.Graph,
                                   entity1: str, entity2: str) -> Optional[str]:
        """确定比较方面"""
        # 获取两个实体的属性
        attrs1 = set(graph.nodes[entity1].keys())
        attrs2 = set(graph.nodes[entity2].keys())
        
        # 找共同属性
        common_attrs = attrs1 & attrs2
        common_attrs -= {'id', 'type'}  # 排除ID和类型
        
        if common_attrs:
            return random.choice(list(common_attrs))
        
        # 如果没有共同属性，看是否有共同的关系类型
        relations1 = {graph.get_edge_data(entity1, n).get('relation', '')
                     for n in graph.neighbors(entity1)}
        relations2 = {graph.get_edge_data(entity2, n).get('relation', '')
                     for n in graph.neighbors(entity2)}
        
        common_relations = relations1 & relations2
        if common_relations:
            return f"在{random.choice(list(common_relations))}方面"
        
        return None
    
    def _build_causal_explanation(self, graph: nx.Graph,
                                chain: List[str]) -> str:
        """构建因果解释"""
        explanation = f"{chain[0]}会影响{chain[-1]}，原因如下：\n"
        
        for i in range(len(chain) - 1):
            edge_data = graph.get_edge_data(chain[i], chain[i+1], {})
            relation = edge_data.get('relation', '影响')
            
            explanation += f"{i+1}. {chain[i]}{relation}{chain[i+1]}"
            
            # 添加额外说明
            if 'description' in edge_data:
                explanation += f"（{edge_data['description']}）"
            
            explanation += "；\n"
        
        explanation += f"因此，{chain[0]}最终会影响{chain[-1]}。"
        
        return explanation
    
    def _build_reasoning_answer(self, graph: nx.Graph,
                              chain: List[str]) -> str:
        """构建推理答案"""
        answer = "根据以下推理过程：\n"
        
        for i in range(len(chain) - 1):
            answer += f"- {chain[i]} → {chain[i+1]}\n"
        
        answer += f"可以得出结论。"
        
        return answer
    
    def _build_process_answer(self, graph: nx.Graph,
                            path: List[str]) -> str:
        """构建流程答案"""
        answer = f"从{path[0]}到{path[-1]}的完整流程如下：\n"
        
        for i, node in enumerate(path):
            answer += f"{i+1}. {node}"
            
            if i < len(path) - 1:
                edge_data = graph.get_edge_data(path[i], path[i+1], {})
                relation = edge_data.get('relation', '')
                if relation:
                    answer += f" ({relation})"
            
            answer += "\n"
        
        return answer
    
    def _build_multi_hop_answer(self, graph: nx.Graph,
                              path: List[str]) -> str:
        """构建多跳答案"""
        answer = "通过以下路径连接：\n"
        
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i+1], {})
            relation = edge_data.get('relation', '→')
            
            answer += f"{path[i]} {relation} {path[i+1]}\n"
        
        return answer
    
    def _build_comparative_answer(self, graph: nx.Graph,
                                entity1: str, entity2: str,
                                aspect: str) -> str:
        """构建比较答案"""
        answer = f"比较{entity1}和{entity2}在{aspect}方面：\n"
        
        # 获取属性值
        if aspect in graph.nodes[entity1] and aspect in graph.nodes[entity2]:
            val1 = graph.nodes[entity1][aspect]
            val2 = graph.nodes[entity2][aspect]
            
            answer += f"- {entity1}的{aspect}：{val1}\n"
            answer += f"- {entity2}的{aspect}：{val2}\n"
            
            # 添加比较结论
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if val1 > val2:
                    answer += f"因此，{entity1}在{aspect}方面更高。"
                elif val1 < val2:
                    answer += f"因此，{entity2}在{aspect}方面更高。"
                else:
                    answer += f"两者在{aspect}方面相同。"
            else:
                answer += f"两者在{aspect}方面有所不同。"
        else:
            # 基于关系比较
            answer += f"两者都涉及{aspect}，但具体实现方式不同。"
        
        return answer
    
    def _extract_relations(self, graph: nx.Graph,
                         path: List[str]) -> List[str]:
        """提取路径中的关系"""
        relations = []
        
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i+1], {})
            relation = edge_data.get('relation', '')
            if relation:
                relations.append(relation)
        
        return relations
    
    def _determine_question_count(self, subgraph: Any) -> int:
        """确定为子图生成的问题数量"""
        # 基于子图大小和复杂度
        graph_size = len(subgraph.graph.nodes())
        
        if graph_size <= 5:
            return random.randint(2, 4)
        elif graph_size <= 10:
            return random.randint(3, 6)
        else:
            return random.randint(4, 8)
    
    def _distribute_counts(self, total: int, weights: List[float]) -> List[int]:
        """根据权重分配数量"""
        import numpy as np
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        counts = (weights * total).astype(int)
        
        # 处理舍入误差
        diff = total - counts.sum()
        if diff > 0:
            indices = np.argsort(weights)[::-1]
            for i in range(diff):
                counts[indices[i % len(indices)]] += 1
        
        return counts.tolist()
    
    def _filter_answerable(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """过滤不可回答的问题"""
        filtered = []
        
        for qa in qa_pairs:
            # 检查答案长度
            if len(qa.answer) < 10:
                continue
                
            # 检查是否包含必要信息
            if not qa.entities:
                continue
                
            # 检查答案是否与问题相关
            if not any(entity in qa.answer for entity in qa.entities):
                continue
                
            filtered.append(qa)
        
        return filtered
    
    def _add_distractors(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """为问题添加干扰项"""
        # 收集所有实体和关系
        all_entities = set()
        all_relations = set()
        
        for qa in qa_pairs:
            all_entities.update(qa.entities)
            all_relations.update(qa.relations)
        
        # 为每个问题添加干扰项
        for qa in qa_pairs:
            if qa.question_type in ['factual', 'comparative']:
                # 为选择题类型的问题添加干扰项
                distractors = self._generate_distractors(
                    qa, all_entities, all_relations
                )
                qa.metadata['distractors'] = distractors
        
        return qa_pairs
    
    def _generate_distractors(self, qa: QAPair,
                            all_entities: set,
                            all_relations: set) -> List[str]:
        """生成干扰项"""
        distractors = []
        
        # 基于实体替换生成干扰项
        other_entities = all_entities - set(qa.entities)
        
        if other_entities and qa.entities:
            # 随机选择其他实体替换
            for _ in range(min(3, len(other_entities))):
                entity = random.choice(list(other_entities))
                # 简单替换第一个实体
                distractor = qa.answer.replace(qa.entities[0], entity, 1)
                if distractor != qa.answer:
                    distractors.append(distractor)
        
        return distractors[:3]  # 最多3个干扰项
    
    def _load_templates(self):
        """加载问题模板"""
        # 这里可以从配置文件或外部文件加载更多模板
        # 当前使用配置中的模板
        pass
    
    def _initialize_qa_validator(self):
        """初始化QA验证模型（用于检测答案错误/无关的问答对）"""
        if self.qa_validator_model is None:
            model_config = self.config.get('models', {})
            qa_model_path = model_config.get('qa_generator_model', {}).get('path', 
                                            '/mnt/storage/models/Qwen/Qwen2.5-14B-Instruct')
            
            self.logger.info(f"加载QA验证模型: {qa_model_path}")
            self.qa_validator_tokenizer = AutoTokenizer.from_pretrained(qa_model_path)
            self.qa_validator_model = AutoModelForCausalLM.from_pretrained(
                qa_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
    
    def _check_answer_relevance(self, question: str, answer: str) -> Tuple[bool, float]:
        """使用大模型检查答案是否相关且正确"""
        self._initialize_qa_validator()
        
        prompt = f"""请评估以下问答对的质量：

问题：{question}
答案：{answer}

请从以下几个方面评估：
1. 答案是否直接回答了问题？
2. 答案内容是否准确合理？
3. 答案是否包含明显的错误或矛盾？

请给出：
- 评分（0-1之间，1表示完全正确相关，0表示完全错误无关）
- 是否通过（是/否）
- 简短理由

格式：
评分：X.X
通过：是/否
理由：XXX
"""
        
        inputs = self.qa_validator_tokenizer(prompt, return_tensors="pt").to(self.qa_validator_model.device)
        
        with torch.no_grad():
            outputs = self.qa_validator_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True
            )
        
        response = self.qa_validator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # 解析响应
        try:
            lines = response.split('\n')
            score = 0.5  # 默认分数
            is_valid = False
            
            for line in lines:
                if '评分' in line:
                    score = float(line.split('：')[-1].strip())
                elif '通过' in line:
                    is_valid = '是' in line
            
            return is_valid, score
        except:
            # 解析失败时的默认值
            return True, 0.7
    
    def _calculate_question_quality_score(self, qa: Dict) -> float:
        """计算问题的质量分数"""
        score = 0.0
        
        # 1. 问题长度分数（20-200字符最佳）
        q_len = len(qa.get('question', ''))
        if 20 <= q_len <= 200:
            score += 2.0
        elif q_len < 20:
            score += 0.5
        else:
            score += 1.0
        
        # 2. 答案长度分数（50-500字符最佳）
        a_len = len(qa.get('answer', ''))
        if 50 <= a_len <= 500:
            score += 2.0
        elif a_len < 50:
            score += 0.5
        else:
            score += 1.0
        
        # 3. 问题类型分数
        q_type = qa.get('type', '')
        type_scores = {
            'reasoning': 2.0,
            'multi_hop': 2.0,
            'comparative': 1.5,
            'factual': 1.0,
            'counterfactual': 1.8,
            'temporal': 1.5,
            'causal': 1.8
        }
        score += type_scores.get(q_type, 1.0)
        
        # 4. 实体数量分数
        entities = qa.get('entities', [])
        if 2 <= len(entities) <= 5:
            score += 1.5
        elif len(entities) == 1:
            score += 0.5
        else:
            score += 1.0
        
        # 5. 合理性分数加成
        validity_score = qa.get('validity_score', 0.5)
        score += validity_score * 2.0
        
        return score
    
    def _balance_question_types(self, qa_pairs: List[Dict]) -> List[Dict]:
        """平衡不同类型问题的分布"""
        # 按类型分组
        type_groups = defaultdict(list)
        for qa in qa_pairs:
            q_type = qa.get('type', 'unknown')
            type_groups[q_type].append(qa)
        
        # 计算每种类型的目标数量
        total_target = min(len(qa_pairs), 100)  # 限制总数
        num_types = len(type_groups)
        if num_types == 0:
            return qa_pairs
        
        target_per_type = max(1, total_target // num_types)
        
        # 平衡选择
        balanced = []
        for q_type, type_qa_pairs in type_groups.items():
            # 按质量分数排序，选择最好的
            type_qa_pairs.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            selected = type_qa_pairs[:target_per_type]
            balanced.extend(selected)
        
        return balanced
    
    def _filter_qa_pairs(self, qa_pairs: List[Dict]) -> List[Dict]:
        """过滤和质量检查QA对（增强版）"""
        if not qa_pairs:
            return []
        
        filtered = []
        
        # 获取质量检查配置
        quality_config = self.config.get('dataset_synthesis', {}).get('quality_checks', {})
        min_q_len = quality_config.get('min_question_length', 20)
        max_q_len = quality_config.get('max_question_length', 600)
        min_a_len = quality_config.get('min_answer_length', 20)
        
        # 用于去重的集合
        seen_questions = set()
        
        # 统计信息
        filter_stats = {
            'total': len(qa_pairs),
            'length_filtered': 0,
            'duplicate_filtered': 0,
            'validity_filtered': 0,
            'answer_filtered': 0,
            'relevance_filtered': 0  # 新增：相关性过滤
        }
        
        for qa in qa_pairs:
            # 1. 长度检查
            q_len = len(qa.get('question', ''))
            if not (min_q_len <= q_len <= max_q_len):
                filter_stats['length_filtered'] += 1
                self.logger.info(f"长度过滤: {q_len} 不在 [{min_q_len}, {max_q_len}] 范围内")
                continue
            
            # 2. 答案验证
            answer = qa.get('answer', '')
            if not answer or len(answer) < min_a_len:
                filter_stats['answer_filtered'] += 1
                self.logger.debug(f"答案过滤: 答案长度 {len(answer)} < {min_a_len}")
                continue
            
            # 3. 合理性分数检查
            validity_score = qa.get('validity_score', 0.85)
            if validity_score < self.validity_threshold:
                filter_stats['validity_filtered'] += 1
                self.logger.debug(f"合理性过滤: 分数 {validity_score} < {self.validity_threshold}")
                continue
            
            # 4. 去重（基于问题内容）
            question_key = qa['question'].lower().strip()
            # 生成更精确的指纹
            import re
            keywords = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', question_key)
            keywords = [w for w in keywords if len(w) > 1]
            fingerprint = ''.join(sorted(keywords[:8]))  # 取前8个关键词
            
            if fingerprint in seen_questions:
                filter_stats['duplicate_filtered'] += 1
                self.logger.debug(f"重复过滤: 问题指纹重复")
                continue
            
            seen_questions.add(fingerprint)
            
            # 5. 使用大模型检查答案相关性和正确性
            if self.config.get('use_llm_validation', True):
                is_relevant, relevance_score = self._check_answer_relevance(
                    qa['question'], qa['answer']
                )
                
                if not is_relevant or relevance_score < 0.6:
                    filter_stats['relevance_filtered'] += 1
                    self.logger.info(f"相关性过滤: 问题 '{qa['question'][:50]}...' 的答案相关性分数 {relevance_score}")
                    continue
                
                # 更新合理性分数
                qa['validity_score'] = (validity_score + relevance_score) / 2
            
            # 6. 质量分数计算和筛选
            quality_score = self._calculate_question_quality_score(qa)
            qa['quality_score'] = quality_score
            
            # 只保留质量分数较高的问题
            if quality_score >= 5.0:  # 设定质量阈值
                filtered.append(qa)
        
        # 确保类型分布均衡
        balanced_qa = self._balance_question_types(filtered)
        
        # 最终排序（质量分数降序）
        balanced_qa.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # 记录过滤统计
        self.logger.info(f"质量过滤统计:")
        self.logger.info(f"  总数: {filter_stats['total']}")
        self.logger.info(f"  长度过滤: {filter_stats['length_filtered']}")
        self.logger.info(f"  答案过滤: {filter_stats['answer_filtered']}")
        self.logger.info(f"  合理性过滤: {filter_stats['validity_filtered']}")
        self.logger.info(f"  重复过滤: {filter_stats['duplicate_filtered']}")
        self.logger.info(f"  相关性过滤: {filter_stats['relevance_filtered']}")
        self.logger.info(f"  最终保留: {len(balanced_qa)}")
        
        return balanced_qa