"""
改进版问题生成器 - 结合模板和LLM生成的优势
"""
import random
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ImprovedQuestionGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.min_question_length = 50  # 最小问题长度
        self.max_question_length = 120  # 最大问题长度
        
    def _generate_complex_factual_questions(self, subgraph, features, lang='zh_cn'):
        """生成复杂的事实型问题"""
        qa_pairs = []
        
        # 复杂问题模板
        complex_templates = {
            'zh_cn': [
                "在{context}的情况下，当{entity1}的{attribute1}出现{symptom}时，"
                "且{entity2}与其存在{relation}关系，请分析可能的{problem_type}原因"
                "并考虑{constraint}的影响。",
                
                "某{domain}系统中，{entity1}通过{relation1}连接到{entity2}，"
                "而{entity2}又{relation2}于{entity3}。当系统出现{symptom}现象时，"
                "如何通过{diagnostic_method}定位问题根源？",
                
                "考虑到{entity1}的{technical_spec}特性，以及其与{entity2}的{relation}关系，"
                "在{operating_condition}条件下出现{failure_mode}，"
                "请分析故障传播路径和影响范围。"
            ]
        }
        
        edges = list(subgraph.edges(data=True))
        
        # 收集更多上下文信息
        for i, (u, v, edge_data) in enumerate(edges[:3]):
            # 获取更多相关节点和边
            related_edges = self._get_related_edges(subgraph, u, v)
            
            if len(related_edges) >= 2:  # 确保有足够的关系用于生成复杂问题
                # 使用LLM增强模板生成
                question = self._generate_enhanced_question(
                    subgraph, u, v, edge_data, related_edges, lang
                )
                
                if question and len(question) >= self.min_question_length:
                    answer = self._generate_detailed_answer(question, subgraph, 'factual')
                    
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'factual_complex',
                        'language': lang,
                        'complexity': 'high'
                    })
        
        return qa_pairs
    
    def _generate_enhanced_question(self, subgraph, source, target, edge_data, 
                                  related_edges, lang='zh_cn'):
        """使用LLM增强问题生成"""
        # 构建丰富的上下文
        context = self._build_rich_context(subgraph, source, target, related_edges)
        
        prompt = f"""基于以下工业场景信息，生成一个复杂的技术问题：

场景信息：
{context}

要求：
1. 问题长度在{self.min_question_length}-{self.max_question_length}字之间
2. 必须包含具体的技术细节和参数
3. 涉及多个组件的相互作用
4. 包含实际的故障现象或性能问题
5. 需要专业知识才能回答
6. 使用专业术语，但表述清晰

示例格式：
"在[具体场景]中，当[组件A]的[参数/状态]出现[具体现象]，同时考虑到[组件B]的[相关特性]，
以及它们之间的[关系类型]，请分析[具体问题]的可能原因和解决方案。"

生成的问题："""

        # 这里应该调用LLM
        # question = self.model.generate(prompt, ...)
        
        # 模拟生成
        return self._simulate_complex_question(source, target, edge_data, related_edges)
    
    def _simulate_complex_question(self, source, target, edge_data, related_edges):
        """模拟生成复杂问题"""
        # 构建复杂问题
        relation = edge_data.get('relation', '关联')
        
        # 添加更多细节
        symptoms = ["异常波动", "性能下降", "响应延迟", "输出不稳定", "间歇性故障"]
        conditions = ["高温环境", "高负载运行", "长时间工作", "频繁启停", "恶劣工况"]
        methods = ["分层诊断", "信号追踪", "参数监测", "对比分析", "故障树分析"]
        
        symptom = random.choice(symptoms)
        condition = random.choice(conditions)
        method = random.choice(methods)
        
        # 获取额外的组件
        extra_components = []
        for edge in related_edges[:2]:
            if edge['target'] not in [source, target]:
                extra_components.append(edge['target'])
        
        question = f"在{condition}下，{source}出现{symptom}现象，"
        
        if extra_components:
            question += f"已知{source}通过{relation}与{target}连接，"
            question += f"同时{extra_components[0]}也参与系统运行。"
        else:
            question += f"其与{target}存在{relation}关系。"
        
        question += f"请运用{method}方法，分析可能的故障原因，"
        question += "并说明诊断步骤和预防措施。"
        
        # 确保问题足够长
        if len(question) < self.min_question_length:
            question += "特别注意考虑组件间的耦合效应和故障传播机制。"
        
        return question
    
    def _get_related_edges(self, subgraph, node1, node2):
        """获取相关的边用于构建复杂问题"""
        related = []
        
        # 获取与node1和node2相关的所有边
        for u, v, data in subgraph.edges(data=True):
            if u == node1 or v == node1 or u == node2 or v == node2:
                related.append({
                    'source': u,
                    'target': v,
                    'relation': data.get('relation', '相关')
                })
        
        return related
    
    def _build_rich_context(self, subgraph, source, target, related_edges):
        """构建丰富的上下文信息"""
        context_lines = []
        
        # 主要关系
        source_data = subgraph.nodes[source]
        target_data = subgraph.nodes[target]
        
        context_lines.append(
            f"主要组件：{source}（{source_data.get('type', '组件')}）"
        )
        context_lines.append(
            f"相关组件：{target}（{target_data.get('type', '组件')}）"
        )
        
        # 添加更多相关信息
        for edge in related_edges[:3]:
            context_lines.append(
                f"- {edge['source']} {edge['relation']} {edge['target']}"
            )
        
        return '\n'.join(context_lines)
    
    def _generate_detailed_answer(self, question, subgraph, q_type):
        """生成详细的答案"""
        # 这里应该使用LLM生成详细答案
        # 暂时返回模拟答案
        return (
            "根据问题描述的故障现象，需要从以下几个方面进行分析：\n"
            "1. 首先检查相关组件的工作参数是否在正常范围内；\n"
            "2. 分析组件间的信号传输和数据交换是否正常；\n"
            "3. 考虑环境因素对系统性能的影响；\n"
            "4. 制定系统的诊断方案并逐步实施。\n"
            "具体的解决方案需要根据实际检测结果进行调整。"
        )