"""
修改后的QuestionGenerator - 生成长问题版本
基于原代码A的架构，但增强了问题复杂度和长度
"""
import torch
import random
from typing import List, Dict, Tuple, Optional
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class EnhancedQuestionGenerator:
    """增强版问题生成器 - 生成60-120字的复杂问题"""
    
    def __init__(self, config: dict):
        self.config = config
        self.qg_config = config.get('question_generation', {})
        
        # 长度控制参数
        self.min_question_length = 60
        self.max_question_length = 120
        
        # 加载模型（保持原有逻辑）
        self._load_qa_generator()
        
        # 初始化复杂问题模板
        self._init_complex_question_templates()
        
    def _init_complex_question_templates(self):
        """初始化复杂问题模板"""
        self.complex_templates = {
            'factual': {
                'zh_cn': [
                    # 故障诊断型
                    "在{scenario}环境下，{entity1}出现{symptom1}现象，同时{entity2}显示{symptom2}，"
                    "考虑到{entity1}与{entity2}之间的{relation}关系，以及{constraint}的限制，"
                    "请分析可能的故障原因并提出诊断方案。",
                    
                    # 性能分析型
                    "某{domain}系统中，当{entity1}的{parameter1}达到{value1}时，"
                    "{entity2}的{parameter2}出现{abnormal_behavior}，"
                    "基于两者的{relation}关系和{technical_spec}，"
                    "请评估系统性能瓶颈并建议优化措施。",
                    
                    # 技术应用型
                    "在{application_scenario}中使用{entity1}技术时，"
                    "需要考虑其与{entity2}的{relation}以及{environmental_factor}的影响，"
                    "请详细说明技术实施要点和潜在风险。"
                ]
            },
            'reasoning': {
                'zh_cn': [
                    # 因果推理型
                    "基于{evidence1}和{evidence2}的观察结果，"
                    "结合{entity1}与{entity2}的{relation}关系，"
                    "在{condition}条件下，推断{entity3}可能出现的{consequence}，"
                    "并说明推理依据和验证方法。",
                    
                    # 故障传播型
                    "当{root_cause}发生在{entity1}时，"
                    "考虑到系统中{entity2}、{entity3}的级联关系，"
                    "分析故障如何传播至{end_point}，"
                    "并提出阻断故障传播的措施。"
                ]
            },
            'multi_hop': {
                'zh_cn': [
                    # 多步关系型
                    "在{system_name}中，{entity1}通过{relation1}连接到{entity2}，"
                    "{entity2}又通过{relation2}影响{entity3}，"
                    "当{trigger_event}发生时，分析从{entity1}到{entity3}的完整影响路径，"
                    "并评估各节点的关键程度。",
                    
                    # 复杂依赖型
                    "{entity1}依赖于{entity2}的{service1}，而{entity2}又需要{entity3}提供{service2}，"
                    "在{failure_scenario}情况下，如何确保系统的{reliability_requirement}？"
                    "请提供详细的冗余设计方案。"
                ]
            }
        }
        
    def _generate_factual_questions(self, subgraph: nx.DiGraph, 
                                  features: Dict, lang: str) -> List[Dict]:
        """生成复杂的事实型问题"""
        qa_pairs = []
        
        # 收集更多上下文信息
        edges = list(subgraph.edges(data=True))
        nodes = list(subgraph.nodes(data=True))
        
        # 确保有足够的信息生成复杂问题
        if len(edges) < 2 or len(nodes) < 3:
            logger.info("子图太小，跳过复杂问题生成")
            return qa_pairs
        
        # 为每个主要边生成复杂问题
        for i in range(min(3, len(edges))):
            # 选择一个中心边
            u, v, edge_data = edges[i]
            
            # 收集相关的其他边和节点
            related_info = self._collect_related_context(subgraph, u, v)
            
            # 使用LLM生成复杂问题
            question = self._generate_complex_question_llm(
                subgraph, u, v, edge_data, related_info, 'factual', lang
            )
            
            # 如果LLM生成失败，使用模板
            if not question or len(question) < self.min_question_length:
                question = self._generate_from_complex_template(
                    subgraph, u, v, edge_data, related_info, 'factual', lang
                )
            
            # 确保问题长度符合要求
            question = self._ensure_question_length(question, related_info)
            
            # 生成详细答案
            answer = self._generate_detailed_answer(question, subgraph, related_info)
            
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'type': 'factual_complex',
                'language': lang,
                'complexity': 'high',
                'evidence': related_info
            })
        
        return qa_pairs
    
    def _generate_complex_question_llm(self, subgraph, source, target, 
                                     edge_data, context, q_type, lang):
        """使用LLM生成复杂问题"""
        # 构建详细的提示
        prompt = f"""作为{self.config.get('domain', '工业')}领域专家，基于以下信息生成一个复杂的技术问题。

核心关系：
- {source} --[{edge_data.get('relation', '关联')}]--> {target}

相关上下文：
{self._format_context_for_prompt(context)}

要求：
1. 问题长度必须在{self.min_question_length}-{self.max_question_length}字之间
2. 必须包含具体的技术场景或故障现象
3. 涉及至少2-3个组件的相互作用
4. 需要专业知识才能回答
5. 包含以下元素之一：故障诊断、性能分析、技术应用、系统优化

良好示例：
"在高负载运行环境下，当主控制器CPU使用率持续超过85%且内存占用异常增长时，考虑到其与数据采集模块的实时通信需求以及缓存机制的限制，请分析可能导致系统响应延迟的原因并提出优化方案。"

请直接生成问题，不要包含其他说明："""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                  max_length=1500)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 提取生成的问题
            question = self._extract_question_from_response(response, prompt)
            
            return question
            
        except Exception as e:
            logger.error(f"LLM生成问题失败: {str(e)}")
            return None
    
    def _generate_from_complex_template(self, subgraph, source, target, 
                                      edge_data, context, q_type, lang):
        """使用复杂模板生成问题"""
        templates = self.complex_templates.get(q_type, {}).get(lang, [])
        if not templates:
            return ""
        
        template = random.choice(templates)
        
        # 准备填充数据
        fill_data = self._prepare_template_data(
            subgraph, source, target, edge_data, context
        )
        
        try:
            question = template.format(**fill_data)
        except KeyError as e:
            logger.error(f"模板填充失败: {str(e)}")
            # 使用备用简单版本
            question = self._generate_fallback_question(source, target, edge_data)
        
        return question
    
    def _collect_related_context(self, subgraph, center_u, center_v):
        """收集相关上下文信息"""
        context = {
            'center_nodes': [center_u, center_v],
            'related_nodes': [],
            'related_edges': [],
            'node_attributes': {},
            'technical_details': []
        }
        
        # 收集与中心节点相关的所有边
        for u, v, data in subgraph.edges(data=True):
            if u in [center_u, center_v] or v in [center_u, center_v]:
                context['related_edges'].append({
                    'source': u,
                    'target': v,
                    'relation': data.get('relation', '相关'),
                    'properties': data.get('properties', {})
                })
                
                # 收集相关节点
                for node in [u, v]:
                    if node not in context['center_nodes'] and node not in context['related_nodes']:
                        context['related_nodes'].append(node)
        
        # 收集节点属性
        for node in context['center_nodes'] + context['related_nodes']:
            if node in subgraph.nodes:
                node_data = subgraph.nodes[node]
                context['node_attributes'][node] = {
                    'type': node_data.get('type', '组件'),
                    'properties': node_data.get('properties', {}),
                    'description': node_data.get('description', '')
                }
        
        return context
    
    def _prepare_template_data(self, subgraph, source, target, edge_data, context):
        """准备模板填充数据"""
        # 基础数据
        data = {
            'entity1': source,
            'entity2': target,
            'relation': edge_data.get('relation', '关联'),
            'domain': self.config.get('domain', '工业系统'),
            'scenario': random.choice(['生产环境', '测试环境', '高负载运行', '极端条件']),
            'symptom1': random.choice(['性能下降', '响应延迟', '输出异常', '间歇故障']),
            'symptom2': random.choice(['数据丢失', '通信中断', '过热警告', '振动超标']),
            'parameter1': random.choice(['处理速度', '温度', '压力', '电流']),
            'parameter2': random.choice(['效率', '稳定性', '精度', '功耗']),
            'value1': random.choice(['80%', '上限值', '临界点', '额定值的1.2倍']),
            'abnormal_behavior': random.choice(['波动', '突降', '失控', '漂移']),
            'constraint': random.choice(['实时性要求', '安全规范', '成本限制', '环境因素']),
            'technical_spec': random.choice(['技术规格', '性能指标', '接口标准', '通信协议']),
            'environmental_factor': random.choice(['温湿度', '电磁干扰', '振动冲击', '粉尘污染']),
            'application_scenario': random.choice(['自动化产线', '过程控制', '数据采集', '远程监控'])
        }
        
        # 添加额外的节点（如果有）
        if context['related_nodes']:
            data['entity3'] = context['related_nodes'][0] if context['related_nodes'] else 'system'
        
        # 添加更多关系
        if len(context['related_edges']) > 1:
            data['relation1'] = context['related_edges'][0]['relation']
            data['relation2'] = context['related_edges'][1]['relation'] if len(context['related_edges']) > 1 else '依赖'
        
        return data
    
    def _ensure_question_length(self, question, context):
        """确保问题长度符合要求"""
        if len(question) >= self.min_question_length:
            # 如果太长，适当截断
            if len(question) > self.max_question_length:
                question = question[:self.max_question_length-3] + "..."
            return question
        
        # 如果太短，添加更多细节
        additions = [
            f"特别需要考虑{random.choice(['实时性', '可靠性', '安全性', '经济性'])}方面的要求。",
            f"请结合{random.choice(['行业标准', '最佳实践', '历史经验', '理论分析'])}进行分析。",
            f"同时评估对{random.choice(['系统性能', '生产效率', '产品质量', '运维成本'])}的影响。",
            f"并提供{random.choice(['具体的实施步骤', '量化的评估指标', '可行的替代方案', '风险控制措施'])}。"
        ]
        
        while len(question) < self.min_question_length and additions:
            addition = additions.pop(0)
            question = question.rstrip('。？') + '，' + addition
        
        return question
    
    def _generate_detailed_answer(self, question, subgraph, context):
        """生成详细的答案"""
        prompt = f"""作为技术专家，请为以下问题提供专业、详细的答案。

问题：{question}

相关技术信息：
{self._format_context_for_prompt(context)}

答案要求：
1. 结构清晰，逻辑严密
2. 包含具体的分析步骤
3. 提供可操作的建议
4. 长度控制在150-250字

请提供答案："""

        # 这里应该调用LLM生成答案
        # 简化示例返回
        return (
            "针对所描述的问题，建议采用以下诊断和解决方案：\n"
            "1. 首先进行系统状态全面检查，记录各关键参数的实时数值；\n"
            "2. 通过对比分析确定异常点，重点关注参数间的相关性；\n"
            "3. 基于故障树分析法，逐步排查可能的原因；\n"
            "4. 实施针对性的优化措施，并建立预防性维护机制。\n"
            "整个过程需要密切监控系统响应，确保不影响正常生产。"
        )
    
    def _format_context_for_prompt(self, context):
        """格式化上下文信息用于提示"""
        lines = []
        
        # 中心节点信息
        lines.append("核心组件：")
        for node in context.get('center_nodes', []):
            attrs = context.get('node_attributes', {}).get(node, {})
            lines.append(f"- {node}: {attrs.get('type', '未知类型')}")
        
        # 相关节点
        if context.get('related_nodes'):
            lines.append("\n相关组件：")
            for node in context['related_nodes'][:3]:
                attrs = context.get('node_attributes', {}).get(node, {})
                lines.append(f"- {node}: {attrs.get('type', '未知类型')}")
        
        # 关系信息
        if context.get('related_edges'):
            lines.append("\n组件关系：")
            for edge in context['related_edges'][:3]:
                lines.append(
                    f"- {edge['source']} --[{edge['relation']}]--> {edge['target']}"
                )
        
        return '\n'.join(lines)
    
    def _extract_question_from_response(self, response, prompt):
        """从LLM响应中提取问题"""
        # 移除原始prompt
        if prompt in response:
            response = response.replace(prompt, '').strip()
        
        # 提取第一个问句
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.endswith('？') or line.endswith('?') or line.endswith('。')):
                return line
        
        # 如果没有找到，返回整个响应
        return response.strip()
    
    def _load_qa_generator(self):
        """加载模型（简化版本）"""
        # 这里应该加载实际的模型
        # 为了示例，使用占位符
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cpu')
        logger.info("模型加载完成（示例）")
    
    def _generate_fallback_question(self, source, target, edge_data):
        """生成备用问题"""
        relation = edge_data.get('relation', '关联')
        return (
            f"在复杂工业系统中，当{source}与{target}存在{relation}关系时，"
            f"系统出现异常表现，请分析可能的原因并提出解决方案，"
            f"特别要考虑两者的相互影响和系统整体性能。"
        )