# 问题生成器改进指南

## 问题分析总结

### 代码A生成短问题的原因：
1. **模板限制**：使用预定义的简短模板
2. **缺少长度要求**：没有明确的长度控制
3. **LLM使用有限**：主要用于验证而非生成
4. **上下文信息少**：模板填充时只使用基础信息

### 代码B生成长问题的原因：
1. **明确的长度要求**：80-120字
2. **复杂度要求**：多组件交互、故障诊断
3. **LLM自由生成**：不受模板限制
4. **丰富的上下文**：提供详细的组件和关系描述

## 具体改进方案

### 1. 修改模板设计

将简短模板改为复杂模板：

```python
# 原模板
"{entity}使用了什么{relation_type}？"

# 改进后的模板
"在{operating_condition}条件下，当{entity1}的{parameter}出现{symptom}时，"
"考虑到其与{entity2}的{relation}关系以及{entity3}的影响，"
"请分析可能的{problem_type}并提出{solution_approach}。"
```

### 2. 增强LLM提示词

在 `_generate_answer_with_llm` 方法中添加问题生成功能：

```python
def _generate_complex_question_with_llm(self, subgraph, q_type):
    prompt = f"""基于以下知识图谱信息，生成一个复杂的{q_type}类型问题。

知识图谱信息：
{self._format_subgraph_for_prompt(subgraph)}

要求：
1. 问题长度在60-120字之间
2. 必须涉及至少2-3个实体的交互
3. 包含具体的场景描述（如故障现象、性能问题等）
4. 需要专业知识和多步推理才能回答
5. 使用行业专业术语

示例：
- 事实型："在高温高湿环境下，当变频器输出频率异常波动且散热风扇转速下降时，考虑到IGBT模块与控制板的连接关系，请分析可能的故障原因及诊断方法。"
- 推理型："基于PLC控制系统中I/O模块频繁重启、通信延迟增加的现象，结合系统拓扑结构和电源供应链路，推断最可能的故障点并说明诊断步骤。"

生成的问题："""
    
    # 调用LLM生成
    return self._call_llm_generate(prompt)
```

### 3. 增加上下文收集

```python
def _collect_rich_context(self, subgraph, center_node):
    """收集丰富的上下文信息用于生成复杂问题"""
    context = {
        'main_entity': center_node,
        'related_entities': [],
        'relations': [],
        'attributes': [],
        'constraints': []
    }
    
    # 收集相关实体（2跳内）
    for node in nx.single_source_shortest_path_length(subgraph, center_node, cutoff=2):
        node_data = subgraph.nodes[node]
        context['related_entities'].append({
            'id': node,
            'type': node_data.get('type'),
            'properties': node_data.get('properties', {})
        })
    
    # 收集关系
    for u, v, data in subgraph.edges(data=True):
        if u == center_node or v == center_node:
            context['relations'].append({
                'source': u,
                'target': v,
                'type': data.get('relation'),
                'properties': data.get('properties', {})
            })
    
    return context
```

### 4. 修改问题生成流程

```python
def _generate_factual_questions_enhanced(self, subgraph, features, lang):
    """增强版事实型问题生成"""
    qa_pairs = []
    
    # 选择关键节点
    key_nodes = self._identify_key_entities(subgraph)
    
    for node in key_nodes[:3]:  # 为前3个关键节点生成问题
        # 收集丰富上下文
        context = self._collect_rich_context(subgraph, node['id'])
        
        # 决定生成方式
        if len(context['related_entities']) >= 3:
            # 有足够上下文，使用LLM生成
            question = self._generate_complex_question_with_llm(subgraph, 'factual')
        else:
            # 使用增强模板
            question = self._generate_from_complex_template(context, lang)
        
        # 确保问题长度
        if len(question) < 50:
            question = self._expand_question(question, context)
        
        # 生成答案
        answer = self._generate_answer_with_llm(question, subgraph, 'factual')
        
        qa_pairs.append({
            'question': question,
            'answer': answer,
            'type': 'factual_complex',
            'language': lang,
            'context': context
        })
    
    return qa_pairs
```

### 5. 添加问题扩展机制

```python
def _expand_question(self, question, context):
    """扩展问题，添加更多细节"""
    expansions = [
        "特别是考虑到系统的实时性要求，",
        "在实际生产环境中，",
        "结合历史故障数据分析，",
        "从维护成本和可靠性角度，",
        "考虑到安全规范和行业标准，"
    ]
    
    details = [
        "请详细说明诊断步骤和预防措施。",
        "并评估对整体系统性能的影响。",
        "同时提供具体的解决方案和实施建议。",
        "分析故障传播路径和潜在风险。",
        "说明检测方法和判断标准。"
    ]
    
    # 在问题前添加场景
    prefix = random.choice(expansions)
    # 在问题后添加要求
    suffix = random.choice(details)
    
    return prefix + question + suffix
```

### 6. 配置参数调整

```python
class QuestionGenerator:
    def __init__(self, config: dict):
        # ... 原有代码 ...
        
        # 添加长度控制参数
        self.min_question_length = 60
        self.max_question_length = 120
        self.prefer_complex_questions = True
        
        # 调整生成参数
        self.generation_params = {
            'temperature': 0.8,  # 提高创造性
            'max_new_tokens': 150,  # 增加生成长度
            'top_p': 0.9,
            'do_sample': True
        }
```

## 实施步骤

1. **第一步**：修改模板系统，添加复杂模板
2. **第二步**：增强上下文收集功能
3. **第三步**：修改LLM调用，支持直接生成问题
4. **第四步**：添加问题长度检查和扩展机制
5. **第五步**：调整配置参数，优化生成效果

## 效果对比

### 改进前：
- "TCL电视使用了什么技术？"
- "显示屏的性能参数是什么？"

### 改进后：
- "在某型号TCL电视机中，用户反映屏幕出现间歇性闪烁并伴有音频输出不稳定，经检测电源模块正常但温度升高时故障加剧，考虑到显示驱动板与主控板的信号传输关系，请分析可能的故障原因并提供系统化的诊断方案。"

- "基于OLED显示技术的电视在长时间显示静态画面后出现残影现象，同时亮度自动调节功能异常，结合像素补偿算法和驱动电路的工作原理，请分析故障机理并提出优化建议。"