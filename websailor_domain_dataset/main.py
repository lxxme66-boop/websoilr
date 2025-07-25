import json
from core.knowledge_graph_builder import KnowledgeGraphBuilder
from core.data_synthesizer import DataSynthesizer

# 示例：TCL工业垂域三元组
triples = [
    {"head": "设备A", "relation": "连接", "tail": "设备B", "head_attr": "传感器", "tail_attr": "执行器", "rel_desc": "有线连接"},
    {"head": "设备B", "relation": "控制", "tail": "设备C", "head_attr": "执行器", "tail_attr": "控制器", "rel_desc": "信号控制"},
    {"head": "设备C", "relation": "监控", "tail": "设备D", "head_attr": "控制器", "tail_attr": "监控终端", "rel_desc": "实时监控"},
    {"head": "设备A", "relation": "供电", "tail": "设备D", "head_attr": "传感器", "tail_attr": "监控终端", "rel_desc": "电源线"},
    {"head": "设备D", "relation": "报警", "tail": "设备E", "head_attr": "监控终端", "tail_attr": "报警器", "rel_desc": "异常报警"},
]

# 问题模板
question_templates = [
    {"type": "entity_attr", "template": "{entity}的主要功能是什么?"},
    {"type": "relation", "template": "{head}和{tail}之间的{relation}关系是什么?"},
    {"type": "path", "template": "从{start}到{end}的信号传递路径是?"}
]

# 模糊化模式
obfuscation_patterns = [
    {"type": "entity", "mapping": {"设备A": "该设备", "设备B": "另一装置"}, "add_noise": True, "noise": "（请注意，部分信息已省略）"},
    {"type": "entity", "mapping": {"设备C": "这位领导人", "设备D": "某终端"}, "add_noise": False}
]

def main():
    # 1. 构建知识图谱
    builder = KnowledgeGraphBuilder()
    graph = builder.build_from_triples(triples)

    # 2. 数据合成
    synthesizer = DataSynthesizer(graph, question_templates, obfuscation_patterns)
    dataset = synthesizer.synthesize(num_subgraphs=5, questions_per_subgraph=3, obfuscate=True)

    # 3. 输出
    with open('websailor_domain_dataset/output_dataset/qa_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print('数据集已生成: output_dataset/qa_pairs.json')

if __name__ == '__main__':
    main()