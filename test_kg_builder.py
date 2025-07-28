#!/usr/bin/env python3
"""
测试优化后的知识图谱构建器
"""

import logging
from typing import Dict, Any
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 模拟配置类
class Config:
    DOMAIN = "材料科学"
    CHUNK_SIZE = 1000
    KG_EXTRACTOR_MODEL_PATH = "model_path"

# 模拟模型管理器
class MockModelManager:
    def load_sentence_transformer(self, path, name):
        """模拟加载句子转换器"""
        return None
    
    def generate_text(self, model_name, prompt, max_new_tokens=512, temperature=0.3, top_p=0.9):
        """模拟文本生成 - 返回更规范的JSON"""
        if "提取关键实体" in prompt:
            # 返回格式良好的实体JSON
            return '''{"id": "IGZO", "name": "铟镓锌氧化物", "type": "材料", "description": "一种透明导电氧化物"}
{"id": "TFT", "name": "薄膜晶体管", "type": "器件", "description": "薄膜场效应晶体管"}
{"id": "mobility", "name": "迁移率", "type": "参数", "description": "载流子迁移率"}'''
        
        elif "提取实体之间的关系" in prompt:
            # 返回格式良好的关系JSON
            return '''{"source": "IGZO", "target": "TFT", "relation": "用于制备", "description": "IGZO材料用于制备TFT器件"}
{"source": "TFT", "target": "mobility", "relation": "具有", "description": "TFT器件具有迁移率参数"}'''
        
        return ""

# 测试优化后的KG构建器
def test_kg_builder():
    # 导入优化后的构建器
    from core.kg_builder import IndustrialKGBuilder
    
    # 初始化
    config = Config()
    model_manager = MockModelManager()
    kg_builder = IndustrialKGBuilder(model_manager, config)
    
    # 测试文档
    test_documents = {
        "test_doc.txt": """
        InGaZnO（IGZO）薄膜晶体管因其高迁移率、低温制备和透明性等优点，
        在显示技术领域得到广泛应用。本研究通过磁控溅射法制备了IGZO薄膜，
        并研究了退火温度对薄膜性能的影响。
        
        实验结果表明，在350°C退火条件下，IGZO-TFT表现出最佳的电学性能，
        场效应迁移率达到15.2 cm²/V·s，开关比超过10⁸。
        """
    }
    
    # 构建知识图谱
    print("开始构建知识图谱...")
    graph = kg_builder.build_from_documents(test_documents)
    
    # 输出结果
    print(f"\n构建完成！")
    print(f"节点数: {len(graph.nodes)}")
    print(f"边数: {len(graph.edges)}")
    
    print("\n节点列表:")
    for node_id, node_data in graph.nodes(data=True):
        print(f"  - {node_id}: {node_data}")
    
    print("\n边列表:")
    for source, target, edge_data in graph.edges(data=True):
        print(f"  - {source} -> {target}: {edge_data}")

if __name__ == "__main__":
    test_kg_builder()