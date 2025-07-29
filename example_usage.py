"""
知识图谱构建器使用示例
展示如何配置和使用不同的大语言模型
"""

from kg_builder_improved import KnowledgeGraphBuilder
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 示例1: 使用ChatGLM3-6B（推荐，中文效果好）
def example_chatglm():
    config = {
        'knowledge_graph': {
            'entity_types': ['技术', '产品', '材料', '工艺', '公司', '人员'],
            'relation_types': ['使用', '包含', '生产', '研发', '依赖', '改进', '替代', '认证', '合作', '应用于'],
            'chunk_size': 1000,
            'max_chunk_overlap': 200,
            'extraction_rules': {
                'confidence_threshold': 0.7
            }
        },
        'tcl_specific': {
            'technical_terms': {
                'display': ['OLED', 'LCD', 'Mini-LED', 'QLED', '量子点', '背光模组'],
                'semiconductor': ['芯片', '半导体', '集成电路', '晶圆', '封装'],
                'appliance': ['空调', '冰箱', '洗衣机', '电视', '显示器'],
                'materials': ['面板', '背光', '驱动IC', '偏光片', '玻璃基板'],
                'manufacturing': ['贴片', '封装', '测试', '组装', '切割']
            }
        },
        'models': {
            'llm_model': {
                'path': 'THUDM/chatglm3-6b'  # ChatGLM3-6B
            }
        }
    }
    
    builder = KnowledgeGraphBuilder(config)
    return builder

# 示例2: 使用Qwen模型
def example_qwen():
    config = {
        'knowledge_graph': {
            'entity_types': ['技术', '产品', '材料', '工艺', '公司', '人员'],
            'relation_types': ['使用', '包含', '生产', '研发', '依赖', '改进', '替代', '认证', '合作', '应用于'],
            'chunk_size': 1000,
            'extraction_rules': {
                'confidence_threshold': 0.7
            }
        },
        'tcl_specific': {
            'technical_terms': {
                'display': ['OLED', 'LCD', 'Mini-LED', 'QLED'],
                'semiconductor': ['芯片', '半导体', '集成电路']
            }
        },
        'models': {
            'llm_model': {
                'path': 'Qwen/Qwen-7B-Chat'  # 通义千问7B
            }
        }
    }
    
    builder = KnowledgeGraphBuilder(config)
    return builder

# 示例3: 使用本地模型
def example_local_model():
    config = {
        'knowledge_graph': {
            'entity_types': ['技术', '产品', '材料', '工艺', '公司', '人员'],
            'relation_types': ['使用', '包含', '生产', '研发', '依赖'],
            'chunk_size': 800,  # 本地模型可能需要更小的块
            'extraction_rules': {
                'confidence_threshold': 0.6  # 可以调整置信度阈值
            }
        },
        'tcl_specific': {
            'technical_terms': {}  # 可以为空
        },
        'models': {
            'llm_model': {
                'path': '/path/to/your/local/model'  # 替换为实际路径
            }
        }
    }
    
    builder = KnowledgeGraphBuilder(config)
    return builder

# 示例4: 使用量化模型（节省显存）
def example_quantized_model():
    """
    如果显存不足，可以使用量化版本的模型
    需要安装额外的库：pip install bitsandbytes accelerate
    """
    config = {
        'knowledge_graph': {
            'entity_types': ['技术', '产品', '材料', '工艺', '公司', '人员'],
            'relation_types': ['使用', '包含', '生产', '研发'],
            'chunk_size': 500,  # 量化模型建议使用更小的块
            'extraction_rules': {
                'confidence_threshold': 0.7
            }
        },
        'tcl_specific': {
            'technical_terms': {
                'display': ['OLED', 'LCD', 'Mini-LED']
            }
        },
        'models': {
            'llm_model': {
                'path': 'THUDM/chatglm3-6b',
                # 可以在代码中添加量化配置
                'quantization': '4bit'  # 或 '8bit'
            }
        }
    }
    
    # 注意：需要修改kg_builder_improved.py中的模型加载代码以支持量化
    builder = KnowledgeGraphBuilder(config)
    return builder

# 示例5: 完整的使用流程
def full_example():
    """完整的知识图谱构建流程"""
    
    # 1. 选择配置（这里使用ChatGLM3）
    config = {
        'knowledge_graph': {
            'entity_types': ['技术', '产品', '材料', '工艺', '公司', '人员', '标准', '应用'],
            'relation_types': [
                '使用', '包含', '生产', '研发', '依赖', 
                '改进', '替代', '认证', '合作', '应用于',
                '符合', '基于', '集成', '供应', '竞争'
            ],
            'chunk_size': 1000,
            'max_chunk_overlap': 200,
            'extraction_rules': {
                'confidence_threshold': 0.75
            }
        },
        'tcl_specific': {
            'technical_terms': {
                'display': ['OLED', 'LCD', 'Mini-LED', 'QLED', '量子点', '背光模组', '液晶面板'],
                'semiconductor': ['芯片', '半导体', '集成电路', '晶圆', '封装', 'IC设计'],
                'appliance': ['空调', '冰箱', '洗衣机', '电视', '显示器', '智能家电'],
                'materials': ['面板', '背光', '驱动IC', '偏光片', '玻璃基板', '彩色滤光片'],
                'manufacturing': ['贴片', '封装', '测试', '组装', '切割', 'SMT', 'COG'],
                'standards': ['ISO9001', 'ISO14001', 'RoHS', 'CE', 'FCC', 'CCC']
            }
        },
        'models': {
            'llm_model': {
                'path': 'THUDM/chatglm3-6b'
            }
        }
    }
    
    # 2. 创建知识图谱构建器
    print("初始化知识图谱构建器...")
    builder = KnowledgeGraphBuilder(config)
    
    # 3. 构建知识图谱
    print("开始构建知识图谱...")
    input_dir = './data/tcl_texts'  # 输入文本目录
    kg = builder.build_from_texts(input_dir)
    
    # 4. 保存知识图谱
    output_path = Path('./output/tcl_knowledge_graph.json')
    print(f"保存知识图谱到: {output_path}")
    builder.save_graph(kg, output_path)
    
    # 5. 打印统计信息
    print(f"\n知识图谱统计信息:")
    print(f"- 节点总数: {kg.number_of_nodes()}")
    print(f"- 边总数: {kg.number_of_edges()}")
    print(f"- 平均度: {sum(dict(kg.degree()).values()) / kg.number_of_nodes():.2f}")
    
    # 6. 展示部分结果
    print("\n部分实体示例:")
    for node, data in list(kg.nodes(data=True))[:5]:
        print(f"- {node} (类型: {data.get('type', 'unknown')})")
    
    print("\n部分关系示例:")
    for source, target, data in list(kg.edges(data=True))[:5]:
        print(f"- {source} --[{data.get('relation', 'unknown')}]--> {target}")
    
    return kg

# 示例6: 自定义模型加载（高级用法）
def example_custom_model_loading():
    """
    如果需要更细粒度的控制，可以自定义模型加载逻辑
    需要修改KnowledgeGraphBuilder类
    """
    
    class CustomKnowledgeGraphBuilder(KnowledgeGraphBuilder):
        def _load_llm_model(self):
            """自定义模型加载逻辑"""
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            logger = logging.getLogger(__name__)
            logger.info(f"加载大语言模型: {self.model_path}")
            
            # 配置量化（4bit）
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    quantization_config=quantization_config,  # 使用量化配置
                    device_map="auto"
                )
                
                self.model.eval()
                logger.info("大语言模型加载成功（4bit量化）")
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                raise
    
    # 使用自定义的构建器
    config = {
        'knowledge_graph': {
            'entity_types': ['技术', '产品', '材料'],
            'relation_types': ['使用', '包含', '生产'],
            'chunk_size': 500,
            'extraction_rules': {
                'confidence_threshold': 0.7
            }
        },
        'tcl_specific': {
            'technical_terms': {}
        },
        'models': {
            'llm_model': {
                'path': 'THUDM/chatglm3-6b'
            }
        }
    }
    
    builder = CustomKnowledgeGraphBuilder(config)
    return builder


if __name__ == "__main__":
    # 运行完整示例
    full_example()
    
    # 或者尝试其他配置
    # builder = example_chatglm()
    # builder = example_qwen()
    # builder = example_quantized_model()