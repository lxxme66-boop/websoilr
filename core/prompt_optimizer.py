# ==================== core/prompt_optimizer.py ====================
"""
提示词优化模块 - 用于改善实体和关系提取的质量
"""

import re
from typing import List, Dict, Tuple

class PromptOptimizer:
    """优化提示词以获得更好的JSON输出"""
    
    def __init__(self, domain: str):
        self.domain = domain
        self.entity_examples = self._get_domain_entity_examples()
        self.relation_examples = self._get_domain_relation_examples()
    
    def _get_domain_entity_examples(self) -> Dict[str, List[Dict]]:
        """获取领域特定的实体示例"""
        examples = {
            "材料科学": [
                {"id": "IGZO", "name": "铟镓锌氧化物", "type": "材料", "description": "一种透明导电氧化物材料"},
                {"id": "TFT", "name": "薄膜晶体管", "type": "器件", "description": "薄膜场效应晶体管器件"},
                {"id": "annealing_temp", "name": "退火温度", "type": "工艺参数", "description": "热处理过程的温度参数"},
                {"id": "mobility", "name": "载流子迁移率", "type": "性能参数", "description": "表征载流子运动能力的参数"}
            ],
            "默认": [
                {"id": "entity1", "name": "实体名称1", "type": "类型1", "description": "描述1"},
                {"id": "entity2", "name": "实体名称2", "type": "类型2", "description": "描述2"}
            ]
        }
        return examples.get(self.domain, examples["默认"])
    
    def _get_domain_relation_examples(self) -> List[Dict]:
        """获取领域特定的关系示例"""
        examples = {
            "材料科学": [
                {"source": "IGZO", "target": "TFT", "relation": "用于制备", "description": "IGZO材料用于制备TFT器件"},
                {"source": "annealing_temp", "target": "mobility", "relation": "影响", "description": "退火温度影响载流子迁移率"}
            ],
            "默认": [
                {"source": "entity1", "target": "entity2", "relation": "关系类型", "description": "关系描述"}
            ]
        }
        return examples.get(self.domain, examples["默认"])
    
    def optimize_entity_prompt(self, text_chunk: str) -> str:
        """优化实体提取提示词"""
        # 预处理文本，提取关键句子
        key_sentences = self._extract_key_sentences(text_chunk)
        
        # 构建示例
        examples = "\n".join([
            f'{{"id": "{ex["id"]}", "name": "{ex["name"]}", "type": "{ex["type"]}", "description": "{ex["description"]}"}}'
            for ex in self.entity_examples[:2]
        ])
        
        prompt = f"""你是{self.domain}领域的专家。请从文本中提取专业实体。

重点关注的文本片段：
{key_sentences}

完整文本：
{text_chunk[:1000]}

实体类型包括：材料、化合物、器件、设备、工艺、方法、参数、性能指标、条件等。

输出格式示例：
{examples}

要求：
1. 每行输出一个完整的JSON对象
2. id用英文或拼音（如IGZO、TFT、temp_350C）
3. name用中文全称
4. type选择合适的类别
5. description简洁说明实体含义

请直接输出JSON对象，每行一个："""
        
        return prompt
    
    def optimize_relation_prompt(self, entities: List[Dict], text_chunk: str) -> str:
        """优化关系提取提示词"""
        # 限制实体数量
        entities = entities[:8]
        
        # 构建实体表
        entity_table = "\n".join([
            f"{i+1}. [{e['id']}] {e['name']} ({e['type']})"
            for i, e in enumerate(entities)
        ])
        
        # 构建示例
        examples = "\n".join([
            f'{{"source": "{ex["source"]}", "target": "{ex["target"]}", "relation": "{ex["relation"]}", "description": "{ex["description"]}"}}'
            for ex in self.relation_examples
        ])
        
        prompt = f"""分析以下实体之间的关系。

实体列表：
{entity_table}

相关文本：
{text_chunk[:800]}

常见关系类型：
- 材料关系：组成、包含、掺杂、合成
- 工艺关系：制备、处理、影响、优化
- 性能关系：提高、降低、决定、表现为
- 结构关系：具有、属于、形成

输出示例：
{examples}

请识别实体间的明确关系，直接输出JSON对象，每行一个："""
        
        return prompt
    
    def _extract_key_sentences(self, text: str) -> str:
        """提取包含关键信息的句子"""
        sentences = re.split(r'[。！？\n]+', text)
        
        # 关键词列表
        keywords = ['制备', '性能', '温度', '材料', '方法', '结果', '表明', '影响', '提高', '优化']
        
        # 筛选包含关键词的句子
        key_sentences = []
        for sent in sentences[:10]:  # 只看前10个句子
            sent = sent.strip()
            if any(kw in sent for kw in keywords) and len(sent) > 10:
                key_sentences.append(sent)
        
        return '\n'.join(key_sentences[:5])  # 返回前5个关键句子
    
    def post_process_entities(self, entities: List[Dict]) -> List[Dict]:
        """后处理实体，确保格式正确"""
        processed = []
        
        for entity in entities:
            # 规范化ID
            if 'id' in entity:
                entity['id'] = self._normalize_id(entity['id'])
            
            # 确保必要字段
            if 'name' not in entity and 'description' in entity:
                entity['name'] = entity['description'][:20]
            
            # 规范化类型
            if 'type' in entity:
                entity['type'] = self._normalize_type(entity['type'])
            
            processed.append(entity)
        
        return processed
    
    def _normalize_id(self, id_str: str) -> str:
        """规范化ID格式"""
        # 移除特殊字符，保留字母数字和下划线
        id_str = re.sub(r'[^\w\-]', '_', id_str)
        # 移除连续的下划线
        id_str = re.sub(r'_+', '_', id_str)
        # 移除首尾下划线
        id_str = id_str.strip('_')
        return id_str or 'unknown'
    
    def _normalize_type(self, type_str: str) -> str:
        """规范化类型名称"""
        type_mapping = {
            '材料': '材料',
            '化合物': '材料',
            '物质': '材料',
            '器件': '器件',
            '设备': '设备',
            '装置': '设备',
            '参数': '参数',
            '性能': '性能指标',
            '指标': '性能指标',
            '方法': '方法',
            '工艺': '工艺',
            '过程': '工艺',
            '条件': '条件',
            '温度': '条件'
        }
        
        for key, value in type_mapping.items():
            if key in type_str:
                return value
        
        return type_str