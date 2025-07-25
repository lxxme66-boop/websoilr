import random
from typing import List, Dict, Any

class ObfuscationProcessor:
    """
    模糊化处理器：
    对问题进行模糊描述，模糊中间实体或关系，添加冗余或干扰信息。
    """
    def __init__(self, obfuscation_patterns: List[Dict[str, Any]]):
        self.patterns = obfuscation_patterns

    def obfuscate(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        对单个QA对进行模糊化处理。
        """
        pattern = random.choice(self.patterns)
        question = qa_pair['question']
        # 例：将实体名替换为“该设备”、“这位领导人”等模糊指代
        if pattern['type'] == 'entity':
            for ent, obf in pattern['mapping'].items():
                question = question.replace(ent, obf)
        # 添加冗余信息
        if pattern.get('add_noise'):
            question += pattern['noise']
        return {**qa_pair, 'question': question, 'obfuscated': True}