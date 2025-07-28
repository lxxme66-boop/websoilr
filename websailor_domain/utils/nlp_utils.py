"""
NLP工具函数
"""

import logging
import spacy
import jieba
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def setup_nlp_models(config: dict):
    """设置NLP模型"""
    logger.info("设置NLP模型...")
    
    # 初始化jieba
    jieba.initialize()
    
    # 加载spaCy模型（如果需要）
    try:
        nlp = spacy.load("zh_core_web_sm")
        logger.info("加载spaCy中文模型成功")
    except:
        logger.warning("未找到spaCy中文模型，部分功能可能受限")
        nlp = None
    
    # 设置全局变量
    globals()['nlp'] = nlp
    
    return nlp


def segment_text(text: str, use_jieba: bool = True) -> List[str]:
    """文本分词"""
    if use_jieba:
        return list(jieba.cut(text))
    else:
        if 'nlp' in globals() and globals()['nlp']:
            doc = globals()['nlp'](text)
            return [token.text for token in doc]
        else:
            # 简单的基于空格的分词
            return text.split()


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """提取关键词"""
    import jieba.analyse
    
    keywords = jieba.analyse.extract_tags(text, topK=top_k, withWeight=False)
    return keywords


def is_chinese(text: str) -> bool:
    """判断文本是否为中文"""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def clean_text(text: str) -> str:
    """清理文本"""
    # 移除多余空格
    text = ' '.join(text.split())
    
    # 移除特殊字符（保留中文、英文、数字和基本标点）
    import re
    text = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9，。！？、；：""''（）\s]', '', text)
    
    return text.strip()


def calculate_text_similarity(text1: str, text2: str) -> float:
    """计算文本相似度（简化版）"""
    # 使用Jaccard相似度
    words1 = set(segment_text(text1))
    words2 = set(segment_text(text2))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)