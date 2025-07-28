"""
NLP工具函数
提供中文文本处理、关键词提取、相似度计算等功能
"""

import re
import jieba
import jieba.analyse
import jieba.posseg as pseg
from typing import List, Tuple, Dict, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging


logger = logging.getLogger(__name__)


def tokenize_chinese(text: str, use_pos: bool = False) -> List[str]:
    """
    中文分词
    
    Args:
        text: 输入文本
        use_pos: 是否返回词性标注
        
    Returns:
        分词结果列表
    """
    if not text:
        return []
    
    # 清理文本
    text = clean_chinese_text(text)
    
    if use_pos:
        # 返回词性标注
        words_pos = pseg.cut(text)
        return [(word, pos) for word, pos in words_pos]
    else:
        # 只返回分词结果
        return list(jieba.cut(text))


def extract_keywords(text: str, topk: int = 10, 
                    method: str = 'tfidf') -> List[Tuple[str, float]]:
    """
    提取关键词
    
    Args:
        text: 输入文本
        topk: 返回前k个关键词
        method: 提取方法 ('tfidf' 或 'textrank')
        
    Returns:
        关键词及其权重列表
    """
    if not text:
        return []
    
    if method == 'tfidf':
        keywords = jieba.analyse.extract_tags(
            text, topK=topk, withWeight=True
        )
    elif method == 'textrank':
        keywords = jieba.analyse.textrank(
            text, topK=topk, withWeight=True
        )
    else:
        raise ValueError(f"不支持的方法: {method}")
    
    return keywords


def calculate_text_similarity(text1: str, text2: str, 
                            method: str = 'tfidf') -> float:
    """
    计算文本相似度
    
    Args:
        text1: 文本1
        text2: 文本2
        method: 相似度计算方法
        
    Returns:
        相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    if method == 'tfidf':
        # 使用TF-IDF计算相似度
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    elif method == 'jaccard':
        # 使用Jaccard相似度
        words1 = set(tokenize_chinese(text1))
        words2 = set(tokenize_chinese(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    else:
        raise ValueError(f"不支持的方法: {method}")


def segment_sentences(text: str) -> List[str]:
    """
    句子分割
    
    Args:
        text: 输入文本
        
    Returns:
        句子列表
    """
    if not text:
        return []
    
    # 中文句子分割规则
    sentence_delimiters = r'[。！？；\n]+'
    
    # 分割句子
    sentences = re.split(sentence_delimiters, text)
    
    # 过滤空句子
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def clean_chinese_text(text: str) -> str:
    """
    清理中文文本
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符（保留中文、英文、数字和基本标点）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s。，！？；：""''（）《》【】、—…·]', '', text)
    
    # 统一标点符号
    punctuation_map = {
        '!': '！',
        '?': '？',
        ';': '；',
        ':': '：',
        ',': '，',
        '.': '。',
        '(': '（',
        ')': '）',
        '[': '【',
        ']': '】',
        '<': '《',
        '>': '》'
    }
    
    for eng, chn in punctuation_map.items():
        text = text.replace(eng, chn)
    
    return text.strip()


def extract_entities_by_pos(text: str, pos_tags: Set[str] = None) -> List[str]:
    """
    基于词性提取实体
    
    Args:
        text: 输入文本
        pos_tags: 目标词性标签集合
        
    Returns:
        实体列表
    """
    if not text:
        return []
    
    if pos_tags is None:
        # 默认提取名词类词性
        pos_tags = {'n', 'nr', 'ns', 'nt', 'nw', 'nz', 'vn'}
    
    words_pos = pseg.cut(text)
    entities = []
    
    for word, pos in words_pos:
        if pos in pos_tags and len(word) >= 2:
            entities.append(word)
    
    return entities


def extract_noun_phrases(text: str) -> List[str]:
    """
    提取名词短语
    
    Args:
        text: 输入文本
        
    Returns:
        名词短语列表
    """
    if not text:
        return []
    
    words_pos = pseg.cut(text)
    noun_phrases = []
    current_phrase = []
    
    for word, pos in words_pos:
        if pos.startswith('n') or pos in ['vn', 'an']:
            current_phrase.append(word)
        else:
            if current_phrase:
                phrase = ''.join(current_phrase)
                if len(phrase) >= 2:
                    noun_phrases.append(phrase)
                current_phrase = []
    
    # 处理最后的短语
    if current_phrase:
        phrase = ''.join(current_phrase)
        if len(phrase) >= 2:
            noun_phrases.append(phrase)
    
    return noun_phrases


def is_chinese_text(text: str) -> bool:
    """
    判断文本是否主要是中文
    
    Args:
        text: 输入文本
        
    Returns:
        是否是中文文本
    """
    if not text:
        return False
    
    chinese_chars = 0
    total_chars = 0
    
    for char in text:
        if '\u4e00' <= char <= '\u9fa5':
            chinese_chars += 1
        if not char.isspace():
            total_chars += 1
    
    if total_chars == 0:
        return False
    
    return chinese_chars / total_chars > 0.5


def extract_numbers_and_units(text: str) -> List[Tuple[str, str]]:
    """
    提取数字和单位
    
    Args:
        text: 输入文本
        
    Returns:
        (数字, 单位)元组列表
    """
    if not text:
        return []
    
    # 匹配数字和单位的正则表达式
    pattern = r'(\d+\.?\d*)\s*([个只台套件米厘米毫米千米公里吨千克克毫克升毫升立方米平方米度摄氏度华氏度瓦千瓦兆瓦伏特安培欧姆赫兹分贝帕斯卡巴毫巴秒分钟小时天周月年元美元欧元英镑日元人民币%％‰]+)'
    
    matches = re.findall(pattern, text)
    
    return [(num, unit) for num, unit in matches]


def normalize_numbers(text: str) -> str:
    """
    规范化文本中的数字
    
    Args:
        text: 输入文本
        
    Returns:
        规范化后的文本
    """
    if not text:
        return ""
    
    # 中文数字到阿拉伯数字的映射
    chinese_num_map = {
        '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
        '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        '十': '10', '百': '100', '千': '1000', '万': '10000',
        '壹': '1', '贰': '2', '叁': '3', '肆': '4', '伍': '5',
        '陆': '6', '柒': '7', '捌': '8', '玖': '9', '拾': '10',
        '佰': '100', '仟': '1000', '萬': '10000'
    }
    
    # 替换中文数字
    for chinese, arabic in chinese_num_map.items():
        text = text.replace(chinese, arabic)
    
    return text


def get_text_statistics(text: str) -> Dict[str, int]:
    """
    获取文本统计信息
    
    Args:
        text: 输入文本
        
    Returns:
        统计信息字典
    """
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'chinese_char_count': 0,
            'english_char_count': 0,
            'digit_count': 0
        }
    
    # 字符统计
    char_count = len(text)
    chinese_char_count = len([c for c in text if '\u4e00' <= c <= '\u9fa5'])
    english_char_count = len([c for c in text if c.isalpha() and c.isascii()])
    digit_count = len([c for c in text if c.isdigit()])
    
    # 词语统计
    words = tokenize_chinese(text)
    word_count = len(words)
    
    # 句子统计
    sentences = segment_sentences(text)
    sentence_count = len(sentences)
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'chinese_char_count': chinese_char_count,
        'english_char_count': english_char_count,
        'digit_count': digit_count
    }