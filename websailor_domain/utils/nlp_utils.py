"""
NLP工具函数模块
提供中英文文本处理、分词、实体识别等功能
"""

import re
import string
from typing import List, Dict, Set, Tuple, Optional
import jieba
import jieba.posseg as pseg
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# 下载必要的NLTK数据
def download_nltk_data():
    """下载NLTK所需的数据包"""
    required_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for data in required_data:
        try:
            nltk.data.find(f'tokenizers/{data}')
        except LookupError:
            logger.info(f"下载NLTK数据: {data}")
            nltk.download(data, quiet=True)

# 初始化
download_nltk_data()

# 中文停用词
CHINESE_STOPWORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上',
    '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己',
    '这', '那', '些', '什么', '如何', '怎么', '为什么', '哪里', '哪些', '谁', '哪个'
])

# 英文停用词
try:
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except:
    ENGLISH_STOPWORDS = set()

# TCL工业领域专有词汇
TCL_DOMAIN_TERMS = {
    'zh': {
        '量子点', '显示屏', '处理器', '音响系统', '智能家电', '空调', '冰箱', '洗衣机',
        '电视', 'QLED', 'Mini LED', '画质引擎', '音质', '能效', '变频', '制冷',
        '智能控制', 'AIoT', '语音助手', '远程控制', '节能模式', '静音技术'
    },
    'en': {
        'quantum dot', 'display', 'processor', 'audio system', 'smart appliance',
        'air conditioner', 'refrigerator', 'washing machine', 'television',
        'picture engine', 'sound quality', 'energy efficiency', 'inverter',
        'cooling', 'smart control', 'voice assistant', 'remote control',
        'energy saving mode', 'silent technology'
    }
}


def detect_language(text: str) -> str:
    """
    检测文本语言
    返回: 'zh' 或 'en'
    """
    # 统计中文字符和英文字符
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    # 根据字符比例判断
    if chinese_chars > english_chars:
        return 'zh'
    else:
        return 'en'


def segment_text(text: str, lang: Optional[str] = None) -> List[str]:
    """
    文本分词
    自动检测语言并使用相应的分词器
    """
    if lang is None:
        lang = detect_language(text)
    
    if lang == 'zh':
        # 中文分词
        return list(jieba.cut(text))
    else:
        # 英文分词
        return word_tokenize(text.lower())


def extract_sentences(text: str, lang: Optional[str] = None) -> List[str]:
    """
    句子切分
    """
    if lang is None:
        lang = detect_language(text)
    
    if lang == 'zh':
        # 中文句子切分
        sentences = re.split(r'[。！？；\n]+', text)
        return [s.strip() for s in sentences if s.strip()]
    else:
        # 英文句子切分
        try:
            return sent_tokenize(text)
        except:
            # 备用方案
            sentences = re.split(r'[.!?\n]+', text)
            return [s.strip() for s in sentences if s.strip()]


def extract_keywords(text: str, top_k: int = 10, lang: Optional[str] = None) -> List[Tuple[str, float]]:
    """
    提取关键词
    返回: [(关键词, 权重), ...]
    """
    if lang is None:
        lang = detect_language(text)
    
    # 分词
    words = segment_text(text, lang)
    
    # 过滤停用词
    stopwords_set = CHINESE_STOPWORDS if lang == 'zh' else ENGLISH_STOPWORDS
    filtered_words = [w for w in words if w not in stopwords_set and len(w) > 1]
    
    # 统计词频
    word_freq = Counter(filtered_words)
    
    # 计算TF权重
    total_words = len(filtered_words)
    keywords = []
    
    for word, freq in word_freq.most_common(top_k * 2):
        tf = freq / total_words
        
        # 领域词汇加权
        domain_terms = TCL_DOMAIN_TERMS.get(lang, set())
        if word in domain_terms:
            tf *= 2.0
            
        keywords.append((word, tf))
    
    # 排序并返回top_k
    keywords.sort(key=lambda x: x[1], reverse=True)
    return keywords[:top_k]


def extract_named_entities(text: str, lang: Optional[str] = None) -> List[Dict[str, str]]:
    """
    命名实体识别
    返回: [{'text': 实体文本, 'type': 实体类型}, ...]
    """
    if lang is None:
        lang = detect_language(text)
    
    entities = []
    
    if lang == 'zh':
        # 使用jieba进行词性标注
        words = pseg.cut(text)
        for word, flag in words:
            entity_type = None
            
            # 根据词性判断实体类型
            if flag.startswith('n'):  # 名词
                if any(term in word for term in ['公司', '集团', '企业']):
                    entity_type = 'ORG'
                elif any(term in word for term in ['市', '省', '区', '国']):
                    entity_type = 'LOC'
                elif flag == 'nr':  # 人名
                    entity_type = 'PER'
                elif re.match(r'TCL-?\w+', word):  # 产品型号
                    entity_type = 'PRODUCT'
                    
            if entity_type:
                entities.append({
                    'text': word,
                    'type': entity_type
                })
                
    else:
        # 英文实体识别（简化版）
        # 识别大写词组作为潜在实体
        pattern = r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b'
        for match in re.finditer(pattern, text):
            entity_text = match.group()
            if len(entity_text) > 2:  # 过滤单个字母
                entities.append({
                    'text': entity_text,
                    'type': 'MISC'
                })
    
    return entities


def calculate_text_similarity(text1: str, text2: str, lang: Optional[str] = None) -> float:
    """
    计算文本相似度（基于词汇重叠）
    返回: 0-1之间的相似度分数
    """
    if lang is None:
        lang1 = detect_language(text1)
        lang2 = detect_language(text2)
        lang = lang1 if lang1 == lang2 else 'en'
    
    # 分词
    words1 = set(segment_text(text1, lang))
    words2 = set(segment_text(text2, lang))
    
    # 过滤停用词
    stopwords_set = CHINESE_STOPWORDS if lang == 'zh' else ENGLISH_STOPWORDS
    words1 = {w for w in words1 if w not in stopwords_set}
    words2 = {w for w in words2 if w not in stopwords_set}
    
    # 计算Jaccard相似度
    if not words1 or not words2:
        return 0.0
        
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


def extract_numbers_and_units(text: str) -> List[Dict[str, str]]:
    """
    提取数字和单位
    用于识别技术参数
    """
    patterns = [
        # 中文数字+单位
        (r'(\d+(?:\.\d+)?)\s*(寸|英寸|Hz|赫兹|瓦|W|分贝|dB|度|°C|升|L)', 'zh'),
        # 英文数字+单位
        (r'(\d+(?:\.\d+)?)\s*(inch|inches|Hz|Hertz|watts|W|decibels|dB|degrees|°C|liters|L)', 'en'),
        # 百分比
        (r'(\d+(?:\.\d+)?)\s*(%|％|percent)', 'percentage'),
        # 分辨率
        (r'(\d+)\s*[xX×]\s*(\d+)', 'resolution'),
        # 比例
        (r'(\d+)\s*[:：]\s*(\d+)', 'ratio')
    ]
    
    results = []
    for pattern, pattern_type in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if pattern_type == 'resolution':
                results.append({
                    'value': f"{match.group(1)}x{match.group(2)}",
                    'type': 'resolution',
                    'text': match.group(0)
                })
            elif pattern_type == 'ratio':
                results.append({
                    'value': f"{match.group(1)}:{match.group(2)}",
                    'type': 'ratio',
                    'text': match.group(0)
                })
            else:
                results.append({
                    'value': match.group(1),
                    'unit': match.group(2) if len(match.groups()) > 1 else '',
                    'type': pattern_type,
                    'text': match.group(0)
                })
    
    return results


def normalize_product_model(text: str) -> str:
    """
    标准化产品型号
    """
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 标准化TCL产品型号
    text = re.sub(r'TCL\s*-?\s*', 'TCL-', text, flags=re.IGNORECASE)
    
    # 统一大写
    if re.match(r'TCL-', text, re.IGNORECASE):
        text = text.upper()
    
    return text


def extract_technical_terms(text: str, lang: Optional[str] = None) -> List[str]:
    """
    提取技术术语
    """
    if lang is None:
        lang = detect_language(text)
    
    technical_patterns = {
        'zh': [
            r'[A-Z]+(?:\s+[A-Z]+)*',  # 英文缩写
            r'\d+K',  # 分辨率
            r'\d+(?:G|GB|T|TB)',  # 存储容量
            r'(?:高|标准|超)清',  # 清晰度
            r'(?:智能|语音|远程)控制',  # 控制方式
            r'(?:节能|省电|低功耗)模式',  # 能效相关
        ],
        'en': [
            r'\b[A-Z]{2,}\b',  # 缩写
            r'\d+K\b',  # 分辨率
            r'\d+(?:G|GB|T|TB)\b',  # 存储
            r'(?:HD|FHD|UHD|4K|8K)\b',  # 清晰度
            r'(?:smart|voice|remote)\s+control',  # 控制
            r'(?:energy|power)\s+(?:saving|efficient)',  # 能效
        ]
    }
    
    terms = set()
    patterns = technical_patterns.get(lang, technical_patterns['en'])
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        terms.update(matches)
    
    # 添加领域特定术语
    domain_terms = TCL_DOMAIN_TERMS.get(lang, set())
    for term in domain_terms:
        if term.lower() in text.lower():
            terms.add(term)
    
    return list(terms)


def clean_text(text: str, remove_punctuation: bool = False) -> str:
    """
    清理文本
    """
    # 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # 移除特殊字符
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    
    if remove_punctuation:
        # 保留基本标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff。！？，、；：""''（）\[\]【】\-\n]', ' ', text)
    
    return text.strip()


def is_valid_sentence(sentence: str, min_length: int = 5) -> bool:
    """
    检查句子是否有效
    """
    # 移除空白后检查长度
    cleaned = sentence.strip()
    if len(cleaned) < min_length:
        return False
    
    # 检查是否包含实质内容
    words = segment_text(cleaned)
    if len(words) < 2:
        return False
    
    # 检查是否只是标点或特殊字符
    if re.match(r'^[\W_]+$', cleaned):
        return False
    
    return True


# 导出所有函数
__all__ = [
    'detect_language',
    'segment_text',
    'extract_sentences',
    'extract_keywords',
    'extract_named_entities',
    'calculate_text_similarity',
    'extract_numbers_and_units',
    'normalize_product_model',
    'extract_technical_terms',
    'clean_text',
    'is_valid_sentence',
    'CHINESE_STOPWORDS',
    'ENGLISH_STOPWORDS',
    'TCL_DOMAIN_TERMS'
]