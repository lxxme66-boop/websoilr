#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Utils - NLP工具函数
提供中文文本处理、关键词提取、相似度计算等功能
"""

import re
import logging
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter

try:
    import jieba
    import jieba.analyse
    import jieba.posseg as pseg
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError as e:
    logging.warning(f"某些NLP库未安装: {e}")


def tokenize_chinese_text(text: str, use_jieba: bool = True) -> List[str]:
    """
    中文文本分词
    
    Args:
        text: 输入文本
        use_jieba: 是否使用jieba分词
        
    Returns:
        分词结果列表
    """
    if not text:
        return []
    
    if use_jieba:
        try:
            # 使用jieba分词
            tokens = list(jieba.cut(text))
            # 过滤空字符串和单字符（除了有意义的单字）
            meaningful_single_chars = {'我', '你', '他', '她', '它', '的', '了', '是', '在', '有', '和', '与'}
            filtered_tokens = []
            for token in tokens:
                token = token.strip()
                if len(token) > 1 or token in meaningful_single_chars:
                    filtered_tokens.append(token)
            return filtered_tokens
        except Exception as e:
            logging.warning(f"jieba分词失败: {e}")
    
    # 备用方案：基于正则表达式的简单分词
    # 分离中文字符、英文单词、数字
    pattern = r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+'
    tokens = re.findall(pattern, text)
    return tokens


def extract_keywords(text: str, method: str = "tfidf", top_k: int = 10) -> List[Tuple[str, float]]:
    """
    提取关键词
    
    Args:
        text: 输入文本
        method: 提取方法 ("tfidf", "textrank", "frequency")
        top_k: 返回前k个关键词
        
    Returns:
        关键词列表，每个元素为(词, 权重)
    """
    if not text:
        return []
    
    try:
        if method == "textrank":
            # 使用jieba的TextRank算法
            keywords = jieba.analyse.textrank(text, topK=top_k, withWeight=True)
            return list(keywords)
        
        elif method == "tfidf":
            # 使用jieba的TF-IDF算法
            keywords = jieba.analyse.extract_tags(text, topK=top_k, withWeight=True)
            return list(keywords)
        
        elif method == "frequency":
            # 基于词频的简单方法
            tokens = tokenize_chinese_text(text)
            # 过滤停用词
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
            filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
            
            # 计算词频
            word_freq = Counter(filtered_tokens)
            total_words = len(filtered_tokens)
            
            # 转换为权重（归一化频率）
            keywords = [(word, freq/total_words) for word, freq in word_freq.most_common(top_k)]
            return keywords
        
    except Exception as e:
        logging.warning(f"关键词提取失败: {e}")
    
    return []


def compute_text_similarity(text1: str, text2: str, method: str = "cosine") -> float:
    """
    计算文本相似度
    
    Args:
        text1: 文本1
        text2: 文本2
        method: 相似度计算方法 ("cosine", "jaccard", "edit_distance")
        
    Returns:
        相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    try:
        if method == "cosine":
            # 使用TF-IDF向量和余弦相似度
            # 分词
            tokens1 = tokenize_chinese_text(text1)
            tokens2 = tokenize_chinese_text(text2)
            
            # 构建文档
            doc1 = ' '.join(tokens1)
            doc2 = ' '.join(tokens2)
            
            if not doc1 or not doc2:
                return 0.0
            
            # TF-IDF向量化
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
            
            # 计算余弦相似度
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        
        elif method == "jaccard":
            # Jaccard相似度
            tokens1 = set(tokenize_chinese_text(text1))
            tokens2 = set(tokenize_chinese_text(text2))
            
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            
            if not union:
                return 0.0
            
            return len(intersection) / len(union)
        
        elif method == "edit_distance":
            # 基于编辑距离的相似度
            distance = levenshtein_distance(text1, text2)
            max_len = max(len(text1), len(text2))
            
            if max_len == 0:
                return 1.0
            
            similarity = 1 - (distance / max_len)
            return max(0.0, similarity)
        
    except Exception as e:
        logging.warning(f"文本相似度计算失败: {e}")
    
    return 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算编辑距离
    
    Args:
        s1: 字符串1
        s2: 字符串2
        
    Returns:
        编辑距离
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def extract_entities_with_pos(text: str) -> List[Tuple[str, str]]:
    """
    提取实体及其词性
    
    Args:
        text: 输入文本
        
    Returns:
        实体列表，每个元素为(实体, 词性)
    """
    entities = []
    
    try:
        # 使用jieba进行词性标注
        words = pseg.cut(text)
        
        for word, flag in words:
            word = word.strip()
            if len(word) > 1:
                # 筛选可能的实体词性
                if flag in ['nr', 'ns', 'nt', 'nz', 'n', 'nrf', 'ng']:
                    entities.append((word, flag))
    
    except Exception as e:
        logging.warning(f"实体提取失败: {e}")
    
    return entities


def detect_question_type(text: str) -> str:
    """
    检测问题类型
    
    Args:
        text: 问题文本
        
    Returns:
        问题类型
    """
    text = text.lower()
    
    # 定义问题类型模式
    patterns = {
        "what": ["什么", "何为", "是什么"],
        "how": ["如何", "怎么", "怎样", "怎么样"],
        "why": ["为什么", "为何", "原因"],
        "where": ["哪里", "何处", "在哪"],
        "when": ["什么时候", "何时", "时间"],
        "who": ["谁", "什么人"],
        "which": ["哪个", "哪些", "哪种"],
        "yes_no": ["是否", "是不是", "有没有", "能否"],
        "count": ["多少", "几个", "数量"]
    }
    
    for q_type, keywords in patterns.items():
        if any(keyword in text for keyword in keywords):
            return q_type
    
    return "unknown"


def clean_and_normalize_text(text: str) -> str:
    """
    清理和标准化文本
    
    Args:
        text: 输入文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符（保留中文、英文、数字、基本标点）
    text = re.sub(r'[^\u4e00-\u9fff\w\s\.\?\!，。？！：；""''（）\[\]{}]', '', text)
    
    # 统一标点符号
    text = text.replace('?', '？')
    text = text.replace('!', '！')
    text = text.replace(',', '，')
    text = text.replace(';', '；')
    text = text.replace(':', '：')
    
    return text.strip()


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
    
    # 基于标点符号分割句子
    sentence_endings = r'[。！？；\.\!\?;]'
    sentences = re.split(sentence_endings, text)
    
    # 过滤空句子并清理
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 2:
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def extract_noun_phrases(text: str) -> List[str]:
    """
    提取名词短语
    
    Args:
        text: 输入文本
        
    Returns:
        名词短语列表
    """
    noun_phrases = []
    
    try:
        # 使用词性标注找到名词短语
        words = list(pseg.cut(text))
        
        current_phrase = []
        for word, flag in words:
            word = word.strip()
            
            # 如果是名词相关词性，添加到当前短语
            if flag.startswith('n') or flag in ['a', 'ad']:  # 名词或形容词
                current_phrase.append(word)
            else:
                # 如果当前短语不为空，保存并重置
                if current_phrase:
                    phrase = ''.join(current_phrase)
                    if len(phrase) > 1:
                        noun_phrases.append(phrase)
                    current_phrase = []
        
        # 处理最后一个短语
        if current_phrase:
            phrase = ''.join(current_phrase)
            if len(phrase) > 1:
                noun_phrases.append(phrase)
    
    except Exception as e:
        logging.warning(f"名词短语提取失败: {e}")
    
    return noun_phrases