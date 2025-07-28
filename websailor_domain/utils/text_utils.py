"""
文本处理工具函数
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import re

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict:
    """加载配置文件"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    logger.info(f"成功加载配置文件: {config_path}")
    return config


def save_json(data: Union[Dict, List], output_path: Union[str, Path], indent: int = 2):
    """保存JSON文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    
    logger.info(f"数据已保存到: {output_path}")


def load_json(input_path: Union[str, Path]) -> Union[Dict, List]:
    """加载JSON文件"""
    input_path = Path(input_path)
    
    if not input_path.exists():
        logger.error(f"文件不存在: {input_path}")
        raise FileNotFoundError(f"File not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def load_text_files(directory: Union[str, Path], pattern: str = "*.txt") -> List[Dict]:
    """加载目录下的所有文本文件"""
    directory = Path(directory)
    
    if not directory.exists():
        logger.error(f"目录不存在: {directory}")
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    text_files = []
    for file_path in directory.glob(pattern):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            text_files.append({
                "filename": file_path.name,
                "path": str(file_path),
                "content": content
            })
    
    logger.info(f"从 {directory} 加载了 {len(text_files)} 个文本文件")
    return text_files


def extract_sentences(text: str) -> List[str]:
    """从文本中提取句子"""
    # 中文句子分割
    sentences = re.split(r'[。！？；\n]+', text)
    
    # 过滤空句子
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def extract_paragraphs(text: str) -> List[str]:
    """从文本中提取段落"""
    # 按空行分割段落
    paragraphs = re.split(r'\n\s*\n', text)
    
    # 过滤空段落
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs


def normalize_whitespace(text: str) -> str:
    """规范化空白字符"""
    # 将多个空格替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 移除首尾空白
    text = text.strip()
    
    return text


def extract_tcl_terms(text: str) -> List[str]:
    """提取TCL相关专业术语"""
    # TCL专业术语模式
    tcl_patterns = [
        r'TCL[A-Za-z0-9\-]*',  # TCL开头的术语
        r'[A-Z]{2,}',  # 大写缩写
        r'\d+[A-Za-z]+',  # 数字+字母组合（如4K、8K）
        r'[A-Za-z]+\d+',  # 字母+数字组合（如LED65）
    ]
    
    terms = []
    for pattern in tcl_patterns:
        matches = re.findall(pattern, text)
        terms.extend(matches)
    
    # 去重
    terms = list(set(terms))
    
    return terms


def mask_numbers(text: str, mask_token: str = "[NUM]") -> str:
    """将文本中的数字替换为掩码"""
    # 保留带单位的数字
    text = re.sub(r'\b\d+\.?\d*\b(?![A-Za-z])', mask_token, text)
    
    return text


def extract_entities_by_pattern(text: str, patterns: Dict[str, str]) -> Dict[str, List[str]]:
    """根据模式提取实体"""
    entities = {}
    
    for entity_type, pattern in patterns.items():
        matches = re.findall(pattern, text)
        entities[entity_type] = list(set(matches))  # 去重
    
    return entities


def calculate_text_statistics(text: str) -> Dict:
    """计算文本统计信息"""
    from utils.nlp_utils import segment_text
    
    words = segment_text(text)
    sentences = extract_sentences(text)
    
    stats = {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        "avg_sentence_length": sum(len(s) for s in sentences) / len(sentences) if sentences else 0,
        "unique_words": len(set(words)),
        "lexical_diversity": len(set(words)) / len(words) if words else 0
    }
    
    return stats


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """截断文本到指定长度"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix