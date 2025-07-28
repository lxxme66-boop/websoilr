"""
文本处理工具函数
提供日志设置、文件IO、文本清理等功能
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import yaml
from datetime import datetime


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    设置日志配置
    
    Args:
        level: 日志级别
        log_file: 日志文件路径（可选）
    """
    # 设置日志级别
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置处理器
    handlers = []
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    handlers.append(console_handler)
    
    # 文件处理器（如果指定）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)
    
    # 配置根日志器
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )
    
    # 设置第三方库的日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('networkx').setLevel(logging.WARNING)


def load_json(file_path: str) -> Any:
    """
    加载JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        解析后的数据
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2,
              ensure_ascii: bool = False) -> None:
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 输出文件路径
        indent: 缩进级别
        ensure_ascii: 是否确保ASCII编码
    """
    file_path = Path(file_path)
    
    # 创建父目录
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def load_yaml(file_path: str) -> Any:
    """
    加载YAML文件
    
    Args:
        file_path: YAML文件路径
        
    Returns:
        解析后的数据
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Any, file_path: str) -> None:
    """
    保存数据到YAML文件
    
    Args:
        data: 要保存的数据
        file_path: 输出文件路径
    """
    file_path = Path(file_path)
    
    # 创建父目录
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def clean_text(text: str, remove_extra_spaces: bool = True,
               remove_special_chars: bool = False) -> str:
    """
    清理文本
    
    Args:
        text: 原始文本
        remove_extra_spaces: 是否移除多余空格
        remove_special_chars: 是否移除特殊字符
        
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 移除前后空白
    text = text.strip()
    
    # 移除多余空格
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
    
    # 移除特殊字符
    if remove_special_chars:
        # 保留中文、英文、数字和基本标点
        text = re.sub(
            r'[^\u4e00-\u9fa5a-zA-Z0-9\s。，！？；：""''（）《》【】、—…·.!?,;:\'"()\[\]{}<>/-]',
            '', text
        )
    
    return text


def format_output(data: Any, format_type: str = 'json') -> str:
    """
    格式化输出数据
    
    Args:
        data: 要格式化的数据
        format_type: 格式类型 ('json', 'yaml', 'text')
        
    Returns:
        格式化后的字符串
    """
    if format_type == 'json':
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    elif format_type == 'yaml':
        return yaml.dump(data, allow_unicode=True, default_flow_style=False)
    
    elif format_type == 'text':
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                lines.append(f"{key}: {value}")
            return '\n'.join(lines)
        elif isinstance(data, list):
            return '\n'.join(str(item) for item in data)
        else:
            return str(data)
    
    else:
        raise ValueError(f"不支持的格式类型: {format_type}")


def read_text_file(file_path: str, encoding: str = 'utf-8') -> str:
    """
    读取文本文件
    
    Args:
        file_path: 文件路径
        encoding: 文件编码
        
    Returns:
        文件内容
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def write_text_file(content: str, file_path: str,
                   encoding: str = 'utf-8') -> None:
    """
    写入文本文件
    
    Args:
        content: 要写入的内容
        file_path: 文件路径
        encoding: 文件编码
    """
    file_path = Path(file_path)
    
    # 创建父目录
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(content)


def read_lines(file_path: str, encoding: str = 'utf-8',
               skip_empty: bool = True) -> List[str]:
    """
    按行读取文件
    
    Args:
        file_path: 文件路径
        encoding: 文件编码
        skip_empty: 是否跳过空行
        
    Returns:
        行列表
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    
    # 处理行
    lines = [line.rstrip('\n\r') for line in lines]
    
    if skip_empty:
        lines = [line for line in lines if line.strip()]
    
    return lines


def create_timestamp() -> str:
    """
    创建时间戳字符串
    
    Returns:
        格式化的时间戳
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_dir(dir_path: str) -> Path:
    """
    确保目录存在
    
    Args:
        dir_path: 目录路径
        
    Returns:
        Path对象
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size(file_path: str) -> int:
    """
    获取文件大小（字节）
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件大小
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    return file_path.stat().st_size


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节大小
        
    Returns:
        格式化的大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} PB"


def truncate_text(text: str, max_length: int,
                  suffix: str = '...') -> str:
    """
    截断文本
    
    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后缀
        
    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    
    truncate_length = max_length - len(suffix)
    return text[:truncate_length] + suffix


def split_text_into_chunks(text: str, chunk_size: int,
                          overlap: int = 0) -> List[str]:
    """
    将文本分割成块
    
    Args:
        text: 原始文本
        chunk_size: 块大小
        overlap: 重叠大小
        
    Returns:
        文本块列表
    """
    if not text or chunk_size <= 0:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # 移动到下一个块的起始位置
        start = end - overlap
        
        # 确保不会无限循环
        if overlap >= chunk_size:
            start = end
    
    return chunks


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个字典
    
    Args:
        *dicts: 要合并的字典
        
    Returns:
        合并后的字典
    """
    result = {}
    
    for d in dicts:
        if d:
            result.update(d)
    
    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = '',
                 sep: str = '.') -> Dict[str, Any]:
    """
    扁平化嵌套字典
    
    Args:
        d: 嵌套字典
        parent_key: 父键
        sep: 分隔符
        
    Returns:
        扁平化的字典
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)