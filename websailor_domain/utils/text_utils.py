"""
文本处理工具函数模块
提供文本预处理、格式化、模板渲染等功能
"""

import re
import json
import yaml
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
import random
from string import Template
import hashlib

logger = logging.getLogger(__name__)


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载JSON文件
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"JSON文件不存在: {file_path}")
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误 {file_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"读取JSON文件失败 {file_path}: {e}")
        return {}


def save_json_file(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    保存数据到JSON文件
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        return True
    except Exception as e:
        logger.error(f"保存JSON文件失败 {file_path}: {e}")
        return False


def load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载YAML文件
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"YAML文件不存在: {file_path}")
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"YAML解析错误 {file_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"读取YAML文件失败 {file_path}: {e}")
        return {}


def save_yaml_file(data: Any, file_path: Union[str, Path]) -> bool:
    """
    保存数据到YAML文件
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        return True
    except Exception as e:
        logger.error(f"保存YAML文件失败 {file_path}: {e}")
        return False


def load_text_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """
    加载文本文件，自动处理编码
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"文本文件不存在: {file_path}")
        return ""
    
    # 尝试多种编码
    encodings = [encoding, 'utf-8', 'gbk', 'gb2312', 'big5', 'latin-1']
    
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                content = f.read()
                logger.debug(f"成功使用编码 {enc} 读取文件: {file_path}")
                return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return ""
    
    logger.error(f"无法使用任何编码读取文件: {file_path}")
    return ""


def save_text_file(content: str, file_path: Union[str, Path], encoding: str = 'utf-8') -> bool:
    """
    保存文本到文件
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"保存文本文件失败 {file_path}: {e}")
        return False


def render_template(template_str: str, variables: Dict[str, Any]) -> str:
    """
    渲染模板字符串
    支持 ${variable} 格式
    """
    try:
        template = Template(template_str)
        return template.safe_substitute(variables)
    except Exception as e:
        logger.error(f"模板渲染失败: {e}")
        return template_str


def extract_json_from_text(text: str) -> List[Dict[str, Any]]:
    """
    从文本中提取JSON对象
    """
    json_objects = []
    
    # 查找JSON块
    json_pattern = r'\{[^{}]*\}'
    matches = re.finditer(json_pattern, text)
    
    for match in matches:
        json_str = match.group()
        try:
            obj = json.loads(json_str)
            json_objects.append(obj)
        except json.JSONDecodeError:
            # 尝试更复杂的JSON块
            pass
    
    # 尝试查找嵌套的JSON
    nested_pattern = r'\{(?:[^{}]|\{[^{}]*\})*\}'
    nested_matches = re.finditer(nested_pattern, text)
    
    for match in nested_matches:
        json_str = match.group()
        try:
            obj = json.loads(json_str)
            if obj not in json_objects:  # 避免重复
                json_objects.append(obj)
        except json.JSONDecodeError:
            pass
    
    return json_objects


def format_json_output(data: Any, indent: int = 2) -> str:
    """
    格式化JSON输出
    """
    return json.dumps(data, ensure_ascii=False, indent=indent)


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    截断文本到指定长度
    """
    if len(text) <= max_length:
        return text
    
    # 在词边界截断
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # 如果空格位置合理
        truncated = truncated[:last_space]
    
    return truncated + suffix


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    将文本分割成重叠的块
    """
    if not text or chunk_size <= 0:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 尝试在句子边界结束
        if end < len(text):
            # 查找句号、问号、感叹号
            sentence_ends = ['.', '。', '!', '！', '?', '？']
            best_end = end
            
            for i in range(end, max(start + chunk_size // 2, start), -1):
                if text[i-1:i] in sentence_ends:
                    best_end = i
                    break
            
            end = best_end
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 下一个块的起始位置（考虑重叠）
        start = end - overlap if end < len(text) else end
    
    return chunks


def normalize_whitespace(text: str) -> str:
    """
    标准化空白字符
    """
    # 替换各种空白字符为标准空格
    text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]', ' ', text)
    
    # 合并多个空格
    text = re.sub(r' +', ' ', text)
    
    # 合并多个换行
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


def remove_special_chars(text: str, keep_chars: str = "") -> str:
    """
    移除特殊字符，保留字母、数字、中文和指定字符
    """
    # 基本字符集：字母、数字、中文、空白
    pattern = f'[^a-zA-Z0-9\u4e00-\u9fff\s{re.escape(keep_chars)}]'
    return re.sub(pattern, '', text)


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    从文本中提取代码块
    """
    code_blocks = []
    
    # Markdown代码块
    markdown_pattern = r'```(\w*)\n(.*?)```'
    matches = re.finditer(markdown_pattern, text, re.DOTALL)
    
    for match in matches:
        language = match.group(1) or 'text'
        code = match.group(2).strip()
        code_blocks.append({
            'language': language,
            'code': code,
            'type': 'markdown'
        })
    
    # 缩进代码块（4个空格或制表符开头）
    lines = text.split('\n')
    indent_code = []
    in_code = False
    
    for line in lines:
        if line.startswith('    ') or line.startswith('\t'):
            if not in_code:
                in_code = True
            indent_code.append(line[4:] if line.startswith('    ') else line[1:])
        else:
            if in_code and indent_code:
                code_blocks.append({
                    'language': 'text',
                    'code': '\n'.join(indent_code),
                    'type': 'indent'
                })
                indent_code = []
                in_code = False
    
    # 处理最后的代码块
    if indent_code:
        code_blocks.append({
            'language': 'text',
            'code': '\n'.join(indent_code),
            'type': 'indent'
        })
    
    return code_blocks


def generate_hash(text: str, algorithm: str = 'md5') -> str:
    """
    生成文本的哈希值
    """
    if algorithm == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'sha256':
        hasher = hashlib.sha256()
    elif algorithm == 'sha1':
        hasher = hashlib.sha1()
    else:
        raise ValueError(f"不支持的哈希算法: {algorithm}")
    
    hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()


def mask_sensitive_info(text: str) -> str:
    """
    遮蔽敏感信息（如邮箱、电话等）
    """
    # 邮箱
    text = re.sub(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        '[EMAIL]',
        text
    )
    
    # 手机号（中国）
    text = re.sub(
        r'1[3-9]\d{9}',
        '[PHONE]',
        text
    )
    
    # IP地址
    text = re.sub(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        '[IP]',
        text
    )
    
    # 身份证号（简单匹配）
    text = re.sub(
        r'\b\d{17}[\dXx]\b',
        '[ID]',
        text
    )
    
    return text


def extract_urls(text: str) -> List[str]:
    """
    从文本中提取URL
    """
    url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    urls = re.findall(url_pattern, text)
    return list(set(urls))  # 去重


def calculate_text_stats(text: str) -> Dict[str, int]:
    """
    计算文本统计信息
    """
    stats = {
        'total_chars': len(text),
        'total_chars_no_space': len(text.replace(' ', '').replace('\n', '').replace('\t', '')),
        'total_lines': text.count('\n') + 1 if text else 0,
        'total_words': len(text.split()),
        'chinese_chars': len(re.findall(r'[\u4e00-\u9fff]', text)),
        'english_chars': len(re.findall(r'[a-zA-Z]', text)),
        'digits': len(re.findall(r'\d', text)),
        'punctuation': len(re.findall(r'[^\w\s]', text)),
    }
    
    # 计算中英文词数
    import jieba
    chinese_words = list(jieba.cut(text))
    stats['chinese_words'] = len([w for w in chinese_words if re.match(r'[\u4e00-\u9fff]+', w)])
    
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    stats['english_words'] = len(english_words)
    
    return stats


def format_number(num: Union[int, float], decimal_places: int = 2) -> str:
    """
    格式化数字显示
    """
    if isinstance(num, int):
        return f"{num:,}"
    else:
        return f"{num:,.{decimal_places}f}"


def random_sample_lines(text: str, n: int = 10, seed: Optional[int] = None) -> List[str]:
    """
    随机采样文本行
    """
    if seed is not None:
        random.seed(seed)
    
    lines = text.strip().split('\n')
    if len(lines) <= n:
        return lines
    
    return random.sample(lines, n)


# 导出所有函数
__all__ = [
    'load_json_file',
    'save_json_file',
    'load_yaml_file',
    'save_yaml_file',
    'load_text_file',
    'save_text_file',
    'render_template',
    'extract_json_from_text',
    'format_json_output',
    'truncate_text',
    'split_into_chunks',
    'normalize_whitespace',
    'remove_special_chars',
    'extract_code_blocks',
    'generate_hash',
    'mask_sensitive_info',
    'extract_urls',
    'calculate_text_stats',
    'format_number',
    'random_sample_lines'
]