"""
模型管理器
支持多个模型并行加载和处理，提高系统效率
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import threading
import queue
import torch
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    pipeline
)
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    model_type: str  # ner, relation_extraction, question_generation, etc.
    model_path: str
    device: str  # cuda:0, cuda:1, cpu
    batch_size: int = 32
    max_length: int = 512
    num_workers: int = 2


class BaseModel(ABC):
    """基础模型类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
    @abstractmethod
    def load(self):
        """加载模型"""
        pass
        
    @abstractmethod
    def predict(self, inputs: Union[str, List[str]]) -> Any:
        """预测"""
        pass
        
    def batch_predict(self, inputs: List[str]) -> List[Any]:
        """批量预测"""
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_results = self.predict(batch)
            results.extend(batch_results)
            
        return results


class NERModel(BaseModel):
    """命名实体识别模型"""
    
    def load(self):
        """加载NER模型"""
        self.logger.info(f"Loading NER model on {self.device}")
        
        # 支持多种NER模型
        if "bert" in self.config.model_path.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.config.model_path
            ).to(self.device)
        else:
            # 使用pipeline简化
            self.model = pipeline(
                "ner",
                model=self.config.model_path,
                device=0 if self.device.type == 'cuda' else -1
            )
            
        self.model.eval()
        self.logger.info("NER model loaded successfully")
        
    def predict(self, inputs: Union[str, List[str]]) -> List[Dict]:
        """执行NER预测"""
        if isinstance(inputs, str):
            inputs = [inputs]
            
        if hasattr(self.model, 'predict'):
            # 使用pipeline
            results = self.model(inputs)
        else:
            # 使用原始模型
            results = []
            with torch.no_grad():
                for text in inputs:
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding=True,
                        max_length=self.config.max_length,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    outputs = self.model(**encoding)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    
                    # 解码预测结果
                    tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
                    labels = [self.model.config.id2label[p.item()] for p in predictions[0]]
                    
                    # 提取实体
                    entities = self._extract_entities(tokens, labels)
                    results.append(entities)
                    
        return results
        
    def _extract_entities(self, tokens: List[str], labels: List[str]) -> List[Dict]:
        """从标签序列提取实体"""
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': token,
                    'type': label[2:],
                    'start': i,
                    'end': i + 1
                }
            elif label.startswith('I-') and current_entity:
                current_entity['text'] += token
                current_entity['end'] = i + 1
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                    
        if current_entity:
            entities.append(current_entity)
            
        return entities


class RelationExtractionModel(BaseModel):
    """关系抽取模型"""
    
    def load(self):
        """加载关系抽取模型"""
        self.logger.info(f"Loading RE model on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_path
        ).to(self.device)
        
        self.model.eval()
        self.logger.info("RE model loaded successfully")
        
    def predict(self, inputs: List[Dict]) -> List[Dict]:
        """
        预测关系
        inputs: [{'text': str, 'entity1': str, 'entity2': str}]
        """
        results = []
        
        with torch.no_grad():
            for item in inputs:
                # 构造输入
                text = f"{item['entity1']} [SEP] {item['entity2']} [SEP] {item['text']}"
                
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**encoding)
                predictions = torch.softmax(outputs.logits, dim=-1)
                
                # 获取最可能的关系
                relation_id = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][relation_id].item()
                
                results.append({
                    'entity1': item['entity1'],
                    'entity2': item['entity2'],
                    'relation': self.model.config.id2label[relation_id],
                    'confidence': confidence
                })
                
        return results


class QuestionGenerationModel(BaseModel):
    """问题生成模型"""
    
    def load(self):
        """加载问题生成模型"""
        self.logger.info(f"Loading QG model on {self.device}")
        
        # 使用T5或BART等生成模型
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.config.model_path
        ).to(self.device)
        
        self.model.eval()
        self.logger.info("QG model loaded successfully")
        
    def predict(self, inputs: List[Dict]) -> List[str]:
        """
        生成问题
        inputs: [{'context': str, 'answer': str}]
        """
        results = []
        
        with torch.no_grad():
            for item in inputs:
                # 构造输入
                input_text = f"generate question: context: {item['context']} answer: {item['answer']}"
                
                encoding = self.tokenizer(
                    input_text,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # 生成问题
                output_ids = self.model.generate(
                    **encoding,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True
                )
                
                question = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                results.append(question)
                
        return results


class ModelManager:
    """
    模型管理器
    负责多个模型的并行加载和调度
    """
    
    def __init__(self, configs: List[ModelConfig], max_workers: int = 4):
        self.configs = configs
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, BaseModel] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._loading_futures: Dict[str, Future] = {}
        
    def load_models_parallel(self):
        """并行加载所有模型"""
        self.logger.info(f"Starting parallel loading of {len(self.configs)} models")
        
        # 提交所有模型加载任务
        for config in self.configs:
            future = self.executor.submit(self._load_single_model, config)
            self._loading_futures[config.name] = future
            
        # 等待所有模型加载完成
        for name, future in self._loading_futures.items():
            try:
                model = future.result(timeout=300)  # 5分钟超时
                with self._lock:
                    self.models[name] = model
                self.logger.info(f"Model {name} loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load model {name}: {str(e)}")
                
        self.logger.info("All models loaded")
        
    def _load_single_model(self, config: ModelConfig) -> BaseModel:
        """加载单个模型"""
        # 根据模型类型创建相应的模型实例
        model_class_map = {
            'ner': NERModel,
            'relation_extraction': RelationExtractionModel,
            'question_generation': QuestionGenerationModel,
        }
        
        model_class = model_class_map.get(config.model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {config.model_type}")
            
        model = model_class(config)
        model.load()
        
        return model
        
    def get_model(self, name: str) -> Optional[BaseModel]:
        """获取指定模型"""
        return self.models.get(name)
        
    def predict_parallel(
        self, 
        model_names: List[str], 
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        并行调用多个模型进行预测
        """
        futures = {}
        
        # 提交预测任务
        for name in model_names:
            model = self.get_model(name)
            if model:
                model_input = inputs.get(name, inputs.get('default'))
                future = self.executor.submit(model.predict, model_input)
                futures[name] = future
            else:
                self.logger.warning(f"Model {name} not found")
                
        # 收集结果
        results = {}
        for name, future in futures.items():
            try:
                result = future.result(timeout=60)
                results[name] = result
            except Exception as e:
                self.logger.error(f"Prediction failed for {name}: {str(e)}")
                results[name] = None
                
        return results
        
    def process_pipeline(
        self, 
        text: str,
        pipeline_config: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        执行处理管道
        pipeline_config: [
            {'model': 'ner', 'input_key': 'text', 'output_key': 'entities'},
            {'model': 're', 'input_key': 'entities', 'output_key': 'relations'},
            ...
        ]
        """
        context = {'text': text}
        
        for step in pipeline_config:
            model_name = step['model']
            input_key = step.get('input_key', 'text')
            output_key = step.get('output_key', 'result')
            
            model = self.get_model(model_name)
            if not model:
                self.logger.error(f"Model {model_name} not found in pipeline")
                continue
                
            # 获取输入
            model_input = context.get(input_key)
            if model_input is None:
                self.logger.error(f"Input {input_key} not found for {model_name}")
                continue
                
            # 执行预测
            try:
                result = model.predict(model_input)
                context[output_key] = result
            except Exception as e:
                self.logger.error(f"Pipeline step {model_name} failed: {str(e)}")
                context[output_key] = None
                
        return context
        
    def shutdown(self):
        """关闭模型管理器"""
        self.executor.shutdown(wait=True)
        
        # 清理GPU内存
        for model in self.models.values():
            if hasattr(model.model, 'to'):
                model.model.to('cpu')
        torch.cuda.empty_cache()


class AsyncModelManager(ModelManager):
    """
    异步模型管理器
    支持流式处理和更高的并发
    """
    
    def __init__(self, configs: List[ModelConfig], max_workers: int = 4):
        super().__init__(configs, max_workers)
        self.request_queues: Dict[str, queue.Queue] = {}
        self.result_queues: Dict[str, queue.Queue] = {}
        self.workers: Dict[str, threading.Thread] = {}
        
    def start_async_workers(self):
        """启动异步工作线程"""
        for name, model in self.models.items():
            # 创建请求和结果队列
            self.request_queues[name] = queue.Queue(maxsize=1000)
            self.result_queues[name] = queue.Queue(maxsize=1000)
            
            # 启动工作线程
            worker = threading.Thread(
                target=self._async_worker,
                args=(name, model),
                daemon=True
            )
            worker.start()
            self.workers[name] = worker
            
    def _async_worker(self, name: str, model: BaseModel):
        """异步工作线程"""
        request_queue = self.request_queues[name]
        result_queue = self.result_queues[name]
        
        while True:
            try:
                # 获取请求
                request_id, data = request_queue.get(timeout=1)
                
                if request_id is None:  # 停止信号
                    break
                    
                # 执行预测
                result = model.predict(data)
                
                # 返回结果
                result_queue.put((request_id, result))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {name} error: {str(e)}")
                result_queue.put((request_id, None))
                
    def async_predict(
        self, 
        model_name: str, 
        data: Any, 
        request_id: str
    ) -> str:
        """异步预测"""
        if model_name not in self.request_queues:
            raise ValueError(f"Model {model_name} not found")
            
        self.request_queues[model_name].put((request_id, data))
        return request_id
        
    def get_result(
        self, 
        model_name: str, 
        request_id: str, 
        timeout: float = None
    ) -> Any:
        """获取异步结果"""
        if model_name not in self.result_queues:
            raise ValueError(f"Model {model_name} not found")
            
        result_queue = self.result_queues[model_name]
        
        try:
            while True:
                rid, result = result_queue.get(timeout=timeout)
                if rid == request_id:
                    return result
                else:
                    # 放回队列
                    result_queue.put((rid, result))
        except queue.Empty:
            return None


# 使用示例配置
def create_default_configs() -> List[ModelConfig]:
    """创建默认的模型配置"""
    configs = [
        ModelConfig(
            name="ner_bert",
            model_type="ner",
            model_path="bert-base-chinese",
            device="cuda:0",
            batch_size=32
        ),
        ModelConfig(
            name="re_bert",
            model_type="relation_extraction",
            model_path="bert-base-chinese",
            device="cuda:1",
            batch_size=16
        ),
        ModelConfig(
            name="qg_t5",
            model_type="question_generation",
            model_path="t5-base",
            device="cuda:0",
            batch_size=8
        )
    ]
    
    return configs