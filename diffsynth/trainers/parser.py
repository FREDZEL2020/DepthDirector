import yaml
import os
import glob
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json

@dataclass
class Config:
    """支持 YAML 包含的配置类"""
    
    # 数据集配置
    dataset_base_path: str = ""
    dataset_metadata_path: Optional[str] = None
    val_dataset_base_path: str = ""
    val_dataset_metadata_path: Optional[str] = None
    dataset_repeat: int = 1
    dataset_num_workers: int = 0
    
    # 数据预处理配置
    max_pixels: int = 1280 * 720
    height: Optional[int] = None
    width: Optional[int] = None
    num_frames: int = 81
    data_file_keys: str = "image,video"
    
    # 模型配置
    model_paths: Optional[str] = None
    model_id_with_origin_paths: Optional[str] = None
    remove_prefix_in_ckpt: str = "pipe.dit."
    trainable_models: Optional[str] = None
    trained_checkpoints: Optional[str] = None
    
    # LoRA 配置
    lora_base_model: Optional[str] = None
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"
    lora_rank: int = 32
    lora_checkpoint: Optional[str] = None
    
    # 训练配置
    learning_rate: float = 1e-4
    num_epochs: int = 1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_timestep_boundary: float = 1.0
    min_timestep_boundary: float = 0.0
    extra_inputs: Optional[str] = None
    enable_val = False
    # 优化器配置
    use_gradient_checkpointing_offload: bool = False
    find_unused_parameters: bool = False
    cpu_offload: bool = False
    
    # 输出配置
    output_path: str = "./models"
    save_steps: Optional[int] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """从 YAML 文件加载配置（支持包含）"""
        loader = YamlIncludeLoader()
        config_dict = loader.load_yaml(yaml_path)
        
        # 展平嵌套的字典结构
        flattened = cls._flatten_config(config_dict)
        
        # 创建 Config 实例
        return cls(**flattened)
    
    @staticmethod
    def _flatten_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """展平配置字典（递归处理嵌套）"""
        flattened = {}
        
        def _flatten(prefix: str, d: Dict[str, Any]):
            for key, value in d.items():
                new_key = f"{prefix}_{key}" if prefix else key
                if isinstance(value, dict):
                    _flatten(new_key, value)
                else:
                    flattened[new_key] = value
        
        _flatten("", config_dict)
        return flattened
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {field.name: getattr(self, field.name) 
                for field in self.__dataclass_fields__.values()}
    
    def to_json(self, json_path: str):
        """保存配置为 JSON 文件"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def merge_with_dict(self, override_dict: Dict[str, Any]) -> 'Config':
        """用字典合并配置"""
        current_dict = self.to_dict()
        current_dict.update(override_dict)
        return Config(**current_dict)

from typing import Dict, Any, Optional
import os
import yaml

class YamlIncludeLoader:
    """支持 YAML 包含功能的配置加载器"""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = base_path or os.getcwd()
        
    def load_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """加载 YAML 文件，支持包含引用"""
        abs_path = os.path.abspath(yaml_path)
        with open(abs_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析 YAML
        config = yaml.safe_load(content)
        
        # 处理包含引用（支持多种格式）
        return self._process_includes(config, os.path.dirname(abs_path))
    
    def _process_includes(self, config: Any, base_dir: str) -> Any:
        """递归处理包含引用"""
        if isinstance(config, dict):
            processed = {}
            for key, value in config.items():
                if key == '__include__':
                    # 处理 __include__ 键
                    include_config = self._load_included(value, base_dir)
                    if isinstance(include_config, dict):
                        processed.update(include_config)
                elif key.startswith('__include_'):
                    # 处理 __include_* 模式
                    pattern = key.replace('__include_', '')
                    include_config = self._load_included(value, base_dir)
                    if isinstance(include_config, dict):
                        for k, v in include_config.items():
                            processed[k] = v
                else:
                    processed[key] = self._process_includes(value, base_dir)
            return processed
        elif isinstance(config, list):
            return [self._process_includes(item, base_dir) for item in config]
        else:
            return config
    
    def _load_included(self, include_path: Any, base_dir: str) -> Any:
        """加载被包含的文件"""
        if isinstance(include_path, str):
            # 解析路径（支持相对路径）
            if not os.path.isabs(include_path):
                include_path = os.path.join(base_dir, include_path)
            
            # 支持通配符
            if '*' in include_path or '?' in include_path:
                import glob
                files = sorted(glob.glob(include_path))
                merged_config = {}
                for file in files:
                    file_config = self.load_yaml(file)
                    if isinstance(file_config, dict):
                        merged_config.update(file_config)
                return merged_config
            else:
                return self.load_yaml(include_path)
        elif isinstance(include_path, list):
            # 包含多个文件
            merged_config = {}
            for path in include_path:
                included = self._load_included(path, base_dir)
                if isinstance(included, dict):
                    merged_config.update(included)
            return merged_config
        else:
            return {}