"""配置加载与管理模块"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class Config:
    """应用配置管理器，支持从 YAML 文件加载配置"""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: Optional[str] = None) -> None:
        """加载配置文件"""
        if config_path is None:
            # 默认查找项目根目录下的 config.yaml
            project_root = Path(__file__).parent.parent
            config_path = str(project_root / "config.yaml")

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持点号分隔的嵌套路径"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def get_word_lists(self) -> Dict[str, Dict[str, Any]]:
        """获取所有敏感词库配置"""
        return self._config.get("word_lists", {})

    def get_categories(self) -> List[str]:
        """获取已启用的违规类别列表"""
        word_lists = self.get_word_lists()
        return [
            category
            for category, config in word_lists.items()
            if config.get("enabled", True)
        ]

    def get_severity(self, category: str) -> str:
        """获取指定类别的严重等级"""
        word_lists = self.get_word_lists()
        return word_lists.get(category, {}).get("severity", "medium")

    def get_model_config(self) -> Dict[str, Any]:
        """获取模型推理配置"""
        return self._config.get("model", {})

    def get_detector_config(self) -> Dict[str, Any]:
        """获取检测引擎配置"""
        return self._config.get("detector", {})

    def get_preprocessor_config(self) -> Dict[str, Any]:
        """获取预处理器配置"""
        return self._config.get("preprocessor", {})

    def get_server_config(self) -> Dict[str, Any]:
        """获取服务器配置"""
        return self._config.get("server", {})

    def get_auth_config(self) -> Dict[str, Any]:
        """获取鉴权配置"""
        return self._config.get("auth", {})

    def get_whitelist_config(self) -> Dict[str, Any]:
        """获取白名单配置"""
        return self._config.get("whitelist", {})

    def get_storage_config(self) -> Dict[str, Any]:
        """获取存储配置"""
        return self._config.get("storage", {})

    def get_metrics_config(self) -> Dict[str, Any]:
        """获取监控指标配置"""
        return self._config.get("metrics", {})

    @classmethod
    def reset(cls):
        """重置单例（用于测试）"""
        cls._instance = None
        cls._config = {}


# 全局配置实例
config = Config()
