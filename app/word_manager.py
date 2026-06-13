"""敏感词库管理器 — 管理敏感词库的生命周期，支持热加载"""

import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from app.config import Config
from app.logger import get_logger
from app.trie import TrieTree

logger = get_logger("word_manager")


class WordManager:
    """
    敏感词库管理器

    功能：
    - 按类别加载/管理敏感词
    - 支持运行时热加载（文件变更后自动重建 Trie 树）
    - 支持动态添加/删除词汇
    """

    def __init__(self, config: Config, base_dir: Optional[str] = None):
        self._config = config
        self._base_dir = base_dir or str(Path(__file__).parent.parent)
        self._trees: Dict[str, TrieTree] = {}
        self._word_sets: Dict[str, Set[str]] = {}
        self._file_mtimes: Dict[str, float] = {}
        self._lock = threading.Lock()
        # 初始化预处理器（用于词汇归一化）
        from app.preprocessor import Preprocessor
        prep_cfg = config.get_preprocessor_config()
        self._preprocessor = Preprocessor(
            skip_chars=prep_cfg.get("skip_chars", ""),
            traditional_to_simplified=prep_cfg.get("traditional_to_simplified", True),
            normalize_variants=prep_cfg.get("normalize_variants", True),
            remove_special_chars=prep_cfg.get("remove_special_chars", True),
        )

    def load_all(self) -> Dict[str, int]:
        """
        加载所有已配置的敏感词库

        Returns:
            各类别加载的词数
        """
        counts = {}
        word_lists = self._config.get_word_lists()

        for category, cfg in word_lists.items():
            if not cfg.get("enabled", True):
                logger.info(f"类别 [{category}] 已禁用，跳过加载")
                continue

            filepath = self._resolve_path(cfg["path"])
            count = self._load_category(category, filepath)
            counts[category] = count
            logger.info(f"类别 [{category}] 加载完成，共 {count} 个词")

        return counts

    def load_category(self, category: str) -> int:
        """加载指定类别的词库"""
        word_lists = self._config.get_word_lists()
        cfg = word_lists.get(category)
        if not cfg:
            raise ValueError(f"未知类别: {category}")

        filepath = self._resolve_path(cfg["path"])
        return self._load_category(category, filepath)

    def _load_category(self, category: str, filepath: str) -> int:
        """内部：加载指定类别的词库"""
        with self._lock:
            tree = TrieTree()
            words = set()

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        word = line.strip()
                        if word and not word.startswith("#"):
                            normalized = self._preprocessor.normalize(word)
                            tree.insert(normalized)
                            words.add(word)  # 保留原始词用于展示
            except FileNotFoundError:
                logger.warning(f"词库文件不存在: {filepath}，将创建空词库")
                # 创建空文件
                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                Path(filepath).touch()

            self._trees[category] = tree
            self._word_sets[category] = words

            # 记录文件修改时间
            try:
                self._file_mtimes[category] = os.path.getmtime(filepath)
            except OSError:
                self._file_mtimes[category] = 0

            return tree.size

    def get_tree(self, category: str) -> Optional[TrieTree]:
        """获取指定类别的 Trie 树"""
        return self._trees.get(category)

    def get_all_trees(self) -> Dict[str, TrieTree]:
        """获取所有已加载的 Trie 树"""
        return self._trees.copy()

    def get_categories(self) -> List[str]:
        """获取已加载的类别列表"""
        return list(self._trees.keys())

    def get_words(self, category: str) -> Set[str]:
        """获取指定类别的所有词汇"""
        return self._word_sets.get(category, set()).copy()

    def get_word_count(self, category: str) -> int:
        """获取指定类别的词库大小"""
        tree = self._trees.get(category)
        return tree.size if tree else 0

    def get_total_count(self) -> int:
        """获取所有词库的总词数"""
        return sum(tree.size for tree in self._trees.values())

    def add_word(self, category: str, word: str) -> bool:
        """
        动态添加敏感词

        Returns:
            是否添加成功（已存在则返回 False）
        """
        word = word.strip()
        if not word:
            return False

        # 归一化词汇（与检测时的预处理保持一致）
        normalized = self._preprocessor.normalize(word)

        with self._lock:
            if category not in self._trees:
                self._trees[category] = TrieTree()
                self._word_sets[category] = set()

            tree = self._trees[category]
            if normalized in self._word_sets[category]:
                return False

            tree.insert(normalized)
            self._word_sets[category].add(normalized)
            logger.info(f"添加词汇: [{category}] {word} -> 归一化: {normalized}")
            return True

    def remove_word(self, category: str, word: str) -> bool:
        """动态删除敏感词"""
        word = word.strip()
        if not word:
            return False

        normalized = self._preprocessor.normalize(word)

        with self._lock:
            tree = self._trees.get(category)
            if not tree:
                return False

            if tree.remove(normalized):
                self._word_sets[category].discard(normalized)
                logger.info(f"删除词汇: [{category}] {word}")
                return True
            return False

    def add_words(self, category: str, words: List[str]) -> int:
        """批量添加词汇，返回实际添加数量"""
        added = 0
        for word in words:
            if self.add_word(category, word):
                added += 1
        return added

    def remove_words(self, category: str, words: List[str]) -> int:
        """批量删除词汇，返回实际删除数量"""
        removed = 0
        for word in words:
            if self.remove_word(category, word):
                removed += 1
        return removed

    def reload(self, category: Optional[str] = None) -> Dict[str, int]:
        """
        重新加载词库

        Args:
            category: 指定类别，None 则重载所有

        Returns:
            各类别加载的词数
        """
        if category:
            count = self.load_category(category)
            return {category: count}
        return self.load_all()

    def check_and_reload(self) -> Dict[str, int]:
        """
        检查文件变更并自动重载

        Returns:
            发生变更的类别及其新词数
        """
        reloaded = {}
        word_lists = self._config.get_word_lists()

        for category, cfg in word_lists.items():
            if not cfg.get("enabled", True):
                continue

            filepath = self._resolve_path(cfg["path"])
            try:
                mtime = os.path.getmtime(filepath)
            except OSError:
                continue

            if mtime > self._file_mtimes.get(category, 0):
                logger.info(f"检测到词库文件变更: [{category}]，正在重载...")
                count = self._load_category(category, filepath)
                reloaded[category] = count

        return reloaded

    def export_words(self, category: str) -> str:
        """导出指定类别的词库内容"""
        words = self.get_words(category)
        return "\n".join(sorted(words))

    def _resolve_path(self, path: str) -> str:
        """解析相对路径为绝对路径"""
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(Path(self._base_dir) / p)
