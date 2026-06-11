"""白名单机制 — 允许特定文本绕过检测"""

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from app.logger import get_logger

logger = get_logger("whitelist")


@dataclass
class WhitelistEntry:
    """白名单条目"""
    text: str
    reason: str = ""
    categories: Optional[List[str]] = None  # None 表示绕过所有类别


class Whitelist:
    """
    白名单管理器

    支持三种匹配模式：
    - exact: 精确匹配（O(1) 查找）
    - contains: 包含匹配（子串检测）
    - regex: 正则表达式匹配

    每个条目可指定仅绕过特定类别。
    """

    def __init__(
        self,
        mode: str = "exact",
        bypass_categories: Optional[List[str]] = None,
    ):
        self._mode = mode
        self._bypass_categories = bypass_categories  # 全局绕过类别限制
        self._lock = threading.Lock()

        # exact 模式用 set
        self._exact_set: Set[str] = set()
        self._exact_entries: Dict[str, WhitelistEntry] = {}

        # contains 模式用列表
        self._contains_entries: List[WhitelistEntry] = []

        # regex 模式用编译后的正则
        self._regex_entries: List[Tuple[re.Pattern, WhitelistEntry]] = []

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def size(self) -> int:
        if self._mode == "exact":
            return len(self._exact_set)
        elif self._mode == "contains":
            return len(self._contains_entries)
        else:
            return len(self._regex_entries)

    def add(self, entry: WhitelistEntry) -> bool:
        """添加白名单条目"""
        with self._lock:
            text = entry.text.strip()
            if not text:
                return False

            if self._mode == "exact":
                if text in self._exact_set:
                    return False
                self._exact_set.add(text)
                self._exact_entries[text] = entry
            elif self._mode == "contains":
                self._contains_entries.append(entry)
            else:  # regex
                try:
                    pattern = re.compile(text)
                    self._regex_entries.append((pattern, entry))
                except re.error as e:
                    logger.warning(f"无效的正则表达式: {text}, 错误: {e}")
                    return False

            logger.debug(f"添加白名单: [{self._mode}] {text}")
            return True

    def remove(self, text: str) -> bool:
        """移除白名单条目"""
        with self._lock:
            text = text.strip()
            if not text:
                return False

            if self._mode == "exact":
                if text not in self._exact_set:
                    return False
                self._exact_set.discard(text)
                self._exact_entries.pop(text, None)
                return True
            elif self._mode == "contains":
                for i, entry in enumerate(self._contains_entries):
                    if entry.text == text:
                        self._contains_entries.pop(i)
                        return True
                return False
            else:  # regex
                for i, (pattern, entry) in enumerate(self._regex_entries):
                    if entry.text == text:
                        self._regex_entries.pop(i)
                        return True
                return False

    def is_whitelisted(
        self,
        text: str,
        categories: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        检查文本是否在白名单中

        Args:
            text: 待检查文本
            categories: 当前检测的类别列表

        Returns:
            (是否白名单, 原因)
        """
        if not text or not text.strip():
            return False, None

        text = text.strip()

        if self._mode == "exact":
            return self._check_exact(text, categories)
        elif self._mode == "contains":
            return self._check_contains(text, categories)
        else:
            return self._check_regex(text, categories)

    def _check_exact(
        self, text: str, categories: Optional[List[str]]
    ) -> Tuple[bool, Optional[str]]:
        """精确匹配检查"""
        if text in self._exact_set:
            entry = self._exact_entries[text]
            if self._should_bypass(entry, categories):
                return True, entry.reason or "whitelisted"
        return False, None

    def _check_contains(
        self, text: str, categories: Optional[List[str]]
    ) -> Tuple[bool, Optional[str]]:
        """包含匹配检查"""
        for entry in self._contains_entries:
            if entry.text in text:
                if self._should_bypass(entry, categories):
                    return True, entry.reason or "whitelisted"
        return False, None

    def _check_regex(
        self, text: str, categories: Optional[List[str]]
    ) -> Tuple[bool, Optional[str]]:
        """正则匹配检查"""
        for pattern, entry in self._regex_entries:
            if pattern.search(text):
                if self._should_bypass(entry, categories):
                    return True, entry.reason or "whitelisted"
        return False, None

    def _should_bypass(
        self,
        entry: WhitelistEntry,
        categories: Optional[List[str]],
    ) -> bool:
        """判断是否应绕过检测"""
        # 条目未指定类别限制 → 总是绕过
        if entry.categories is None:
            return True

        # 当前未指定检测类别 → 总是绕过
        if categories is None:
            return True

        # 检查条目的类别与当前检测类别是否有交集
        return bool(set(entry.categories) & set(categories))

    def get_all_entries(self) -> List[WhitelistEntry]:
        """获取所有白名单条目"""
        if self._mode == "exact":
            return list(self._exact_entries.values())
        elif self._mode == "contains":
            return list(self._contains_entries)
        else:
            return [entry for _, entry in self._regex_entries]

    def load_from_file(self, filepath: str) -> int:
        """从文件加载白名单"""
        count = 0
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text and not text.startswith("#"):
                        if self.add(WhitelistEntry(text=text)):
                            count += 1
        except FileNotFoundError:
            logger.info(f"白名单文件不存在: {filepath}，将创建空文件")
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            Path(filepath).touch()
        except Exception as e:
            logger.error(f"加载白名单失败: {e}")
        return count

    def load_from_config(self, entries: List[Dict]) -> int:
        """从配置加载白名单条目"""
        count = 0
        for item in entries:
            text = item.get("text", "").strip()
            if not text:
                continue
            entry = WhitelistEntry(
                text=text,
                reason=item.get("reason", ""),
                categories=item.get("categories"),
            )
            if self.add(entry):
                count += 1
        return count

    def clear(self) -> None:
        """清空白名单"""
        with self._lock:
            self._exact_set.clear()
            self._exact_entries.clear()
            self._contains_entries.clear()
            self._regex_entries.clear()
