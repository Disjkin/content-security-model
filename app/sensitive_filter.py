"""敏感词过滤器 — 基于 Trie 树的敏感词匹配引擎"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.logger import get_logger
from app.preprocessor import Preprocessor
from app.trie import MatchResult, TrieTree
from app.word_manager import WordManager

logger = get_logger("sensitive_filter")


@dataclass
class FilterMatch:
    """过滤匹配结果"""
    category: str
    word: str
    start: int
    end: int
    severity: str = "medium"


@dataclass
class FilterResult:
    """过滤结果"""
    is_violation: bool = False
    matches: List[FilterMatch] = field(default_factory=list)
    categories_hit: List[str] = field(default_factory=list)

    @property
    def total_matches(self) -> int:
        return len(self.matches)


class SensitiveFilter:
    """
    敏感词过滤器

    使用 Trie 树对文本进行多模式匹配，检测是否包含敏感词汇。
    支持多类别词库、严重等级标记。
    """

    def __init__(self, word_manager: WordManager, preprocessor: Optional[Preprocessor] = None):
        self._word_manager = word_manager
        self._preprocessor = preprocessor or Preprocessor()

    def filter(
        self,
        text: str,
        categories: Optional[List[str]] = None,
    ) -> FilterResult:
        """
        对文本执行敏感词过滤

        Args:
            text: 待检测文本
            categories: 指定检测类别，None 则检测所有类别

        Returns:
            过滤结果
        """
        if not text or not text.strip():
            return FilterResult()

        # 预处理文本
        normalized = self._preprocessor.normalize(text)

        # 获取要检测的 Trie 树
        trees = self._word_manager.get_all_trees()
        if categories:
            trees = {cat: tree for cat, tree in trees.items() if cat in categories}

        all_matches: List[FilterMatch] = []
        categories_hit: set = set()

        for category, tree in trees.items():
            matches = tree.search(normalized)
            severity = self._word_manager._config.get_severity(category)

            for match in matches:
                all_matches.append(
                    FilterMatch(
                        category=category,
                        word=match.word,
                        start=match.start,
                        end=match.end,
                        severity=severity,
                    )
                )
                categories_hit.add(category)

        result = FilterResult(
            is_violation=len(all_matches) > 0,
            matches=all_matches,
            categories_hit=sorted(categories_hit),
        )

        if result.is_violation:
            logger.debug(
                f"敏感词命中: {result.total_matches} 个词, "
                f"类别: {result.categories_hit}"
            )

        return result

    def get_stats(self) -> Dict[str, int]:
        """获取各类别词库统计"""
        stats = {}
        for category in self._word_manager.get_categories():
            stats[category] = self._word_manager.get_word_count(category)
        return stats
