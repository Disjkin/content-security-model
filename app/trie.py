"""Trie 树（前缀树）数据结构 — 用于高效多模式字符串匹配"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class MatchResult:
    """匹配结果"""
    word: str
    start: int
    end: int
    category: str


class TrieNode:
    """Trie 树节点"""

    __slots__ = ("children", "is_end", "word")

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.is_end: bool = False
        self.word: Optional[str] = None  # 存储完整词，用于回溯


class TrieTree:
    """
    Trie 树实现

    支持：
    - 批量插入词汇
    - 最长匹配搜索（贪心策略）
    - 从文件加载词库
    """

    def __init__(self):
        self._root = TrieNode()
        self._size: int = 0
        self._words: Set[str] = set()

    @property
    def size(self) -> int:
        """词库大小"""
        return self._size

    def insert(self, word: str) -> None:
        """插入一个词到 Trie 树"""
        word = word.strip()
        if not word:
            return

        node = self._root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        if not node.is_end:
            node.is_end = True
            node.word = word
            self._size += 1
            self._words.add(word)

    def remove(self, word: str) -> bool:
        """
        删除一个词（软删除，标记为非词尾）
        返回是否成功删除
        """
        word = word.strip()
        if word not in self._words:
            return False

        node = self._root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]

        if node.is_end:
            node.is_end = False
            node.word = None
            self._size -= 1
            self._words.discard(word)
            return True
        return False

    def search(self, text: str) -> List[MatchResult]:
        """
        在文本中搜索所有匹配的敏感词（最长匹配策略）

        使用贪心最长匹配：从每个位置开始，尽可能匹配最长的词。

        Args:
            text: 待搜索文本

        Returns:
            匹配结果列表，包含词、位置信息
        """
        results: List[MatchResult] = []
        text_len = len(text)
        i = 0

        while i < text_len:
            node = self._root
            last_match: Optional[MatchResult] = None
            j = i

            while j < text_len:
                char = text[j]
                if char not in node.children:
                    break
                node = node.children[char]
                if node.is_end and node.word:
                    last_match = MatchResult(
                        word=node.word,
                        start=i,
                        end=j + 1,
                        category="",  # 由上层填充
                    )
                j += 1

            if last_match:
                results.append(last_match)
                # 跳过已匹配的字符，避免重叠匹配
                i = last_match.end
            else:
                i += 1

        return results

    def contains(self, word: str) -> bool:
        """检查词是否在 Trie 树中"""
        node = self._root
        for char in word.strip():
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def build_from_file(self, filepath: str) -> int:
        """
        从文件批量构建 Trie 树

        文件格式：每行一个词

        Args:
            filepath: 词库文件路径

        Returns:
            成功加载的词数
        """
        count = 0
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith("#"):
                        self.insert(word)
                        count += 1
        except FileNotFoundError:
            raise FileNotFoundError(f"词库文件不存在: {filepath}")
        except Exception as e:
            raise RuntimeError(f"加载词库失败: {filepath}, 错误: {e}")
        return count

    def build_from_list(self, words: List[str]) -> int:
        """从列表批量构建 Trie 树"""
        count = 0
        for word in words:
            word = word.strip()
            if word:
                self.insert(word)
                count += 1
        return count

    def get_all_words(self) -> Set[str]:
        """获取所有词汇"""
        return self._words.copy()

    def clear(self) -> None:
        """清空 Trie 树"""
        self._root = TrieNode()
        self._size = 0
        self._words.clear()
