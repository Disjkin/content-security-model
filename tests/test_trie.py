"""Trie 树单元测试"""

import pytest

from app.trie import MatchResult, TrieNode, TrieTree


class TestTrieNode:
    """TrieNode 节点测试"""

    def test_init(self):
        node = TrieNode()
        assert node.children == {}
        assert node.is_end is False
        assert node.word is None


class TestTrieTree:
    """TrieTree 树操作测试"""

    def test_empty_tree(self):
        tree = TrieTree()
        assert tree.size == 0
        assert tree.get_all_words() == set()

    def test_insert_single_word(self):
        tree = TrieTree()
        tree.insert("赌博")
        assert tree.size == 1
        assert tree.contains("赌博")

    def test_insert_multiple_words(self):
        tree = TrieTree()
        tree.insert("赌博")
        tree.insert("色情")
        tree.insert("暴力")
        assert tree.size == 3

    def test_insert_duplicate_word(self):
        tree = TrieTree()
        tree.insert("赌博")
        tree.insert("赌博")
        assert tree.size == 1

    def test_insert_empty_string(self):
        tree = TrieTree()
        tree.insert("")
        tree.insert("   ")
        assert tree.size == 0

    def test_insert_strips_whitespace(self):
        tree = TrieTree()
        tree.insert("  赌博  ")
        assert tree.contains("赌博")
        assert tree.size == 1

    def test_remove_existing_word(self):
        tree = TrieTree()
        tree.insert("赌博")
        assert tree.remove("赌博") is True
        assert tree.size == 0
        assert tree.contains("赌博") is False

    def test_remove_nonexistent_word(self):
        tree = TrieTree()
        tree.insert("赌博")
        assert tree.remove("色情") is False
        assert tree.size == 1

    def test_remove_word_not_in_tree(self):
        tree = TrieTree()
        assert tree.remove("不存在") is False

    def test_search_no_match(self):
        tree = TrieTree()
        tree.insert("赌博")
        results = tree.search("今天天气很好")
        assert len(results) == 0

    def test_search_single_match(self):
        tree = TrieTree()
        tree.insert("赌博")
        results = tree.search("他沉迷于赌博无法自拔")
        assert len(results) == 1
        assert results[0].word == "赌博"
        assert results[0].start == 4
        assert results[0].end == 6

    def test_search_multiple_matches(self):
        tree = TrieTree()
        tree.insert("赌博")
        tree.insert("色情")
        results = tree.search("赌博和色情都是违法的")
        assert len(results) == 2

    def test_search_longest_match(self):
        """测试最长匹配策略"""
        tree = TrieTree()
        tree.insert("赌")
        tree.insert("赌博")
        tree.insert("赌博网")
        results = tree.search("这个赌博网很好")
        assert len(results) == 1
        assert results[0].word == "赌博网"

    def test_search_overlapping_skip(self):
        """已匹配的字符应被跳过，避免重叠"""
        tree = TrieTree()
        tree.insert("赌博")
        tree.insert("博彩")
        results = tree.search("赌博博彩")
        assert len(results) == 2

    def test_search_at_start(self):
        tree = TrieTree()
        tree.insert("色情")
        results = tree.search("色情内容")
        assert len(results) == 1
        assert results[0].start == 0

    def test_search_at_end(self):
        tree = TrieTree()
        tree.insert("色情")
        results = tree.search("传播色情")
        assert len(results) == 1
        assert results[0].end == 4

    def test_contains(self):
        tree = TrieTree()
        tree.insert("赌博")
        assert tree.contains("赌博") is True
        assert tree.contains("赌") is False
        assert tree.contains("色情") is False

    def test_build_from_list(self):
        tree = TrieTree()
        words = ["赌博", "色情", "暴力"]
        count = tree.build_from_list(words)
        assert count == 3
        assert tree.size == 3

    def test_build_from_list_with_empty(self):
        tree = TrieTree()
        words = ["赌博", "", "  ", "色情"]
        count = tree.build_from_list(words)
        assert count == 2

    def test_get_all_words(self):
        tree = TrieTree()
        tree.insert("赌博")
        tree.insert("色情")
        words = tree.get_all_words()
        assert words == {"赌博", "色情"}

    def test_get_all_words_returns_copy(self):
        tree = TrieTree()
        tree.insert("赌博")
        words = tree.get_all_words()
        words.add("修改")
        assert tree.size == 1  # 原始数据不受影响

    def test_clear(self):
        tree = TrieTree()
        tree.insert("赌博")
        tree.insert("色情")
        tree.clear()
        assert tree.size == 0
        assert tree.get_all_words() == set()

    def test_build_from_file(self, tmp_path):
        """测试从文件加载词库"""
        word_file = tmp_path / "test_words.txt"
        word_file.write_text("赌博\n色情\n# 这是注释\n暴力\n", encoding="utf-8")

        tree = TrieTree()
        count = tree.build_from_file(str(word_file))
        assert count == 3
        assert tree.size == 3
        assert tree.contains("赌博")
        assert tree.contains("色情")
        assert tree.contains("暴力")

    def test_build_from_file_not_found(self):
        tree = TrieTree()
        with pytest.raises(FileNotFoundError):
            tree.build_from_file("/不存在/path/words.txt")
