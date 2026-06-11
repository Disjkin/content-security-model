"""白名单模块测试"""

import pytest

from app.whitelist import Whitelist, WhitelistEntry


class TestWhitelistExact:
    """精确匹配模式测试"""

    def test_empty_whitelist(self):
        wl = Whitelist(mode="exact")
        is_wl, reason = wl.is_whitelisted("任何文本")
        assert is_wl is False
        assert reason is None

    def test_add_and_match(self):
        wl = Whitelist(mode="exact")
        wl.add(WhitelistEntry(text="官方公告", reason="official"))
        is_wl, reason = wl.is_whitelisted("官方公告")
        assert is_wl is True
        assert reason == "official"

    def test_no_match(self):
        wl = Whitelist(mode="exact")
        wl.add(WhitelistEntry(text="官方公告"))
        is_wl, _ = wl.is_whitelisted("其他文本")
        assert is_wl is False

    def test_partial_no_match(self):
        """精确模式不支持部分匹配"""
        wl = Whitelist(mode="exact")
        wl.add(WhitelistEntry(text="公告"))
        is_wl, _ = wl.is_whitelisted("官方公告")
        assert is_wl is False

    def test_remove(self):
        wl = Whitelist(mode="exact")
        wl.add(WhitelistEntry(text="测试"))
        assert wl.remove("测试") is True
        is_wl, _ = wl.is_whitelisted("测试")
        assert is_wl is False

    def test_remove_nonexistent(self):
        wl = Whitelist(mode="exact")
        assert wl.remove("不存在") is False

    def test_size(self):
        wl = Whitelist(mode="exact")
        assert wl.size == 0
        wl.add(WhitelistEntry(text="a"))
        wl.add(WhitelistEntry(text="b"))
        assert wl.size == 2

    def test_duplicate_add(self):
        wl = Whitelist(mode="exact")
        assert wl.add(WhitelistEntry(text="a")) is True
        assert wl.add(WhitelistEntry(text="a")) is False
        assert wl.size == 1

    def test_add_empty(self):
        wl = Whitelist(mode="exact")
        assert wl.add(WhitelistEntry(text="")) is False
        assert wl.add(WhitelistEntry(text="  ")) is False


class TestWhitelistContains:
    """包含匹配模式测试"""

    def test_contains_match(self):
        wl = Whitelist(mode="contains")
        wl.add(WhitelistEntry(text="官方", reason="official"))
        is_wl, reason = wl.is_whitelisted("这是官方公告")
        assert is_wl is True
        assert reason == "official"

    def test_contains_no_match(self):
        wl = Whitelist(mode="contains")
        wl.add(WhitelistEntry(text="官方"))
        is_wl, _ = wl.is_whitelisted("这是普通文本")
        assert is_wl is False

    def test_contains_multiple_entries(self):
        wl = Whitelist(mode="contains")
        wl.add(WhitelistEntry(text="官方"))
        wl.add(WhitelistEntry(text="公告"))
        is_wl, _ = wl.is_whitelisted("这是公告内容")
        assert is_wl is True

    def test_contains_remove(self):
        wl = Whitelist(mode="contains")
        wl.add(WhitelistEntry(text="官方"))
        assert wl.remove("官方") is True
        is_wl, _ = wl.is_whitelisted("官方公告")
        assert is_wl is False


class TestWhitelistRegex:
    """正则匹配模式测试"""

    def test_regex_match(self):
        wl = Whitelist(mode="regex")
        wl.add(WhitelistEntry(text=r"官方.*公告", reason="official"))
        is_wl, reason = wl.is_whitelisted("这是官方紧急公告")
        assert is_wl is True

    def test_regex_no_match(self):
        wl = Whitelist(mode="regex")
        wl.add(WhitelistEntry(text=r"^官方.*$"))
        is_wl, _ = wl.is_whitelisted("普通文本")
        assert is_wl is False

    def test_invalid_regex(self):
        wl = Whitelist(mode="regex")
        result = wl.add(WhitelistEntry(text="[invalid"))
        assert result is False

    def test_regex_remove(self):
        wl = Whitelist(mode="regex")
        wl.add(WhitelistEntry(text=r"test\d+"))
        assert wl.remove(r"test\d+") is True
        assert wl.size == 0


class TestWhitelistCategories:
    """类别过滤测试"""

    def test_bypass_all_categories(self):
        """未指定类别 → 绕过所有检测"""
        wl = Whitelist(mode="exact")
        wl.add(WhitelistEntry(text="测试", categories=None))
        is_wl, _ = wl.is_whitelisted("测试", categories=["politics", "gambling"])
        assert is_wl is True

    def test_bypass_specific_category(self):
        """仅绕过指定类别"""
        wl = Whitelist(mode="exact")
        wl.add(WhitelistEntry(text="彩票", categories=["gambling"]))

        # gambling 类别被绕过
        is_wl, _ = wl.is_whitelisted("彩票", categories=["gambling"])
        assert is_wl is True

        # politics 类别不被绕过
        is_wl, _ = wl.is_whitelisted("彩票", categories=["politics"])
        assert is_wl is False

    def test_bypass_with_no_categories_filter(self):
        """当前检测未指定类别 → 总是绕过"""
        wl = Whitelist(mode="exact")
        wl.add(WhitelistEntry(text="测试", categories=["gambling"]))
        is_wl, _ = wl.is_whitelisted("测试", categories=None)
        assert is_wl is True


class TestWhitelistFile:
    """文件加载测试"""

    def test_load_from_file(self, tmp_path):
        wl_file = tmp_path / "whitelist.txt"
        wl_file.write_text("官方公告\n# 注释\n测试词\n", encoding="utf-8")

        wl = Whitelist(mode="exact")
        count = wl.load_from_file(str(wl_file))
        assert count == 2
        assert wl.size == 2

    def test_load_from_nonexistent_file(self, tmp_path):
        wl = Whitelist(mode="exact")
        count = wl.load_from_file(str(tmp_path / "不存在.txt"))
        assert count == 0


class TestWhitelistConfig:
    """配置加载测试"""

    def test_load_from_config(self):
        wl = Whitelist(mode="exact")
        entries = [
            {"text": "官方公告", "reason": "official"},
            {"text": "彩票", "categories": ["gambling"]},
        ]
        count = wl.load_from_config(entries)
        assert count == 2

    def test_load_from_config_skip_empty(self):
        wl = Whitelist(mode="exact")
        entries = [{"text": ""}, {"text": "有效"}]
        count = wl.load_from_config(entries)
        assert count == 1


class TestWhitelistEdgeCases:
    """边界情况测试"""

    def test_is_whitelisted_empty_text(self):
        wl = Whitelist(mode="exact")
        wl.add(WhitelistEntry(text="测试"))
        is_wl, _ = wl.is_whitelisted("")
        assert is_wl is False

    def test_is_whitelisted_none_text(self):
        wl = Whitelist(mode="exact")
        is_wl, _ = wl.is_whitelisted(None)
        assert is_wl is False

    def test_clear(self):
        wl = Whitelist(mode="exact")
        wl.add(WhitelistEntry(text="a"))
        wl.add(WhitelistEntry(text="b"))
        wl.clear()
        assert wl.size == 0

    def test_get_all_entries(self):
        wl = Whitelist(mode="exact")
        wl.add(WhitelistEntry(text="a", reason="r1"))
        wl.add(WhitelistEntry(text="b", reason="r2"))
        entries = wl.get_all_entries()
        assert len(entries) == 2
