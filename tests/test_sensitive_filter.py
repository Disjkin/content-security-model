"""敏感词过滤器测试"""

import pytest

from app.config import Config
from app.preprocessor import Preprocessor
from app.sensitive_filter import FilterMatch, FilterResult, SensitiveFilter
from app.word_manager import WordManager


@pytest.fixture
def config():
    """创建测试用配置"""
    cfg = Config()
    cfg.load()
    return cfg


@pytest.fixture
def word_manager(config, tmp_path):
    """创建测试用词库管理器"""
    # 创建临时词库文件
    politics_file = tmp_path / "politics.txt"
    politics_file.write_text("分裂国家\n颠覆政权\n反动\n", encoding="utf-8")

    gambling_file = tmp_path / "gambling.txt"
    gambling_file.write_text("网络赌博\n在线赌博\n博彩\n", encoding="utf-8")

    # 修改配置指向临时文件
    config._config["word_lists"]["politics"]["path"] = str(politics_file)
    config._config["word_lists"]["gambling"]["path"] = str(gambling_file)

    wm = WordManager(config, str(tmp_path))
    wm.load_all()
    return wm


@pytest.fixture
def sensitive_filter(word_manager):
    """创建测试用敏感词过滤器"""
    preprocessor = Preprocessor(
        traditional_to_simplified=False,
        normalize_variants=False,
        remove_special_chars=False,
    )
    return SensitiveFilter(word_manager, preprocessor)


class TestFilterResult:
    """FilterResult 数据类测试"""

    def test_default_values(self):
        result = FilterResult()
        assert result.is_violation is False
        assert result.matches == []
        assert result.categories_hit == []
        assert result.total_matches == 0

    def test_total_matches_property(self):
        result = FilterResult(
            matches=[
                FilterMatch(category="politics", word="test", start=0, end=2),
                FilterMatch(category="gambling", word="test2", start=3, end=5),
            ]
        )
        assert result.total_matches == 2


class TestSensitiveFilter:
    """SensitiveFilter 过滤器测试"""

    def test_filter_empty_text(self, sensitive_filter):
        result = sensitive_filter.filter("")
        assert result.is_violation is False
        assert result.total_matches == 0

    def test_filter_safe_text(self, sensitive_filter):
        result = sensitive_filter.filter("今天天气真好")
        assert result.is_violation is False
        assert result.total_matches == 0

    def test_filter_violation_text(self, sensitive_filter):
        result = sensitive_filter.filter("他参与了网络赌博活动")
        assert result.is_violation is True
        assert result.total_matches >= 1
        assert "gambling" in result.categories_hit

    def test_filter_multiple_categories(self, sensitive_filter):
        result = sensitive_filter.filter("分裂国家的网络赌博集团")
        assert result.is_violation is True
        assert "politics" in result.categories_hit
        assert "gambling" in result.categories_hit

    def test_filter_with_categories_filter(self, sensitive_filter):
        """测试指定类别过滤"""
        result = sensitive_filter.filter(
            "分裂国家的网络赌博集团", categories=["politics"]
        )
        assert result.is_violation is True
        # 只检测 politics 类别
        assert all(m.category == "politics" for m in result.matches)

    def test_filter_match_position(self, sensitive_filter):
        result = sensitive_filter.filter("他参与了网络赌博")
        assert result.is_violation is True
        match = result.matches[0]
        assert match.start >= 0
        assert match.end > match.start
        # 验证位置对应的原文确实是匹配词
        text = "他参与了网络赌博"
        assert text[match.start : match.end] == match.word

    def test_filter_match_severity(self, sensitive_filter):
        result = sensitive_filter.filter("分裂国家")
        assert result.is_violation is True
        assert result.matches[0].severity == "high"

    def test_filter_none_text(self, sensitive_filter):
        result = sensitive_filter.filter(None)
        assert result.is_violation is False

    def test_get_stats(self, sensitive_filter):
        stats = sensitive_filter.get_stats()
        assert "politics" in stats
        assert "gambling" in stats
        assert stats["politics"] > 0
        assert stats["gambling"] > 0
