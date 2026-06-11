"""预处理器单元测试"""

import pytest

from app.preprocessor import Preprocessor


class TestPreprocessor:
    """Preprocessor 预处理器测试"""

    def test_normalize_empty_text(self):
        p = Preprocessor(traditional_to_simplified=False, normalize_variants=False)
        assert p.normalize("") == ""

    def test_normalize_none_text(self):
        p = Preprocessor(traditional_to_simplified=False, normalize_variants=False)
        assert p.normalize(None) == ""

    def test_unicode_normalize(self):
        p = Preprocessor(traditional_to_simplified=False, normalize_variants=False)
        # 全角数字应被归一化
        result = p.normalize("１２３")
        assert result == "123"

    def test_lowercase(self):
        p = Preprocessor(traditional_to_simplified=False, normalize_variants=False)
        result = p.normalize("Hello WORLD")
        assert result == "hello world"

    def test_remove_skip_chars(self):
        p = Preprocessor(
            skip_chars="!@#",
            traditional_to_simplified=False,
            normalize_variants=False,
        )
        result = p.normalize("hello!@#world")
        assert result == "helloworld"

    def test_normalize_variants(self):
        p = Preprocessor(traditional_to_simplified=False, normalize_variants=True)
        # 数字变体还原
        result = p.normalize("h3llo")
        assert "e" in result  # 3 → e

    def test_full_width_to_half_width(self):
        p = Preprocessor(traditional_to_simplified=False, normalize_variants=False)
        result = p.normalize("ＡＢＣ")
        assert result == "abc"

    def test_mixed_text(self):
        p = Preprocessor(
            skip_chars=" ",
            traditional_to_simplified=False,
            normalize_variants=False,
        )
        result = p.normalize("Hello World TEST")
        assert result == "helloworldtest"
