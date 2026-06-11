"""检测引擎集成测试"""

import pytest

from app.config import Config
from app.detector import ContentDetector
from app.models import RiskLevel


@pytest.fixture
def detector(tmp_path):
    """创建测试用检测引擎"""
    cfg = Config()
    cfg.load()

    # 创建临时词库文件
    politics_file = tmp_path / "politics.txt"
    politics_file.write_text("分裂国家\n颠覆政权\n反动\n煽动颠覆\n", encoding="utf-8")

    gambling_file = tmp_path / "gambling.txt"
    gambling_file.write_text("网络赌博\n在线赌博\n博彩\n赌博平台\n", encoding="utf-8")

    pornography_file = tmp_path / "pornography.txt"
    pornography_file.write_text("色情\n淫秽\n黄色\n", encoding="utf-8")

    cfg._config["word_lists"]["politics"]["path"] = str(politics_file)
    cfg._config["word_lists"]["gambling"]["path"] = str(gambling_file)
    cfg._config["word_lists"]["pornography"]["path"] = str(pornography_file)

    # 禁用模型（测试中不依赖模型文件）
    cfg._config["detector"]["strategy"] = "sensitive_only"

    det = ContentDetector(cfg)
    return det


class TestContentDetector:
    """ContentDetector 检测引擎测试"""

    def test_detect_safe_text(self, detector):
        result = detector.detect("今天天气真好，适合出去散步")
        assert result.is_violation is False
        assert result.risk_level == RiskLevel.LOW
        assert len(result.violations) == 0
        assert result.processing_time_ms > 0

    def test_detect_politics_violation(self, detector):
        result = detector.detect("企图分裂国家的阴谋")
        assert result.is_violation is True
        assert result.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert any(v.category == "politics" for v in result.violations)

    def test_detect_gambling_violation(self, detector):
        result = detector.detect("参与网络赌博活动")
        assert result.is_violation is True
        assert any(v.category == "gambling" for v in result.violations)

    def test_detect_pornography_violation(self, detector):
        result = detector.detect("传播淫秽内容")
        assert result.is_violation is True
        assert any(v.category == "pornography" for v in result.violations)

    def test_detect_multiple_categories(self, detector):
        result = detector.detect("分裂国家的网络赌博集团传播淫秽内容")
        assert result.is_violation is True
        categories = set(v.category for v in result.violations)
        assert len(categories) >= 2

    def test_detect_with_category_filter(self, detector):
        result = detector.detect(
            "分裂国家的网络赌博集团", categories=["politics"]
        )
        assert result.is_violation is True
        assert all(v.category == "politics" for v in result.violations)

    def test_detect_violation_detail_fields(self, detector):
        result = detector.detect("参与网络赌博")
        assert result.is_violation is True
        violation = result.violations[0]
        assert violation.category == "gambling"
        assert violation.source == "sensitive_word"
        assert violation.matched_word is not None
        assert violation.position is not None
        assert violation.confidence == 1.0

    def test_detect_summary(self, detector):
        result = detector.detect("参与网络赌博")
        assert result.summary.total_matches > 0
        assert "gambling" in result.summary.categories_hit
        assert result.summary.word_hits > 0

    def test_detect_batch(self, detector):
        texts = [
            "今天天气好",
            "参与网络赌博",
            "传播淫秽内容",
        ]
        results = detector.detect_batch(texts)
        assert len(results) == 3
        assert results[0].is_violation is False
        assert results[1].is_violation is True
        assert results[2].is_violation is True

    def test_detect_batch_limit(self, detector):
        detector._max_batch_size = 2
        texts = ["文本1", "文本2", "文本3"]
        results = detector.detect_batch(texts)
        assert len(results) == 2

    def test_detect_text_truncation(self, detector):
        detector._max_text_length = 10
        result = detector.detect("a" * 100)
        assert result.is_violation is False  # 不会崩溃

    def test_risk_level_critical_politics_multiple(self, detector):
        """涉政命中≥2词应为 critical"""
        result = detector.detect("分裂国家和颠覆政权的行为")
        assert result.is_violation is True
        politics_hits = [
            v for v in result.violations if v.category == "politics"
        ]
        if len(politics_hits) >= 2:
            assert result.risk_level == RiskLevel.CRITICAL

    def test_stats(self, detector):
        stats = detector.get_stats()
        assert "word_counts" in stats
        assert "strategy" in stats
        assert "total_requests" in stats
        assert "total_violations" in stats

    def test_reload(self, detector):
        counts = detector.reload()
        assert isinstance(counts, dict)

    def test_request_counter(self, detector):
        initial = detector._total_requests
        detector.detect("测试")
        assert detector._total_requests == initial + 1

    def test_violation_counter(self, detector):
        initial = detector._total_violations
        detector.detect("参与网络赌博")
        assert detector._total_violations == initial + 1

        detector.detect("今天天气好")
        assert detector._total_violations == initial + 1  # 安全文本不增加
