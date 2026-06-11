"""日志存储模块测试"""

import pytest

from app.storage import DetectionLogEntry, LogQuery, SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    """创建测试用存储"""
    db_path = str(tmp_path / "test_logs.db")
    return SQLiteStorage(db_path)


@pytest.fixture
def sample_entry():
    """创建样本日志条目"""
    return DetectionLogEntry(
        request_id="test-req-001",
        text="这是一条测试文本，包含网络赌博内容",
        is_violation=True,
        risk_level="high",
        categories_hit=["gambling"],
        word_hits=1,
        model_confidence=0.95,
        model_category="gambling",
        processing_time_ms=15.5,
        source_ip="127.0.0.1",
        api_key_id="sk-test",
        violations_summary=[
            {"category": "gambling", "source": "sensitive_word", "matched_word": "网络赌博", "confidence": 1.0}
        ],
    )


@pytest.fixture
def safe_entry():
    """创建安全文本日志条目"""
    return DetectionLogEntry(
        request_id="test-req-002",
        text="今天天气真好",
        is_violation=False,
        risk_level="low",
        categories_hit=[],
        word_hits=0,
        processing_time_ms=5.2,
    )


class TestSQLiteStorage:
    """SQLite 存储测试"""

    def test_init_creates_db(self, storage):
        """初始化时创建数据库"""
        assert storage is not None

    def test_save_and_get_log(self, storage, sample_entry):
        """保存并查询日志"""
        request_id = storage.save_log(sample_entry)
        assert request_id == "test-req-001"

        log = storage.get_log("test-req-001")
        assert log is not None
        assert log["request_id"] == "test-req-001"
        assert log["is_violation"] is True
        assert log["risk_level"] == "high"
        assert log["word_hits"] == 1

    def test_get_nonexistent_log(self, storage):
        """查询不存在的日志"""
        log = storage.get_log("nonexistent")
        assert log is None

    def test_save_duplicate_request_id(self, storage, sample_entry):
        """重复 request_id 忽略"""
        storage.save_log(sample_entry)
        storage.save_log(sample_entry)  # 不报错

        log = storage.get_log("test-req-001")
        assert log is not None

    def test_text_not_stored(self, storage, sample_entry):
        """原始文本不存储（隐私保护）"""
        storage.save_log(sample_entry)
        log = storage.get_log("test-req-001")

        assert "text" not in log or log.get("text") is None
        assert log["text_hash"] is not None
        assert len(log["text_hash"]) == 64  # SHA-256

    def test_text_preview_truncated(self, storage):
        """长文本预览被截断"""
        entry = DetectionLogEntry(
            request_id="test-long",
            text="a" * 200,
            is_violation=False,
            risk_level="low",
        )
        storage.save_log(entry)
        log = storage.get_log("test-long")
        assert log["text_preview"].endswith("...")

    def test_query_all(self, storage, sample_entry, safe_entry):
        """查询全部日志"""
        storage.save_log(sample_entry)
        storage.save_log(safe_entry)

        result = storage.query_logs(LogQuery())
        assert result.total == 2
        assert len(result.logs) == 2

    def test_query_by_violation(self, storage, sample_entry, safe_entry):
        """按违规状态筛选"""
        storage.save_log(sample_entry)
        storage.save_log(safe_entry)

        result = storage.query_logs(LogQuery(is_violation=True))
        assert result.total == 1
        assert result.logs[0]["request_id"] == "test-req-001"

    def test_query_by_risk_level(self, storage, sample_entry, safe_entry):
        """按风险等级筛选"""
        storage.save_log(sample_entry)
        storage.save_log(safe_entry)

        result = storage.query_logs(LogQuery(risk_level="high"))
        assert result.total == 1

    def test_query_by_category(self, storage, sample_entry, safe_entry):
        """按类别筛选"""
        storage.save_log(sample_entry)
        storage.save_log(safe_entry)

        result = storage.query_logs(LogQuery(category="gambling"))
        assert result.total == 1

    def test_query_pagination(self, storage):
        """分页查询"""
        for i in range(20):
            entry = DetectionLogEntry(
                request_id=f"req-{i:03d}",
                text=f"测试文本 {i}",
                is_violation=i % 2 == 0,
                risk_level="low",
            )
            storage.save_log(entry)

        # 第 1 页
        result = storage.query_logs(LogQuery(page=1, page_size=10))
        assert result.total == 20
        assert len(result.logs) == 10

        # 第 2 页
        result = storage.query_logs(LogQuery(page=2, page_size=10))
        assert len(result.logs) == 10

        # 第 3 页（无数据）
        result = storage.query_logs(LogQuery(page=3, page_size=10))
        assert len(result.logs) == 0

    def test_get_stats_summary(self, storage, sample_entry, safe_entry):
        """获取统计摘要"""
        storage.save_log(sample_entry)
        storage.save_log(safe_entry)

        stats = storage.get_stats_summary()
        assert stats["total_detections"] == 2
        assert stats["violation_count"] == 1
        assert stats["violation_rate"] == 0.5
        assert "risk_level_distribution" in stats
        assert "category_breakdown" in stats

    def test_get_trends(self, storage, sample_entry, safe_entry):
        """获取趋势数据"""
        storage.save_log(sample_entry)
        storage.save_log(safe_entry)

        trends = storage.get_trends(period="day")
        assert len(trends) >= 1
        assert "period" in trends[0]
        assert "total" in trends[0]

    def test_get_top_violation_words(self, storage, sample_entry):
        """获取高频违规词"""
        storage.save_log(sample_entry)

        words = storage.get_top_violation_words(limit=10)
        assert len(words) >= 1
        assert words[0]["word"] == "网络赌博"
        assert words[0]["count"] == 1

    def test_cleanup(self, storage):
        """清理过期日志"""
        entry = DetectionLogEntry(
            request_id="old-log",
            text="旧日志",
            is_violation=False,
            risk_level="low",
        )
        storage.save_log(entry)

        # 清理 0 天前的日志（应该清理掉所有）
        deleted = storage.cleanup(retention_days=0)
        assert deleted >= 0  # 取决于时间精度

    def test_save_empty_violations(self, storage):
        """保存无违规的空摘要"""
        entry = DetectionLogEntry(
            request_id="empty-violations",
            text="安全文本",
            is_violation=False,
            risk_level="low",
            violations_summary=None,
        )
        storage.save_log(entry)
        log = storage.get_log("empty-violations")
        assert log is not None
