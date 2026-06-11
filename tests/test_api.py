"""API 接口集成测试"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Config


@pytest.fixture
def client(tmp_path):
    """创建测试客户端，使用临时配置"""
    # 创建临时词库文件
    for name in ["politics", "gambling", "pornography"]:
        (tmp_path / f"{name}.txt").write_text("测试词\n", encoding="utf-8")

    # 创建临时配置文件
    import yaml

    test_config = {
        "server": {"host": "0.0.0.0", "port": 8000, "workers": 1, "reload": False},
        "auth": {"enabled": False, "api_keys": ["sk-content-sec-dev-001"]},
        "word_lists": {
            "politics": {
                "path": str(tmp_path / "politics.txt"),
                "enabled": True,
                "severity": "high",
            },
            "gambling": {
                "path": str(tmp_path / "gambling.txt"),
                "enabled": True,
                "severity": "medium",
            },
            "pornography": {
                "path": str(tmp_path / "pornography.txt"),
                "enabled": True,
                "severity": "high",
            },
        },
        "preprocessor": {
            "traditional_to_simplified": False,
            "normalize_variants": False,
            "remove_special_chars": False,
            "skip_chars": "",
        },
        "detector": {
            "max_batch_size": 100,
            "max_text_length": 10000,
            "strategy": "sensitive_only",
            "risk_thresholds": {"critical_min_matches": 5},
        },
        "model": {
            "enabled": True,
            "model_path": str(tmp_path / "model.pkl"),
            "vectorizer_path": str(tmp_path / "vectorizer.pkl"),
            "training_data": str(tmp_path / "train.json"),
            "confidence_threshold": 0.6,
        },
        "logging": {"level": "INFO"},
    }

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(test_config, allow_unicode=True), encoding="utf-8")

    # 重置配置单例并加载测试配置
    Config.reset()
    from app.config import config

    config.load(str(config_file))

    # 重置全局检测引擎实例，确保使用新配置
    import app.api as api_module

    api_module._detector = None

    app = create_app()
    yield TestClient(app)

    # 清理
    api_module._detector = None
    Config.reset()


class TestHealthAPI:
    """健康检查接口测试"""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "word_count" in data


class TestDetectAPI:
    """内容检测接口测试"""

    def test_detect_safe(self, client):
        response = client.post(
            "/api/v1/detect",
            json={"text": "今天天气真好"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_violation"] is False
        assert data["risk_level"] == "low"
        assert "processing_time_ms" in data

    def test_detect_violation(self, client):
        response = client.post(
            "/api/v1/detect",
            json={"text": "测试词"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_violation"] is True
        assert len(data["violations"]) > 0

    def test_detect_with_categories(self, client):
        response = client.post(
            "/api/v1/detect",
            json={"text": "测试词", "categories": ["politics"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_violation"] is True

    def test_detect_empty_text(self, client):
        response = client.post(
            "/api/v1/detect",
            json={"text": ""},
        )
        # min_length=1 校验
        assert response.status_code == 422

    def test_detect_missing_text(self, client):
        response = client.post(
            "/api/v1/detect",
            json={},
        )
        assert response.status_code == 422

    def test_detect_disable_model(self, client):
        response = client.post(
            "/api/v1/detect",
            json={"text": "测试文本", "enable_model": False},
        )
        assert response.status_code == 200


class TestBatchDetectAPI:
    """批量检测接口测试"""

    def test_batch_detect(self, client):
        response = client.post(
            "/api/v1/detect/batch",
            json={
                "texts": ["今天天气好", "测试词", "另一个测试词"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 3
        assert "total_time_ms" in data

    def test_batch_detect_empty_list(self, client):
        response = client.post(
            "/api/v1/detect/batch",
            json={"texts": []},
        )
        # min_length=1 校验
        assert response.status_code == 422

    def test_batch_detect_single_item(self, client):
        response = client.post(
            "/api/v1/detect/batch",
            json={"texts": ["安全文本"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1


class TestWordsAPI:
    """词库管理接口测试"""

    def test_get_words(self, client):
        response = client.get("/api/v1/words/politics")
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "politics"
        assert "words" in data
        assert "count" in data

    def test_get_words_nonexistent_category(self, client):
        response = client.get("/api/v1/words/nonexistent")
        assert response.status_code == 404

    def test_add_words(self, client):
        response = client.post(
            "/api/v1/words/politics",
            json={"words": ["测试敏感词"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "politics"
        assert "added" in data

    def test_remove_words(self, client):
        # 先添加
        client.post(
            "/api/v1/words/politics",
            json={"words": ["待删除词"]},
        )
        # 再删除
        response = client.request(
            "DELETE",
            "/api/v1/words/politics",
            json={"words": ["待删除词"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "removed" in data


class TestReloadAPI:
    """热加载接口测试"""

    def test_reload(self, client):
        response = client.post("/api/v1/reload")
        assert response.status_code == 200
        data = response.json()
        assert "reloaded" in data


class TestStatsAPI:
    """统计接口测试"""

    def test_get_stats(self, client):
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "word_counts" in data
        assert "strategy" in data
        assert "total_requests" in data


class TestAuthAPI:
    """鉴权测试"""

    def test_with_valid_api_key(self, tmp_path):
        """测试有效 API Key"""
        import yaml

        for name in ["politics", "gambling", "pornography"]:
            (tmp_path / f"{name}.txt").write_text("测试\n", encoding="utf-8")

        test_config = {
            "server": {"host": "0.0.0.0", "port": 8000, "workers": 1, "reload": False},
            "auth": {"enabled": True, "api_keys": ["sk-content-sec-dev-001"]},
            "word_lists": {
                "politics": {"path": str(tmp_path / "politics.txt"), "enabled": True, "severity": "high"},
                "gambling": {"path": str(tmp_path / "gambling.txt"), "enabled": True, "severity": "medium"},
                "pornography": {"path": str(tmp_path / "pornography.txt"), "enabled": True, "severity": "high"},
            },
            "preprocessor": {"traditional_to_simplified": False, "normalize_variants": False, "remove_special_chars": False, "skip_chars": ""},
            "detector": {"max_batch_size": 100, "max_text_length": 10000, "strategy": "sensitive_only", "risk_thresholds": {}},
            "model": {"enabled": False, "model_path": "", "vectorizer_path": "", "confidence_threshold": 0.6},
            "logging": {"level": "INFO"},
        }

        config_file = tmp_path / "config_auth.yaml"
        config_file.write_text(yaml.dump(test_config, allow_unicode=True), encoding="utf-8")

        Config.reset()
        from app.config import config

        config.load(str(config_file))

        import app.api as api_module

        api_module._detector = None
        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/api/v1/detect",
            json={"text": "测试"},
            headers={"X-API-Key": "sk-content-sec-dev-001"},
        )
        assert response.status_code == 200

        api_module._detector = None
        Config.reset()

    def test_with_invalid_api_key(self, tmp_path):
        """测试无效 API Key"""
        import yaml

        for name in ["politics", "gambling", "pornography"]:
            (tmp_path / f"{name}.txt").write_text("测试\n", encoding="utf-8")

        test_config = {
            "server": {"host": "0.0.0.0", "port": 8000, "workers": 1, "reload": False},
            "auth": {"enabled": True, "api_keys": ["sk-content-sec-dev-001"]},
            "word_lists": {
                "politics": {"path": str(tmp_path / "politics.txt"), "enabled": True, "severity": "high"},
                "gambling": {"path": str(tmp_path / "gambling.txt"), "enabled": True, "severity": "medium"},
                "pornography": {"path": str(tmp_path / "pornography.txt"), "enabled": True, "severity": "high"},
            },
            "preprocessor": {"traditional_to_simplified": False, "normalize_variants": False, "remove_special_chars": False, "skip_chars": ""},
            "detector": {"max_batch_size": 100, "max_text_length": 10000, "strategy": "sensitive_only", "risk_thresholds": {}},
            "model": {"enabled": False, "model_path": "", "vectorizer_path": "", "confidence_threshold": 0.6},
            "logging": {"level": "INFO"},
        }

        config_file = tmp_path / "config_auth2.yaml"
        config_file.write_text(yaml.dump(test_config, allow_unicode=True), encoding="utf-8")

        Config.reset()
        from app.config import config

        config.load(str(config_file))

        import app.api as api_module

        api_module._detector = None
        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/api/v1/detect",
            json={"text": "测试"},
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401

        api_module._detector = None
        Config.reset()

    def test_without_api_key(self, tmp_path):
        """测试无 API Key"""
        import yaml

        for name in ["politics", "gambling", "pornography"]:
            (tmp_path / f"{name}.txt").write_text("测试\n", encoding="utf-8")

        test_config = {
            "server": {"host": "0.0.0.0", "port": 8000, "workers": 1, "reload": False},
            "auth": {"enabled": True, "api_keys": ["sk-content-sec-dev-001"]},
            "word_lists": {
                "politics": {"path": str(tmp_path / "politics.txt"), "enabled": True, "severity": "high"},
                "gambling": {"path": str(tmp_path / "gambling.txt"), "enabled": True, "severity": "medium"},
                "pornography": {"path": str(tmp_path / "pornography.txt"), "enabled": True, "severity": "high"},
            },
            "preprocessor": {"traditional_to_simplified": False, "normalize_variants": False, "remove_special_chars": False, "skip_chars": ""},
            "detector": {"max_batch_size": 100, "max_text_length": 10000, "strategy": "sensitive_only", "risk_thresholds": {}},
            "model": {"enabled": False, "model_path": "", "vectorizer_path": "", "confidence_threshold": 0.6},
            "logging": {"level": "INFO"},
        }

        config_file = tmp_path / "config_auth3.yaml"
        config_file.write_text(yaml.dump(test_config, allow_unicode=True), encoding="utf-8")

        Config.reset()
        from app.config import config

        config.load(str(config_file))

        import app.api as api_module

        api_module._detector = None
        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/api/v1/detect",
            json={"text": "测试"},
        )
        assert response.status_code == 401

        api_module._detector = None
        Config.reset()
