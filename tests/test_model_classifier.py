"""模型推理分类器测试"""

import pytest
import numpy as np

from app.model_classifier import ClassificationResult, ModelClassifier


class TestClassificationResult:
    """ClassificationResult 数据类测试"""

    def test_default_values(self):
        result = ClassificationResult()
        assert result.predicted_category == "safe"
        assert result.confidence == 0.0
        assert result.probabilities == {}
        assert result.is_violation is False


class TestModelClassifier:
    """ModelClassifier 分类器测试"""

    def test_init_without_model(self):
        classifier = ModelClassifier()
        assert classifier.is_available is False

    def test_classify_without_model(self):
        classifier = ModelClassifier()
        result = classifier.classify("测试文本")
        assert result.predicted_category == "safe"
        assert result.is_violation is False

    def test_classify_empty_text(self):
        classifier = ModelClassifier()
        result = classifier.classify("")
        assert result.predicted_category == "safe"

    def test_classify_none_text(self):
        classifier = ModelClassifier()
        result = classifier.classify(None)
        assert result.predicted_category == "safe"

    def test_get_labels_default(self):
        classifier = ModelClassifier()
        labels = classifier.get_labels()
        assert "safe" in labels
        assert "politics" in labels
        assert "gambling" in labels
        assert "pornography" in labels

    def test_train_and_classify(self, tmp_path):
        """测试训练后分类"""
        model_path = str(tmp_path / "model.pkl")
        vectorizer_path = str(tmp_path / "vectorizer.pkl")

        classifier = ModelClassifier(
            model_path=model_path,
            vectorizer_path=vectorizer_path,
        )

        # 准备训练数据
        texts = [
            "今天天气真好",
            "学习Python很有趣",
            "去公园散步",
            "网络赌博平台",
            "在线赌博网站",
            "赌博软件下载",
            "分裂国家的行为",
            "颠覆政权的阴谋",
            "反动组织活动",
            "色情网站传播",
            "淫秽视频内容",
            "成人影片观看",
        ] * 5  # 重复以满足 min_df 要求

        labels = [
            "safe", "safe", "safe",
            "gambling", "gambling", "gambling",
            "politics", "politics", "politics",
            "pornography", "pornography", "pornography",
        ] * 5

        metrics = classifier.train(texts, labels, save=True)
        assert classifier.is_available is True
        assert "f1_macro" in metrics

        # 分类测试
        result = classifier.classify("网络赌博害人害己")
        assert result.predicted_category != "safe"
        assert result.confidence > 0

    def test_model_persistence(self, tmp_path):
        """测试模型保存和加载"""
        model_path = str(tmp_path / "model.pkl")
        vectorizer_path = str(tmp_path / "vectorizer.pkl")

        # 训练并保存
        classifier1 = ModelClassifier(
            model_path=model_path,
            vectorizer_path=vectorizer_path,
        )

        texts = ["安全文本"] * 10 + ["赌博网站"] * 10
        labels = ["safe"] * 10 + ["gambling"] * 10
        classifier1.train(texts, labels, save=True)

        # 重新加载
        classifier2 = ModelClassifier(
            model_path=model_path,
            vectorizer_path=vectorizer_path,
        )
        assert classifier2.is_available is True

    def test_tokenize(self):
        """测试分词功能"""
        classifier = ModelClassifier()
        tokenized = classifier._tokenize("网络赌博害人害己")
        assert isinstance(tokenized, str)
        assert len(tokenized) > 0
