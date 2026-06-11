"""模型推理分类器 — 基于 TF-IDF + scikit-learn 的语义违规检测"""

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.logger import get_logger

logger = get_logger("model_classifier")


@dataclass
class ClassificationResult:
    """分类结果"""
    predicted_category: str = "safe"
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    is_violation: bool = False


class ModelClassifier:
    """
    基于机器学习的文本违规分类器

    使用 TF-IDF 向量化 + 分类模型（LinearSVC / LogisticRegression）
    对文本进行语义级别的违规分类。

    支持类别：safe / politics / gambling / pornography
    """

    LABELS = ["safe", "politics", "gambling", "pornography"]

    def __init__(
        self,
        model_path: Optional[str] = None,
        vectorizer_path: Optional[str] = None,
        confidence_threshold: float = 0.6,
    ):
        self._model_path = model_path
        self._vectorizer_path = vectorizer_path
        self._confidence_threshold = confidence_threshold
        self._model = None
        self._vectorizer = None
        self._loaded = False

        # 尝试加载已有模型
        if model_path and vectorizer_path:
            self._try_load()

    def _try_load(self) -> None:
        """尝试加载已训练的模型"""
        try:
            model_file = Path(self._model_path)
            vec_file = Path(self._vectorizer_path)

            if model_file.exists() and vec_file.exists():
                with open(model_file, "rb") as f:
                    self._model = pickle.load(f)
                with open(vec_file, "rb") as f:
                    self._vectorizer = pickle.load(f)
                self._loaded = True
                logger.info("模型加载成功")
            else:
                logger.info(
                    "模型文件不存在，模型分类器未启用。"
                    "运行训练脚本生成模型: python -m training.train"
                )
        except Exception as e:
            logger.warning(f"模型加载失败: {e}")
            self._loaded = False

    @property
    def is_available(self) -> bool:
        """模型是否已加载可用"""
        return self._loaded

    def classify(self, text: str) -> ClassificationResult:
        """
        对文本进行分类推理

        Args:
            text: 待分类文本

        Returns:
            分类结果
        """
        if not self._loaded:
            return ClassificationResult()

        if not text or not text.strip():
            return ClassificationResult()

        try:
            # 分词
            tokenized = self._tokenize(text)
            if not tokenized:
                return ClassificationResult()

            # TF-IDF 向量化
            tfidf_vec = self._vectorizer.transform([tokenized])

            # 模型推理
            if hasattr(self._model, "predict_proba"):
                # LogisticRegression 等有 predict_proba
                proba = self._model.predict_proba(tfidf_vec)[0]
            else:
                # LinearSVC 使用 decision_function 并转换为概率
                decision = self._model.decision_function(tfidf_vec)[0]
                # softmax-like 归一化
                exp_decision = np.exp(decision - np.max(decision))
                proba = exp_decision / exp_decision.sum()

            # 获取各类别概率
            labels = self._model.classes_ if hasattr(self._model, "classes_") else self.LABELS
            probabilities = {}
            for label, prob in zip(labels, proba):
                probabilities[label] = float(prob)

            # 取概率最高的类别
            predicted_idx = int(np.argmax(proba))
            predicted_category = labels[predicted_idx]
            confidence = float(proba[predicted_idx])

            is_violation = (
                predicted_category != "safe"
                and confidence >= self._confidence_threshold
            )

            return ClassificationResult(
                predicted_category=predicted_category,
                confidence=confidence,
                probabilities=probabilities,
                is_violation=is_violation,
            )

        except Exception as e:
            logger.error(f"模型推理失败: {e}")
            return ClassificationResult()

    def _tokenize(self, text: str) -> str:
        """
        文本分词（使用 jieba）

        Returns:
            空格分隔的分词结果字符串
        """
        try:
            import jieba
            words = jieba.lcut(text)
            # 过滤空白和单字符
            words = [w.strip() for w in words if len(w.strip()) > 1]
            return " ".join(words)
        except ImportError:
            logger.warning("jieba 未安装，使用简单空格分词")
            return text

    def train(
        self,
        texts: List[str],
        labels: List[str],
        save: bool = True,
    ) -> Dict[str, float]:
        """
        训练分类模型

        Args:
            texts: 训练文本列表
            labels: 对应标签列表
            save: 是否保存模型到文件

        Returns:
            训练评估指标
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        logger.info(f"开始训练模型，样本数: {len(texts)}")

        # 分词
        tokenized_texts = [self._tokenize(t) for t in texts]

        # TF-IDF 向量化
        vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(tokenized_texts)

        # 训练分类器
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
        model.fit(tfidf_matrix, labels)

        # 交叉验证评估
        try:
            scores = cross_val_score(model, tfidf_matrix, labels, cv=5, scoring="f1_macro")
            avg_f1 = float(scores.mean())
            logger.info(f"交叉验证 F1 均值: {avg_f1:.4f}")
        except Exception as e:
            avg_f1 = 0.0
            logger.warning(f"交叉验证失败: {e}")

        self._model = model
        self._vectorizer = vectorizer
        self._loaded = True

        # 保存模型
        if save and self._model_path and self._vectorizer_path:
            self._save_model()

        return {"f1_macro": avg_f1, "sample_count": len(texts)}

    def _save_model(self) -> None:
        """保存模型到文件"""
        try:
            model_dir = Path(self._model_path).parent
            model_dir.mkdir(parents=True, exist_ok=True)

            with open(self._model_path, "wb") as f:
                pickle.dump(self._model, f)
            with open(self._vectorizer_path, "wb") as f:
                pickle.dump(self._vectorizer, f)

            logger.info(f"模型已保存: {self._model_path}")
        except Exception as e:
            logger.error(f"模型保存失败: {e}")

    def get_labels(self) -> List[str]:
        """获取模型支持的类别标签"""
        if self._model and hasattr(self._model, "classes_"):
            return list(self._model.classes_)
        return self.LABELS
