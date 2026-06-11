#!/usr/bin/env python3
"""
模型训练脚本

使用训练数据训练内容安全分类模型（TF-IDF + LogisticRegression）

用法:
    python -m training.train
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import config
from app.model_classifier import ModelClassifier


def load_training_data(data_path: str) -> tuple:
    """
    加载训练数据

    数据格式 (JSON):
    [
        {"text": "文本内容", "label": "safe|politics|gambling|pornography"},
        ...
    ]
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]

    print(f"加载训练数据: {len(texts)} 条")
    print(f"类别分布:")
    from collections import Counter
    label_counts = Counter(labels)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(texts)*100:.1f}%)")

    return texts, labels


def main():
    """主训练流程"""
    config.load()
    model_cfg = config.get_model_config()

    # 解析路径
    data_path = model_cfg.get("training_data", "training/data/train_data.json")
    model_path = model_cfg.get("model_path", "models/content_security_model.pkl")
    vectorizer_path = model_cfg.get("vectorizer_path", "models/tfidf_vectorizer.pkl")

    # 相对路径处理
    if not Path(data_path).is_absolute():
        data_path = str(project_root / data_path)
    if not Path(model_path).is_absolute():
        model_path = str(project_root / model_path)
    if not Path(vectorizer_path).is_absolute():
        vectorizer_path = str(project_root / vectorizer_path)

    data_file = Path(data_path)
    if not data_file.exists():
        print(f"训练数据不存在: {data_path}")
        print("请先准备训练数据，格式参考 training/data/train_data.json")
        sys.exit(1)

    # 加载数据
    texts, labels = load_training_data(data_path)

    if len(texts) < 20:
        print("训练数据不足（至少需要 20 条），跳过训练")
        sys.exit(1)

    # 创建分类器并训练
    classifier = ModelClassifier(
        model_path=model_path,
        vectorizer_path=vectorizer_path,
    )

    print("\n开始训练模型...")
    metrics = classifier.train(texts, labels, save=True)

    print(f"\n训练完成!")
    print(f"  F1 均值: {metrics.get('f1_macro', 0):.4f}")
    print(f"  样本数: {metrics.get('sample_count', 0)}")
    print(f"  模型保存: {model_path}")


if __name__ == "__main__":
    main()
