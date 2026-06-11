"""检测引擎主逻辑 — 协调敏感词过滤与模型推理的组合检测策略"""

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from app.config import Config, config
from app.logger import get_logger
from app.model_classifier import ClassificationResult, ModelClassifier
from app.models import (
    DetectionResult,
    DetectionSummary,
    RiskLevel,
    ViolationDetail,
)
from app.preprocessor import Preprocessor
from app.sensitive_filter import FilterResult, SensitiveFilter
from app.storage import DetectionLogEntry, SQLiteStorage
from app.whitelist import Whitelist, WhitelistEntry
from app.word_manager import WordManager

logger = get_logger("detector")


class ContentDetector:
    """
    内容安全检测引擎

    支持三种检测策略：
    - sensitive_only: 仅使用敏感词过滤
    - model_only: 仅使用模型推理
    - combined: 双层组合检测（默认）
    """

    def __init__(self, config: Optional[Config] = None):
        self._config = config or Config()
        self._project_root = str(Path(__file__).parent.parent)

        # 初始化预处理器
        prep_cfg = self._config.get_preprocessor_config()
        self._preprocessor = Preprocessor(
            skip_chars=prep_cfg.get("skip_chars", ""),
            traditional_to_simplified=prep_cfg.get("traditional_to_simplified", True),
            normalize_variants=prep_cfg.get("normalize_variants", True),
            remove_special_chars=prep_cfg.get("remove_special_chars", True),
        )

        # 初始化词库管理器 + 敏感词过滤器
        self._word_manager = WordManager(self._config, self._project_root)
        self._sensitive_filter = SensitiveFilter(
            self._word_manager, self._preprocessor
        )

        # 初始化模型分类器
        model_cfg = self._config.get_model_config()
        self._model_classifier = ModelClassifier(
            model_path=self._resolve_path(model_cfg.get("model_path", "")),
            vectorizer_path=self._resolve_path(model_cfg.get("vectorizer_path", "")),
            confidence_threshold=model_cfg.get("confidence_threshold", 0.6),
        )

        # 检测策略
        detector_cfg = self._config.get_detector_config()
        self._strategy = detector_cfg.get("strategy", "combined")
        self._max_batch_size = detector_cfg.get("max_batch_size", 100)
        self._max_text_length = detector_cfg.get("max_text_length", 10000)
        self._risk_thresholds = detector_cfg.get("risk_thresholds", {})

        # 批量检测优化
        self._batch_parallel = detector_cfg.get("batch_parallel", True)
        self._batch_max_workers = detector_cfg.get("batch_max_workers", 4)
        self._batch_sequential_threshold = detector_cfg.get(
            "batch_sequential_threshold", 10
        )

        # 初始化白名单
        self._whitelist: Optional[Whitelist] = None
        self._init_whitelist()

        # 初始化存储
        self._storage: Optional[SQLiteStorage] = None
        self._log_queue: Optional[queue.Queue] = None
        self._log_thread: Optional[threading.Thread] = None
        self._init_storage()

        # 统计
        self._total_requests = 0
        self._total_violations = 0

        # 加载词库
        self._load_word_lists()

    def _init_whitelist(self) -> None:
        """初始化白名单"""
        wl_cfg = self._config.get_whitelist_config()
        if not wl_cfg.get("enabled", False):
            return

        mode = wl_cfg.get("mode", "exact")
        bypass_categories = wl_cfg.get("bypass_categories", None)
        self._whitelist = Whitelist(mode=mode, bypass_categories=bypass_categories)

        # 从文件加载
        wl_file = wl_cfg.get("file", "")
        if wl_file:
            filepath = self._resolve_path(wl_file)
            count = self._whitelist.load_from_file(filepath)
            logger.info(f"白名单文件加载: {count} 条")

        # 从配置加载
        entries = wl_cfg.get("entries", [])
        if entries:
            count = self._whitelist.load_from_config(entries)
            logger.info(f"白名单配置加载: {count} 条")

    def _init_storage(self) -> None:
        """初始化日志存储"""
        storage_cfg = self._config.get_storage_config()
        if not storage_cfg.get("enabled", False):
            return

        db_path = storage_cfg.get("db_path", "data/detection_logs.db")
        db_path = self._resolve_path(db_path)

        try:
            self._storage = SQLiteStorage(db_path)

            # 异步日志刷写
            if storage_cfg.get("async_flush", True):
                self._log_queue = queue.Queue()
                self._log_thread = threading.Thread(
                    target=self._flush_logs, daemon=True
                )
                self._log_thread.start()

            logger.info("日志存储初始化完成")
        except Exception as e:
            logger.error(f"日志存储初始化失败: {e}")
            self._storage = None

    def _flush_logs(self) -> None:
        """后台线程：批量刷写日志"""
        while True:
            try:
                # 阻塞等待，最多 5 秒
                try:
                    entry = self._log_queue.get(timeout=5.0)
                except queue.Empty:
                    continue

                # 收集队列中剩余的条目
                entries = [entry]
                while not self._log_queue.empty():
                    try:
                        entries.append(self._log_queue.get_nowait())
                    except queue.Empty:
                        break

                # 批量写入
                if self._storage:
                    for e in entries:
                        self._storage.save_log(e)

            except Exception as e:
                logger.error(f"日志刷写异常: {e}")
                time.sleep(1)

    def _resolve_path(self, path: str) -> str:
        """解析路径"""
        if not path:
            return ""
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(Path(self._project_root) / p)

    def _load_word_lists(self) -> None:
        """加载所有敏感词库"""
        try:
            counts = self._word_manager.load_all()
            total = sum(counts.values())
            logger.info(f"词库加载完成: {counts}, 总计 {total} 个词")
        except Exception as e:
            logger.error(f"词库加载失败: {e}")

    def detect(
        self,
        text: str,
        categories: Optional[List[str]] = None,
        enable_model: bool = True,
        request_id: Optional[str] = None,
    ) -> DetectionResult:
        """
        对单条文本执行内容安全检测

        Args:
            text: 待检测文本
            categories: 指定检测类别
            enable_model: 是否启用模型推理
            request_id: 请求 ID（用于日志关联）

        Returns:
            检测结果
        """
        start_time = time.perf_counter()
        self._total_requests += 1

        # 文本长度限制
        if len(text) > self._max_text_length:
            text = text[: self._max_text_length]

        # === 白名单检查 ===
        if self._whitelist:
            is_wl, reason = self._whitelist.is_whitelisted(text, categories)
            if is_wl:
                elapsed = (time.perf_counter() - start_time) * 1000
                result = DetectionResult(
                    is_violation=False,
                    risk_level=RiskLevel.LOW,
                    violations=[],
                    summary=DetectionSummary(),
                    processing_time_ms=round(elapsed, 2),
                )
                self._persist_log(request_id, text, result)
                return result

        violations: List[ViolationDetail] = []
        word_hits = 0
        model_confidence: Optional[float] = None
        model_category: Optional[str] = None

        # === 敏感词检测层 ===
        if self._strategy in ("sensitive_only", "combined"):
            filter_result = self._sensitive_filter.filter(text, categories)
            word_hits = filter_result.total_matches

            for match in filter_result.matches:
                violations.append(
                    ViolationDetail(
                        category=match.category,
                        source="sensitive_word",
                        matched_word=match.word,
                        position=[match.start, match.end],
                        confidence=1.0,
                        severity=match.severity,
                    )
                )

        # === 模型推理层 ===
        use_model = (
            enable_model
            and self._model_classifier.is_available
            and self._strategy in ("model_only", "combined")
        )

        if use_model:
            cls_result = self._model_classifier.classify(text)
            model_confidence = cls_result.confidence
            model_category = cls_result.predicted_category

            if cls_result.is_violation:
                # 检查类别是否被过滤
                if categories is None or cls_result.predicted_category in categories:
                    violations.append(
                        ViolationDetail(
                            category=cls_result.predicted_category,
                            source="model",
                            confidence=cls_result.confidence,
                            severity="medium",
                        )
                    )

        # === 结果聚合 ===
        is_violation = len(violations) > 0
        categories_hit = list(set(v.category for v in violations))
        risk_level = self._calculate_risk_level(
            violations=violations,
            word_hits=word_hits,
            model_result=(
                self._model_classifier.classify(text)
                if use_model
                else ClassificationResult()
            ),
        )

        if is_violation:
            self._total_violations += 1

        elapsed = (time.perf_counter() - start_time) * 1000  # ms

        result = DetectionResult(
            is_violation=is_violation,
            risk_level=risk_level,
            violations=violations,
            summary=DetectionSummary(
                total_matches=len(violations),
                categories_hit=sorted(categories_hit),
                word_hits=word_hits,
                model_confidence=model_confidence,
                model_category=model_category,
            ),
            processing_time_ms=round(elapsed, 2),
        )

        # 持久化日志
        self._persist_log(request_id, text, result)

        logger.debug(
            f"检测完成: violation={is_violation}, risk={risk_level}, "
            f"time={result.processing_time_ms}ms"
        )
        return result

    def _persist_log(
        self,
        request_id: Optional[str],
        text: str,
        result: DetectionResult,
    ) -> None:
        """持久化检测日志"""
        if not self._storage or not request_id:
            return

        entry = DetectionLogEntry(
            request_id=request_id,
            text=text,
            is_violation=result.is_violation,
            risk_level=result.risk_level,
            categories_hit=result.summary.categories_hit,
            word_hits=result.summary.word_hits,
            model_confidence=result.summary.model_confidence,
            model_category=result.summary.model_category,
            processing_time_ms=result.processing_time_ms,
            violations_summary=[
                {
                    "category": v.category,
                    "source": v.source,
                    "matched_word": v.matched_word,
                    "confidence": v.confidence,
                }
                for v in result.violations
            ],
        )

        if self._log_queue:
            try:
                self._log_queue.put_nowait(entry)
            except queue.Full:
                # 队列满时同步写入
                self._storage.save_log(entry)
        else:
            self._storage.save_log(entry)

    def detect_batch(
        self,
        texts: List[str],
        categories: Optional[List[str]] = None,
        enable_model: bool = True,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> List[DetectionResult]:
        """
        批量检测（支持并行处理）

        Args:
            texts: 待检测文本列表
            categories: 指定检测类别
            enable_model: 是否启用模型推理
            parallel: 是否并行处理
            max_workers: 最大并行数

        Returns:
            各文本检测结果列表
        """
        batch_size = min(len(texts), self._max_batch_size)
        texts = texts[:batch_size]

        if not texts:
            return []

        # 小批量或未启用并行时顺序执行
        if not parallel or not self._batch_parallel or batch_size <= self._batch_sequential_threshold:
            return [self.detect(t, categories, enable_model) for t in texts]

        # 并行执行
        workers = max_workers or self._batch_max_workers
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self.detect, t, categories, enable_model)
                    for t in texts
                ]
                results = []
                for future in futures:
                    try:
                        results.append(future.result(timeout=30.0))
                    except Exception as e:
                        logger.error(f"批量检测子任务失败: {e}")
                        results.append(
                            DetectionResult(
                                is_violation=False,
                                risk_level=RiskLevel.LOW,
                                processing_time_ms=0.0,
                            )
                        )
                return results
        except Exception as e:
            logger.error(f"并行批量检测失败，回退到顺序执行: {e}")
            return [self.detect(t, categories, enable_model) for t in texts]

    def _calculate_risk_level(
        self,
        violations: List[ViolationDetail],
        word_hits: int,
        model_result: ClassificationResult,
    ) -> str:
        """
        计算风险等级

        规则：
        - critical: 涉政类命中≥2词，或双层均判定违规
        - high: 敏感词命中≥1词，或模型置信度>0.8
        - medium: 仅模型判定违规且置信度 0.5-0.8
        - low: 未命中
        """
        if not violations:
            return RiskLevel.LOW

        # 统计涉政词命中数
        politics_word_hits = sum(
            1
            for v in violations
            if v.category == "politics" and v.source == "sensitive_word"
        )

        # 是否双层均命中
        has_word_violation = any(v.source == "sensitive_word" for v in violations)
        has_model_violation = any(v.source == "model" for v in violations)

        # Critical 条件
        if politics_word_hits >= 2:
            return RiskLevel.CRITICAL
        if has_word_violation and has_model_violation:
            return RiskLevel.CRITICAL

        critical_min = self._risk_thresholds.get("critical_min_matches", 5)
        if word_hits >= critical_min:
            return RiskLevel.CRITICAL

        # High 条件
        if word_hits >= 1:
            return RiskLevel.HIGH
        if model_result.confidence > 0.8 and model_result.is_violation:
            return RiskLevel.HIGH

        # Medium 条件
        if model_result.is_violation and model_result.confidence >= 0.5:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def reload(self) -> Dict[str, int]:
        """热加载词库"""
        logger.info("正在重载词库...")
        counts = self._word_manager.reload()
        logger.info(f"词库重载完成: {counts}")
        return counts

    def get_stats(self) -> Dict:
        """获取引擎统计"""
        return {
            "word_counts": self._sensitive_filter.get_stats(),
            "model_loaded": self._model_classifier.is_available,
            "strategy": self._strategy,
            "total_requests": self._total_requests,
            "total_violations": self._total_violations,
        }

    @property
    def word_manager(self) -> WordManager:
        """获取词库管理器实例"""
        return self._word_manager

    @property
    def model_classifier(self) -> ModelClassifier:
        """获取模型分类器实例"""
        return self._model_classifier
