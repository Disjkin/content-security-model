"""Prometheus 监控指标定义"""

from prometheus_client import Counter, Gauge, Histogram, Info

# ==================== 请求级指标 ====================

REQUEST_COUNT = Counter(
    "cs_detect_requests_total",
    "Total detection requests",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "cs_detect_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ==================== 检测级指标 ====================

DETECTION_COUNT = Counter(
    "cs_detect_detections_total",
    "Total text detections",
    ["is_violation", "risk_level"],
)

WORD_HITS = Counter(
    "cs_detect_word_hits_total",
    "Total sensitive word matches",
    ["category"],
)

MODEL_INFERENCE_LATENCY = Histogram(
    "cs_detect_model_inference_seconds",
    "Model inference latency",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

BATCH_SIZE_HISTOGRAM = Histogram(
    "cs_detect_batch_size",
    "Batch detection sizes",
    buckets=[1, 5, 10, 25, 50, 100],
)

# ==================== 系统级指标 ====================

WORD_LIST_SIZE = Gauge(
    "cs_word_list_size",
    "Current word list size",
    ["category"],
)

MODEL_LOADED = Gauge(
    "cs_model_loaded",
    "Whether the ML model is loaded (1=yes, 0=no)",
)

WHITELIST_SIZE = Gauge(
    "cs_whitelist_size",
    "Current whitelist size",
)

STORAGE_LOGS_TOTAL = Gauge(
    "cs_storage_logs_total",
    "Total stored detection logs",
)

APP_INFO = Info("cs_app", "Application info")
