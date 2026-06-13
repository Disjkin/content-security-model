"""Pydantic 数据模型定义"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ViolationCategory(str, Enum):
    """违规类别枚举"""
    POLITICS = "politics"
    GAMBLING = "gambling"
    PORNOGRAPHY = "pornography"


class ViolationSource(str, Enum):
    """违规检测来源"""
    SENSITIVE_WORD = "sensitive_word"
    MODEL = "model"


class ViolationDetail(BaseModel):
    """单条违规详情"""
    category: str = Field(..., description="违规类别")
    source: str = Field(..., description="检测来源: sensitive_word / model")
    matched_word: Optional[str] = Field(None, description="命中的敏感词")
    position: Optional[List[int]] = Field(None, description="命中词在原文中的位置 [start, end]")
    confidence: float = Field(..., description="置信度 (0-1)", ge=0.0, le=1.0)
    severity: Optional[str] = Field(None, description="严重等级")


class DetectionSummary(BaseModel):
    """检测结果摘要"""
    total_matches: int = Field(0, description="总命中数")
    categories_hit: List[str] = Field(default_factory=list, description="命中的违规类别")
    word_hits: int = Field(0, description="敏感词命中数")
    model_confidence: Optional[float] = Field(None, description="模型最高置信度")
    model_category: Optional[str] = Field(None, description="模型预测类别")


class DetectionRequest(BaseModel):
    """单条检测请求"""
    text: str = Field(..., description="待检测文本", min_length=1)
    categories: Optional[List[str]] = Field(
        None,
        description="指定检测类别，不传则检测所有类别",
    )
    enable_model: bool = Field(True, description="是否启用模型推理")


class BatchDetectionRequest(BaseModel):
    """批量检测请求"""
    texts: List[str] = Field(..., description="待检测文本列表", min_length=1, max_length=100)
    categories: Optional[List[str]] = Field(None, description="指定检测类别")
    enable_model: bool = Field(True, description="是否启用模型推理")


class DetectionResult(BaseModel):
    """单条检测结果"""
    is_violation: bool = Field(..., description="是否违规")
    risk_level: str = Field(..., description="风险等级: low/medium/high/critical")
    violations: List[ViolationDetail] = Field(
        default_factory=list, description="违规详情列表"
    )
    summary: DetectionSummary = Field(
        default_factory=DetectionSummary, description="检测摘要"
    )
    processing_time_ms: float = Field(..., description="处理耗时(毫秒)")


class BatchDetectionResult(BaseModel):
    """批量检测结果"""
    results: List[DetectionResult] = Field(..., description="各文本检测结果")
    total_count: int = Field(..., description="总文本数")
    violation_count: int = Field(..., description="违规文本数")
    total_time_ms: float = Field(..., description="总耗时(毫秒)")


class WordManageRequest(BaseModel):
    """词库管理请求"""
    words: List[str] = Field(..., description="敏感词列表", min_length=1)


class StatsResponse(BaseModel):
    """统计信息响应"""
    word_counts: Dict[str, int] = Field(
        default_factory=dict, description="各类别词库大小"
    )
    model_loaded: bool = Field(False, description="模型是否已加载")
    strategy: str = Field(..., description="当前检测策略")
    total_requests: int = Field(0, description="总请求数")
    total_violations: int = Field(0, description="总违规数")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "0.1.0"
    model_loaded: bool = False
    word_count: int = 0


# ==================== 白名单模型 ====================

class WhitelistEntryModel(BaseModel):
    """白名单条目"""
    text: str = Field(..., description="白名单文本", min_length=1)
    reason: str = Field("", description="白名单原因")
    categories: Optional[List[str]] = Field(None, description="仅绕过指定类别")


class WhitelistManageRequest(BaseModel):
    """白名单管理请求"""
    entries: List[WhitelistEntryModel] = Field(..., min_length=1)


class WhitelistResponse(BaseModel):
    """白名单查询响应"""
    mode: str
    total: int
    entries: List[WhitelistEntryModel]


# ==================== 检测日志模型 ====================

class LogQueryRequest(BaseModel):
    """日志查询请求"""
    start_date: Optional[str] = Field(None, description="开始日期 YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="结束日期 YYYY-MM-DD")
    is_violation: Optional[bool] = Field(None, description="是否违规")
    risk_level: Optional[str] = Field(None, description="风险等级")
    category: Optional[str] = Field(None, description="违规类别")
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=500)


class DetectionLogResponse(BaseModel):
    """检测日志详情"""
    request_id: str
    text_hash: str
    text_preview: Optional[str] = None
    is_violation: bool
    risk_level: str
    categories_hit: Optional[List[str]] = None
    word_hits: int = 0
    model_confidence: Optional[float] = None
    model_category: Optional[str] = None
    processing_time_ms: Optional[float] = None
    source_ip: Optional[str] = None
    created_at: Optional[str] = None


class LogQueryResponse(BaseModel):
    """日志查询响应"""
    total: int
    page: int
    page_size: int
    logs: List[Dict]


# ==================== 分析统计模型 ====================

class AnalyticsSummary(BaseModel):
    """统计摘要"""
    total_detections: int = 0
    violation_count: int = 0
    violation_rate: float = 0.0
    avg_processing_time_ms: float = 0.0
    total_word_hits: int = 0
    risk_level_distribution: Dict[str, int] = Field(default_factory=dict)
    category_breakdown: Dict[str, int] = Field(default_factory=dict)


class TrendPoint(BaseModel):
    """趋势数据点"""
    period: str
    total: int
    violations: int
    avg_latency_ms: float


class WordStat(BaseModel):
    """高频违规词统计"""
    word: str
    count: int
    category: str


class CategoryStat(BaseModel):
    """类别统计"""
    category: str
    total_hits: int
    violation_count: int
    percentage: float
