"""检测日志持久化存储 — SQLite 实现"""

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.logger import get_logger

logger = get_logger("storage")


@dataclass
class DetectionLogEntry:
    """检测日志条目"""
    request_id: str
    text: str  # 原始文本（用于计算 hash，不持久化）
    is_violation: bool
    risk_level: str
    categories_hit: List[str] = field(default_factory=list)
    word_hits: int = 0
    model_confidence: Optional[float] = None
    model_category: Optional[str] = None
    processing_time_ms: float = 0.0
    source_ip: Optional[str] = None
    api_key_id: Optional[str] = None
    violations_summary: Optional[List[Dict]] = None


@dataclass
class LogQuery:
    """日志查询条件"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_violation: Optional[bool] = None
    risk_level: Optional[str] = None
    category: Optional[str] = None
    page: int = 1
    page_size: int = 50


@dataclass
class LogQueryResult:
    """日志查询结果"""
    total: int
    page: int
    page_size: int
    logs: List[Dict[str, Any]]


class SQLiteStorage:
    """
    SQLite 检测日志存储

    使用线程安全的连接池，支持异步刷写。
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS detection_logs (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id      TEXT NOT NULL UNIQUE,
        text_hash       TEXT NOT NULL,
        text_preview    TEXT,
        is_violation    INTEGER NOT NULL,
        risk_level      TEXT NOT NULL,
        categories_hit  TEXT,
        word_hits       INTEGER DEFAULT 0,
        model_confidence REAL,
        model_category  TEXT,
        processing_time_ms REAL,
        source_ip       TEXT,
        api_key_id      TEXT,
        violations_summary TEXT,
        created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_logs_created_at ON detection_logs(created_at);
    CREATE INDEX IF NOT EXISTS idx_logs_is_violation ON detection_logs(is_violation);
    CREATE INDEX IF NOT EXISTS idx_logs_risk_level ON detection_logs(risk_level);
    CREATE INDEX IF NOT EXISTS idx_logs_text_hash ON detection_logs(text_hash);
    """

    def __init__(self, db_path: str = "data/detection_logs.db"):
        self._db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()

        # 确保目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库表结构"""
        with self._get_connection() as conn:
            conn.executescript(self.SCHEMA)
            logger.info(f"数据库初始化完成: {self._db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """获取线程本地数据库连接"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                timeout=10.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _text_hash(self, text: str) -> str:
        """计算文本 SHA-256 哈希"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _text_preview(self, text: str, max_len: int = 100) -> str:
        """截取文本预览"""
        return text[:max_len] if len(text) <= max_len else text[:max_len] + "..."

    def save_log(self, entry: DetectionLogEntry) -> str:
        """
        保存检测日志

        Returns:
            request_id
        """
        try:
            conn = self._get_connection()
            conn.execute(
                """INSERT OR IGNORE INTO detection_logs
                   (request_id, text_hash, text_preview, is_violation, risk_level,
                    categories_hit, word_hits, model_confidence, model_category,
                    processing_time_ms, source_ip, api_key_id, violations_summary)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.request_id,
                    self._text_hash(entry.text),
                    self._text_preview(entry.text),
                    1 if entry.is_violation else 0,
                    entry.risk_level,
                    json.dumps(entry.categories_hit, ensure_ascii=False),
                    entry.word_hits,
                    entry.model_confidence,
                    entry.model_category,
                    entry.processing_time_ms,
                    entry.source_ip,
                    entry.api_key_id,
                    json.dumps(entry.violations_summary, ensure_ascii=False)
                    if entry.violations_summary
                    else None,
                ),
            )
            conn.commit()
            return entry.request_id
        except Exception as e:
            logger.error(f"保存检测日志失败: {e}")
            return entry.request_id

    def get_log(self, request_id: str) -> Optional[Dict[str, Any]]:
        """获取单条日志"""
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM detection_logs WHERE request_id = ?",
                (request_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None
        except Exception as e:
            logger.error(f"查询日志失败: {e}")
            return None

    def query_logs(self, query: LogQuery) -> LogQueryResult:
        """查询日志列表"""
        try:
            conn = self._get_connection()

            # 构建 WHERE 子句
            conditions = []
            params = []

            if query.start_date:
                conditions.append("created_at >= ?")
                params.append(query.start_date)

            if query.end_date:
                conditions.append("created_at <= ?")
                params.append(query.end_date + " 23:59:59")

            if query.is_violation is not None:
                conditions.append("is_violation = ?")
                params.append(1 if query.is_violation else 0)

            if query.risk_level:
                conditions.append("risk_level = ?")
                params.append(query.risk_level)

            if query.category:
                conditions.append("categories_hit LIKE ?")
                params.append(f"%{query.category}%")

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # 查询总数
            count_sql = f"SELECT COUNT(*) FROM detection_logs WHERE {where_clause}"
            total = conn.execute(count_sql, params).fetchone()[0]

            # 分页查询
            offset = (query.page - 1) * query.page_size
            data_sql = (
                f"SELECT * FROM detection_logs WHERE {where_clause} "
                f"ORDER BY created_at DESC LIMIT ? OFFSET ?"
            )
            rows = conn.execute(data_sql, params + [query.page_size, offset]).fetchall()

            logs = [self._row_to_dict(row) for row in rows]

            return LogQueryResult(
                total=total,
                page=query.page,
                page_size=query.page_size,
                logs=logs,
            )
        except Exception as e:
            logger.error(f"查询日志失败: {e}")
            return LogQueryResult(total=0, page=query.page, page_size=query.page_size, logs=[])

    def get_stats_summary(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """获取统计摘要"""
        try:
            conn = self._get_connection()

            conditions = []
            params = []
            if start_date:
                conditions.append("created_at >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("created_at <= ?")
                params.append(end_date + " 23:59:59")

            where = " AND ".join(conditions) if conditions else "1=1"

            row = conn.execute(
                f"""SELECT
                        COUNT(*) as total,
                        SUM(is_violation) as violations,
                        AVG(processing_time_ms) as avg_latency,
                        SUM(word_hits) as total_word_hits
                    FROM detection_logs WHERE {where}""",
                params,
            ).fetchone()

            total = row[0] or 0
            violations = row[1] or 0
            avg_latency = round(row[2] or 0, 2)
            total_word_hits = row[3] or 0

            # 风险等级分布
            risk_rows = conn.execute(
                f"SELECT risk_level, COUNT(*) FROM detection_logs "
                f"WHERE {where} GROUP BY risk_level",
                params,
            ).fetchall()
            risk_distribution = {row[0]: row[1] for row in risk_rows}

            # 类别分布
            cat_rows = conn.execute(
                f"SELECT categories_hit FROM detection_logs "
                f"WHERE {where} AND is_violation = 1",
                params,
            ).fetchall()
            category_counts: Dict[str, int] = {}
            for row in cat_rows:
                try:
                    cats = json.loads(row[0]) if row[0] else []
                    for cat in cats:
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass

            return {
                "total_detections": total,
                "violation_count": violations,
                "violation_rate": round(violations / total, 4) if total > 0 else 0.0,
                "avg_processing_time_ms": avg_latency,
                "total_word_hits": total_word_hits,
                "risk_level_distribution": risk_distribution,
                "category_breakdown": category_counts,
            }
        except Exception as e:
            logger.error(f"获取统计摘要失败: {e}")
            return {}

    def get_trends(
        self,
        period: str = "day",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """获取趋势数据"""
        try:
            conn = self._get_connection()

            conditions = []
            params = []
            if start_date:
                conditions.append("created_at >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("created_at <= ?")
                params.append(end_date + " 23:59:59")

            where = " AND ".join(conditions) if conditions else "1=1"

            # SQLite 日期格式化
            if period == "hour":
                date_fmt = "%Y-%m-%d %H:00"
            elif period == "day":
                date_fmt = "%Y-%m-%d"
            elif period == "month":
                date_fmt = "%Y-%m"
            else:
                date_fmt = "%Y-%m-%d"

            rows = conn.execute(
                f"""SELECT
                        strftime('{date_fmt}', created_at) as period,
                        COUNT(*) as total,
                        SUM(is_violation) as violations,
                        AVG(processing_time_ms) as avg_latency
                    FROM detection_logs
                    WHERE {where}
                    GROUP BY strftime('{date_fmt}', created_at)
                    ORDER BY period""",
                params,
            ).fetchall()

            return [
                {
                    "period": row[0],
                    "total": row[1],
                    "violations": row[2] or 0,
                    "avg_latency_ms": round(row[3] or 0, 2),
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"获取趋势数据失败: {e}")
            return []

    def get_top_violation_words(
        self,
        limit: int = 20,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """获取高频违规词"""
        try:
            conn = self._get_connection()

            conditions = ["is_violation = 1", "violations_summary IS NOT NULL"]
            params: list = []
            if start_date:
                conditions.append("created_at >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("created_at <= ?")
                params.append(end_date + " 23:59:59")

            where = " AND ".join(conditions)

            rows = conn.execute(
                f"SELECT violations_summary FROM detection_logs WHERE {where}",
                params,
            ).fetchall()

            word_counts: Dict[str, Dict[str, int]] = {}
            for row in rows:
                try:
                    violations = json.loads(row[0])
                    for v in (violations or []):
                        word = v.get("matched_word")
                        if word:
                            if word not in word_counts:
                                word_counts[word] = {"count": 0, "category": v.get("category", "")}
                            word_counts[word]["count"] += 1
                except (json.JSONDecodeError, TypeError):
                    pass

            sorted_words = sorted(
                word_counts.items(), key=lambda x: x[1]["count"], reverse=True
            )[:limit]

            return [
                {"word": word, "count": info["count"], "category": info["category"]}
                for word, info in sorted_words
            ]
        except Exception as e:
            logger.error(f"获取高频违规词失败: {e}")
            return []

    def cleanup(self, retention_days: int = 90) -> int:
        """清理过期日志"""
        try:
            conn = self._get_connection()
            cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
            cursor = conn.execute(
                "DELETE FROM detection_logs WHERE created_at < ?", (cutoff,)
            )
            conn.commit()
            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"清理了 {deleted} 条过期日志")
            return deleted
        except Exception as e:
            logger.error(f"清理日志失败: {e}")
            return 0

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """将数据库行转换为字典"""
        d = dict(row)
        # JSON 字段解析
        for key in ("categories_hit", "violations_summary"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        # Boolean 字段
        d["is_violation"] = bool(d.get("is_violation", 0))
        return d
