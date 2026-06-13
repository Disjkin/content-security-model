"""FastAPI 路由与接口定义"""

import json
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Header, Query, Request, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse


class UnicodeJSONResponse(JSONResponse):
    """JSON 响应类 — 保留中文字符，不进行 Unicode 转义"""

    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

from app.config import Config, config
from app.detector import ContentDetector
from app.logger import get_logger, setup_logger
from app.models import (
    AnalyticsSummary,
    BatchDetectionRequest,
    BatchDetectionResult,
    DetectionRequest,
    DetectionResult,
    HealthResponse,
    LogQueryRequest,
    LogQueryResponse,
    StatsResponse,
    WhitelistEntryModel,
    WhitelistManageRequest,
    WhitelistResponse,
    WordManageRequest,
)
from app.storage import LogQuery

logger = get_logger("api")

# 全局检测引擎实例
_detector: Optional[ContentDetector] = None


def get_detector() -> ContentDetector:
    """获取检测引擎实例"""
    global _detector
    if _detector is None:
        if not config._config:
            config.load()
        log_cfg = config._config.get("logging", {})
        setup_logger(
            level=log_cfg.get("level", "INFO"),
            log_format=log_cfg.get("format"),
            log_file=log_cfg.get("file"),
            max_bytes=log_cfg.get("max_bytes", 10485760),
            backup_count=log_cfg.get("backup_count", 5),
        )
        _detector = ContentDetector(config)
    return _detector


def verify_api_key(
    request: Request,
    x_api_key: str = Header(None),
    x_admin_request: str = Header(None),
    api_key: str = Query(None, description="API Key（浏览器访问时可用 URL 参数）"),
) -> str:
    """
    API Key 鉴权（支持三种方式）：
    1. Header: X-API-Key: sk-xxx
    2. URL 参数: ?api_key=sk-xxx
    3. 管理后台自动放行: X-Admin-Request: true
    """
    auth_cfg = config.get_auth_config()
    if not auth_cfg.get("enabled", False):
        return ""

    # 管理后台请求自动放行
    if x_admin_request == "true":
        return ""

    valid_keys = auth_cfg.get("api_keys", [])

    # 优先从 Header 取，其次从 URL 参数取
    key = x_api_key or api_key
    if not key or key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="无效的 API Key。请在 Header 中添加 X-API-Key，或在 URL 中添加 ?api_key=你的密钥",
        )
    return key


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""

    # 加载配置（仅在未加载时）
    if not config._config:
        config.load()

    app = FastAPI(
        title="内容安全检测 API",
        description="基于敏感词过滤 + 模型推理的内容安全检测服务",
        version="0.2.0",
        docs_url=None,       # 禁用默认 /docs，改为自定义
        redoc_url=None,      # 禁用默认 /redoc
        default_response_class=UnicodeJSONResponse,
    )

    # 声明 API Key 鉴权方案（让 Swagger UI 显示 Authorize 按钮）
    from fastapi.security import APIKeyHeader
    from fastapi.openapi.utils import get_openapi

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            description=app.description,
            version=app.version,
            routes=app.routes,
        )
        schema["components"]["securitySchemes"] = {
            "ApiKeyHeader": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API Key 鉴权，值在 config.yaml 中配置",
            }
        }
        # 所有接口默认需要 ApiKeyHeader
        for path in schema["paths"]:
            for method in schema["paths"][path]:
                if method in ("get", "post", "put", "delete", "patch"):
                    schema["paths"][path][method]["security"] = [{"ApiKeyHeader": []}]
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi

    # ==================== 静态文件 ====================

    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/admin/static", StaticFiles(directory=str(static_dir)), name="admin_static")

        @app.get("/admin", include_in_schema=False)
        async def admin_page():
            return FileResponse(str(static_dir / "index.html"))

    @app.get("/", include_in_schema=False)
    async def root():
        """根路径重定向到管理后台"""
        return RedirectResponse(url="/admin")

    # ==================== 自定义 /docs（本地静态资源） ====================

    swagger_dir = static_dir / "swagger-ui"
    if swagger_dir.exists():
        app.mount("/docs/assets", StaticFiles(directory=str(swagger_dir)), name="swagger_assets")

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui():
        """使用本地静态资源加载 Swagger UI，无需依赖外部 CDN"""
        return HTMLResponse(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{app.title} - API 文档</title>
    <link rel="stylesheet" href="/docs/assets/swagger-ui.css">
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="/docs/assets/swagger-ui-bundle.js"></script>
    <script>
        SwaggerUIBundle({{
            url: '/openapi.json',
            dom_id: '#swagger-ui',
            layout: 'BaseLayout',
            deepLinking: true,
            showExtensions: true,
            showCommonExtensions: true,
        }});
    </script>
</body>
</html>""")

    # ==================== CORS ====================

    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ==================== 中间件 ====================

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """为每个请求分配唯一 ID"""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # ==================== 健康检查 ====================

    @app.get("/health", response_model=HealthResponse, tags=["系统"])
    async def health_check():
        """健康检查接口"""
        detector = get_detector()
        stats = detector.get_stats()
        total_words = sum(stats["word_counts"].values())
        return HealthResponse(
            status="ok",
            version="0.2.0",
            model_loaded=stats["model_loaded"],
            word_count=total_words,
        )

    # ==================== 检测接口 ====================

    @app.post(
        "/api/v1/detect",
        response_model=DetectionResult,
        tags=["内容检测"],
    )
    async def detect_single(
        request: DetectionRequest,
        req: Request,
        _: str = Depends(verify_api_key),
    ):
        """
        单条文本内容检测

        对输入文本执行敏感词匹配和模型推理，返回违规检测结果。
        """
        detector = get_detector()
        result = detector.detect(
            text=request.text,
            categories=request.categories,
            enable_model=request.enable_model,
            request_id=getattr(req.state, "request_id", None),
        )
        return result

    @app.post(
        "/api/v1/detect/batch",
        response_model=BatchDetectionResult,
        tags=["内容检测"],
    )
    async def detect_batch(
        request: BatchDetectionRequest,
        _: str = Depends(verify_api_key),
    ):
        """
        批量文本内容检测

        支持同时检测多条文本，最大批量数由配置决定。
        """
        detector = get_detector()
        start = time.perf_counter()

        results = detector.detect_batch(
            texts=request.texts,
            categories=request.categories,
            enable_model=request.enable_model,
        )

        total_time = (time.perf_counter() - start) * 1000
        violation_count = sum(1 for r in results if r.is_violation)

        return BatchDetectionResult(
            results=results,
            total_count=len(results),
            violation_count=violation_count,
            total_time_ms=round(total_time, 2),
        )

    # ==================== 词库管理接口 ====================

    @app.get("/api/v1/words", tags=["词库管理"])
    async def get_all_words_overview(_: str = Depends(verify_api_key)):
        """查询所有类别词库概览"""
        detector = get_detector()
        wm = detector.word_manager
        categories = wm.get_categories()
        return {
            "categories": [
                {
                    "category": cat,
                    "count": wm.get_word_count(cat),
                }
                for cat in categories
            ],
            "total": wm.get_total_count(),
        }

    @app.get(
        "/api/v1/words/{category}",
        tags=["词库管理"],
    )
    async def get_words(
        category: str,
        _: str = Depends(verify_api_key),
    ):
        """查询指定类别的词库"""
        detector = get_detector()
        wm = detector.word_manager

        if category not in wm.get_categories():
            raise HTTPException(status_code=404, detail=f"类别不存在: {category}")

        words = sorted(wm.get_words(category))
        return {
            "category": category,
            "count": len(words),
            "words": words,
        }

    @app.post(
        "/api/v1/words/{category}",
        tags=["词库管理"],
    )
    async def add_words(
        category: str,
        request: WordManageRequest,
        _: str = Depends(verify_api_key),
    ):
        """向指定类别添加敏感词"""
        detector = get_detector()
        wm = detector.word_manager

        added = wm.add_words(category, request.words)
        return {
            "category": category,
            "requested": len(request.words),
            "added": added,
            "total": wm.get_word_count(category),
        }

    @app.delete(
        "/api/v1/words/{category}",
        tags=["词库管理"],
    )
    async def remove_words(
        category: str,
        request: WordManageRequest,
        _: str = Depends(verify_api_key),
    ):
        """从指定类别删除敏感词"""
        detector = get_detector()
        wm = detector.word_manager

        removed = wm.remove_words(category, request.words)
        return {
            "category": category,
            "requested": len(request.words),
            "removed": removed,
            "total": wm.get_word_count(category),
        }

    @app.get("/api/v1/words/{category}/export", tags=["词库管理"])
    async def export_words(
        category: str,
        _: str = Depends(verify_api_key),
    ):
        """导出指定类别的词库"""
        detector = get_detector()
        wm = detector.word_manager

        if category not in wm.get_categories():
            raise HTTPException(status_code=404, detail=f"类别不存在: {category}")

        content = wm.export_words(category)
        return PlainTextResponse(
            content,
            headers={
                "Content-Disposition": f"attachment; filename={category}.txt"
            },
        )

    @app.post("/api/v1/words/{category}/import", tags=["词库管理"])
    async def import_words(
        category: str,
        file: UploadFile = File(...),
        _: str = Depends(verify_api_key),
    ):
        """从文件导入敏感词"""
        detector = get_detector()
        wm = detector.word_manager

        try:
            content = await file.read()
            text = content.decode("utf-8")
            words = [w.strip() for w in text.splitlines() if w.strip() and not w.startswith("#")]
            added = wm.add_words(category, words)
            return {
                "category": category,
                "imported": len(words),
                "added": added,
                "total": wm.get_word_count(category),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"导入失败: {e}")

    @app.post("/api/v1/words/{category}/search", tags=["词库管理"])
    async def search_words(
        category: str,
        keyword: str = Query("", description="搜索关键词"),
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=500),
        _: str = Depends(verify_api_key),
    ):
        """搜索指定类别词库"""
        detector = get_detector()
        wm = detector.word_manager

        if category not in wm.get_categories():
            raise HTTPException(status_code=404, detail=f"类别不存在: {category}")

        words = sorted(wm.get_words(category))
        if keyword:
            words = [w for w in words if keyword in w]

        total = len(words)
        start = (page - 1) * page_size
        return {
            "category": category,
            "total": total,
            "page": page,
            "page_size": page_size,
            "words": words[start : start + page_size],
        }

    @app.post("/api/v1/reload", tags=["词库管理"])
    async def reload_words(_: str = Depends(verify_api_key)):
        """手动触发词库热加载"""
        detector = get_detector()
        counts = detector.reload()
        return {"reloaded": counts}

    # ==================== 白名单管理接口 ====================

    @app.get("/api/v1/whitelist", tags=["白名单管理"])
    async def get_whitelist(_: str = Depends(verify_api_key)):
        """查询白名单"""
        detector = get_detector()
        wl = detector._whitelist

        if not wl:
            return WhitelistResponse(mode="exact", total=0, entries=[])

        entries = wl.get_all_entries()
        return WhitelistResponse(
            mode=wl.mode,
            total=wl.size,
            entries=[
                WhitelistEntryModel(
                    text=e.text, reason=e.reason, categories=e.categories
                )
                for e in entries
            ],
        )

    @app.post("/api/v1/whitelist", tags=["白名单管理"])
    async def add_whitelist(
        request: WhitelistManageRequest,
        _: str = Depends(verify_api_key),
    ):
        """添加白名单条目"""
        detector = get_detector()
        wl = detector._whitelist

        if not wl:
            raise HTTPException(status_code=400, detail="白名单未启用")

        added = 0
        for entry in request.entries:
            from app.whitelist import WhitelistEntry

            if wl.add(
                WhitelistEntry(
                    text=entry.text,
                    reason=entry.reason,
                    categories=entry.categories,
                )
            ):
                added += 1

        return {"requested": len(request.entries), "added": added, "total": wl.size}

    @app.delete("/api/v1/whitelist", tags=["白名单管理"])
    async def remove_whitelist(
        request: WhitelistManageRequest,
        _: str = Depends(verify_api_key),
    ):
        """删除白名单条目"""
        detector = get_detector()
        wl = detector._whitelist

        if not wl:
            raise HTTPException(status_code=400, detail="白名单未启用")

        removed = 0
        for entry in request.entries:
            if wl.remove(entry.text):
                removed += 1

        return {
            "requested": len(request.entries),
            "removed": removed,
            "total": wl.size,
        }

    # ==================== 检测日志接口 ====================

    @app.get("/api/v1/logs", tags=["检测日志"])
    async def query_logs(
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        is_violation: Optional[bool] = Query(None),
        risk_level: Optional[str] = Query(None),
        category: Optional[str] = Query(None),
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=500),
        _: str = Depends(verify_api_key),
    ):
        """查询检测日志"""
        detector = get_detector()
        storage = detector._storage

        if not storage:
            raise HTTPException(status_code=400, detail="日志存储未启用")

        query = LogQuery(
            start_date=start_date,
            end_date=end_date,
            is_violation=is_violation,
            risk_level=risk_level,
            category=category,
            page=page,
            page_size=page_size,
        )

        result = storage.query_logs(query)
        return {
            "total": result.total,
            "page": result.page,
            "page_size": result.page_size,
            "logs": result.logs,
        }

    @app.get("/api/v1/logs/{request_id}", tags=["检测日志"])
    async def get_log(
        request_id: str,
        _: str = Depends(verify_api_key),
    ):
        """获取单条检测日志"""
        detector = get_detector()
        storage = detector._storage

        if not storage:
            raise HTTPException(status_code=400, detail="日志存储未启用")

        log = storage.get_log(request_id)
        if not log:
            raise HTTPException(status_code=404, detail="日志不存在")

        return log

    # ==================== 分析统计接口 ====================

    @app.get("/api/v1/analytics/summary", tags=["分析统计"])
    async def get_analytics_summary(
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        _: str = Depends(verify_api_key),
    ):
        """获取检测统计摘要"""
        detector = get_detector()
        storage = detector._storage

        if not storage:
            raise HTTPException(status_code=400, detail="日志存储未启用")

        summary = storage.get_stats_summary(start_date, end_date)
        return summary

    @app.get("/api/v1/analytics/trends", tags=["分析统计"])
    async def get_analytics_trends(
        period: str = Query("day", pattern="^(hour|day|month)$"),
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        _: str = Depends(verify_api_key),
    ):
        """获取检测趋势数据"""
        detector = get_detector()
        storage = detector._storage

        if not storage:
            raise HTTPException(status_code=400, detail="日志存储未启用")

        trends = storage.get_trends(period, start_date, end_date)
        return {"period": period, "data": trends}

    @app.get("/api/v1/analytics/top-violations", tags=["分析统计"])
    async def get_top_violations(
        limit: int = Query(20, ge=1, le=100),
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        _: str = Depends(verify_api_key),
    ):
        """获取高频违规词"""
        detector = get_detector()
        storage = detector._storage

        if not storage:
            raise HTTPException(status_code=400, detail="日志存储未启用")

        words = storage.get_top_violation_words(limit, start_date, end_date)
        return {"words": words}

    # ==================== 模型管理接口 ====================

    @app.post("/api/v1/model/train", tags=["模型管理"])
    async def train_model(_: str = Depends(verify_api_key)):
        """
        触发模型训练

        使用 training/data/train_data.json 中的数据训练模型
        """
        import json
        from pathlib import Path

        detector = get_detector()
        model_cfg = config.get_model_config()
        data_path = model_cfg.get("training_data", "training/data/train_data.json")

        # 解析路径
        p = Path(data_path)
        if not p.is_absolute():
            p = Path(detector._project_root) / p

        if not p.exists():
            raise HTTPException(
                status_code=404,
                detail=f"训练数据不存在: {data_path}",
            )

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

            texts = [item["text"] for item in data]
            labels = [item["label"] for item in data]

            result = detector.model_classifier.train(texts, labels)
            return {"status": "success", "metrics": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"模型训练失败: {e}")

    # ==================== 统计接口 ====================

    @app.get("/api/v1/stats", response_model=StatsResponse, tags=["系统"])
    async def get_stats(_: str = Depends(verify_api_key)):
        """获取检测统计信息"""
        detector = get_detector()
        stats = detector.get_stats()
        return StatsResponse(**stats)

    return app
