#!/usr/bin/env python3
"""内容安全检测服务 — 启动入口"""

import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn

from app.api import create_app
from app.config import config

# 加载配置
config.load()
server_cfg = config.get_server_config()


def main():
    """启动服务"""
    app = create_app()

    print("=" * 60)
    print("  内容安全检测服务 v0.1.0")
    print("=" * 60)
    print(f"  地址: http://{server_cfg.get('host', '0.0.0.0')}:{server_cfg.get('port', 8000)}")
    print(f"  文档: http://localhost:{server_cfg.get('port', 8000)}/docs")
    print("=" * 60)

    uvicorn.run(
        app,
        host=server_cfg.get("host", "0.0.0.0"),
        port=server_cfg.get("port", 8000),
        workers=server_cfg.get("workers", 1),
        reload=server_cfg.get("reload", False),
        log_level="info",
    )


if __name__ == "__main__":
    main()
