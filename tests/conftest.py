"""测试配置与共享 fixtures"""

import pytest

from app.config import Config


@pytest.fixture(autouse=True)
def reset_config():
    """每个测试前重置配置单例"""
    Config.reset()
    yield
    # 不在 teardown 中 reset，因为部分 fixture 已自行管理
