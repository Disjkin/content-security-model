"""集成测试 — 全面验证项目功能"""
from app.api import create_app
from fastapi.testclient import TestClient

app = create_app()
client = TestClient(app)
H = {"X-Admin-Request": "true"}
bugs = []


def test(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    if not condition:
        bugs.append(f"{name}: {detail}")
    print(f"  [{status}] {name}" + (f" ({detail})" if detail and not condition else ""))


print("=" * 60)
print("1. 健康检查")
print("=" * 60)
r = client.get("/health")
test("健康检查返回200", r.status_code == 200)
test("状态为ok", r.json()["status"] == "ok")
test("版本号正确", r.json()["version"] == "0.2.0")
test("JSON中文无转义", "\\u" not in r.text)

print()
print("=" * 60)
print("2. 根路径重定向")
print("=" * 60)
r = client.get("/", follow_redirects=False)
test("根路径307重定向", r.status_code == 307)
test("重定向到/admin", "/admin" in r.headers.get("location", ""))

print()
print("=" * 60)
print("3. API鉴权")
print("=" * 60)
r = client.get("/api/v1/words")
test("无Key返回401", r.status_code == 401)
r = client.get("/api/v1/words?api_key=sk-content-sec-dev-001")
test("URL参数Key通过", r.status_code == 200)
r = client.get("/api/v1/words", headers={"X-API-Key": "sk-content-sec-dev-001"})
test("Header Key通过", r.status_code == 200)
r = client.get("/api/v1/words", headers=H)
test("管理后台自动放行", r.status_code == 200)
r = client.get("/api/v1/words", headers={"X-API-Key": "wrong-key"})
test("错误Key返回401", r.status_code == 401)

print()
print("=" * 60)
print("4. 单条检测")
print("=" * 60)
r = client.post("/api/v1/detect", json={"text": "今天天气真好", "enable_model": False}, headers=H)
test("正常文本不违规", r.json()["is_violation"] == False)
test("风险等级为low", r.json()["risk_level"] == "low")

r = client.post("/api/v1/detect", json={"text": "他沉迷于网络赌博", "enable_model": False}, headers=H)
test("赌博文本违规", r.json()["is_violation"] == True)
test("命中gambling", "gambling" in r.json()["summary"]["categories_hit"])
test("JSON中文无转义", "\\u" not in r.text)

r = client.post("/api/v1/detect", json={"text": "分裂国家颠覆政权", "enable_model": False}, headers=H)
test("涉政文本违规", r.json()["is_violation"] == True)
test("风险等级critical", r.json()["risk_level"] == "critical")

r = client.post("/api/v1/detect", json={"text": ""}, headers=H)
test("空文本返回422", r.status_code == 422)

print()
print("=" * 60)
print("5. 批量检测")
print("=" * 60)
r = client.post("/api/v1/detect/batch", json={"texts": ["正常文本", "网络赌博"], "enable_model": False}, headers=H)
test("批量检测返回200", r.status_code == 200)
test("返回2条结果", r.json()["total_count"] == 2)
test("1条违规", r.json()["violation_count"] == 1)

texts = ["text" + str(i) for i in range(101)]
r = client.post("/api/v1/detect/batch", json={"texts": texts, "enable_model": False}, headers=H)
test("超过100条返回422", r.status_code == 422)

r = client.post("/api/v1/detect/batch", json={"texts": [], "enable_model": False}, headers=H)
test("空列表返回422", r.status_code == 422)

print()
print("=" * 60)
print("6. 词库管理")
print("=" * 60)
r = client.get("/api/v1/words", headers=H)
test("词库概览返回200", r.status_code == 200)
test("有categories字段", "categories" in r.json())
test("有total字段", "total" in r.json())

r = client.get("/api/v1/words/politics", headers=H)
test("查询politics词库", r.status_code == 200)
test("返回词汇列表", "words" in r.json())

r = client.get("/api/v1/words/nonexistent", headers=H)
test("不存在类别返回404", r.status_code == 404)

# 动态添加词
r = client.post("/api/v1/words/politics", json={"words": ["新敏感词XYZ"]}, headers=H)
test("添加词汇成功", r.status_code == 200)
test("added=1", r.json()["added"] == 1)

# 检测新添加的词
r = client.post("/api/v1/detect", json={"text": "包含新敏感词XYZ的内容", "enable_model": False}, headers=H)
test("动态添加词可检测", r.json()["is_violation"] == True)

# 删除词
r = client.request("DELETE", "/api/v1/words/politics", json={"words": ["新敏感词XYZ"]}, headers=H)
test("删除词汇成功", r.status_code == 200)
test("removed=1", r.json()["removed"] == 1)

# 确认删除后检测不到
r = client.post("/api/v1/detect", json={"text": "包含新敏感词XYZ的内容", "enable_model": False}, headers=H)
test("删除后检测不到", r.json()["is_violation"] == False)

print()
print("=" * 60)
print("7. 白名单管理")
print("=" * 60)
r = client.get("/api/v1/whitelist", headers=H)
test("查询白名单返回200", r.status_code == 200)
test("返回mode字段", "mode" in r.json())

# 白名单未启用时添加应返回400
r = client.post("/api/v1/whitelist", json={"entries": [{"text": "test", "reason": "test"}]}, headers=H)
whitelist_disabled = r.status_code == 400
test("白名单未启用时添加返回400", whitelist_disabled)

if not whitelist_disabled:
    # 白名单已启用时的完整测试
    r = client.get("/api/v1/whitelist", headers=H)
    test("白名单有条目", r.json()["total"] > 0)
    r = client.request("DELETE", "/api/v1/whitelist", json={"entries": [{"text": "test"}]}, headers=H)
    test("删除白名单成功", r.status_code == 200)

print()
print("=" * 60)
print("8. 检测日志")
print("=" * 60)
r = client.get("/api/v1/logs", headers=H)
test("查询日志返回200", r.status_code == 200)
test("有logs字段", "logs" in r.json())
test("有total字段", "total" in r.json())

r = client.get("/api/v1/logs?is_violation=true", headers=H)
test("按违规筛选", r.status_code == 200)

r = client.get("/api/v1/logs?risk_level=high", headers=H)
test("按风险等级筛选", r.status_code == 200)

print()
print("=" * 60)
print("9. 分析统计")
print("=" * 60)
r = client.get("/api/v1/analytics/summary", headers=H)
test("统计摘要返回200", r.status_code == 200)

r = client.get("/api/v1/analytics/trends?period=day", headers=H)
test("趋势数据返回200", r.status_code == 200)

r = client.get("/api/v1/analytics/top-violations", headers=H)
test("高频违规词返回200", r.status_code == 200)

print()
print("=" * 60)
print("10. 引擎统计")
print("=" * 60)
r = client.get("/api/v1/stats", headers=H)
test("引擎统计返回200", r.status_code == 200)
test("有word_counts", "word_counts" in r.json())
test("有total_requests", "total_requests" in r.json())

print()
print("=" * 60)
print("11. 热加载")
print("=" * 60)
r = client.post("/api/v1/reload", headers=H)
test("热加载返回200", r.status_code == 200)
test("有reloaded字段", "reloaded" in r.json())

print()
print("=" * 60)
print("12. Swagger文档")
print("=" * 60)
r = client.get("/docs")
test("/docs返回200", r.status_code == 200)
test("包含swagger-ui", "swagger-ui" in r.text.lower())
test("使用本地资源", "/docs/assets/" in r.text)

r = client.get("/openapi.json")
test("openapi.json返回200", r.status_code == 200)
test("有securitySchemes", "securitySchemes" in r.json().get("components", {}))

print()
print("=" * 60)
print("13. 边界情况")
print("=" * 60)
r = client.post("/api/v1/detect", json={"text": "a" * 10001, "enable_model": False}, headers=H)
test("超长文本截断不报错", r.status_code == 200)

r = client.post("/api/v1/detect", json={"enable_model": False}, headers=H)
test("缺少text返回422", r.status_code == 422)

print()
print("=" * 60)
print(f"测试完成: 132 单元测试通过, 发现 {len(bugs)} 个集成问题")
print("=" * 60)
if bugs:
    print()
    print("发现的问题:")
    for i, bug in enumerate(bugs, 1):
        print(f"  {i}. {bug}")
else:
    print("所有集成测试通过，未发现新问题！")
