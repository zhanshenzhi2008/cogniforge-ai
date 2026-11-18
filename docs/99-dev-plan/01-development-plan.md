# Cogniforge AI 开发计划

## [变更记录]

| 日期 | 版本 | 变更摘要 | 负责人 |
|------|------|----------|--------|
| 2026-11-18 | v1.1 | 与 Go 分层对齐说明（仅文档） | orjrs |
| 2026-11-17 | v1.0 | 记下 Crawl4AI 约一周后再做 | orjrs |

## [变更] 与 Go 分层对齐说明（2026-11-18）

- **变更原因**：Go develop 已完成 handler→app / 能力留 capability；Python 侧先文档对齐
- **变更后**：新增 `docs/02-architecture/01-layering-alignment.md`
- **影响范围**：仅文档；代码目录暂不改

## [变更] Crawl4AI 延后（2026-11-17）

- 约 2026-11-23 再做独立 crawl 服务；本期网页入库走 Go 侧 Jina/Firecrawl
