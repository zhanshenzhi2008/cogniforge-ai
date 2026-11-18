# 与 Go 仓分层对齐（概念映射）

## [变更记录]

| 日期 | 版本 | 变更摘要 | 负责人 |
|------|------|----------|--------|
| 2026-11-18 | v1.0 | 初版：对照 cogniforge develop 分层，不强制立刻改目录 | orjrs |

## [变更] 初版对齐说明（2026-11-18）

- **变更原因**：Go 已按 gateway/app/runtime/capability… 分层；Python 侧先记清对应关系，避免乱搬目录
- **原则**：本期只写映射，**不强制物理改名**；行为不变（仍回调 Go，不自持 API Key）

---

## 1. 对照表

| Go（cogniforge） | Python（cogniforge-ai）现状 | 说明 |
|---|---|---|
| `gateway` | （无对等；入口在 FastAPI） | 鉴权/限流在 Go |
| `app` | `app/routers/*` | HTTP 路由/用例 |
| `runtime` | `agent/`、部分 `llm/` 调用链 | 执行循环；聊天最终仍走 Go |
| `capability/memory` | `memory/` | 记忆能力 |
| `capability/knowledge` + RAG | `services/rag/` | 解析/分块/向量 |
| `capability/provider` | **不存 Key**；`llm/go_gateway.py` 回调 Go | 模型配置唯一在 Go 控制台 |
| `governance` | （无） | 配额/评测在 Go |
| `orchestration` | （无） | 工作流在 Go |

## 2. 硬约束（不变）

1. 明文 API Key 不进 Python / Redis  
2. LLM / embeddings 经 Go：`COGNIFORGE_API_URL`  
3. Crawl4AI 另服务，约 2026-11-23 再做  

## 3. 后续（可选）

若要物理对齐，建议顺序：`routers` 明确为 app → `agent`/`llm` 执行核标为 runtime 概念 → `memory`/`services/rag` 标为 capability 概念；仍保持单仓，不拆微服务。
