# 开发交接

## 当前快照

截至 2026-09-03：

- 代码目录：`seek/`
- 本地数据库：366 个专辑发行组、366 个发行版本、366 张封面、4,136 条曲目。
- 本地模型：Qwen3-VL-Embedding-2B，约 4.0 GB。
- 向量维数：2048。
- 本地索引：366 条，FAISS 已验证重启复用。
- Mac M4 Pro 已验证 MPS 推理。
- 查询“很多杂乱的小皮鞋，像学校鞋柜，背景颜色记不清了”时，《健全な社会》为第一名，实测余弦分数约 0.6444。
- 自动化测试：33 项通过，另有两个第三方测试库弃用警告。

## 主要代码

- `app/domain.py`：领域对象。
- `app/ports.py`：模型和仓库接口。
- `app/adapters/sqlite_repository.py`：SQLite 仓库。
- `app/adapters/musicbrainz.py`：MusicBrainz/CAA 导入、限速、重试和图片兼容下载。
- `app/adapters/qwen_vl.py`：Qwen 图文向量。
- `app/adapters/faiss_index.py`：持久化和增量索引。
- `app/indexed_search.py`：真实封面检索。
- `app/search.py`：假数据上的问题和三态反馈原型。
- `app/runtime.py`：运行时组装与健康状态。
- `app/api.py`：FastAPI 和当前 Gradio 挂载。

## 本地运行

```bash
cd seek
uv sync
uv run python -m app
```

打开 `http://127.0.0.1:8000`。首次加载本地模型需要等待；已有索引会直接复用。停止服务使用 `Control-C`。

测试：

```bash
uv run ruff check .
uv run pytest
```

## Git 不包含的资产

`.gitignore` 排除了：

- `models/`
- `data/`
- `indexes/`
- `.env`

因此只执行 `git push` 不会把本机的 4 GB 模型、29 MB 当前数据和 5.3 MB 索引传到服务器。这是有意设计，避免提交权重、版权图片、生成物和密钥。

## 在服务器恢复

### 代码和依赖

```bash
git clone <repository-url>
cd <repository>/seek
uv sync
```

### 模型

```bash
uv run python -m app.download_models --model qwen
```

也可以通过专用对象存储或安全文件传输复制 `models/qwen3-vl-embedding-2b/`，不要提交 Git。

### 数据

当前 366 条数据是多轮实验查询和人工清理后的本地状态，不能仅靠一次 `--limit 366` 精确重现。服务器有两种选择：

1. 临时验证：使用安全文件传输复制 `data/` 和 `indexes/`，并注意这些目录包含第三方封面，不应公开分发。
2. 推荐方案：按 `docs/data-strategy.md` 实现基于 MusicBrainz 转储的可重复数据快照，再在服务器或离线机器重新生成索引。

仅做小规模演示可执行：

```bash
uv run python -m app.import_music --limit 100
uv run python -m app
```

这不会得到与当前 366 条完全相同的数据。

## 当前已验证行为

- 中文和英文封面描述检索。
- 最多返回 20 个候选。
- 模型不存在时明确报错，不调用在线 API。
- 首次建立索引，后续重启复用。
- 新增或变化封面只重新计算差异向量。
- SHA-256 完全重复封面不进入索引。
- CAA 重定向在部分 macOS TLS 环境下有兼容路径。
- 页面展示封面、专辑、艺人和相似度。

## 已知限制

- 当前 366 条数据不是全球均衡数据集，主要来自日语摇滚相关实验。
- 通过标签和模糊艺人名称导入曾产生错误匹配；已移除 14 条明显错误，但未做完整审计。
- 数据库仍以 `Album` 简化承载 ReleaseGroup，缺少完整 Artist、Recording、Genre 和来源许可表。
- 真实 Qwen + FAISS 路径尚未接入 MemoryClue 解析和主动追问。
- 当前只有封面向量单路检索，没有元数据向量、全文召回和结构化融合。
- 当前 Gradio 页面不是微信小程序。
- 当前 API 无公网鉴权、限流、匿名会话和反馈存储。
- 当前封面版权仅适合本地原型，公开展示方案未确定。

## 下一步任务顺序

### P0：参赛和上线前置

1. 确认微信小程序 AppID、主体、赛区和可用 HTTPS 域名。
2. 确认公开封面的合法展示和下架方案。
3. 创建 30 条真实模糊记忆基准，覆盖正确记忆、错误颜色、错误数量和混合记忆。

### P1：领域模型与检索闭环

1. 引入 Artist、ReleaseGroup、Recording、GenreTag、CoverAnalysis 和 DataSource。
2. 实现带置信度、极性和来源的 MemoryClue 解析。
3. 实现封面、元数据、全文和结构化多路召回。
4. 实现候选去重、发行组归并和动态重排。
5. 将信息增益问题选择接入真实搜索。
6. 增加专辑到曲目确认的完整 API。

### P2：可扩展数据管线

1. 停止依赖逐条模糊 MusicBrainz 搜索。
2. 使用官方数据库转储构建本地 PostgreSQL 暂存库。
3. 使用艺人 MBID 和实体关系精确导入。
4. 建立流派、地区、语言、年代和热度覆盖矩阵。
5. 生成第一批 10,000 条可重复快照，再扩至 50,000–100,000。

### P3：小程序和服务器

1. 将 Gradio 与 API 运行模式拆开。
2. 部署常驻 Qwen 文本向量服务和只读索引服务。
3. 使用对象存储/CDN 提供压缩封面。
4. 创建微信小程序原生 TypeScript 项目。
5. 实现首页、候选、详情、追问、找到和错误状态。
6. 加入匿名反馈和基础指标。

## 不要直接做的事

- 不把当前 `data/`、`models/` 或 `indexes/` 提交到公开 Git。
- 不继续用模糊艺人名盲目扩容。
- 不在完成混合检索前用数据量掩盖检索问题。
- 不把相似度当成正确概率显示给用户。
- 不先加入音频、哼唱、播放、社区或账号体系。

