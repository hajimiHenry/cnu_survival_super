# 首都师范大学在校生生存助手

基于 LangChain 的 RAG 问答系统，专为首都师范大学在校本科生设计。

## 功能特点

### 基础功能
- **查规定** - 查询学校的规章制度（学分、绩点、处分、奖学金等）
- **问流程** - 询问办事流程（缓考、休学、转专业、补办学生证等）
- **求经验** - 获取学习生活建议（选课、考试、社团、时间管理等）
- **要模板** - 获取文本模板（请假邮件、申请书等）
- **查培养方案** - 查询专业培养方案、课程设置、毕业要求

### 增强功能
- **用户数据持久化** - 自动保存用户信息和对话历史
- **长期记忆** - 自动从对话中提取重要信息，提供个性化建议
- **追问上下文** - 智能识别追问，自动关联上轮对话内容
- **迭代查询** - 复杂问题自动分解为子问题，多轮检索聚合证据
- **智能重排** - LLM 对检索结果打分重排，确保高质量证据优先
- **联网搜索** - 本地知识不足时自动联网搜索补充（需配置 Tavily API）
- **多智能体协作** - 本地智能体 + 联网智能体 + 裁决智能体协同工作
- **来源标注** - 每个回答自动标注【来源说明】，明确本地/网络信息使用情况
- **结构化日志** - 完整记录检索过程，支持追踪和调试

## 界面模式

### Web 图形界面（推荐）
- 拟物化设计风格
- 侧边栏管理用户信息和长期记忆
- 实时显示证据来源（本地/网络）
- API 配置设置面板
- 响应式设计，支持移动端

### 命令行界面
适合开发者和高级用户的终端交互方式。

## 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/hajimiHenry/cnu_survival_super.git
cd cnu_survival_super

# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API

复制配置模板：
```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的配置：

```ini
# 必填：LLM API 配置
OPENAI_API_KEY=sk-your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini

# 可选：联网搜索（增强功能）
TAVILY_API_KEY=tvly-your-api-key
```

**API 配置说明：**
| 配置项 | 说明 | 必填 |
|--------|------|------|
| `OPENAI_API_KEY` | LLM API 密钥 | 是 |
| `OPENAI_BASE_URL` | API 地址（支持 OpenAI 官方、中转站、Deepseek 等） | 是 |
| `MODEL_NAME` | 模型名称 | 是 |
| `TAVILY_API_KEY` | Tavily 联网搜索 API（[免费获取](https://tavily.com/)） | 否 |

### 3. 构建向量索引

```bash
python build_index.py
```

首次运行会自动下载嵌入模型（约 500MB）。

### 4. 启动系统

**方式一：Web 界面（推荐）**
```bash
python run_web.py
```
然后访问 http://localhost:8080

**方式二：命令行界面**
```bash
python -m src.cli_app
```

**方式三：Windows 一键启动**
双击 `启动Web界面.bat`

## 项目结构

```
cnu_survival_super/
├── data/                       # 知识库数据
│   ├── rules/                  # 制度规定
│   ├── flows/                  # 办事流程
│   ├── experience/             # 经验建议
│   ├── templates/              # 文本模板
│   └── docs/                   # 培养方案
├── src/                        # 源代码
│   ├── config.py               # 配置管理
│   ├── loader.py               # 文档加载
│   ├── vector_store.py         # 向量存储（支持混合检索）
│   ├── intent_router.py        # 意图识别
│   ├── rag_chain.py            # 基础 RAG 问答链
│   ├── enhanced_rag_chain.py   # 增强版 RAG（v2.0）
│   ├── streaming_rag.py        # 流式 RAG 处理器（SSE 支持）
│   ├── retrieval_judge.py      # 检索质量判定器
│   ├── query_transformer.py    # 查询改写/扩展（防跑偏校验）
│   ├── reranker.py             # LLM 重排器
│   ├── task_decomposer.py      # 任务分解器
│   ├── evidence_aggregator.py  # 证据聚合器
│   ├── web_search.py           # 联网搜索（Tavily）
│   ├── logger.py               # 结构化日志
│   ├── stream_events.py        # SSE 事件定义
│   ├── agents/                 # 多智能体模块
│   │   ├── local_agent.py      # 本地证据智能体
│   │   ├── web_agent.py        # 联网研究智能体
│   │   └── arbiter_agent.py    # 裁决智能体
│   ├── state.py                # 对话状态与记忆管理
│   ├── cli_app.py              # 命令行界面
│   └── web_server.py           # Web API 服务
├── static/                     # 前端静态文件
│   └── index.html              # Web 界面
├── index/                      # 向量索引（自动生成）
├── logs/                       # 追踪日志（自动生成）
├── build_index.py              # 索引构建脚本
├── run_web.py                  # Web 服务启动脚本
├── requirements.txt            # 依赖列表
├── .env.example                # 配置模板
└── README.md
```

## 增强功能详解

### 迭代查询

对于复杂问题，系统自动分解为子问题：

```
"奖学金怎么评定？"
  → 子问题1: 奖学金有哪些类型？
  → 子问题2: 申请条件是什么？
  → 子问题3: 评审流程是怎样的？
  → 聚合所有证据 → 生成综合回答
```

### 联网搜索

当本地知识库无法覆盖问题时，自动触发联网搜索：

| 阈值设置 | 说明 |
|---------|------|
| 覆盖度 < 70% | 触发联网搜索补充信息 |

可在 `src/config.py` 中调整 `WEB_SEARCH_TRIGGER_THRESHOLD`。

### 多智能体协作

```
用户问题
    │
    ├──────────────┬──────────────┐
    ▼              ▼              │
[本地智能体]   [联网智能体]       │
    │              │              │
    └──────┬───────┘              │
           ▼                      │
      [裁决智能体]                │
           │                      │
           ▼                      │
       最终答案 ◄─────────────────┘
```

裁决规则：
- 不新增事实，只基于证据
- 冲突时默认以本地证据为准
- 网络信息作为补充

### 来源标注

每个回答都会自动追加来源说明：

```
【来源说明】本地: 已使用 网络: 未使用
【来源说明】本地: 未找到 网络: 已使用
【来源说明】本地: 已使用 网络: 已使用
```

状态说明：
- `已使用`: 成功从该来源获取了信息
- `未找到`: 在该来源中未找到相关信息
- `未使用`: 未启用或未触发该来源

### 追问上下文

系统智能识别追问场景，自动关联上轮对话：

```
用户: 奖学金怎么申请？
助手: [回答奖学金申请流程...]

用户: 那材料要准备哪些？  ← 识别为追问
助手: [基于"奖学金申请"上下文回答材料清单]
```

识别规则：
- 问题长度 ≤ 6 字
- 包含追问标记词（"那"、"这样"、"怎么办"、"还有"等）

## 配置参数

在 `src/config.py` 中可调整以下参数：

```python
# 检索判定阈值
SIMILARITY_THRESHOLD = 0.5      # 向量相似度阈值
RELEVANT_DOCS_THRESHOLD = 2     # 相关片段数阈值
COVERAGE_THRESHOLD = 0.6        # 证据覆盖度阈值

# 迭代上限
MAX_ITERATION_ROUNDS = 5        # 最多迭代轮数

# 联网搜索
WEB_SEARCH_ENABLED = True                # 是否启用
WEB_SEARCH_TRIGGER_THRESHOLD = 0.7       # 触发阈值（越高越积极）
```

## Web 界面功能

### 聊天界面
- 输入问题，按 Enter 发送
- Shift + Enter 换行
- 显示证据来源（本地/网络分开展示）
- 显示使用的增强功能标签

### 侧边栏
- **用户信息**：填写年级、学院、专业
- **长期记忆**：查看/添加/删除记忆
- **操作**：清空对话历史、清空记忆

### 设置面板
- 配置 API 地址、密钥、模型
- 测试连接功能

## 命令行可用命令

| 命令 | 说明 |
|------|------|
| `help` / `帮助` | 显示帮助信息 |
| `memory` / `记忆` | 查看和管理长期记忆 |
| `info` / `信息` | 显示当前用户信息 |
| `clear` / `清空` | 清空对话历史 |
| `reset` / `重置` | 重新设置个人信息 |
| `save` / `保存` | 手动保存数据 |
| `exit` / `quit` | 退出系统（自动保存） |

## 添加知识库内容

在 `data/` 对应目录下添加 `.txt` 文件，格式：

```
# 条目: 标题
# 类别: 制度/流程/经验/模板
# 主题: 具体主题
# 来源: 资料来源

正文内容...

---

# 条目: 下一条标题
...
```

添加后重新运行 `python build_index.py` 更新索引。

## 依赖

- Python 3.8+
- LangChain
- FAISS
- Sentence Transformers
- FastAPI + Uvicorn
- Tavily Python（联网搜索）
- OpenAI 兼容 API
- jieba（中文分词）
- rank-bm25（BM25 检索）

## License

MIT
