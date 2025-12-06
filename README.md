# 首都师范大学在校生生存助手

基于 LangChain 的 RAG 问答系统，专为首都师范大学在校本科生设计。

## 功能特点

- **查规定** - 查询学校的规章制度（学分、绩点、处分、奖学金等）
- **问流程** - 询问办事流程（缓考、休学、转专业、补办学生证等）
- **求经验** - 获取学习生活建议（选课、考试、社团、时间管理等）
- **要模板** - 获取文本模板（请假邮件、申请书等）
- **用户数据持久化** - 自动保存用户信息和对话历史，下次打开继续
- **长期记忆** - 自动从对话中提取重要信息，提供个性化建议

## 技术架构

- **RAG 检索增强生成** - 基于知识库检索，减少 AI 幻觉
- **本地嵌入模型** - 使用 HuggingFace 模型，免费无限制
- **FAISS 向量检索** - 高效的语义相似度搜索
- **意图识别** - 自动判断问题类型，差异化回答策略
- **对话状态管理** - 支持多轮对话，记录用户信息
- **长期记忆提取** - 自动识别并存储用户重要信息（如挂科、转专业意向等）

## 项目结构

```
cnu_survival_assistant/
├── data/                    # 知识库数据
│   ├── rules/              # 制度规定
│   ├── flows/              # 办事流程
│   ├── experience/         # 经验建议
│   ├── templates/          # 文本模板
│   └── docs/               # 培养方案
├── src/                     # 源代码
│   ├── config.py           # 配置管理
│   ├── loader.py           # 文档加载
│   ├── vector_store.py     # 向量存储
│   ├── intent_router.py    # 意图识别
│   ├── rag_chain.py        # RAG 问答链
│   ├── state.py            # 对话状态与记忆管理
│   └── cli_app.py          # 命令行界面
├── index/                   # 向量索引（需构建）
├── user_data.json          # 用户数据（自动生成）
├── build_index.py          # 索引构建脚本
├── requirements.txt        # 依赖列表
└── .env.example            # 配置模板
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/hajimiHenry/cnu_survival_assistant.git
cd cnu_survival_assistant
```

### 2. 创建虚拟环境

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置 API

复制配置模板并填入你的 API 信息：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=https://你的API地址/v1
MODEL_NAME=模型名称
```

### 5. 构建向量索引

```bash
python build_index.py
```

首次运行会自动下载嵌入模型（约 100MB）。

### 6. 启动系统

```bash
python -m src.cli_app
```

## 使用示例

```
您: 挂科了会怎么样？

[识别意图: 查规定]
正在思考...

助手:
  根据首都师范大学的规定，必修课程总评成绩不及格的，
  学生必须在下一年级选课并跟班重修...
```

### 长期记忆功能

系统会自动从对话中提取重要信息并记住：

```
您: 我高数挂了，怎么补考？

助手:
  根据学校规定...

  ----------------------------------------
  [系统已自动记录以下信息]
    * 高等数学挂科
  ----------------------------------------
```

下次对话时，系统会参考这些记忆提供个性化建议。

### 可用命令

| 命令 | 说明 |
|------|------|
| `help` / `帮助` | 显示帮助信息 |
| `memory` / `记忆` | 查看和管理长期记忆 |
| `info` / `信息` | 显示当前用户信息 |
| `clear` / `清空` | 清空对话历史 |
| `reset` / `重置` | 重新设置个人信息 |
| `save` / `保存` | 手动保存数据 |
| `exit` / `quit` | 退出系统（自动保存） |

## 配置说明

| 配置项 | 说明 |
|--------|------|
| `OPENAI_API_KEY` | API 密钥 |
| `OPENAI_BASE_URL` | API 地址（兼容 OpenAI 格式） |
| `MODEL_NAME` | 对话模型名称 |

嵌入模型使用本地的 `paraphrase-multilingual-MiniLM-L12-v2`，无需额外配置。

## 添加知识库内容

在 `data/` 对应目录下添加 `.txt` 文件，格式如下：

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
- OpenAI 兼容 API

## License

MIT
