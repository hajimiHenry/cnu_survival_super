"""
结构化日志模块 - 记录问答过程的详细追踪信息
"""
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

import numpy as np

from .config import PROJECT_ROOT


# 日志存储目录
LOGS_DIR = PROJECT_ROOT / "logs" / "query_traces"


@dataclass
class RetrievalLog:
    """单次检索的日志"""
    query: str                          # 检索查询
    method: str                         # 检索方法 (vector/bm25/hybrid)
    top_k: int                          # 请求数量
    docs: List[Dict[str, Any]]          # 命中文档 [{content, source, score}]
    duration_ms: float                  # 耗时(毫秒)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FailureJudgmentLog:
    """检索失败判定的日志"""
    similarity_score: float             # Top1相似度
    similarity_threshold: float         # 相似度阈值
    similarity_failed: bool             # 相似度是否失败
    relevant_count: int                 # 相关片段数量
    relevant_threshold: int             # 相关片段阈值
    relevant_failed: bool               # 相关数量是否失败
    coverage_score: float               # 覆盖度分数
    coverage_threshold: float           # 覆盖度阈值
    coverage_failed: bool               # 覆盖度是否失败
    failed_signals: int                 # 失败信号数量
    is_failed: bool                     # 最终判定是否失败
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RerankLog:
    """重排的日志"""
    before_ranking: List[Dict[str, Any]]   # 重排前 [{content_preview, source, score}]
    after_ranking: List[Dict[str, Any]]    # 重排后
    duration_ms: float                     # 耗时
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FallbackLog:
    """回退策略的日志"""
    round_num: int                      # 第几轮回退
    strategy: str                       # 使用的策略 (rewrite/expand/hybrid)
    original_query: str                 # 原始查询
    transformed_query: str              # 转换后的查询
    success: bool                       # 回退后是否成功
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DecompositionLog:
    """任务分解的日志"""
    original_question: str              # 原始问题
    sub_questions: List[str]            # 分解后的子问题
    should_decompose: bool              # 是否需要分解
    duration_ms: float                  # 耗时
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class IterationLog:
    """单轮迭代的日志"""
    round_num: int                      # 第几轮
    sub_question: str                   # 本轮子问题
    retrieval: Optional[RetrievalLog]   # 检索日志
    evidence_count: int                 # 本轮新增证据数
    total_evidence_count: int           # 累计证据数
    converged: bool                     # 是否收敛
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class WebSearchLog:
    """联网搜索的日志"""
    query: str                          # 搜索查询
    results: List[Dict[str, Any]]       # 搜索结果 [{title, url, snippet}]
    duration_ms: float                  # 耗时
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentLog:
    """智能体执行的日志"""
    agent_type: str                     # 智能体类型 (local/web/arbiter)
    input_summary: str                  # 输入摘要
    output_summary: str                 # 输出摘要
    evidence_count: int                 # 产出证据数
    duration_ms: float                  # 耗时
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvidenceItem:
    """证据条目"""
    content: str                        # 内容
    source: str                         # 来源 (文件名或URL)
    source_type: str                    # 来源类型 (local/web)
    score: float                        # 相关性分数
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryTrace:
    """完整的问答追踪记录"""
    trace_id: str                       # 追踪ID
    question: str                       # 用户问题原文
    intent: str                         # 识别的意图

    # 迭代查询相关
    used_iteration: bool = False        # 是否使用迭代查询
    decomposition: Optional[DecompositionLog] = None
    iterations: List[IterationLog] = field(default_factory=list)

    # 失败回退相关
    used_fallback: bool = False         # 是否触发回退
    fallbacks: List[FallbackLog] = field(default_factory=list)
    failure_judgments: List[FailureJudgmentLog] = field(default_factory=list)

    # 检索相关
    retrievals: List[RetrievalLog] = field(default_factory=list)
    reranks: List[RerankLog] = field(default_factory=list)

    # 多智能体相关
    used_web_search: bool = False       # 是否使用联网搜索
    web_searches: List[WebSearchLog] = field(default_factory=list)
    agent_logs: List[AgentLog] = field(default_factory=list)

    # 最终结果
    answer: str = ""                    # 最终答案
    evidence_local: List[EvidenceItem] = field(default_factory=list)
    evidence_web: List[EvidenceItem] = field(default_factory=list)

    # 时间统计
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = ""
    total_duration_ms: float = 0.0

    # LLM调用统计
    llm_calls: int = 0
    llm_total_duration_ms: float = 0.0


class StructuredLogger:
    """结构化日志记录器"""

    def __init__(self, question: str = "", intent: str = ""):
        self.trace = QueryTrace(
            trace_id=self._generate_trace_id(),
            question=question,
            intent=intent
        )
        self._start_time = time.time()

    @staticmethod
    def _generate_trace_id() -> str:
        """生成追踪ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        return f"{timestamp}_{short_uuid}"

    @property
    def trace_id(self) -> str:
        return self.trace.trace_id

    def set_question(self, question: str):
        """设置问题"""
        self.trace.question = question

    def set_intent(self, intent: str):
        """设置意图"""
        self.trace.intent = intent

    def log_retrieval(self, query: str, method: str, top_k: int,
                      docs: List[Dict[str, Any]], duration_ms: float):
        """记录检索"""
        log = RetrievalLog(
            query=query,
            method=method,
            top_k=top_k,
            docs=docs,
            duration_ms=duration_ms
        )
        self.trace.retrievals.append(log)

    def log_failure_judgment(self,
                             similarity_score: float, similarity_threshold: float,
                             relevant_count: int, relevant_threshold: int,
                             coverage_score: float, coverage_threshold: float,
                             is_failed: bool):
        """记录失败判定"""
        similarity_failed = similarity_score < similarity_threshold
        relevant_failed = relevant_count < relevant_threshold
        coverage_failed = coverage_score < coverage_threshold
        failed_signals = sum([similarity_failed, relevant_failed, coverage_failed])

        log = FailureJudgmentLog(
            similarity_score=similarity_score,
            similarity_threshold=similarity_threshold,
            similarity_failed=similarity_failed,
            relevant_count=relevant_count,
            relevant_threshold=relevant_threshold,
            relevant_failed=relevant_failed,
            coverage_score=coverage_score,
            coverage_threshold=coverage_threshold,
            coverage_failed=coverage_failed,
            failed_signals=failed_signals,
            is_failed=is_failed
        )
        self.trace.failure_judgments.append(log)

    def log_fallback(self, round_num: int, strategy: str,
                     original_query: str, transformed_query: str, success: bool):
        """记录回退策略"""
        log = FallbackLog(
            round_num=round_num,
            strategy=strategy,
            original_query=original_query,
            transformed_query=transformed_query,
            success=success
        )
        self.trace.fallbacks.append(log)
        self.trace.used_fallback = True

    def log_rerank(self, before: List[Dict[str, Any]],
                   after: List[Dict[str, Any]], duration_ms: float):
        """记录重排"""
        log = RerankLog(
            before_ranking=before,
            after_ranking=after,
            duration_ms=duration_ms
        )
        self.trace.reranks.append(log)

    def log_decomposition(self, original: str, sub_questions: List[str],
                          should_decompose: bool, duration_ms: float):
        """记录任务分解"""
        self.trace.decomposition = DecompositionLog(
            original_question=original,
            sub_questions=sub_questions,
            should_decompose=should_decompose,
            duration_ms=duration_ms
        )
        if should_decompose:
            self.trace.used_iteration = True

    def log_iteration(self, round_num: int, sub_question: str,
                      retrieval: Optional[RetrievalLog],
                      evidence_count: int, total_evidence_count: int,
                      converged: bool):
        """记录迭代轮次"""
        log = IterationLog(
            round_num=round_num,
            sub_question=sub_question,
            retrieval=retrieval,
            evidence_count=evidence_count,
            total_evidence_count=total_evidence_count,
            converged=converged
        )
        self.trace.iterations.append(log)

    def log_web_search(self, query: str, results: List[Dict[str, Any]],
                       duration_ms: float):
        """记录联网搜索"""
        log = WebSearchLog(
            query=query,
            results=results,
            duration_ms=duration_ms
        )
        self.trace.web_searches.append(log)
        self.trace.used_web_search = True

    def log_agent(self, agent_type: str, input_summary: str,
                  output_summary: str, evidence_count: int, duration_ms: float):
        """记录智能体执行"""
        log = AgentLog(
            agent_type=agent_type,
            input_summary=input_summary,
            output_summary=output_summary,
            evidence_count=evidence_count,
            duration_ms=duration_ms
        )
        self.trace.agent_logs.append(log)

    def log_llm_call(self, duration_ms: float):
        """记录LLM调用"""
        self.trace.llm_calls += 1
        self.trace.llm_total_duration_ms += duration_ms

    def set_answer(self, answer: str):
        """设置最终答案"""
        self.trace.answer = answer

    def add_local_evidence(self, content: str, source: str,
                           score: float, metadata: Dict[str, Any] = None):
        """添加本地证据"""
        evidence = EvidenceItem(
            content=content,
            source=source,
            source_type="local",
            score=score,
            metadata=metadata or {}
        )
        self.trace.evidence_local.append(evidence)

    def add_web_evidence(self, content: str, source: str,
                         score: float, metadata: Dict[str, Any] = None):
        """添加网络证据"""
        evidence = EvidenceItem(
            content=content,
            source=source,
            source_type="web",
            score=score,
            metadata=metadata or {}
        )
        self.trace.evidence_web.append(evidence)

    def finalize(self):
        """完成追踪，计算总耗时"""
        self.trace.end_time = datetime.now().isoformat()
        self.trace.total_duration_ms = (time.time() - self._start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._dataclass_to_dict(self.trace)

    def _dataclass_to_dict(self, obj) -> Any:
        """递归转换dataclass为字典，处理numpy类型"""
        if hasattr(obj, '__dataclass_fields__'):
            return {k: self._dataclass_to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [self._dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._dataclass_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def export_json(self, pretty: bool = True) -> str:
        """导出为JSON字符串"""
        self.finalize()
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self) -> Path:
        """保存到文件"""
        self.finalize()

        # 确保目录存在
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        filename = f"{self.trace.trace_id}.json"
        filepath = LOGS_DIR / filename

        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

        return filepath

    @classmethod
    def load(cls, trace_id: str) -> Optional['StructuredLogger']:
        """从文件加载"""
        filepath = LOGS_DIR / f"{trace_id}.json"
        if not filepath.exists():
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger = cls()
        # 重建trace对象（简化处理，只保留原始数据）
        logger.trace = QueryTrace(
            trace_id=data.get('trace_id', ''),
            question=data.get('question', ''),
            intent=data.get('intent', '')
        )
        # 复制其他字段
        for key, value in data.items():
            if hasattr(logger.trace, key):
                setattr(logger.trace, key, value)

        return logger

    @classmethod
    def list_traces(cls, limit: int = 50) -> List[Dict[str, Any]]:
        """列出最近的追踪记录"""
        if not LOGS_DIR.exists():
            return []

        traces = []
        files = sorted(LOGS_DIR.glob("*.json"), reverse=True)[:limit]

        for filepath in files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    traces.append({
                        'trace_id': data.get('trace_id', ''),
                        'question': data.get('question', '')[:100],
                        'intent': data.get('intent', ''),
                        'start_time': data.get('start_time', ''),
                        'total_duration_ms': data.get('total_duration_ms', 0),
                        'used_fallback': data.get('used_fallback', False),
                        'used_iteration': data.get('used_iteration', False),
                        'used_web_search': data.get('used_web_search', False)
                    })
            except Exception:
                continue

        return traces


def get_trace_detail(trace_id: str) -> Optional[Dict[str, Any]]:
    """获取追踪详情"""
    filepath = LOGS_DIR / f"{trace_id}.json"
    if not filepath.exists():
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
