"""
流式事件定义 - 用于 SSE 实时推送智能体状态
"""
import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


class EventType(Enum):
    """事件类型"""
    # 流程控制
    START = "start"                     # 开始处理
    COMPLETE = "complete"               # 处理完成
    ERROR = "error"                     # 发生错误

    # 意图识别
    INTENT_START = "intent_start"       # 开始识别意图
    INTENT_DONE = "intent_done"         # 意图识别完成

    # 任务分解
    DECOMPOSE_START = "decompose_start"     # 开始任务分解
    DECOMPOSE_DONE = "decompose_done"       # 任务分解完成

    # 检索阶段
    RETRIEVAL_START = "retrieval_start"     # 开始检索
    RETRIEVAL_DONE = "retrieval_done"       # 检索完成
    FALLBACK_START = "fallback_start"       # 开始回退
    FALLBACK_DONE = "fallback_done"         # 回退完成

    # LocalAgent
    LOCAL_AGENT_START = "local_agent_start"     # LocalAgent 开始
    LOCAL_AGENT_SEARCH = "local_agent_search"   # LocalAgent 检索中
    LOCAL_AGENT_RERANK = "local_agent_rerank"   # LocalAgent 重排中
    LOCAL_AGENT_THINK = "local_agent_think"     # LocalAgent 思考中
    LOCAL_AGENT_DONE = "local_agent_done"       # LocalAgent 完成

    # WebAgent
    WEB_AGENT_START = "web_agent_start"         # WebAgent 开始
    WEB_AGENT_SEARCH = "web_agent_search"       # WebAgent 搜索中
    WEB_AGENT_THINK = "web_agent_think"         # WebAgent 思考中
    WEB_AGENT_DONE = "web_agent_done"           # WebAgent 完成
    WEB_AGENT_SKIP = "web_agent_skip"           # WebAgent 跳过

    # Arbiter
    ARBITER_START = "arbiter_start"             # Arbiter 开始
    ARBITER_CONFLICT = "arbiter_conflict"       # Arbiter 检测冲突
    ARBITER_THINK = "arbiter_think"             # Arbiter 裁决中
    ARBITER_DONE = "arbiter_done"               # Arbiter 完成

    # 答案生成
    GENERATE_START = "generate_start"           # 开始生成答案
    GENERATE_DONE = "generate_done"             # 答案生成完成


@dataclass
class StreamEvent:
    """流式事件"""
    event_type: EventType
    agent: str = ""                     # 智能体名称
    message: str = ""                   # 用户可见消息
    detail: str = ""                    # 详细信息/思考过程
    data: Dict[str, Any] = field(default_factory=dict)  # 附加数据
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """转换为 SSE 格式"""
        payload = {
            "type": self.event_type.value,
            "agent": self.agent,
            "message": self.message,
            "detail": self.detail,
            "data": self.data,
            "timestamp": self.timestamp
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class EventEmitter:
    """事件发射器 - 用于在处理过程中收集事件"""

    def __init__(self):
        self.events: List[StreamEvent] = []
        self.start_time = time.time()

    def emit(
        self,
        event_type: EventType,
        agent: str = "",
        message: str = "",
        detail: str = "",
        **data
    ) -> StreamEvent:
        """发射事件"""
        event = StreamEvent(
            event_type=event_type,
            agent=agent,
            message=message,
            detail=detail,
            data=data
        )
        self.events.append(event)
        return event

    def elapsed_ms(self) -> float:
        """获取已用时间(毫秒)"""
        return (time.time() - self.start_time) * 1000


# ========== 便捷函数 ==========

def event_start(question: str) -> StreamEvent:
    """开始处理事件"""
    return StreamEvent(
        event_type=EventType.START,
        message="开始处理问题",
        data={"question": question}
    )

def event_intent(intent: str, confidence: float) -> StreamEvent:
    """意图识别完成事件"""
    return StreamEvent(
        event_type=EventType.INTENT_DONE,
        agent="意图识别",
        message=f"识别意图: {intent}",
        detail=f"置信度: {confidence:.2f}",
        data={"intent": intent, "confidence": confidence}
    )

def event_local_start() -> StreamEvent:
    """LocalAgent 开始"""
    return StreamEvent(
        event_type=EventType.LOCAL_AGENT_START,
        agent="LocalAgent",
        message="开始检索本地知识库"
    )

def event_local_search(query: str, doc_count: int) -> StreamEvent:
    """LocalAgent 检索完成"""
    return StreamEvent(
        event_type=EventType.LOCAL_AGENT_SEARCH,
        agent="LocalAgent",
        message=f"检索到 {doc_count} 条相关文档",
        detail=f"查询: {query[:50]}..."
    )

def event_local_rerank(before: int, after: int) -> StreamEvent:
    """LocalAgent 重排完成"""
    return StreamEvent(
        event_type=EventType.LOCAL_AGENT_RERANK,
        agent="LocalAgent",
        message=f"重排序完成: {before} -> {after} 条",
        detail="按相关性重新排序"
    )

def event_local_think(thinking: str) -> StreamEvent:
    """LocalAgent 思考中"""
    return StreamEvent(
        event_type=EventType.LOCAL_AGENT_THINK,
        agent="LocalAgent",
        message="正在分析证据...",
        detail=thinking[:200] if thinking else ""
    )

def event_local_done(summary: str, key_points: List[str], evidence_count: int) -> StreamEvent:
    """LocalAgent 完成"""
    return StreamEvent(
        event_type=EventType.LOCAL_AGENT_DONE,
        agent="LocalAgent",
        message=f"本地研究完成，找到 {evidence_count} 条证据",
        detail=summary[:200] if summary else "",
        data={"key_points": key_points, "evidence_count": evidence_count}
    )

def event_web_start() -> StreamEvent:
    """WebAgent 开始"""
    return StreamEvent(
        event_type=EventType.WEB_AGENT_START,
        agent="WebAgent",
        message="开始联网搜索"
    )

def event_web_search(query: str, result_count: int) -> StreamEvent:
    """WebAgent 搜索完成"""
    return StreamEvent(
        event_type=EventType.WEB_AGENT_SEARCH,
        agent="WebAgent",
        message=f"找到 {result_count} 条网络结果",
        detail=f"搜索: {query[:50]}..."
    )

def event_web_think(thinking: str) -> StreamEvent:
    """WebAgent 思考中"""
    return StreamEvent(
        event_type=EventType.WEB_AGENT_THINK,
        agent="WebAgent",
        message="正在分析网络信息...",
        detail=thinking[:200] if thinking else ""
    )

def event_web_done(summary: str, key_points: List[str], evidence_count: int) -> StreamEvent:
    """WebAgent 完成"""
    return StreamEvent(
        event_type=EventType.WEB_AGENT_DONE,
        agent="WebAgent",
        message=f"网络研究完成，找到 {evidence_count} 条信息",
        detail=summary[:200] if summary else "",
        data={"key_points": key_points, "evidence_count": evidence_count}
    )

def event_web_skip(reason: str) -> StreamEvent:
    """WebAgent 跳过"""
    return StreamEvent(
        event_type=EventType.WEB_AGENT_SKIP,
        agent="WebAgent",
        message="跳过联网搜索",
        detail=reason
    )

def event_arbiter_start() -> StreamEvent:
    """Arbiter 开始"""
    return StreamEvent(
        event_type=EventType.ARBITER_START,
        agent="Arbiter",
        message="开始综合裁决"
    )

def event_arbiter_conflict(has_conflict: bool, details: str) -> StreamEvent:
    """Arbiter 冲突检测"""
    if has_conflict:
        return StreamEvent(
            event_type=EventType.ARBITER_CONFLICT,
            agent="Arbiter",
            message="检测到信息冲突",
            detail=details
        )
    else:
        return StreamEvent(
            event_type=EventType.ARBITER_CONFLICT,
            agent="Arbiter",
            message="无信息冲突",
            detail="本地与网络信息一致"
        )

def event_arbiter_think(thinking: str) -> StreamEvent:
    """Arbiter 思考中"""
    return StreamEvent(
        event_type=EventType.ARBITER_THINK,
        agent="Arbiter",
        message="正在综合分析...",
        detail=thinking[:200] if thinking else ""
    )

def event_arbiter_done(confidence: str) -> StreamEvent:
    """Arbiter 完成"""
    confidence_text = {"high": "高", "medium": "中", "low": "低"}.get(confidence, confidence)
    return StreamEvent(
        event_type=EventType.ARBITER_DONE,
        agent="Arbiter",
        message=f"裁决完成，置信度: {confidence_text}",
        data={"confidence": confidence}
    )

def event_generate_start() -> StreamEvent:
    """开始生成答案"""
    return StreamEvent(
        event_type=EventType.GENERATE_START,
        agent="生成器",
        message="正在组织最终答案..."
    )

def event_generate_done() -> StreamEvent:
    """答案生成完成"""
    return StreamEvent(
        event_type=EventType.GENERATE_DONE,
        agent="生成器",
        message="答案生成完成"
    )

def event_complete(result: Dict[str, Any]) -> StreamEvent:
    """处理完成事件"""
    return StreamEvent(
        event_type=EventType.COMPLETE,
        message="处理完成",
        data=result
    )

def event_error(error: str) -> StreamEvent:
    """错误事件"""
    return StreamEvent(
        event_type=EventType.ERROR,
        message="处理出错",
        detail=error
    )
