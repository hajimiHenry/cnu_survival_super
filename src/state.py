"""
对话状态管理 - 保存用户信息、对话历史和长期记忆
支持数据持久化到本地文件
"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from .config import PROJECT_ROOT

# 用户数据保存路径
USER_DATA_FILE = PROJECT_ROOT / "user_data.json"

# 最多保存的历史对话条数
MAX_HISTORY_SAVE = 20

# 最多保存的长期记忆条数
MAX_MEMORIES = 50


@dataclass
class Memory:
    """长期记忆条目"""
    content: str                # 记忆内容
    created_at: datetime = field(default_factory=datetime.now)
    source: str = ""            # 来源（哪次对话提取的）

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "source": self.source
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """从字典创建"""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()

        return cls(
            content=data.get("content", ""),
            created_at=created_at,
            source=data.get("source", "")
        )


@dataclass
class UserProfile:
    """用户基本信息"""
    university: str = "首都师范大学"
    college: str = ""      # 学院
    major: str = ""        # 专业
    grade: str = ""        # 年级
    notes: Dict[str, str] = field(default_factory=dict)  # 额外标记，如 {"已挂科": "高数"}

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "university": self.university,
            "college": self.college,
            "major": self.major,
            "grade": self.grade,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        """从字典创建"""
        return cls(
            university=data.get("university", "首都师范大学"),
            college=data.get("college", ""),
            major=data.get("major", ""),
            grade=data.get("grade", ""),
            notes=data.get("notes", {})
        )

    def has_basic_info(self) -> bool:
        """检查是否已填写基本信息"""
        return bool(self.grade or self.college or self.major)


@dataclass
class Message:
    """单条对话消息"""
    role: str           # "user" 或 "assistant"
    content: str        # 消息内容
    timestamp: datetime = field(default_factory=datetime.now)
    intent: str = ""    # 识别到的意图

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """从字典创建"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        else:
            timestamp = datetime.now()

        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=timestamp,
            intent=data.get("intent", "")
        )


@dataclass
class ConversationState:
    """对话状态"""
    profile: UserProfile = field(default_factory=UserProfile)
    history: List[Message] = field(default_factory=list)
    memories: List[Memory] = field(default_factory=list)  # 长期记忆
    session_start: datetime = field(default_factory=datetime.now)

    def add_user_message(self, content: str, intent: str = "") -> None:
        """添加用户消息"""
        self.history.append(Message(
            role="user",
            content=content,
            intent=intent
        ))

    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        self.history.append(Message(
            role="assistant",
            content=content
        ))

    def add_memory(self, content: str, source: str = "") -> bool:
        """
        添加长期记忆

        Args:
            content: 记忆内容
            source: 来源描述

        Returns:
            是否添加成功（重复则不添加）
        """
        # 检查是否已存在相似记忆
        content_lower = content.lower().strip()
        for mem in self.memories:
            if mem.content.lower().strip() == content_lower:
                return False  # 已存在，不重复添加

        # 添加新记忆
        self.memories.append(Memory(
            content=content.strip(),
            source=source
        ))

        # 如果超过上限，删除最早的记忆
        while len(self.memories) > MAX_MEMORIES:
            self.memories.pop(0)

        return True

    def remove_memory(self, index: int) -> bool:
        """
        删除指定索引的记忆

        Args:
            index: 记忆索引（从1开始）

        Returns:
            是否删除成功
        """
        idx = index - 1  # 转为0-based
        if 0 <= idx < len(self.memories):
            self.memories.pop(idx)
            return True
        return False

    def get_memories_text(self) -> str:
        """
        获取所有记忆的文本形式（用于提示词）

        Returns:
            记忆文本
        """
        if not self.memories:
            return "暂无记录"

        lines = []
        for mem in self.memories:
            lines.append(f"- {mem.content}")
        return "\n".join(lines)

    def get_memories_display(self) -> str:
        """
        获取记忆的显示形式（带编号，用于用户查看）

        Returns:
            带编号的记忆列表
        """
        if not self.memories:
            return "暂无长期记忆"

        lines = []
        for i, mem in enumerate(self.memories, 1):
            date_str = mem.created_at.strftime("%Y-%m-%d")
            lines.append(f"  {i}. {mem.content} ({date_str})")
        return "\n".join(lines)

    def clear_memories(self) -> None:
        """清空所有记忆"""
        self.memories.clear()

    def get_recent_history(self, n: int = 5) -> List[Message]:
        """获取最近的 n 条对话记录"""
        return self.history[-n:] if len(self.history) >= n else self.history

    def get_history_text(self, n: int = 5) -> str:
        """获取最近对话历史的文本形式"""
        recent = self.get_recent_history(n)
        lines = []
        for msg in recent:
            role_name = "用户" if msg.role == "user" else "助手"
            lines.append(f"{role_name}: {msg.content}")
        return "\n".join(lines)

    def add_note(self, key: str, value: str) -> None:
        """添加用户备注信息"""
        self.profile.notes[key] = value

    def get_profile_summary(self) -> str:
        """获取用户信息摘要"""
        parts = []
        if self.profile.university:
            parts.append(f"学校: {self.profile.university}")
        if self.profile.college:
            parts.append(f"学院: {self.profile.college}")
        if self.profile.major:
            parts.append(f"专业: {self.profile.major}")
        if self.profile.grade:
            parts.append(f"年级: {self.profile.grade}")
        if self.profile.notes:
            notes_str = ", ".join(f"{k}: {v}" for k, v in self.profile.notes.items())
            parts.append(f"备注: {notes_str}")

        return " | ".join(parts) if parts else "暂无用户信息"

    def clear_history(self) -> None:
        """清空对话历史"""
        self.history.clear()

    def has_discussed(self, keyword: str) -> bool:
        """检查对话历史中是否讨论过某个话题"""
        for msg in self.history:
            if keyword in msg.content:
                return True
        return False

    def to_dict(self) -> dict:
        """转换为字典（用于保存）"""
        return {
            "profile": self.profile.to_dict(),
            "history": [msg.to_dict() for msg in self.history[-MAX_HISTORY_SAVE:]],
            "memories": [mem.to_dict() for mem in self.memories],
            "last_updated": datetime.now().isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationState":
        """从字典创建"""
        state = cls()
        state.profile = UserProfile.from_dict(data.get("profile", {}))
        state.history = [
            Message.from_dict(msg) for msg in data.get("history", [])
        ]
        state.memories = [
            Memory.from_dict(mem) for mem in data.get("memories", [])
        ]
        return state

    def save(self, file_path: Optional[Path] = None) -> bool:
        """保存状态到文件"""
        if file_path is None:
            file_path = USER_DATA_FILE

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存用户数据失败: {e}")
            return False

    @classmethod
    def load(cls, file_path: Optional[Path] = None) -> "ConversationState":
        """从文件加载状态"""
        if file_path is None:
            file_path = USER_DATA_FILE

        if not file_path.exists():
            return cls()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            print(f"加载用户数据失败: {e}")
            return cls()

    def reset_profile(self) -> None:
        """重置用户信息（但保留学校）"""
        self.profile = UserProfile()

    def reset_all(self) -> None:
        """重置所有数据"""
        self.profile = UserProfile()
        self.history.clear()
        self.memories.clear()


def load_user_state() -> ConversationState:
    """加载用户状态的便捷函数"""
    return ConversationState.load()


def save_user_state(state: ConversationState) -> bool:
    """保存用户状态的便捷函数"""
    return state.save()


if __name__ == "__main__":
    # 测试状态管理和持久化
    state = ConversationState()

    # 设置用户信息
    state.profile.college = "信息工程学院"
    state.profile.major = "计算机科学与技术"
    state.profile.grade = "大二"

    # 添加对话
    state.add_user_message("挂科了怎么办？", intent="查规定")
    state.add_assistant_message("根据学校规定，必修课不及格需要重修...")

    # 添加长期记忆
    state.add_memory("高等数学挂科，需要重修", source="用户提及")
    state.add_memory("计划申请国家励志奖学金", source="用户提及")
    state.add_memory("考虑转专业到软件工程", source="用户提及")

    print("=== 用户信息 ===")
    print(state.get_profile_summary())

    print("\n=== 长期记忆 ===")
    print(state.get_memories_display())

    print("\n=== 记忆文本（用于提示词）===")
    print(state.get_memories_text())

    # 测试保存
    print("\n=== 测试保存 ===")
    if state.save():
        print(f"数据已保存到: {USER_DATA_FILE}")

    # 测试加载
    print("\n=== 测试加载 ===")
    loaded_state = ConversationState.load()
    print(f"加载的用户信息: {loaded_state.get_profile_summary()}")
    print(f"加载的记忆条数: {len(loaded_state.memories)}")
