"""
RAG 问答链 - 根据意图调用向量检索，组装提示词，调用大模型得到回答
支持自动提取和利用长期记忆
"""
import re
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

from .config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME,
    TEMPERATURE, TOP_K
)
from .intent_router import Intent, detect_intent, get_category_for_intent
from .state import ConversationState
from .vector_store import similarity_search


# 记忆提取的标记
MEMORY_TAG_PATTERN = r'\[记忆:([^\]]+)\]'


def get_llm() -> ChatOpenAI:
    """获取 LLM 实例"""
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL
    )


def format_documents(docs: List[Document]) -> str:
    """将检索到的文档格式化为上下文文本"""
    if not docs:
        return "未找到相关参考资料。"

    context_parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get('title', '未知')
        category = doc.metadata.get('category', '未知')
        source = doc.metadata.get('source', '未知')
        content = doc.page_content

        part = f"""【参考资料 {i}】
标题: {title}
类别: {category}
来源: {source}
内容: {content}
"""
        context_parts.append(part)

    return "\n".join(context_parts)


def build_memory_instruction() -> str:
    """构建记忆提取指令（添加到每个提示词末尾）"""
    return """

【重要】记忆提取指令：
如果用户在问题中提到了关于自己的重要个人情况，请在回答的最末尾用特殊格式标记：
[记忆:简短描述]

需要记录的情况包括：
- 学业情况：挂科、重修、休学、转专业等
- 个人计划：考研、出国、就业意向等
- 特殊身份：贫困生、获奖情况、学生干部等
- 其他重要信息

示例：
- 用户说"我高数挂了" → 回答末尾加 [记忆:高等数学挂科]
- 用户说"我想转到软件工程" → 回答末尾加 [记忆:有转专业到软件工程的意向]
- 用户说"我是贫困生" → 回答末尾加 [记忆:贫困生身份]

注意：
1. 只记录用户明确提到的个人情况，不要猜测
2. 如果没有需要记录的信息，不要添加任何标记
3. 记忆内容要简短，10-20字以内
4. 可以添加多条记忆，每条单独一行"""


def build_prompt_for_rule(question: str, context: str, state: ConversationState) -> str:
    """构建查规定类问题的提示词"""
    user_info = state.get_profile_summary()
    history = state.get_history_text(3)
    memories = state.get_memories_text()

    return f"""你是首都师范大学的学业咨询助手。用户正在询问学校的制度规定。

【用户信息】
{user_info}

【用户的重要记录】
{memories}

【近期对话】
{history}

【参考资料】
{context}

【用户问题】
{question}

【回答要求】
1. 首先用简洁的自然语言解释相关规定
2. 然后明确指出关键条款或数字（如学分、年限、比例等）
3. 如果用户的记录中有相关信息（如挂科记录），要结合这些情况给出针对性建议
4. 最后强调这是首都师范大学的规定，如有疑问建议咨询相关部门
5. 回答要准确、清晰，避免模糊表述
{build_memory_instruction()}

请回答："""


def build_prompt_for_flow(question: str, context: str, state: ConversationState) -> str:
    """构建问流程类问题的提示词"""
    user_info = state.get_profile_summary()
    history = state.get_history_text(3)
    memories = state.get_memories_text()

    return f"""你是首都师范大学的学业咨询助手。用户正在询问办事流程。

【用户信息】
{user_info}

【用户的重要记录】
{memories}

【近期对话】
{history}

【参考资料】
{context}

【用户问题】
{question}

【回答要求】
1. 按步骤顺序清晰列出办事流程，使用"第一步"、"第二步"等标记
2. 说明每一步需要做什么、找谁、去哪里
3. 提醒需要准备的材料和注意事项
4. 如果有时间节点或截止日期，要特别强调
5. 如果参考资料中信息不完整，建议咨询具体部门
{build_memory_instruction()}

请回答："""


def build_prompt_for_experience(question: str, context: str, state: ConversationState) -> str:
    """构建求经验类问题的提示词"""
    user_info = state.get_profile_summary()
    history = state.get_history_text(3)
    memories = state.get_memories_text()

    return f"""你是首都师范大学的学业咨询助手。用户正在寻求经验建议。

【用户信息】
{user_info}

【用户的重要记录】
{memories}

【近期对话】
{history}

【参考资料】
{context}

【用户问题】
{question}

【回答要求】
1. 首先简要提及相关的学校规定底线（如有）
2. 结合用户的个人记录（如有挂科、转专业意向等），给出针对性建议
3. 建议应该切实可行，考虑到大学生的实际情况
4. 最后加一句"以上建议仅供参考，请根据个人情况酌情调整"
5. 语气要友善、鼓励，像学长学姐给建议
{build_memory_instruction()}

请回答："""


def build_prompt_for_template(question: str, context: str, state: ConversationState) -> str:
    """构建要模板类问题的提示词"""
    user_info = state.get_profile_summary()
    memories = state.get_memories_text()

    return f"""你是首都师范大学的学业咨询助手。用户需要一个文本模板。

【用户信息】
{user_info}

【用户的重要记录】
{memories}

【参考资料】
{context}

【用户问题】
{question}

【回答要求】
1. 首先简要说明这类文本的注意事项和礼仪要点
2. 然后提供一个完整的模板，用【】标记需要替换的内容，如【姓名】、【课程名】
3. 模板要正式得体，符合学生向老师或学校沟通的场合
4. 最后提醒用户根据实际情况修改相应内容
5. 如果是邮件，要包含主题行
{build_memory_instruction()}

请回答："""


def build_prompt_for_curriculum(question: str, context: str, state: ConversationState) -> str:
    """构建查培养方案类问题的提示词"""
    user_info = state.get_profile_summary()
    memories = state.get_memories_text()

    return f"""你是首都师范大学的学业咨询助手。用户正在询问专业培养方案相关内容。

【用户信息】
{user_info}

【用户的重要记录】
{memories}

【参考资料】
{context}

【用户问题】
{question}

【回答要求】
1. 准确引用培养方案中的相关内容
2. 清晰说明学分要求、课程设置等具体信息
3. 如果涉及必修和选修，要分别说明
4. 提醒用户具体要求可能因年级不同有所变化，建议查看最新版培养方案
5. 如果信息不完整，建议咨询学院教务
{build_memory_instruction()}

请回答："""


def build_prompt_for_general(question: str, context: str, state: ConversationState) -> str:
    """构建通用问题的提示词"""
    user_info = state.get_profile_summary()
    history = state.get_history_text(3)
    memories = state.get_memories_text()

    return f"""你是首都师范大学的学业咨询助手，帮助在校本科生解答学习和校园生活相关问题。

【用户信息】
{user_info}

【用户的重要记录】
{memories}

【近期对话】
{history}

【参考资料】
{context}

【用户问题】
{question}

【回答要求】
1. 基于参考资料，准确回答用户问题
2. 结合用户的个人记录，给出针对性建议
3. 如果参考资料中没有相关信息，请诚实说明
4. 回答要清晰、有条理
5. 保持友善、专业的语气
{build_memory_instruction()}

请回答："""


def extract_memories(answer: str) -> Tuple[str, List[str]]:
    """
    从回答中提取记忆标记

    Args:
        answer: AI 的原始回答

    Returns:
        (清理后的回答, 提取到的记忆列表)
    """
    # 查找所有记忆标记
    memories = re.findall(MEMORY_TAG_PATTERN, answer)

    # 移除记忆标记，得到干净的回答
    clean_answer = re.sub(MEMORY_TAG_PATTERN, '', answer)

    # 清理多余的空行
    clean_answer = re.sub(r'\n{3,}', '\n\n', clean_answer)
    clean_answer = clean_answer.strip()

    return clean_answer, memories


def retrieve_documents(
    vector_store: FAISS,
    question: str,
    intent: Intent,
    k: int = TOP_K
) -> List[Document]:
    """根据意图检索相关文档"""
    category = get_category_for_intent(intent)

    if category:
        docs = similarity_search(vector_store, question, k=k, category_filter=category)
        if len(docs) < k // 2:
            additional = similarity_search(vector_store, question, k=k)
            seen_contents = {doc.page_content for doc in docs}
            for doc in additional:
                if doc.page_content not in seen_contents:
                    docs.append(doc)
                    if len(docs) >= k:
                        break
    else:
        docs = similarity_search(vector_store, question, k=k)

    return docs


def answer_question(
    vector_store: FAISS,
    question: str,
    state: ConversationState
) -> Tuple[str, List[str]]:
    """
    回答用户问题

    Args:
        vector_store: 向量存储
        question: 用户问题
        state: 对话状态

    Returns:
        (回答文本, 新提取的记忆列表)
    """
    # 1. 意图识别
    intent, confidence = detect_intent(question)

    # 2. 检索相关文档
    docs = retrieve_documents(vector_store, question, intent)

    # 3. 格式化上下文
    context = format_documents(docs)

    # 4. 根据意图选择提示构造函数
    prompt_builders = {
        Intent.RULE: build_prompt_for_rule,
        Intent.FLOW: build_prompt_for_flow,
        Intent.EXPERIENCE: build_prompt_for_experience,
        Intent.TEMPLATE: build_prompt_for_template,
        Intent.CURRICULUM: build_prompt_for_curriculum,
        Intent.UNKNOWN: build_prompt_for_general
    }

    prompt_builder = prompt_builders.get(intent, build_prompt_for_general)
    prompt = prompt_builder(question, context, state)

    # 5. 调用 LLM
    llm = get_llm()
    response = llm.invoke(prompt)
    # 兼容处理：某些第三方 API 可能返回字符串而非 AIMessage 对象
    if isinstance(response, str):
        raw_answer = response
    else:
        raw_answer = response.content

    # 6. 提取记忆
    clean_answer, new_memories = extract_memories(raw_answer)

    # 7. 保存新记忆到状态
    added_memories = []
    for mem in new_memories:
        if state.add_memory(mem.strip(), source=f"从对话中提取"):
            added_memories.append(mem.strip())

    # 8. 更新对话状态
    state.add_user_message(question, intent=intent.value)
    state.add_assistant_message(clean_answer)

    return clean_answer, added_memories


def answer_without_rag(question: str, state: ConversationState) -> str:
    """不使用知识库，直接调用 LLM 回答（用于对比测试）"""
    user_info = state.get_profile_summary()

    prompt = f"""你是一个大学学业咨询助手。请根据你的通用知识回答以下问题。

【用户信息】
{user_info}

【用户问题】
{question}

请回答（注意：你没有访问首都师范大学的具体规定和文件）："""

    llm = get_llm()
    response = llm.invoke(prompt)
    # 兼容处理：某些第三方 API 可能返回字符串而非 AIMessage 对象
    if isinstance(response, str):
        return response
    return response.content


if __name__ == "__main__":
    # 测试模块
    from .vector_store import load_vector_store

    try:
        vs = load_vector_store()
        state = ConversationState()
        state.profile.college = "信息工程学院"
        state.profile.major = "计算机科学与技术"
        state.profile.grade = "大二"

        test_questions = [
            "我高数挂了，怎么办？",
            "奖学金怎么申请？",
        ]

        for q in test_questions:
            print(f"\n问题: {q}")
            print("-" * 50)
            answer, memories = answer_question(vs, q, state)
            print(f"回答:\n{answer}")
            if memories:
                print(f"\n[新增记忆: {', '.join(memories)}]")
            print("=" * 50)

        print(f"\n所有记忆:\n{state.get_memories_display()}")

    except FileNotFoundError:
        print("请先运行 build_index.py 构建向量索引")
