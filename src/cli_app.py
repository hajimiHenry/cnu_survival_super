"""
命令行界面 - 实现交互式问答
支持用户数据持久化和增强版RAG功能
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_store import load_vector_store
from src.rag_chain import answer_question
from src.enhanced_rag_chain import EnhancedRAGChain
from src.state import ConversationState, load_user_state, save_user_state
from src.intent_router import detect_intent


# 全局设置
USE_ENHANCED = True  # 是否使用增强版RAG
ENABLE_WEB_SEARCH = True  # 是否启用联网搜索


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔═════════════════════════════════════════════════════════════╗
║                                                             ║
║     首都师范大学在校生生存助手 (CNU Survival Assistant)     ║
║                                                             ║
║ 本系统可以回答关于学校制度、办事流程、学习经验和文本模板    ║
║ 的问题。输入 'exit' 或 'quit' 退出系统。                    ║
║                                                             ║
╚═════════════════════════════════════════════════════════════╝
"""
    print(banner)


def collect_user_info(state: ConversationState):
    """收集用户基本信息"""
    print("\n为了更好地为您服务，请先提供一些基本信息（直接回车可跳过）：\n")

    # 询问年级
    grade = input("请输入您的年级（如：大一、大二、大三、大四）: ").strip()
    if grade:
        state.profile.grade = grade

    # 询问学院
    college = input("请输入您所在的学院: ").strip()
    if college:
        state.profile.college = college

    # 询问专业
    major = input("请输入您的专业: ").strip()
    if major:
        state.profile.major = major

    print("\n" + "=" * 60)
    print("您的信息已记录：")
    print(state.get_profile_summary())
    print("=" * 60 + "\n")


def show_welcome_back(state: ConversationState):
    """显示欢迎回来信息"""
    print("\n" + "=" * 60)
    print("欢迎回来！已加载您的信息：")
    print(state.get_profile_summary())

    if state.history:
        print(f"\n已加载 {len(state.history)} 条历史对话记录")

    if state.memories:
        print(f"已加载 {len(state.memories)} 条长期记忆")

    print("=" * 60)
    print("\n提示：输入 'memory' 或 '记忆' 查看和管理长期记忆\n")


def format_answer(answer: str) -> str:
    """格式化回答，添加适当的分隔"""
    lines = answer.split('\n')
    formatted = []
    for line in lines:
        formatted.append("  " + line)
    return '\n'.join(formatted)


def main():
    """主程序入口"""
    print_banner()

    # 加载向量索引
    print("正在加载知识库...")
    try:
        vector_store = load_vector_store()
        print("知识库加载完成！")
    except FileNotFoundError:
        print("\n错误: 向量索引不存在！")
        print("请先运行以下命令构建索引：")
        print("  python build_index.py")
        return
    except Exception as e:
        print(f"\n错误: 加载向量索引失败 - {e}")
        return

    # 初始化增强版RAG链
    enhanced_chain = None
    if USE_ENHANCED:
        try:
            enhanced_chain = EnhancedRAGChain(vector_store)
            print("增强版RAG已启用（检索回退 | 迭代查询 | 重排 | 联网搜索）")
        except Exception as e:
            print(f"警告: 增强版RAG初始化失败，将使用基础版 - {e}")
            enhanced_chain = None

    # 加载用户状态（如果存在）
    state = load_user_state()

    # 检查是否有已保存的用户信息
    if state.profile.has_basic_info():
        # 老用户，显示欢迎回来
        show_welcome_back(state)
    else:
        # 新用户，收集信息
        collect_user_info(state)
        # 立即保存
        save_user_state(state)

    # 主循环
    print("请输入您的问题（输入 'exit' 或 'quit' 退出）：\n")

    while True:
        try:
            # 获取用户输入
            user_input = input("您: ").strip()

            # 检查退出命令
            if user_input.lower() in ['exit', 'quit', '退出', '再见']:
                # 退出前保存数据
                print("\n正在保存对话数据...")
                if save_user_state(state):
                    print("数据已保存！")
                print("感谢使用首都师范大学在校生生存助手，再见！")
                break

            # 检查空输入
            if not user_input:
                print("请输入您的问题。\n")
                continue

            # 检查帮助命令
            if user_input.lower() in ['help', '帮助', '?', '？']:
                print("""
可用命令：
  help / 帮助      - 显示帮助信息
  clear / 清空     - 清空对话历史
  info / 信息      - 显示当前用户信息
  memory / 记忆    - 查看和管理长期记忆
  reset / 重置     - 重新设置个人信息
  save / 保存      - 手动保存当前数据
  exit / quit      - 退出系统（自动保存）

您可以询问以下类型的问题：
  1. 查规定 - 询问学校的规章制度
     例如：挂科了会怎么样？奖学金怎么评定？

  2. 问流程 - 询问办事流程和步骤
     例如：缓考怎么申请？转专业的流程是什么？

  3. 求经验 - 获取学习生活建议
     例如：怎么安排期末复习？如何处理与室友的关系？

  4. 要模板 - 获取文本模板
     例如：帮我写一封请假邮件。申请书怎么写？

长期记忆功能：
  系统会自动从对话中提取您的重要个人信息（如挂科、转专业意向等），
  并在后续对话中为您提供更个性化的建议。
""")
                continue

            # 检查清空命令
            if user_input.lower() in ['clear', '清空']:
                state.clear_history()
                save_user_state(state)
                print("对话历史已清空。\n")
                continue

            # 检查信息命令
            if user_input.lower() in ['info', '信息']:
                print(f"\n{state.get_profile_summary()}")
                print(f"对话历史: {len(state.history)} 条记录\n")
                continue

            # 检查重置命令
            if user_input.lower() in ['reset', '重置']:
                confirm = input("确定要重新设置个人信息吗？(y/n): ").strip().lower()
                if confirm == 'y':
                    state.reset_profile()
                    collect_user_info(state)
                    save_user_state(state)
                else:
                    print("已取消。\n")
                continue

            # 检查保存命令
            if user_input.lower() in ['save', '保存']:
                if save_user_state(state):
                    print("数据保存成功！\n")
                else:
                    print("数据保存失败。\n")
                continue

            # 检查记忆管理命令
            if user_input.lower() in ['memory', '记忆']:
                print("\n" + "=" * 50)
                print("【长期记忆管理】")
                print("=" * 50)
                print(state.get_memories_display())
                print("\n可用操作：")
                print("  输入数字删除对应记忆（如输入 1 删除第一条）")
                print("  输入 'add 内容' 手动添加记忆")
                print("  输入 'clear' 清空所有记忆")
                print("  直接回车返回对话")
                print("-" * 50)

                while True:
                    mem_input = input("记忆管理> ").strip()

                    if not mem_input:
                        print("返回对话模式。\n")
                        break

                    if mem_input.lower() == 'clear':
                        confirm = input("确定要清空所有记忆吗？(y/n): ").strip().lower()
                        if confirm == 'y':
                            state.clear_memories()
                            save_user_state(state)
                            print("所有记忆已清空。\n")
                        else:
                            print("已取消。")
                        break

                    if mem_input.lower().startswith('add '):
                        content = mem_input[4:].strip()
                        if content:
                            if state.add_memory(content, source="用户手动添加"):
                                save_user_state(state)
                                print(f"已添加记忆: {content}")
                            else:
                                print("该记忆已存在。")
                        else:
                            print("请输入要添加的内容。")
                        continue

                    try:
                        idx = int(mem_input)
                        if state.remove_memory(idx):
                            save_user_state(state)
                            print(f"已删除第 {idx} 条记忆。")
                            print("\n当前记忆：")
                            print(state.get_memories_display())
                        else:
                            print(f"无效的序号: {idx}")
                    except ValueError:
                        print("无效输入。请输入数字、'add 内容'、'clear' 或直接回车。")

                continue

            # 意图识别（用于显示）
            intent, confidence = detect_intent(user_input)

            # 获取回答
            print(f"\n[识别意图: {intent.value}]")
            print("正在思考...\n")

            try:
                # 使用增强版RAG
                if USE_ENHANCED and enhanced_chain is not None:
                    result = enhanced_chain.answer(
                        user_input,
                        state=state,
                        enable_web_search=ENABLE_WEB_SEARCH
                    )
                    answer = result.answer
                    new_memories = result.new_memories

                    # 显示增强版信息
                    print("助手:")
                    print(format_answer(answer))

                    # 显示检索统计
                    tags = []
                    if result.used_fallback:
                        tags.append("回退检索")
                    if result.used_iteration:
                        tags.append("迭代查询")
                    if result.used_web_search:
                        tags.append("联网搜索")

                    if tags:
                        print(f"\n  [使用功能: {' | '.join(tags)}]")

                    # 显示证据来源
                    if result.evidence_local or result.evidence_web:
                        print("\n  " + "-" * 40)
                        print("  [证据来源]")
                        if result.evidence_local:
                            print(f"    本地知识库: {len(result.evidence_local)} 条")
                            for i, ev in enumerate(result.evidence_local[:3], 1):
                                source = ev.get('source', '未知')
                                print(f"      {i}. {source}")
                        if result.evidence_web:
                            print(f"    网络搜索: {len(result.evidence_web)} 条")
                            for i, ev in enumerate(result.evidence_web[:2], 1):
                                source = ev.get('source', '未知')
                                print(f"      {i}. {source[:50]}...")
                        print(f"    耗时: {result.total_duration_ms:.0f}ms")
                        print("  " + "-" * 40)

                else:
                    # 使用基础版RAG
                    answer, new_memories = answer_question(vector_store, user_input, state)
                    print("助手:")
                    print(format_answer(answer))

                # 显示新提取的记忆
                if new_memories:
                    print("\n  " + "-" * 40)
                    print("  [系统已自动记录以下信息]")
                    for mem in new_memories:
                        print(f"    * {mem}")
                    print("  " + "-" * 40)

                print()

                # 每次对话后自动保存（可选，防止意外退出丢失数据）
                save_user_state(state)

            except Exception as e:
                print(f"抱歉，处理您的问题时出现错误: {e}")
                print("请尝试重新提问或换一种方式表述。\n")

        except KeyboardInterrupt:
            # Ctrl+C 退出时也保存数据
            print("\n\n正在保存数据...")
            save_user_state(state)
            print("感谢使用，再见！")
            break
        except EOFError:
            print("\n\n正在保存数据...")
            save_user_state(state)
            print("感谢使用，再见！")
            break


if __name__ == "__main__":
    main()
