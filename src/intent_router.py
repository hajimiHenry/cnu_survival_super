"""
意图识别 - 根据问题文本判断用户意图
"""
from enum import Enum
from typing import Tuple
import re


class Intent(Enum):
    """用户意图类型"""
    RULE = "查规定"           # 查询制度规定
    FLOW = "问流程"           # 询问办事流程
    EXPERIENCE = "求经验"      # 寻求经验建议
    TEMPLATE = "要模板"        # 需要文本模板
    CURRICULUM = "查培养方案"  # 查询培养方案
    UNKNOWN = "未知"          # 无法判断


# 关键词规则定义
INTENT_KEYWORDS = {
    Intent.RULE: [
        # 制度相关
        "规定", "规则", "条例", "制度", "政策", "要求",
        # 学业相关
        "重修", "挂科", "不及格", "处分", "记过", "警告", "开除",
        "学分", "绩点", "学籍", "退学", "休学", "复学",
        "会不会", "能不能", "可不可以", "允许", "禁止",
        # 奖惩相关
        "奖学金", "助学金", "奖励", "惩罚",
        # 考试相关
        "考试", "作弊", "违纪", "缺考",
        # 住宿相关
        "宿舍", "住宿", "晚归", "留宿",
    ],

    Intent.FLOW: [
        # 流程相关
        "流程", "步骤", "程序", "手续", "办理",
        "怎么申请", "如何申请", "申请流程", "办事流程",
        "去哪里", "到哪", "找谁", "哪个部门",
        "需要什么材料", "需要准备", "提交什么",
        "什么时候办", "截止", "期限",
        # 具体事项
        "缓考", "补考", "请假", "销假",
        "补办", "挂失", "学生证",
        "转专业", "转学", "保研", "推免",
        "贫困认定", "困难认定",
    ],

    Intent.EXPERIENCE: [
        # 建议相关
        "建议", "经验", "技巧", "心得", "体会",
        "怎么安排", "如何安排", "怎么规划", "如何规划",
        "合适", "不合适", "值得", "不值得",
        "要不要", "应该", "应不应该", "该不该",
        "好不好", "有没有必要",
        # 学习生活
        "期末周", "复习", "备考",
        "社团", "学生会", "志愿", "兼职", "实习",
        "选课", "老师", "评价",
        "室友", "人际关系", "沟通",
        "时间管理", "效率",
    ],

    Intent.TEMPLATE: [
        # 模板相关
        "模板", "范文", "格式", "文案",
        "怎么写", "如何写", "写一封", "帮我写",
        "邮件", "请假条", "申请书", "申请信",
        "给老师", "给辅导员", "给导师",
        "措辞", "表达", "回复",
    ],

    Intent.CURRICULUM: [
        # 培养方案相关
        "培养方案", "课程设置", "必修课", "选修课",
        "毕业要求", "毕业学分", "专业课",
        "大类培养", "专业方向",
        "学什么", "开什么课", "课程体系",
    ]
}


def detect_intent(question: str) -> Tuple[Intent, float]:
    """
    根据问题文本判断意图

    Args:
        question: 用户输入的问题

    Returns:
        (意图类型, 置信度)
    """
    question = question.strip()

    if not question:
        return Intent.UNKNOWN, 0.0

    # 统计各意图的匹配分数
    scores = {intent: 0 for intent in Intent}

    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in question:
                scores[intent] += 1
                # 如果关键词较长，给予额外权重
                if len(keyword) >= 4:
                    scores[intent] += 0.5

    # 找出最高分的意图
    max_score = max(scores.values())

    if max_score == 0:
        return Intent.UNKNOWN, 0.0

    # 找出得分最高的意图
    best_intent = max(scores, key=scores.get)

    # 计算置信度（简单归一化）
    total_score = sum(scores.values())
    confidence = scores[best_intent] / total_score if total_score > 0 else 0.0

    return best_intent, confidence


def get_category_for_intent(intent: Intent) -> str:
    """
    根据意图获取对应的文档类别

    Args:
        intent: 用户意图

    Returns:
        对应的文档类别名称
    """
    intent_to_category = {
        Intent.RULE: "制度",
        Intent.FLOW: "流程",
        Intent.EXPERIENCE: "经验",
        Intent.TEMPLATE: "模板",
        Intent.CURRICULUM: "培养方案",
        Intent.UNKNOWN: None
    }
    return intent_to_category.get(intent)


def get_intent_description(intent: Intent) -> str:
    """
    获取意图的描述信息

    Args:
        intent: 用户意图

    Returns:
        意图描述
    """
    descriptions = {
        Intent.RULE: "查询学校制度规定",
        Intent.FLOW: "询问办事流程步骤",
        Intent.EXPERIENCE: "寻求经验建议",
        Intent.TEMPLATE: "获取文本模板",
        Intent.CURRICULUM: "查询培养方案",
        Intent.UNKNOWN: "综合咨询"
    }
    return descriptions.get(intent, "综合咨询")


if __name__ == "__main__":
    # 测试意图识别
    test_questions = [
        "挂科了会怎么样？",
        "重修怎么办理？需要什么材料？",
        "怎么安排期末周的复习时间？",
        "帮我写一封给老师请假的邮件",
        "计算机专业需要学什么课程？",
        "今天天气怎么样？",
        "缓考申请流程是什么？",
        "奖学金的评定规定是什么？",
        "选课时应该注意什么？有什么建议？",
    ]

    print("=== 意图识别测试 ===\n")
    for q in test_questions:
        intent, confidence = detect_intent(q)
        print(f"问题: {q}")
        print(f"意图: {intent.value} (置信度: {confidence:.2f})")
        print(f"类别: {get_category_for_intent(intent)}")
        print()
