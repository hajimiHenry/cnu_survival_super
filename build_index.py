#!/usr/bin/env python
"""
索引构建脚本 - 离线构建向量索引
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.loader import load_all_documents
from src.vector_store import split_documents, build_vector_store, save_vector_store
from src.config import INDEX_DIR


def main():
    """构建向量索引的主函数"""
    print("=" * 60)
    print("首都师范大学在校生生存助手 - 向量索引构建工具")
    print("=" * 60)

    # 1. 加载文档
    print("\n第一步: 加载文档...")
    documents = load_all_documents()

    if not documents:
        print("错误: 没有找到任何文档！")
        print("请确保 data 目录下有正确格式的文本文件。")
        return

    print(f"\n共加载 {len(documents)} 条文档")

    # 2. 切分文档
    print("\n第二步: 切分文档...")
    split_docs = split_documents(documents)

    # 3. 构建向量索引
    print("\n第三步: 构建向量索引...")
    print("（此步骤需要调用嵌入模型，请确保已正确配置 API）")

    try:
        vector_store = build_vector_store(split_docs)
    except Exception as e:
        print(f"\n构建向量索引失败: {e}")
        print("\n可能的原因：")
        print("1. 未设置 OPENAI_API_KEY 环境变量")
        print("2. API 地址不正确")
        print("3. 网络连接问题")
        print("\n请检查配置后重试。")
        return

    # 4. 保存索引
    print("\n第四步: 保存向量索引...")
    save_vector_store(vector_store, INDEX_DIR)

    print("\n" + "=" * 60)
    print("向量索引构建完成！")
    print(f"索引保存位置: {INDEX_DIR}")
    print("\n您现在可以运行以下命令启动问答系统：")
    print("  python -m src.cli_app")
    print("=" * 60)


if __name__ == "__main__":
    main()
