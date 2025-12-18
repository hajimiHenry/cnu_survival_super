#!/usr/bin/env python
"""
Web GUI 启动脚本
自动打开浏览器并启动服务
"""
import sys
import webbrowser
import threading
import time
import urllib.request
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

PORT = 8080  # 服务端口


def open_browser():
    """等待服务启动成功后再打开浏览器"""
    url = f"http://localhost:{PORT}"
    
    # 最多等待60秒
    for i in range(60):
        try:
            urllib.request.urlopen(url, timeout=1)
            print("服务已启动，正在打开浏览器...")
            webbrowser.open(url)
            return
        except:
            time.sleep(1)
    
    print("服务启动超时，请手动打开浏览器访问:", url)


def main():
    """主入口"""
    import uvicorn

    print("=" * 60)
    print("    首都师范大学在校生生存助手 - Web GUI")
    print("=" * 60)
    print()

    # 后台线程打开浏览器
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # 启动 Web 服务
    try:
        uvicorn.run(
            "src.web_server:app",
            host="127.0.0.1",
            port=PORT,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n服务已停止。")
    except Exception as e:
        print(f"\n启动失败: {e}")
        print("\n可能的原因：")
        print(f"1. 端口 {PORT} 被占用")
        print("2. 依赖未安装，请运行: pip install fastapi uvicorn")
        input("\n按回车键退出...")


if __name__ == "__main__":
    main()
