"""
多智能体模块 - 本地证据智能体、联网研究智能体、裁决智能体
"""
from .local_agent import LocalEvidenceAgent
from .web_agent import WebResearchAgent
from .arbiter_agent import ArbiterAgent

__all__ = ['LocalEvidenceAgent', 'WebResearchAgent', 'ArbiterAgent']
