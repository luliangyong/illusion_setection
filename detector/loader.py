"""
数据加载模块 — 读取replies和ground_truth
"""

import json
from pathlib import Path
from .types import ReplyItem, GroundTruthItem

_DATA_DIR = Path(__file__).resolve().parent.parent


def load_replies(path: str = None) -> list[ReplyItem]:
    """加载回复数据"""
    if path is None:
        path = _DATA_DIR / "task4_replies.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        ReplyItem(
            id=item["id"],
            user_question=item["user_question"],
            system_reply=item["system_reply"],
            knowledge_base=item["knowledge_base"],
        )
        for item in raw
    ]


def load_ground_truth(path: str = None) -> list[GroundTruthItem]:
    """加载人工标注"""
    if path is None:
        path = _DATA_DIR / "task4_ground_truth.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        GroundTruthItem(
            id=item["id"],
            is_hallucination=item["is_hallucination"],
            hallucination_type=item.get("hallucination_type"),
            detail=item.get("detail", ""),
        )
        for item in raw
    ]
