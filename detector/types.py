"""
幻觉分类体系定义

5类幻觉 + 严重程度枚举
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class Severity(str, Enum):
    """严重程度"""
    CRITICAL = "严重"   # 涉及人身安全/健康
    HIGH = "高"         # 误导消费决策
    MEDIUM = "中"       # 部分误导但后果可控
    LOW = "低"          # 信息不完整但影响较小


class HallucinationType(str, Enum):
    """幻觉类型"""
    FABRICATION = "事实编造"       # 知识库完全不存在对应信息，凭空杜撰
    DISTORTION = "数值歪曲"        # 知识库有对应信息但数值/参数被篡改
    OVERCLAIM = "能力越界"         # 系统假装执行了不具备的操作能力
    SAFETY = "安全误导"            # 忽略或歪曲安全/健康风险信息
    OMISSION = "信息遗漏"          # 知识库有关键信息但回复未提及


# 类型 -> 严重程度映射
TYPE_SEVERITY_MAP = {
    HallucinationType.FABRICATION: Severity.HIGH,
    HallucinationType.DISTORTION: Severity.HIGH,
    HallucinationType.OVERCLAIM: Severity.HIGH,
    HallucinationType.SAFETY: Severity.CRITICAL,
    HallucinationType.OMISSION: Severity.LOW,
}

# 类型详细定义（用于Prompt）
TYPE_DEFINITIONS = {
    HallucinationType.FABRICATION: {
        "definition": "知识库中完全不存在对应信息，回复凭空杜撰了不存在的事实、政策、优惠、地址、品牌关联等。",
        "examples": [
            "编造不存在的退货政策（如30天无理由退货）",
            "杜撰不存在的优惠券（如满300减50）",
            "编造线下门店地址",
            "编造品牌关联关系",
        ],
        "severity": "高 — 直接误导用户做出错误决策",
    },
    HallucinationType.DISTORTION: {
        "definition": "知识库中有对应信息，但回复中的数值、参数、规格、时间等关键数据被歪曲或篡改。",
        "examples": [
            "蓝牙版本从5.0歪曲为5.3",
            "材质从PU合成革歪曲为头层牛皮",
            "发货时间从24小时歪曲为48小时",
            "接口类型从USB-A歪曲为Type-C",
        ],
        "severity": "高 — 误导用户对产品核心属性的认知",
    },
    HallucinationType.OVERCLAIM: {
        "definition": "系统声称执行了它实际上没有能力执行的操作，即假装具备不存在的系统能力。",
        "examples": [
            "系统未接入物流接口，却声称查到了快递位置",
            "系统无法修改订单，却声称已修改地址",
            "系统不具备工单升级功能，却声称已升级",
        ],
        "severity": "高 — 给用户虚假承诺，造成信任危机",
    },
    HallucinationType.SAFETY: {
        "definition": "回复忽略或歪曲了安全/健康相关的风险信息，可能对用户造成身体伤害。",
        "examples": [
            "产品含孕妇禁忌成分，回复却说孕妇可以放心使用",
            "忽略药物禁忌或过敏风险",
        ],
        "severity": "严重 — 可能造成人身伤害，最高优先级",
    },
    HallucinationType.OMISSION: {
        "definition": "知识库中有对用户决策重要的信息，但回复未提及，导致用户无法做出最优选择。",
        "examples": [
            "知识库有30%用户反馈偏大半码，回复却说尺码标准",
            "遗漏重要的使用注意事项",
        ],
        "severity": "低 — 信息不完整，但不直接造成错误决策",
    },
}


@dataclass
class ReplyItem:
    """单条回复数据"""
    id: str
    user_question: str
    system_reply: str
    knowledge_base: str


@dataclass
class DetectionResult:
    """单条检测结果"""
    id: str
    is_hallucination: bool
    hallucination_type: Optional[HallucinationType] = None
    severity: Optional[Severity] = None
    detail: str = ""
    confidence: float = 1.0  # 置信度 0-1


@dataclass
class GroundTruthItem:
    """单条人工标注"""
    id: str
    is_hallucination: bool
    hallucination_type: Optional[str] = None
    detail: str = ""


@dataclass
class EvaluationReport:
    """评估报告"""
    total: int = 0
    true_positive: int = 0   # 正确检出幻觉
    true_negative: int = 0   # 正确判定非幻觉
    false_positive: int = 0  # 误报（实际非幻觉，检测为幻觉）
    false_negative: int = 0  # 漏检（实际幻觉，检测为非幻觉）
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    false_positives_detail: list = field(default_factory=list)
    false_negatives_detail: list = field(default_factory=list)
    type_confusion: dict = field(default_factory=dict)  # 类型混淆矩阵
