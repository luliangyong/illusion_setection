"""
LLM检测引擎 — 调用OpenAI API进行幻觉检测

支持两种模式：
- real: 真实API调用
- mock: 基于规则的模拟返回（不需要API key）
"""

import json
import os
from .types import (
    ReplyItem, DetectionResult, HallucinationType, Severity,
    TYPE_SEVERITY_MAP, TYPE_DEFINITIONS,
)
from .mock_engine import detect as mock_detect


def _build_prompt(item: ReplyItem) -> str:
    """构建检测prompt"""
    type_desc = "\n".join([
        f"- **{t.value}**（严重程度: {info['severity']}）: {info['definition']}"
        for t, info in TYPE_DEFINITIONS.items()
    ])

    return f"""你是一个客服回复质量审核专家。你的任务是判断客服系统的回复是否存在"幻觉"——即回复内容与知识库信息不一致。

## 幻觉类型定义

{type_desc}

## 判断规则

1. 仔细对比"知识库"和"系统回复"，找出所有不一致之处
2. 如果知识库明确说"无/不支持/未接入"，但回复声称"有/支持/已执行"，则存在幻觉
3. 如果知识库中的数值/参数与回复中的不一致，则存在数值歪曲
4. 如果知识库中含有安全/健康警告但回复忽略或给出相反建议，则属于安全误导
5. 如果回复与知识库完全一致，则无幻觉

## 输入数据

**用户问题**: {item.user_question}
**系统回复**: {item.system_reply}
**知识库**: {item.knowledge_base}

## 输出格式

请严格按以下JSON格式输出（不要包含其他内容）:
```json
{{
    "is_hallucination": true/false,
    "hallucination_type": "事实编造|数值歪曲|能力越界|安全误导|信息遗漏|null",
    "detail": "判断理由，说明回复与知识库的具体矛盾之处",
    "confidence": 0.0-1.0
}}
```
"""


def detect_with_llm(item: ReplyItem, model: str = "gpt-3.5-turbo") -> DetectionResult:
    """使用真实LLM API检测"""
    try:
        from openai import OpenAI
    except ImportError:
        return _mock_llm_detect(item)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # 尝试从streamlit secrets获取
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            pass

    if not api_key:
        return _mock_llm_detect(item)

    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if not base_url:
        try:
            import streamlit as st
            base_url = st.secrets.get("OPENAI_BASE_URL", "")
        except Exception:
            pass

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    prompt = _build_prompt(item)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个严格的客服回复幻觉检测专家。只输出JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        content = response.choices[0].message.content.strip()
        return _parse_llm_response(item.id, content)
    except Exception:
        return _mock_llm_detect(item)


def _parse_llm_response(item_id: str, content: str) -> DetectionResult:
    """解析LLM返回的JSON"""
    try:
        # 提取JSON块
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content)

        htype = data.get("hallucination_type")
        if htype and htype != "null":
            try:
                htype = HallucinationType(htype)
            except ValueError:
                htype = HallucinationType.FABRICATION
        else:
            htype = None

        return DetectionResult(
            id=item_id,
            is_hallucination=data.get("is_hallucination", False),
            hallucination_type=htype,
            severity=TYPE_SEVERITY_MAP.get(htype) if htype else None,
            detail=data.get("detail", ""),
            confidence=float(data.get("confidence", 0.5)),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return DetectionResult(
            id=item_id,
            is_hallucination=False,
            detail=f"LLM返回解析失败，原始内容: {content[:200]}",
            confidence=0.3,
        )


def _mock_llm_detect(item: ReplyItem) -> DetectionResult:
    """
    Mock模式：内部使用规则引擎结果，但标记来源为LLM-mock。
    这样用户可以在没有API key时也能看到检测效果。
    """
    result = mock_detect(item)
    result.detail = f"[Mock-LLM] {result.detail}"
    result.confidence = min(result.confidence, 0.85)
    return result


def detect_batch_llm(
    items: list[ReplyItem],
    mode: str = "mock",
    model: str = "gpt-3.5-turbo",
) -> list[DetectionResult]:
    """
    批量LLM检测

    Args:
        items: 待检测回复列表
        mode: "real" 或 "mock"
        model: LLM模型名（仅real模式使用）
    """
    results = []
    for item in items:
        if mode == "real":
            results.append(detect_with_llm(item, model))
        else:
            results.append(_mock_llm_detect(item))
    return results
