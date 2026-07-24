"""
Mock检测引擎 — 基于规则的幻觉检测

检测策略：
1. 能力越界: KB显示"系统未接入/不具备/需人工" + 回复出现"已帮您/已修改/已升级"等动作词
2. 事实编造: KB明确说"无/不支持/暂无"但回复声称"有/支持/可以"
3. 数值歪曲: 提取KB和回复中的关键数值/参数进行对比
4. 安全误导: KB含安全/健康警告但回复忽略或给出相反建议
5. 信息遗漏: KB含重要限定信息但回复未提及
"""

import re
from typing import Optional
from .types import (
    ReplyItem, DetectionResult, HallucinationType, Severity,
    TYPE_SEVERITY_MAP, TYPE_DEFINITIONS,
)


# ============================================================
# 检测规则定义
# ============================================================

def _check_capability_overclaim(item: ReplyItem) -> Optional[DetectionResult]:
    """检测能力越界：系统声称做了它做不到的事"""
    kb = item.knowledge_base
    reply = item.system_reply

    # KB中表明系统无此能力的关键词
    no_capability_patterns = [
        r'未接入.*接口',
        r'不具备.*功能',
        r'无\s*[（(].*未接入',
        r'需人工.*操作',
        r'不可.*告知',
        r'需转人工',
        r'需由.*系统.*自动',
        r'人工客服不可',
        r'不可口头告知',
        r'需由客服系统',
    ]

    # KB含具体操作限制（如"短信发送"），检测回复是否绕过了该限制
    restriction_patterns = [
        (r'以短信方式发送', r'短信'),
        (r'不可口头告知', r'(?:地址|电话|收件人|邮编)'),
    ]

    # 回复中声称执行了操作的关键词
    action_claim_patterns = [
        r'已帮您',
        r'已为您',
        r'我帮您查',
        r'已经帮您',
        r'已修改',
        r'已升级',
        r'已处理',
        r'已经.*处理',
        r'直接发到您账户',
    ]

    kb_no_cap = any(re.search(p, kb) for p in no_capability_patterns)
    reply_claims = any(re.search(p, reply) for p in action_claim_patterns)

    if kb_no_cap and reply_claims:
        return DetectionResult(
            id=item.id,
            is_hallucination=True,
            hallucination_type=HallucinationType.OVERCLAIM,
            severity=Severity.HIGH,
            detail="系统不具备该操作能力（知识库明确说明），但回复声称已执行了相关操作。",
            confidence=0.95,
        )

    # 检测绕过限制：KB要求特定渠道（如短信），回复却直接给了信息
    for kb_restrict, reply_trigger in restriction_patterns:
        if re.search(kb_restrict, kb) and re.search(reply_trigger, reply):
            # 确认回复确实给出了具体信息（而不只是提到"短信"这个词）
            # h07: KB说短信发送，回复却给出了具体地址+邮编+收件人
            has_concrete_info = bool(re.search(
                r'(?:省|市|区|路|号|收|邮编|电话|地址)',
                reply
            ))
            if has_concrete_info:
                return DetectionResult(
                    id=item.id,
                    is_hallucination=True,
                    hallucination_type=HallucinationType.OVERCLAIM,
                    severity=Severity.HIGH,
                    detail="知识库要求通过特定渠道（如短信）提供信息，但回复直接给出了具体内容，绕过了系统限制。",
                    confidence=0.90,
                )

    return None


def _check_fabrication(item: ReplyItem) -> Optional[DetectionResult]:
    """检测事实编造：KB明确否定但回复肯定，或KB无此信息但回复编造"""
    kb = item.knowledge_base
    reply = item.system_reply

    # 模式1：KB明确说"无/不支持/暂无"，回复说"有/支持/可以"
    kb_negative_patterns = [
        (r'无线下门店|纯线上.*无线下', r'线下体验店|门店.*查看|到店'),
        (r'无.*优惠|无.*满\d+减|无.*折扣|无.*学生', r'满\d+减\d+|学生.*优惠|学生.*折扣|学生认证'),
        (r'不支持纸质发票', r'纸质发票'),
        (r'未标注NFC|未.*NFC', r'支持NFC|NFC功能'),
        (r'未提及其他品牌|未.*品牌关联', r'旗下.*品牌|子品牌|共享.*供应链'),
    ]

    for kb_pat, reply_pat in kb_negative_patterns:
        if re.search(kb_pat, kb) and re.search(reply_pat, reply):
            return DetectionResult(
                id=item.id,
                is_hallucination=True,
                hallucination_type=HallucinationType.FABRICATION,
                severity=Severity.HIGH,
                detail=f"知识库表明无此信息（'{kb_pat}'），但回复编造了相关内容（'{reply_pat}'）。",
                confidence=0.90,
            )

    # 模式2：KB明确说"不支持"，回复却说"支持"/"可以"
    kb_no_support = re.findall(r'不支持(.+?)(?:[，。；\n]|$)', kb)
    for item_str in kb_no_support:
        # 检查回复中是否有相反表述
        keyword = item_str.strip().rstrip('等')
        if keyword and len(keyword) > 1:
            if keyword in reply and not re.search(rf'不支持{re.escape(keyword)}', reply):
                # 回复提到了但没说不支持 — 可能是部分编造
                pass  # 这个太模糊，交给其他规则

    return None


def _extract_numbers(text: str) -> dict[str, list[str]]:
    """从文本中提取数值及上下文"""
    patterns = [
        (r'(\d+)\s*天.*?(?:无理由|退货|退换)', '退货天数'),
        (r'蓝牙\s*(\d+\.?\d*)', '蓝牙版本'),
        (r'(\d+)\s*ms', '延迟'),
        (r'(\d+)\s*小时.*?(?:发货|内发货)', '发货时间'),
        (r'保修.*?(\d+)\s*(?:年|个月)', '保修期'),
        (r'满\s*(\d+)\s*减\s*(\d+)', '满减优惠'),
        (r'(\d+)\s*折', '折扣'),
        (r'(\d+)\s*天.*?到货', '到货时间'),
        (r'(\d+)\s*小时.*?联系', '联系时间'),
    ]
    result = {}
    for pat, label in patterns:
        matches = re.findall(pat, text)
        if matches:
            result[label] = matches if isinstance(matches[0], tuple) else matches
    return result


def _extract_material_keywords(text: str) -> set[str]:
    """提取材质关键词"""
    materials = {'头层牛皮', 'PU合成革', '真皮', 'PU', '牛皮', '合成革',
                 'USB-A', 'Type-C', 'USB-C', '顺丰', '中通', '韵达', '圆通'}
    found = set()
    for m in materials:
        if m in text:
            found.add(m)
    return found


def _check_distortion(item: ReplyItem) -> Optional[DetectionResult]:
    """检测数值歪曲：KB和回复中的数值/参数不一致"""
    kb = item.knowledge_base
    reply = item.system_reply

    kb_nums = _extract_numbers(kb)
    reply_nums = _extract_numbers(reply)

    # 比较共同标签下的数值
    distortions = []
    for label in kb_nums:
        if label in reply_nums:
            kb_vals = kb_nums[label]
            reply_vals = reply_nums[label]
            if kb_vals != reply_vals:
                distortions.append(f"{label}: KB={kb_vals} vs Reply={reply_vals}")

    # 材质/快递等关键词对比
    kb_mat = _extract_material_keywords(kb)
    reply_mat = _extract_material_keywords(reply)

    # 定义对立关系
    material_conflicts = [
        ({'PU合成革', 'PU'}, {'头层牛皮', '真皮', '牛皮'}),
        ({'USB-A'}, {'Type-C', 'USB-C'}),
        ({'中通', '韵达', '圆通'}, {'顺丰'}),
    ]

    for kb_set, reply_set in material_conflicts:
        if (kb_mat & kb_set) and (reply_mat & reply_set):
            distortions.append(f"材质/参数冲突: KB含{ kb_mat & kb_set} vs Reply含{reply_mat & reply_set}")

    # 检查单设备 vs 多设备
    if '单设备' in kb and ('多设备' in reply or '同时连接' in reply):
        distortions.append("连接方式: KB=单设备 vs Reply=多设备")

    if distortions:
        return DetectionResult(
            id=item.id,
            is_hallucination=True,
            hallucination_type=HallucinationType.DISTORTION,
            severity=Severity.HIGH,
            detail="; ".join(distortions),
            confidence=0.92,
        )
    return None


def _check_safety_misguidance(item: ReplyItem) -> Optional[DetectionResult]:
    """检测安全误导：KB有安全/健康警告，回复给出相反建议"""
    kb = item.knowledge_base
    reply = item.system_reply

    # KB含安全/健康风险关键词
    safety_keywords = [
        r'孕妇.*咨询医生',
        r'孕妇.*慎用',
        r'孕妇.*禁忌',
        r'哺乳期.*咨询',
        r'建议咨询医生',
        r'不建议.*使用',
        r'过敏.*慎用',
    ]

    # 回复中忽略风险的表述
    ignore_patterns = [
        r'可以放心使用',
        r'孕妇可以',
        r'完全可以',
        r'没有任何问题',
        r'绝对安全',
    ]

    kb_has_warning = any(re.search(p, kb) for p in safety_keywords)
    reply_ignores = any(re.search(p, reply) for p in ignore_patterns)

    if kb_has_warning and reply_ignores:
        return DetectionResult(
            id=item.id,
            is_hallucination=True,
            hallucination_type=HallucinationType.SAFETY,
            severity=Severity.CRITICAL,
            detail="知识库含有安全/健康风险警告，但回复忽略警告并给出相反建议，可能造成人身伤害。",
            confidence=0.98,
        )
    return None


def _check_omission(item: ReplyItem) -> Optional[DetectionResult]:
    """检测信息遗漏：KB有关键限定信息但回复未提及"""
    kb = item.knowledge_base
    reply = item.system_reply

    # 模式：KB含定量限定（百分比、多数/少数），回复给出绝对化表述
    kb_has_qualifier = bool(re.search(r'\d+%的用户|部分用户|约\d+%', kb))
    reply_absolute = bool(re.search(r'尺码标准|不偏大也不偏小|完全不|绝对不|一定', reply))

    # KB含明确结论但回复给出相反结论
    if kb_has_qualifier and reply_absolute:
        return DetectionResult(
            id=item.id,
            is_hallucination=True,
            hallucination_type=HallucinationType.OMISSION,
            severity=Severity.LOW,
            detail="知识库含有重要限定信息（如用户反馈统计数据），回复使用了绝对化表述且遗漏了关键限定。",
            confidence=0.75,
        )

    # KB含"温馨提示"或注意事项但回复未提及
    if '温馨提示' in kb or '注意事项' in kb:
        # 检查回复是否覆盖了关键注意点
        kb_points = re.findall(r'[：:]\s*(.+?)(?:[。；\n]|$)', kb)
        reply_covered = sum(1 for p in kb_points if len(p) > 5 and p[:6] in reply)
        if reply_covered < len(kb_points) * 0.5 and len(kb_points) >= 2:
            # 这是比较激进的检测，可能误报，降低置信度
            pass  # 暂不触发，避免过多误报

    return None


def _check_fabrication_general(item: ReplyItem) -> Optional[DetectionResult]:
    """通用事实编造检测：KB明确否定 + 回复肯定"""
    kb = item.knowledge_base
    reply = item.system_reply

    # KB中的否定表述
    kb_no_items = []
    # 匹配 "无XXX" 或 "暂无XXX"或 "不支持XXX"
    for m in re.finditer(r'(?:无|暂无|不支持)([^，。；\n]{1,30})', kb):
        kb_no_items.append(m.group().strip())

    # 回复中的肯定表述
    reply_yes_patterns = [
        r'有的[，。,\.\s]',
        r'支持的[，。,\.\s]',
        r'可以的[，。,\.\s]',
        r'可以.*使用',
    ]

    # 如果KB至少有一个明确的否定，且回复有肯定模式
    if kb_no_items and any(re.search(p, reply) for p in reply_yes_patterns):
        for no_item in kb_no_items:
            # 提取否定对象的关键词
            keyword = re.sub(r'^(?:无|暂无|不支持)', '', no_item).strip()
            if keyword and len(keyword) >= 2:
                # 检查回复中是否有该关键词且上下文是肯定的
                if keyword in reply:
                    # 检查回复中是否真的肯定了这个关键词
                    context_match = re.search(
                        rf'(?:支持|有|可以|提供).*?{re.escape(keyword)}|{re.escape(keyword)}.*?(?:支持|有|可以|提供)',
                        reply
                    )
                    if context_match:
                        return DetectionResult(
                            id=item.id,
                            is_hallucination=True,
                            hallucination_type=HallucinationType.FABRICATION,
                            severity=Severity.HIGH,
                            detail=f"知识库表明'{no_item}'，但回复给出了相反信息。",
                            confidence=0.88,
                        )

    return None


# ============================================================
# 主检测函数
# ============================================================

def detect(item: ReplyItem) -> DetectionResult:
    """
    对单条回复执行幻觉检测。
    按优先级依次运行各检测器，返回第一个命中的结果。
    如果都未命中，返回"非幻觉"。
    """
    detectors = [
        (_check_safety_misguidance, "安全误导"),       # 最高优先级
        (_check_capability_overclaim, "能力越界"),
        (_check_distortion, "数值歪曲"),
        (_check_fabrication, "事实编造"),
        (_check_fabrication_general, "事实编造(通用)"),
        (_check_omission, "信息遗漏"),                 # 最低优先级
    ]

    for detector_fn, _name in detectors:
        result = detector_fn(item)
        if result is not None:
            return result

    # 无幻觉
    return DetectionResult(
        id=item.id,
        is_hallucination=False,
        hallucination_type=None,
        severity=None,
        detail="回复与知识库一致，未检测到幻觉。",
        confidence=0.85,
    )


def detect_batch(items: list[ReplyItem]) -> list[DetectionResult]:
    """批量检测"""
    return [detect(item) for item in items]
