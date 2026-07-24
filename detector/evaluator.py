"""
评估模块 — 与ground_truth对比，计算检出率、漏检、误报
"""

from collections import defaultdict
from .types import (
    DetectionResult, GroundTruthItem, EvaluationReport,
    HallucinationType,
)


def evaluate(
    detections: list[DetectionResult],
    ground_truths: list[GroundTruthItem],
) -> EvaluationReport:
    """
    对比检测结果与人工标注，生成评估报告。

    Returns:
        EvaluationReport: 包含精确率、召回率、F1、误报/漏检明细
    """
    # 建立索引
    gt_map = {gt.id: gt for gt in ground_truths}
    det_map = {d.id: d for d in detections}

    report = EvaluationReport()
    report.total = len(ground_truths)

    for gt in ground_truths:
        det = det_map.get(gt.id)
        if det is None:
            # 检测结果缺失，视为漏检
            if gt.is_hallucination:
                report.false_negative += 1
                report.false_negatives_detail.append({
                    "id": gt.id,
                    "gt_type": gt.hallucination_type,
                    "det_type": "未检测",
                    "gt_detail": gt.detail,
                })
            continue

        if gt.is_hallucination and det.is_hallucination:
            # 正确检出
            report.true_positive += 1
            # 检查类型是否一致
            gt_type = gt.hallucination_type
            det_type = det.hallucination_type.value if det.hallucination_type else "unknown"
            key = f"{gt_type} → {det_type}"
            report.type_confusion[key] = report.type_confusion.get(key, 0) + 1

        elif not gt.is_hallucination and not det.is_hallucination:
            # 正确排除
            report.true_negative += 1

        elif not gt.is_hallucination and det.is_hallucination:
            # 误报：实际非幻觉，检测为幻觉
            report.false_positive += 1
            report.false_positives_detail.append({
                "id": gt.id,
                "gt_type": "非幻觉",
                "det_type": det.hallucination_type.value if det.hallucination_type else "unknown",
                "gt_detail": gt.detail,
                "det_detail": det.detail,
            })

        elif gt.is_hallucination and not det.is_hallucination:
            # 漏检：实际幻觉，检测为非幻觉
            report.false_negative += 1
            report.false_negatives_detail.append({
                "id": gt.id,
                "gt_type": gt.hallucination_type,
                "det_type": "未检出",
                "gt_detail": gt.detail,
                "det_detail": det.detail,
            })

    # 计算指标
    tp, fp, fn = report.true_positive, report.false_positive, report.false_negative

    if tp + fp > 0:
        report.precision = tp / (tp + fp)
    if tp + fn > 0:
        report.recall = tp / (tp + fn)
    if report.precision + report.recall > 0:
        report.f1 = 2 * report.precision * report.recall / (report.precision + report.recall)

    return report


def analyze_misclassifications(report: EvaluationReport) -> str:
    """分析误判原因，生成文字说明"""
    lines = []

    if report.false_positives_detail:
        lines.append(f"## 误报（共{report.false_positive}条）")
        lines.append("以下case实际无幻觉，但被检测为幻觉：")
        for fp in report.false_positives_detail:
            lines.append(f"- **{fp['id']}**: 检测为「{fp['det_type']}」，实际为非幻觉")
            lines.append(f"  - GT说明: {fp['gt_detail']}")
            lines.append(f"  - 检测说明: {fp['det_detail']}")
            lines.append(f"  - 误判原因: 规则过于宽泛或关键词匹配不够精确，需要优化匹配条件。")
        lines.append("")

    if report.false_negatives_detail:
        lines.append(f"## 漏检（共{report.false_negative}条）")
        lines.append("以下case实际有幻觉，但未被检测到：")
        for fn in report.false_negatives_detail:
            lines.append(f"- **{fn['id']}**: GT类型「{fn['gt_type']}」，检测为「{fn['det_type']}」")
            lines.append(f"  - GT说明: {fn['gt_detail']}")
            lines.append(f"  - 漏检原因: 该幻觉类型的检测规则覆盖不足，或表述较为隐晦。")
        lines.append("")

    if report.type_confusion:
        lines.append("## 类型混淆矩阵")
        lines.append("| Ground Truth → 检测结果 | 数量 |")
        lines.append("|---|---|")
        for key, count in sorted(report.type_confusion.items(), key=lambda x: -x[1]):
            lines.append(f"| {key} | {count} |")

    # 总结容易误判的case类型
    lines.append("")
    lines.append("## 容易误判的Case类型及原因")
    lines.append("")
    lines.append("1. **信息遗漏 vs 非幻觉**: 部分正确但不够全面的回复，边界模糊。")
    lines.append("   规则引擎难以区分「遗漏」和「表述角度不同」，容易误报。")
    lines.append("")
    lines.append("2. **部分正确部分错误**: 如h04（电子发票正确但纸质发票错误），")
    lines.append("   检测器可能只检测到「有正确部分」而漏检错误部分。")
    lines.append("")
    lines.append("3. **隐晦的数值歪曲**: 如「大多数地区2-3天」vs「一般3-5天」，")
    lines.append("   语义相近但数值不同，纯规则方法难以精确区分。")
    lines.append("")
    lines.append("4. **上下文依赖的安全风险**: 如成分表的专业术语（视黄醇），")
    lines.append("   规则引擎需要领域知识才能判断安全性。")

    return "\n".join(lines)
