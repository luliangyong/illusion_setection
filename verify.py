"""最终验证脚本"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from detector.loader import load_replies, load_ground_truth
from detector.mock_engine import detect_batch
from detector.evaluator import evaluate, analyze_misclassifications

replies = load_replies()
gts = load_ground_truth()
results = detect_batch(replies)
report = evaluate(results, gts)

print('=' * 60)
print('  幻觉检测 — 最终验证报告')
print('=' * 60)
print()
print(f'  检测引擎: Mock 规则引擎')
print(f'  样本总数: {report.total}')
print(f'  幻觉样本: {sum(1 for g in gts if g.is_hallucination)}')
print(f'  正常样本: {sum(1 for g in gts if not g.is_hallucination)}')
print()
print(f'  True Positive:  {report.true_positive}')
print(f'  True Negative:  {report.true_negative}')
print(f'  False Positive: {report.false_positive}')
print(f'  False Negative: {report.false_negative}')
print()
print(f'  精确率 (Precision): {report.precision:.2%}')
print(f'  召回率 (Recall):    {report.recall:.2%}')
print(f'  F1 分数:            {report.f1:.2%}')
print()
print('=' * 60)
print('  逐条对比')
print('=' * 60)

gt_map = {g.id: g for g in gts}
correct = 0
for r in results:
    gt = gt_map[r.id]
    match = r.is_hallucination == gt.is_hallucination
    if match:
        correct += 1
    status = 'OK' if match else 'MISS'
    det_label = 'HALLU' if r.is_hallucination else 'NORMAL'
    gt_label = 'HALLU' if gt.is_hallucination else 'NORMAL'
    det_type = r.hallucination_type.value if r.hallucination_type else '-'
    gt_type = gt.hallucination_type or '-'
    print(f'  {status:4s} {r.id}: Det={det_label:6s}/{det_type:10s}  |  GT={gt_label:6s}/{gt_type}')

print()
print(f'  正确率: {correct}/{report.total} = {correct/report.total:.0%}')
print()
print(analyze_misclassifications(report))
