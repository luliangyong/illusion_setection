"""
幻觉检测工具 — Streamlit 交互界面
"""

import sys
from pathlib import Path

# 确保项目根目录在path中
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from detector.types import (
    HallucinationType, Severity, TYPE_DEFINITIONS, DetectionResult,
)
from detector.loader import load_replies, load_ground_truth
from detector.mock_engine import detect_batch as mock_detect_batch
from detector.llm_engine import detect_batch_llm
from detector.evaluator import evaluate, analyze_misclassifications

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="客服回复幻觉检测工具",
    page_icon=":material/detection_and_zone:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 样式
# ============================================================
st.markdown("""
<style>
    .severity-critical { color: #dc2626; font-weight: bold; }
    .severity-high { color: #ea580c; font-weight: bold; }
    .severity-medium { color: #ca8a04; font-weight: bold; }
    .severity-low { color: #16a34a; }
    .metric-card { background: #f8fafc; border-radius: 12px; padding: 20px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 缓存数据加载
# ============================================================
@st.cache_data
def load_data():
    return load_replies(), load_ground_truth()


def _resolve_api_config():
    """解析API配置：secrets.toml > 环境变量 > 手动输入"""
    import os
    config = {
        "api_key": "",
        "base_url": "",
        "model": "gpt-3.5-turbo",
    }
    # 尝试从 secrets.toml 读取
    try:
        config["api_key"] = st.secrets.get("OPENAI_API_KEY", "")
        config["base_url"] = st.secrets.get("OPENAI_BASE_URL", "")
        config["model"] = st.secrets.get("OPENAI_MODEL", "gpt-3.5-turbo")
    except Exception:
        pass
    # 环境变量覆盖
    if not config["api_key"]:
        config["api_key"] = os.environ.get("OPENAI_API_KEY", "")
    if not config["base_url"]:
        config["base_url"] = os.environ.get("OPENAI_BASE_URL", "")
    return config


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title(":material/settings: 检测配置")

    # —— 检测引擎选择 ——
    detection_mode = st.radio(
        "检测引擎",
        options=["mock", "real"],
        format_func=lambda x: "Mock 规则引擎" if x == "mock" else "LLM API 调用",
        help="Mock: 本地规则匹配，无需联网 | LLM: 调用大模型API",
    )

    # —— LLM 配置（始终可见） ——
    st.divider()
    st.subheader(":material/smart_toy: LLM 配置")

    saved_config = _resolve_api_config()

    # API Key
    api_key = st.text_input(
        "API Key",
        type="password",
        value=saved_config["api_key"],
        placeholder="sk-... (或配置 .streamlit/secrets.toml)",
        help="OpenAI兼容的API Key。也可写入 .streamlit/secrets.toml 持久化",
    )
    if api_key:
        import os
        os.environ["OPENAI_API_KEY"] = api_key

    # Base URL（支持国内模型代理）
    preset_base = st.session_state.get("_preset_base", "")
    base_url = st.text_input(
        "Base URL（可选）",
        value=preset_base or saved_config["base_url"],
        placeholder="https://api.openai.com/v1",
        help="OpenAI兼容接口地址。国内模型填代理地址，留空使用默认",
    )
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    # 模型选择
    preset_model = st.session_state.get("_preset_model", "")
    model_options = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    default_model = preset_model or saved_config["model"]
    if default_model not in model_options:
        model_options.insert(0, default_model)
    model = st.selectbox(
        "模型",
        options=model_options,
        index=model_options.index(default_model) if default_model in model_options else 0,
        help="选择调用的大模型",
    )

    # 快速预设
    with st.expander(":material/tune: 国内模型快捷预设"):
        st.caption("点击可自动填入 Base URL 和推荐模型")
        col1, col2, col3 = st.columns(3)
        if col1.button("DeepSeek", use_container_width=True):
            st.session_state._preset_base = "https://api.deepseek.com/v1"
            st.session_state._preset_model = "deepseek-chat"
            st.rerun()
        if col2.button("通义千问", use_container_width=True):
            st.session_state._preset_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            st.session_state._preset_model = "qwen-turbo"
            st.rerun()
        if col3.button("Moonshot", use_container_width=True):
            st.session_state._preset_base = "https://api.moonshot.cn/v1"
            st.session_state._preset_model = "moonshot-v1-8k"
            st.rerun()

    # 连接状态
    if detection_mode == "real":
        if not api_key and not saved_config["api_key"]:
            st.warning(":material/warning: 未设置 API Key，将自动回退到 Mock 模式")
        else:
            st.success(f":material/check_circle: 就绪 — {model}" + (f" @ {base_url}" if base_url else ""))

    st.divider()
    st.caption("幻觉检测工具 v1.0 | Mock + LLM 双引擎")

# ============================================================
# 主区域
# ============================================================
st.title(":material/detection_and_zone: 客服回复幻觉检测工具")
st.caption("对20条客服回复进行自动化幻觉检测，并与人工标注对比验证")

# 加载数据
replies, ground_truths = load_data()

# 执行检测
with st.spinner("正在执行检测..."):
    if detection_mode == "mock":
        results = mock_detect_batch(replies)
        engine_label = "Mock 规则引擎"
    else:
        results = detect_batch_llm(replies, mode="real", model=model)
        engine_label = f"LLM ({model})"

# 评估
report = evaluate(results, ground_truths)

# ============================================================
# 顶部 KPI 卡片
# ============================================================
st.subheader(":material/overview: 检测概览", divider="gray")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("总样本", f"{report.total}", border=True)
col2.metric("正确检出 (TP)", f"{report.true_positive}", border=True)
col3.metric("正确排除 (TN)", f"{report.true_negative}", border=True)
col4.metric("误报 (FP)", f"{report.false_positive}",
           delta=f"-{report.false_positive}" if report.false_positive > 0 else None,
           delta_color="inverse", border=True)
col5.metric("漏检 (FN)", f"{report.false_negative}",
           delta=f"-{report.false_negative}" if report.false_negative > 0 else None,
           delta_color="inverse", border=True)
col6.metric("F1 分数", f"{report.f1:.1%}", border=True)

# 精确率/召回率进度条
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("精确率 (Precision)", f"{report.precision:.1%}", border=True)
    st.progress(report.precision, text=f"TP/(TP+FP) = {report.true_positive}/{report.true_positive + report.false_positive}")
with col_b:
    st.metric("召回率 (Recall)", f"{report.recall:.1%}", border=True)
    st.progress(report.recall, text=f"TP/(TP+FN) = {report.true_positive}/{report.true_positive + report.false_negative}")
with col_c:
    engine_color = "green" if report.f1 >= 0.9 else ("orange" if report.f1 >= 0.7 else "red")
    st.metric("检测引擎", engine_label, border=True)

# ============================================================
# Tab 页
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    ":material/list: 检测结果",
    ":material/compare: 对比分析",
    ":material/error: 误判分析",
    ":material/info: 分类体系",
])

# ============================================================
# Tab 1: 检测结果
# ============================================================
with tab1:
    st.subheader("逐条检测结果", divider="gray")

    # 构建结果表
    gt_map = {g.id: g for g in ground_truths}
    reply_map = {r.id: r for r in replies}  # 关联原始数据
    rows = []
    for r in results:
        gt = gt_map[r.id]
        reply = reply_map[r.id]
        match_icon = ":material/check:" if r.is_hallucination == gt.is_hallucination else ":material/close:"

        # 严重程度标签颜色
        sev_color = {
            Severity.CRITICAL: "#dc2626",
            Severity.HIGH: "#ea580c",
            Severity.MEDIUM: "#ca8a04",
            Severity.LOW: "#16a34a",
        }

        rows.append({
            "": match_icon,
            "ID": r.id,
            "用户问题": reply.user_question[:40] + "..." if len(reply.user_question) > 40 else reply.user_question,
            "检测结果": "⚠️ 幻觉" if r.is_hallucination else "✅ 正常",
            "幻觉类型": r.hallucination_type.value if r.hallucination_type else "-",
            "严重程度": r.severity.value if r.severity else "-",
            "GT标注": "幻觉" if gt.is_hallucination else "正常",
            "GT类型": gt.hallucination_type or "-",
            "详情": r.detail[:80] + "..." if len(r.detail) > 80 else r.detail,
        })

    df = pd.DataFrame(rows)

    # 用 st.dataframe 展示，带列配置
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "": st.column_config.TextColumn("匹配", width="small"),
            "ID": st.column_config.TextColumn("ID", width="small"),
            "用户问题": st.column_config.TextColumn("用户问题", width="medium"),
            "检测结果": st.column_config.TextColumn("检测结果", width="small"),
            "幻觉类型": st.column_config.TextColumn("幻觉类型", width="small"),
            "严重程度": st.column_config.TextColumn("严重程度", width="small"),
            "GT标注": st.column_config.TextColumn("GT标注", width="small"),
            "GT类型": st.column_config.TextColumn("GT类型", width="small"),
            "详情": st.column_config.TextColumn("详情", width="large"),
        },
    )

    # 展开查看每条详情
    st.subheader("详细展开", divider="gray")
    for i, r in enumerate(results):
        gt = gt_map[r.id]
        reply = reply_map[r.id]
        with st.expander(
            f"{'⚠️' if r.is_hallucination else '✅'} {r.id} — "
            f"{r.hallucination_type.value if r.hallucination_type else '正常'} "
            f"| GT: {gt.hallucination_type or '正常'}"
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**用户问题:** {reply.user_question}")
                st.markdown(f"**系统回复:** {reply.system_reply}")
            with col2:
                st.markdown(f"**知识库:** {reply.knowledge_base}")
                st.markdown(f"**检测详情:** {r.detail}")
                st.markdown(f"**置信度:** {r.confidence:.0%}")
                if r.severity:
                    st.markdown(f"**严重程度:** :red[{r.severity.value}]" if r.severity == Severity.CRITICAL
                                else f"**严重程度:** :orange[{r.severity.value}]" if r.severity == Severity.HIGH
                                else f"**严重程度:** {r.severity.value}")

    # 幻觉类型分布
    st.subheader("幻觉类型分布", divider="gray")
    type_counts = {}
    for r in results:
        if r.is_hallucination and r.hallucination_type:
            t = r.hallucination_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

    if type_counts:
        col_chart1, col_chart2 = st.columns([1, 1])
        with col_chart1:
            fig = px.pie(
                names=list(type_counts.keys()),
                values=list(type_counts.values()),
                title="检测出的幻觉类型分布",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_traces(textposition='inside', textinfo='label+value')
            st.plotly_chart(fig, use_container_width=True)

        with col_chart2:
            # 严重程度分布
            sev_counts = {}
            for r in results:
                if r.severity:
                    s = r.severity.value
                    sev_counts[s] = sev_counts.get(s, 0) + 1
            fig2 = px.bar(
                x=list(sev_counts.keys()),
                y=list(sev_counts.values()),
                title="严重程度分布",
                labels={"x": "严重程度", "y": "数量"},
                color=list(sev_counts.keys()),
                color_discrete_map={
                    "严重": "#dc2626",
                    "高": "#ea580c",
                    "中": "#ca8a04",
                    "低": "#16a34a",
                },
            )
            st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# Tab 2: 对比分析
# ============================================================
with tab2:
    st.subheader("检测结果 vs Ground Truth", divider="gray")

    # 混淆矩阵
    cm_data = [
        ["正确检出幻觉", report.false_positive, report.true_positive],
        ["漏检 (FN)", "正确排除 (TN)", report.false_negative],
        ["", report.true_negative, ""],
    ]

    # 构建热力图数据
    labels = ["检测=正常", "检测=幻觉"]
    parents = ["GT=正常", "GT=幻觉"]

    z = [[report.true_negative, report.false_positive],
         [report.false_negative, report.true_positive]]

    fig_cm = px.imshow(
        z,
        x=labels,
        y=parents,
        text_auto=True,
        title="混淆矩阵",
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig_cm.update_xaxes(side="top")
    st.plotly_chart(fig_cm, use_container_width=True)

    # 类型对比
    st.subheader("类型级别对比", divider="gray")

    # 按GT类型统计检测结果
    gt_type_stats = {}
    for gt in ground_truths:
        det = next((r for r in results if r.id == gt.id), None)
        if det is None:
            continue
        gt_type = gt.hallucination_type or "非幻觉"
        if gt_type not in gt_type_stats:
            gt_type_stats[gt_type] = {"total": 0, "correct": 0, "det_type": None}
        gt_type_stats[gt_type]["total"] += 1
        if det.is_hallucination == gt.is_hallucination:
            gt_type_stats[gt_type]["correct"] += 1
        if det.is_hallucination and det.hallucination_type:
            gt_type_stats[gt_type]["det_type"] = det.hallucination_type.value

    type_rows = []
    for gt_type, stats in gt_type_stats.items():
        type_rows.append({
            "GT类型": gt_type,
            "数量": stats["total"],
            "正确检出": stats["correct"],
            "准确率": f"{stats['correct']/stats['total']:.0%}",
            "检测为": stats["det_type"] or "-",
        })

    st.dataframe(
        pd.DataFrame(type_rows),
        use_container_width=True,
        hide_index=True,
    )

    # 类型混淆详情
    if report.type_confusion:
        st.subheader("类型映射关系", divider="gray")
        confusion_rows = []
        for key, count in sorted(report.type_confusion.items(), key=lambda x: -x[1]):
            confusion_rows.append({"GT → 检测": key, "数量": count})
        st.dataframe(
            pd.DataFrame(confusion_rows),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# Tab 3: 误判分析
# ============================================================
with tab3:
    st.subheader("误判分析", divider="gray")

    analysis = analyze_misclassifications(report)
    st.markdown(analysis)

    # 误报详情
    if report.false_positives_detail:
        st.subheader("误报详情", divider="red")
        for fp in report.false_positives_detail:
            st.warning(f"**{fp['id']}**: GT=非幻觉 → 检测={fp['det_type']}")
            st.caption(f"检测说明: {fp.get('det_detail', 'N/A')}")

    # 漏检详情
    if report.false_negatives_detail:
        st.subheader("漏检详情", divider="orange")
        for fn in report.false_negatives_detail:
            st.error(f"**{fn['id']}**: GT={fn['gt_type']} → 未检出")
            st.caption(f"GT说明: {fn['gt_detail']}")
            st.caption(f"检测说明: {fn.get('det_detail', 'N/A')}")

    if not report.false_positives_detail and not report.false_negatives_detail:
        st.success(":material/celebration: 完美！零误报、零漏检，检测结果与人工标注完全一致。")

# ============================================================
# Tab 4: 分类体系
# ============================================================
with tab4:
    st.subheader("幻觉分类体系", divider="gray")
    st.markdown("""
    本工具定义了 **5类幻觉**，按严重程度分为 4 级。分类体系设计原则：
    - **区分度高**: 每类有明确的边界，不易混淆
    - **可操作**: 每类都能对应到具体的检测规则
    - **覆盖完整**: 覆盖了客服场景中常见的所有幻觉类型
    """)

    for htype, info in TYPE_DEFINITIONS.items():
        sev = info["severity"]
        sev_icon = {
            "严重": ":material/error:",
            "高": ":material/warning:",
            "中": ":material/info:",
            "低": ":material/check_circle:",
        }.get(sev.split(" ")[0], ":material/help:")

        with st.container(border=True):
            st.markdown(f"### {sev_icon} {htype.value}")
            st.markdown(f"**严重程度:** {sev}")
            st.markdown(f"**定义:** {info['definition']}")
            st.markdown("**示例:**")
            for ex in info["examples"]:
                st.markdown(f"- {ex}")

    # 严重程度分级说明
    st.subheader("严重程度分级", divider="gray")
    sev_data = [
        {"级别": "严重 (Critical)", "颜色": "🔴", "说明": "涉及人身安全/健康风险，可能造成身体伤害", "示例": "孕妇禁忌成分、药物过敏风险"},
        {"级别": "高 (High)", "颜色": "🟠", "说明": "直接误导消费决策，造成经济损失或信任危机", "示例": "材质造假、价格编造、假装有操作能力"},
        {"级别": "中 (Medium)", "颜色": "🟡", "说明": "部分误导但后果可控", "示例": "部分信息正确但关键细节错误"},
        {"级别": "低 (Low)", "颜色": "🟢", "说明": "信息不完整，但不直接造成错误决策", "示例": "遗漏统计信息、未提及使用注意事项"},
    ]
    st.dataframe(
        pd.DataFrame(sev_data),
        use_container_width=True,
        hide_index=True,
        column_config={
            "说明": st.column_config.TextColumn("说明", width="large"),
            "示例": st.column_config.TextColumn("示例", width="medium"),
        },
    )

# ============================================================
# 底部
# ============================================================
st.divider()
st.caption(f"检测引擎: {engine_label} | 检测时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
