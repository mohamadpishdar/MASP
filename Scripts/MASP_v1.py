import streamlit as st
import pandas as pd
import os
import time

import masp_core

fitz_display_errors_silenced = True  # masp_core already silences PyMuPDF errors on import

# =====================================================
# STREAMLIT INIT
# =====================================================
st.set_page_config(page_title="MASP Batch Processor", layout="wide")
st.title(" MASP – High‑Precision Extraction")

if "logs" not in st.session_state:
    st.session_state.logs = []

def log(msg):
    t = time.strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{t}] {msg}")

# Route masp_core's internal log() calls into this app's own session log,
# so every "✅ FIX" comment and behavior inside masp_core.py logs exactly
# where it always did -- only the destination (Streamlit session state
# instead of a plain list) changes.
masp_core.set_log_function(log)

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    target_fields = st.text_input("Fields", "filename, project_identity, auditor, high_vulnerability_count")
    mission_logic = st.text_area("Logic", "Extract audit metadata strictly from evidence.")

    all_models = masp_core.ALL_MODELS

    # Agents: 1) Gemini  2) OpenRouter (replaces Groq)  3) Claude  |  Judge: Claude (user-selectable)
    m1 = st.selectbox("Miner 1 (Gemini)", all_models, index=all_models.index("gemini/gemini-2.0-flash-001"))
    m2 = st.selectbox("Miner 2 (OpenRouter)", all_models, index=all_models.index("openrouter/deepseek/deepseek-chat"))
    m3 = st.selectbox("Miner 3 (Claude)", all_models, index=all_models.index("claude/claude-sonnet-5"))
    m_judge = st.selectbox("Judge", all_models, index=all_models.index("claude/claude-sonnet-5"))

    input_dir, output_dir = "./input", "./structured"
    max_iter = st.slider("Max Iterations", 1, 3, 2)

get_lm = masp_core.get_lm

# =====================================================
# BATCH EXECUTION
# =====================================================
if st.button(" Start Processing All Files"):
    lms = {"agent1": get_lm(m1), "agent2": get_lm(m2), "agent3": get_lm(m3), "judge": get_lm(m_judge)}

    demos = masp_core.build_demos("./shots")
    base_task = f"Extract JSON with fields: {target_fields}\nLogic: {mission_logic}\nExamples:\n{demos}"

    if os.path.exists(input_dir):
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_file_start(idx, total, filename):
            status_text.text(f"در حال پردازش {idx + 1}/{total}: {filename}")

        def on_file_done(idx, total):
            progress_bar.progress((idx + 1) / total)

        results = masp_core.run_masp_batch(
            input_dir=input_dir,
            lms=lms,
            base_task=base_task,
            target_fields=target_fields,
            max_iter=max_iter,
            on_file_start=on_file_start,
            on_file_done=on_file_done,
        )
        status_text.text(f"✅ پردازش {len(files)} سند کامل شد.")

        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)
            os.makedirs(output_dir, exist_ok=True)
            df.to_excel(os.path.join(output_dir, "full_audit_results.xlsx"), index=False)
            st.success(f"Processed {len(results)} reports.")

        # =====================================================
        # TOKEN USAGE SUMMARY
        # =====================================================
        st.subheader("📊 Token Usage")
        role_labels = {"agent1": "Miner 1", "agent2": "Miner 2", "agent3": "Miner 3", "judge": "Judge"}
        model_choice = {"agent1": m1, "agent2": m2, "agent3": m3, "judge": m_judge}
        usage_rows = []
        grand_input, grand_output = 0, 0
        for k, lm in lms.items():
            u = getattr(lm, "usage", {"input": 0, "output": 0, "calls": 0})
            grand_input += u["input"]
            grand_output += u["output"]
            usage_rows.append({
                "Role": role_labels.get(k, k),
                "Model": model_choice.get(k, ""),
                "Calls": u["calls"],
                "Input Tokens": u["input"],
                "Output Tokens": u["output"],
                "Total Tokens": u["input"] + u["output"],
            })

        usage_df = pd.DataFrame(usage_rows)
        st.dataframe(usage_df, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Input Tokens", f"{grand_input:,}")
        c2.metric("Total Output Tokens", f"{grand_output:,}")
        c3.metric("Total Tokens", f"{grand_input + grand_output:,}")

        log(f"📊 Token usage — input: {grand_input}, output: {grand_output}, total: {grand_input + grand_output}")
    else:
        st.error("Input directory not found.")

if st.session_state.logs:
    for l in reversed(st.session_state.logs):
        st.caption(l)
