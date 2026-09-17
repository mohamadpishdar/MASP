import streamlit as st
import pandas as pd
import os
import time

import baseline_core

# ======================================================
# UI
# ======================================================
st.set_page_config(page_title="Audit JSON Extractor", layout="wide")
st.title("Audit Extraction — Single LLM")

if "token_usage" not in st.session_state:
    st.session_state.token_usage = {"input": 0, "output": 0, "calls": 0}
# NOTE: this dict is intentionally re-zeroed inside the "Run Extraction"
# button block below (not just here), so that repeated runs in the same
# Streamlit session never accumulate token/call counts from earlier runs.

# Route baseline_core's retry warnings into this app's own warning box,
# so call_with_retry's behavior is unchanged -- only the destination of
# its warning (a Streamlit warning box instead of stdout) changes.
baseline_core.set_warn_function(st.warning)

# ======================================================
# Sidebar - API Keys
# ======================================================
st.sidebar.header("API Keys")
openrouter_api_key = st.sidebar.text_input(
    "OpenRouter API Key",
    value=baseline_core.OPENROUTER_API_KEY,
    type="password"
)
gemini_api_key = st.sidebar.text_input(
    "Gemini API Key",
    value=baseline_core.GEMINI_API_KEY,
    type="password"
)
anthropic_api_key = st.sidebar.text_input(
    "Claude (Anthropic) API Key",
    value=baseline_core.ANTHROPIC_API_KEY,
    type="password"
)
# Let any key entered/edited directly in the sidebar take effect for
# this run, exactly as before (the sidebar was always the final source
# of truth for a running session, api_keys.json only supplies defaults).
baseline_core.GEMINI_API_KEY = gemini_api_key
baseline_core.OPENROUTER_API_KEY = openrouter_api_key
baseline_core.ANTHROPIC_API_KEY = anthropic_api_key

# ======================================================
# Sidebar - Model Selection
# ======================================================
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Select Model", baseline_core.BASELINE_MODELS)

# ======================================================
# Sidebar - Prompt Template
# ======================================================
st.sidebar.header("Extraction Prompt")
prompt_template = st.sidebar.text_area(
    "Prompt Template",
    baseline_core.DEFAULT_PROMPT_TEMPLATE,
    height=420,
)

# ======================================================
# Sidebar - Few-Shot Training
# ======================================================
st.sidebar.header("Few-Shot Training")
use_training = st.sidebar.checkbox("Use few-shot examples")
train_folder = st.sidebar.text_input("Few-Shot Folder Path", "./shots")

# ======================================================
# MAIN EXECUTION
# ======================================================
if st.sidebar.button("Run Extraction"):

    # ✅ FIX: zero out the session-level counters at the start of THIS
    # run. Without this, token/call counts from any earlier click of
    # "Run Extraction" in the same browser session kept accumulating
    # into this dict forever, silently inflating totals by however many
    # times the button had been pressed before (e.g. one prior test run
    # plus reruns after errors could 5-20x the reported numbers).
    st.session_state.token_usage = {"input": 0, "output": 0, "calls": 0}

    input_folder = "./input"
    few_shots = baseline_core.load_few_shot_examples(train_folder) if use_training else []

    if not os.path.exists(input_folder):
        st.error("پوشه reports یافت نشد")
        results = []
    else:
        pdfs = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]
        progress_bar = st.progress(0)

        def on_file_start(pdf):
            st.write(f"در حال پردازش: {pdf}")

        def on_error(pdf, e):
            st.error(f"خطا در {pdf}: {e}")

        def on_progress(idx, total):
            progress_bar.progress((idx + 1) / total)

        results, usage = baseline_core.run_baseline_batch(
            input_folder=input_folder,
            model_name=model_name,
            prompt_template=prompt_template,
            few_shots=few_shots,
            on_file_start=on_file_start,
            on_error=on_error,
            on_progress=on_progress,
        )
        st.session_state.token_usage = usage

    if results:
        df = pd.DataFrame(results)
        st.dataframe(df)
        os.makedirs("./structured", exist_ok=True)
        timestamp = int(time.time())
        out_path = f"./structured/extraction_{timestamp}.xlsx"
        df.to_excel(out_path, index=False)
        st.success(f"✅ فایل خروجی ذخیره شد: {out_path}")

    # =====================================================
    # TOKEN USAGE SUMMARY (this run only)
    # =====================================================
    st.subheader("📊 Token Usage (this run only)")
    u = st.session_state.token_usage
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Input Tokens", f"{u['input']:,}")
    c2.metric("Total Output Tokens", f"{u['output']:,}")
    c3.metric("Total Tokens", f"{u['input'] + u['output']:,}")
    st.caption(f"Model: {model_name} — API calls (incl. retries): {u['calls']} — Documents processed: {len(results)}")
