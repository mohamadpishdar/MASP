import streamlit as st
import fitz  # PyMuPDF
import json
import os
import pandas as pd
import time
import re
from openai import OpenAI
import google.generativeai as genai
import anthropic
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

# ======================================================
# Sidebar - API Keys
# ======================================================
def load_api_keys(path="api_keys.json"):
    """
    Loads API keys from a local JSON file instead of hardcoding them in
    the script. If the file doesn't exist yet, it is created with empty
    values so the user can fill them in themselves.
    """
    defaults = {"GEMINI_API_KEY": "", "OPENROUTER_API_KEY": "", "ANTHROPIC_API_KEY": ""}
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as jf:
            json.dump(defaults, jf, indent=2)
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as jf:
            loaded = json.load(jf)
        defaults.update(loaded)
        return defaults
    except Exception:
        return defaults

_api_keys = load_api_keys()

st.sidebar.header("API Keys")
openrouter_api_key = st.sidebar.text_input(
    "OpenRouter API Key",
    value=_api_keys.get("OPENROUTER_API_KEY", ""),
    type="password"
)
gemini_api_key = st.sidebar.text_input(
    "Gemini API Key",
    value=_api_keys.get("GEMINI_API_KEY", ""),
    type="password"
)
anthropic_api_key = st.sidebar.text_input(
    "Claude (Anthropic) API Key",
    value=_api_keys.get("ANTHROPIC_API_KEY", ""),
    type="password"
)



# ======================================================
# Sidebar - Model Selection
# ======================================================

st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.0-flash",
        "gemini/gemini-1.5-flash",
        "openrouter/deepseek/deepseek-chat",
        "openrouter/qwen/qwen-2.5-72b-instruct",
        "openrouter/meta-llama/llama-3.3-70b-instruct",
        "openrouter/mistralai/mixtral-8x7b-instruct",

        "claude/claude-sonnet-4-6",
        "claude/claude-opus-4-8",
    ],
)

# ======================================================
# Sidebar - Prompt Template
# ======================================================
st.sidebar.header("Extraction Prompt")
prompt_template = st.sidebar.text_area(
    "Prompt Template",
    """You are given several CORRECT extraction examples above.
You MUST strictly follow the same extraction logic, field meanings,
null-handling behavior, and formatting shown in the examples.

If there is any ambiguity, follow the examples exactly.

Now extract the following fields from the document.

FIELDS:
filename, project_identity, auditor,
high_vulnerability_count, medium_vulnerability_count, low_vulnerability_count,
commit_hash, year, fixed_items, first_vulnerability_title

DOCUMENT:
{{DOCUMENT}}

OUTPUT FORMAT (STRICT):
{
  "filename": null,
  "project_identity": null,
  "auditor": null,
  "high_vulnerability_count": null,
  "medium_vulnerability_count": null,
  "low_vulnerability_count": null,
  "commit_hash": null,
  "year": null,
  "fixed_items": null,
  "first_vulnerability_title": null
}

RULES:
- Output ONLY valid JSON
- Do not explain
- Do not add or remove fields
- Use null if missing
""",
    height=420,
)

# ======================================================
# Sidebar - Few-Shot Training
# ======================================================
st.sidebar.header("Few-Shot Training")
use_training = st.sidebar.checkbox("Use few-shot examples")
train_folder = st.sidebar.text_input("Few-Shot Folder Path", "./shots")

# ======================================================
# Utilities
# ======================================================

# ✅ Reduce token usage
def read_pdf(path, max_pages=10):
    try:
        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc[:max_pages])
    except:
        return ""

# ✅ FIX: matches MASP's context handling (app4.py's MASP_PROMPT_CHARS /
# _truncate). The baseline previously read only 3 pages and hard-cut the
# text at the first 1500 characters, meaning it never saw the findings
# table, severity breakdown, or appendix on most real audit reports
# (see Section III / the motivating example). MASP reads up to 10 pages
# and truncates only if the text exceeds 60000 characters, keeping the
# first 70% and last 30% so an appendix near the end of the document is
# not silently dropped. This was a confound between the baseline and
# MASP configurations, not a deliberate part of either method, so both
# now use the same document-reading budget.
MAX_PROMPT_CHARS = 60000

def truncate_document(text, limit=MAX_PROMPT_CHARS):
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.7)]
    tail = text[-int(limit * 0.3):]
    return head + "\n...[truncated for size limit]...\n" + tail

def load_few_shot_examples(folder):
    examples = []
    if not os.path.exists(folder):
        return examples

    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(".pdf"):
            continue

        base = f[:-4]
        pdf_path = os.path.join(folder, f)
        json_path = os.path.join(folder, base + ".json")

        if not os.path.exists(json_path):
            continue

        document = truncate_document(read_pdf(pdf_path, max_pages=10))

        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                output = json.dumps(json.load(jf), indent=2)
                examples.append({
                    "document": document,
                    "output": output
                })
        except:
            continue

    return examples

def build_prompt(prompt_template, examples, document_text):
    prompt = ""
    for i, ex in enumerate(examples, 1):
        prompt += (
            f"\n### EXAMPLE {i}\n\n"
            f"DOCUMENT:\n{ex['document']}\n\n"
            f"EXPECTED OUTPUT:\n{ex['output']}\n"
        )
    prompt += "\n### NOW PROCESS THIS DOCUMENT\n"
    return prompt + prompt_template.replace("{{DOCUMENT}}", document_text)

def clean_json(output):
    text = output.strip()
    text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\n```$', '', text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return text.strip()

# ======================================================
# ✅ Retry Wrapper (fixes 429 rate-limit errors)
# ======================================================
def call_with_retry(model, prompt, retries=3):
    """
    Returns (response_text, usage_dict, attempts_used).
    attempts_used lets the caller log exactly how many API calls this
    single document actually consumed (1, unless it hit rate limits).
    """

    for i in range(retries):
        try:
            text, usage = call_llm(model, prompt)
            return text, usage, i + 1

        except Exception as e:

            err = str(e).lower()

            if (
                "429" in err
                or "rate" in err
                or "timeout" in err
                or "timed out" in err
            ):

                wait = 20 + (i * 10)

                st.warning(
                    f"⏳ Retry {i+1}/{retries} after {wait}s"
                )

                time.sleep(wait)

            else:
                raise e

    raise RuntimeError("❌ Failed after retries")

# ======================================================
def call_llm(model, prompt):
    """
    Returns a tuple: (response_text, usage_dict)
    usage_dict = {"input": int, "output": int}

    IMPORTANT: this function no longer writes directly into
    st.session_state. Writing directly into a session-level dict from
    inside a function that gets called many times (and potentially
    retried) made it impossible to tell how many of the accumulated
    tokens/calls belonged to the current run versus earlier runs in
    the same Streamlit session. Usage is now returned per-call and
    aggregated explicitly by the caller (see the main loop below),
    which also attaches per-document usage to each result row.
    """

    # =========================
    # Gemini
    # =========================
    if model.startswith("gemini/"):

        if not gemini_api_key:
            raise RuntimeError("Gemini API Key وارد نشده")

        genai.configure(
            api_key=gemini_api_key
        )

        m = genai.GenerativeModel(
            model.replace("gemini/", "")
        )

        response = m.generate_content(prompt)

        usage = getattr(response, "usage_metadata", None)
        usage_dict = {
            "input": (getattr(usage, "prompt_token_count", 0) or 0) if usage else 0,
            "output": (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0,
        }

        return response.text, usage_dict

    # =========================
    # OpenRouter (replaces Groq)
    # =========================
    elif model.startswith("openrouter/"):

        if not openrouter_api_key:
            raise RuntimeError("OpenRouter API Key وارد نشده")

        client = OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        r = client.chat.completions.create(
            model=model.replace("openrouter/", ""),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        usage_dict = {"input": 0, "output": 0}
        if getattr(r, "usage", None):
            usage_dict["input"] = r.usage.prompt_tokens or 0
            usage_dict["output"] = r.usage.completion_tokens or 0

        return r.choices[0].message.content, usage_dict

    # =========================
    # Claude
    # =========================
    elif model.startswith("claude/"):

        if not anthropic_api_key:
            raise RuntimeError("Claude API Key وارد نشده")

        try:

            client = anthropic.Anthropic(
                api_key=anthropic_api_key,
                timeout=120.0
            )

            response = client.messages.create(
                model=model.replace("claude/", ""),
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            if not response.content:
                raise RuntimeError(
                    "Claude returned empty response"
                )

            usage_dict = {"input": 0, "output": 0}
            if getattr(response, "usage", None):
                usage_dict["input"] = response.usage.input_tokens or 0
                usage_dict["output"] = response.usage.output_tokens or 0

            return response.content[0].text, usage_dict

        except Exception as e:
            raise RuntimeError(
                f"Claude API Error: {e}"
            )

    raise RuntimeError("Model not supported")

    
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
    results = []

    few_shots = load_few_shot_examples(train_folder) if use_training else []

    if not os.path.exists(input_folder):
        st.error("پوشه reports یافت نشد")
    else:
        pdfs = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]
        progress_bar = st.progress(0)

        for idx, pdf in enumerate(pdfs):
            st.write(f"در حال پردازش: {pdf}")

            text = read_pdf(os.path.join(input_folder, pdf), max_pages=10)
            if not text:
                continue

            prompt = build_prompt(prompt_template, few_shots, truncate_document(text))

            try:
                # ✅ Uses retry — now also returns per-document usage
                raw_output, usage, attempts = call_with_retry(model_name, prompt)

                cleaned = clean_json(raw_output)

                try:
                    parsed = json.loads(cleaned)
                except:
                    st.warning(f"⚠️ JSON خراب: {pdf}")
                    parsed = {k: None for k in [
                        "filename","project_identity","auditor",
                        "high_vulnerability_count","medium_vulnerability_count","low_vulnerability_count",
                        "commit_hash","year","fixed_items","first_vulnerability_title"
                    ]}

                parsed["filename"] = pdf

                # ✅ per-document token/attempt logging — this survives
                # regardless of how many times the button gets pressed,
                # and lets you audit which specific documents needed
                # retries instead of only seeing one opaque session total.
                parsed["_input_tokens"] = usage["input"]
                parsed["_output_tokens"] = usage["output"]
                parsed["_api_attempts"] = attempts

                results.append(parsed)

                # update the (now correctly-scoped) session totals too,
                # for the quick on-screen summary below
                st.session_state.token_usage["input"] += usage["input"]
                st.session_state.token_usage["output"] += usage["output"]
                st.session_state.token_usage["calls"] += attempts

                time.sleep(3)  # ✅ Increased delay

            except Exception as e:
                st.error(f"خطا در {pdf}: {e}")

            progress_bar.progress((idx + 1) / len(pdfs))

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