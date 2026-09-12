"""
Core single-model baseline logic: PDF reading, few-shot prompt assembly,
JSON cleanup, the retry wrapper, and the per-provider call_llm dispatch.

This module has no Streamlit import, so it can be used identically from
baseline.py (the Streamlit app) and baseline_cli.py (the command-line
tool). Every function below is copied unchanged from baseline.py, with
two structural exceptions required to remove the Streamlit dependency:

1. The three API key variables are now module-level (GEMINI_API_KEY,
   OPENROUTER_API_KEY, ANTHROPIC_API_KEY), sourced from api_keys.json /
   environment variables, instead of being read from sidebar
   st.text_input widgets. The source of the values (api_keys.json,
   overridable by environment variable) is unchanged from baseline.py.
2. call_with_retry() reports a retry through a settable warn() function
   instead of calling st.warning() directly, so the same retry logic
   can print to a terminal (CLI) or a Streamlit warning box (UI).

No prompting, truncation, retry, or provider-dispatch logic has been
altered.
"""

import fitz  # PyMuPDF
import json
import os
import re
import time
from openai import OpenAI
import google.generativeai as genai
import anthropic

# =====================================================
# API KEYS
# =====================================================
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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", _api_keys.get("GEMINI_API_KEY", ""))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", _api_keys.get("OPENROUTER_API_KEY", ""))
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", _api_keys.get("ANTHROPIC_API_KEY", ""))

# =====================================================
# WARN HOOK
# =====================================================
# call_with_retry() previously called st.warning() directly. Both the
# Streamlit app and the CLI now install their own warn() function here
# at import time, so the exact same retry logic can surface a warning to
# either a terminal or a Streamlit warning box without any change to its
# own body.
def warn(msg):
    print(msg)

def set_warn_function(fn):
    """Lets a caller (Streamlit UI or CLI) redirect warn() to its own sink."""
    global warn
    warn = fn

# =====================================================
# Model list and default prompt (shared between the Streamlit sidebar
# and the CLI, so both offer the same choices and default text)
# =====================================================
BASELINE_MODELS = [
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-1.5-flash",
    "openrouter/deepseek/deepseek-chat",
    "openrouter/qwen/qwen-2.5-72b-instruct",
    "openrouter/meta-llama/llama-3.3-70b-instruct",
    "openrouter/mistralai/mixtral-8x7b-instruct",
    "claude/claude-sonnet-4-6",
    "claude/claude-opus-4-8",
]

DEFAULT_PROMPT_TEMPLATE = """You are given several CORRECT extraction examples above.
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
"""

TARGET_FIELDS = [
    "filename", "project_identity", "auditor",
    "high_vulnerability_count", "medium_vulnerability_count", "low_vulnerability_count",
    "commit_hash", "year", "fixed_items", "first_vulnerability_title",
]

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

                warn(
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

        if not GEMINI_API_KEY:
            raise RuntimeError("Gemini API Key وارد نشده")

        genai.configure(
            api_key=GEMINI_API_KEY
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

        if not OPENROUTER_API_KEY:
            raise RuntimeError("OpenRouter API Key وارد نشده")

        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
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

        if not ANTHROPIC_API_KEY:
            raise RuntimeError("Claude API Key وارد نشده")

        try:

            client = anthropic.Anthropic(
                api_key=ANTHROPIC_API_KEY,
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
# BATCH PROCESSING (shared by the Streamlit UI and the CLI)
# ======================================================
def run_baseline_batch(input_folder, model_name, prompt_template, few_shots,
                        on_file_start=None, on_error=None, on_progress=None):
    """
    Runs the single-model baseline over every PDF in input_folder and
    returns (results, token_usage), where token_usage is
    {"input": int, "output": int, "calls": int} aggregated across every
    document processed in this call -- the same aggregation the
    Streamlit app previously did inline against st.session_state.

    on_file_start(pdf_name), on_error(pdf_name, exception), and
    on_progress(idx, total) are optional callbacks so the Streamlit UI
    can drive its progress bar and message log, and the CLI can print to
    stdout, without duplicating the extraction loop itself in two
    places.
    """
    results = []
    token_usage = {"input": 0, "output": 0, "calls": 0}

    if not os.path.exists(input_folder):
        return results, token_usage

    pdfs = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]

    for idx, pdf in enumerate(pdfs):
        if on_file_start:
            on_file_start(pdf)

        text = read_pdf(os.path.join(input_folder, pdf), max_pages=10)
        if not text:
            if on_progress:
                on_progress(idx, len(pdfs))
            continue

        prompt = build_prompt(prompt_template, few_shots, truncate_document(text))

        try:
            # ✅ Uses retry — now also returns per-document usage
            raw_output, usage, attempts = call_with_retry(model_name, prompt)

            cleaned = clean_json(raw_output)

            try:
                parsed = json.loads(cleaned)
            except:
                warn(f"⚠️ JSON خراب: {pdf}")
                parsed = {k: None for k in TARGET_FIELDS}

            parsed["filename"] = pdf

            # ✅ per-document token/attempt logging — this survives
            # regardless of how many times the button gets pressed,
            # and lets you audit which specific documents needed
            # retries instead of only seeing one opaque session total.
            parsed["_input_tokens"] = usage["input"]
            parsed["_output_tokens"] = usage["output"]
            parsed["_api_attempts"] = attempts

            results.append(parsed)

            token_usage["input"] += usage["input"]
            token_usage["output"] += usage["output"]
            token_usage["calls"] += attempts

            time.sleep(3)  # ✅ Increased delay

        except Exception as e:
            if on_error:
                on_error(pdf, e)

        if on_progress:
            on_progress(idx, len(pdfs))

    return results, token_usage
