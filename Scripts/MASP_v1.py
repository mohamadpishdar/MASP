import streamlit as st
import dspy
import fitz  # PyMuPDF
import json
import os
import re
import time
import pandas as pd
from google import genai
from openai import OpenAI
import anthropic


fitz.TOOLS.mupdf_display_errors(False)

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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", _api_keys["GEMINI_API_KEY"])
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", _api_keys["OPENROUTER_API_KEY"])
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", _api_keys["ANTHROPIC_API_KEY"])

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

# =====================================================
# DSPy SIGNATURES
# =====================================================
class ExtractionSignature(dspy.Signature):
    """Extract the requested fields from the document according to the
    task instructions. Before giving your final output, briefly explain
    your reasoning in about 300 words or fewer -- a concise summary of
    which parts of the document you relied on and why, not a full
    transcription of your thought process. This summary will be shown to
    a separate reviewer, so it should stand on its own and end with your
    conclusion rather than being cut off mid-thought."""
    context = dspy.InputField()
    task = dspy.InputField()
    output = dspy.OutputField()

class JudgeSignature(dspy.Signature):
    """You are given the same task instructions that were given to three
    independent miners, along with each miner's proposed output AND the
    reasoning it used to arrive at that output.

    Do not simply pick the value that the majority of miners agree on.
    Instead, evaluate each miner's REASONING against the task
    instructions above. A miner whose reasoning explicitly follows a
    stated rule (for example, correctly excluding an item that the
    instructions say should be excluded) should be preferred over
    miners who agree with each other but whose reasoning shows they
    overlooked or misapplied that rule -- even if those miners are in
    the majority. Numeric or textual agreement between miners is a weak
    signal on its own; explicit, rule-compliant reasoning is a stronger
    signal and should take priority when the two conflict."""
    options = dspy.InputField(desc="each candidate's output together with the reasoning it used to produce that output")
    task = dspy.InputField()
    final_json = dspy.OutputField()
    fault_report = dspy.OutputField()

# =====================================================
#  GEMINI LM (Matches your list)
# =====================================================
class GeminiGenAIClientLM(dspy.LM):
    def __init__(self, model_name, api_key=None):
        # Strip any "models/" prefix so the Google client gets a bare model name
        self.raw_model_name = model_name.replace("models/", "")
        super().__init__(model=f"models/{self.raw_model_name}")
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
        self.provider = "google"
        self.kwargs = {"temperature": 0.1, "max_output_tokens": 4096}
        self.usage = {"input": 0, "output": 0, "calls": 0}

    def __call__(self, prompt=None, messages=None, **kwargs):
        query = prompt if prompt else ""
        if messages:
            query = "\n".join([m.get('content', '') for m in messages])
            
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.raw_model_name, 
                    contents=query
                )
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    self.usage["input"] += getattr(usage, "prompt_token_count", 0) or 0
                    self.usage["output"] += getattr(usage, "candidates_token_count", 0) or 0
                    self.usage["calls"] += 1
                if response and response.text:
                    return [response.text]
            except Exception as e:
                err_msg = str(e).lower()
                if "503" in err_msg or "high demand" in err_msg:
                    time.sleep(5)
                    continue
                log(f"❌ Gemini Error: {str(e)}")
                break
        return ["Reasoning: API Error.\nOutput: {}"]

# =====================================================
#  OPENROUTER LM (Replacement for Groq — no strict TPM cap)
# =====================================================
class OpenRouterLM(dspy.LM):
    MAX_PROMPT_CHARS = 60000  # generous cap; OpenRouter limits are far higher than Groq's free tier

    def __init__(self, model_name="deepseek/deepseek-chat", api_key=None, max_prompt_chars=None):
        super().__init__(model=model_name)
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.model_name = model_name
        self.provider = "openrouter"
        self.usage = {"input": 0, "output": 0, "calls": 0}
        self.max_prompt_chars = max_prompt_chars or self.MAX_PROMPT_CHARS

    def _truncate(self, text, limit):
        if len(text) <= limit:
            return text
        head = text[: int(limit * 0.7)]
        tail = text[-int(limit * 0.3):]
        return head + "\n...[truncated for size limit]...\n" + tail

    def __call__(self, prompt=None, messages=None, **kwargs):
        query = prompt if prompt else ""
        if messages:
            query = "\n".join([m.get('content', '') for m in messages])

        limit = self.max_prompt_chars
        for attempt in range(3):
            trimmed = self._truncate(query, limit)
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": trimmed}],
                    temperature=0.1,
                )
                if getattr(completion, "usage", None):
                    self.usage["input"] += completion.usage.prompt_tokens or 0
                    self.usage["output"] += completion.usage.completion_tokens or 0
                    self.usage["calls"] += 1
                return [completion.choices[0].message.content]
            except Exception as e:
                err_msg = str(e)
                if "413" in err_msg or "rate_limit" in err_msg.lower() or "too large" in err_msg.lower():
                    limit = int(limit * 0.6)
                    log(f"⚠️ OpenRouter size/rate limit hit; shrinking prompt to ~{limit} chars and retrying.")
                    continue
                log(f"❌ OpenRouter Error: {err_msg}")
                break
        return ["Reasoning: OpenRouter Error.\nOutput: {}"]

# =====================================================
#  CLAUDE LM (Anthropic)
# =====================================================
class ClaudeLM(dspy.LM):
    def __init__(self, model_name="claude-sonnet-5", api_key=None):
        super().__init__(model=model_name)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.provider = "anthropic"
        self.usage = {"input": 0, "output": 0, "calls": 0}

    def __call__(self, prompt=None, messages=None, **kwargs):
        query = prompt if prompt else ""
        if messages:
            query = "\n".join([m.get('content', '') for m in messages])

        create_kwargs = {
            "model": self.model_name,
            "max_tokens": 4096,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": query}],
        }
        for attempt in range(3):
            try:
                try:
                    response = self.client.messages.create(**create_kwargs)
                except TypeError as te:
                    if "temperature" in str(te) and "temperature" in create_kwargs:
                        # Old/incompatible anthropic SDK — retry without it.
                        log("⚠️ Claude SDK rejected 'temperature'; retrying without it. Run: pip install -U anthropic")
                        create_kwargs.pop("temperature")
                        response = self.client.messages.create(**create_kwargs)
                    else:
                        raise
                if getattr(response, "usage", None):
                    self.usage["input"] += response.usage.input_tokens or 0
                    self.usage["output"] += response.usage.output_tokens or 0
                    self.usage["calls"] += 1
                text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                if text:
                    return [text]
            except anthropic.APIStatusError as e:
                if e.status_code in (429, 503, 529):
                    time.sleep(5)
                    continue
                log(f"❌ Claude Error: {str(e)}")
                break
            except Exception as e:
                log(f"❌ Claude Error: {str(e)}")
                break
        return ["Reasoning: Claude Error.\nOutput: {}"]

# =====================================================
# MASP ENGINE
# =====================================================
class MASPHighPrecision(dspy.Module):
    def __init__(self, max_iter=2):
        super().__init__()
        self.miner = dspy.ChainOfThought(ExtractionSignature)
        self.judge = dspy.ChainOfThought(JudgeSignature)
        self.max_iter = max_iter

    def forward(self, context, base_task, lms):
        logic = base_task
        best = None
        for i in range(self.max_iter):
            log(f"Iteration {i+1}")
            candidates = []
            for k in ["agent1", "agent2", "agent3"]:
                with dspy.context(lm=lms[k]):
                    try:
                        res = self.miner(context=context, task=logic)
                        # ✅ FIX (reasoning-aware judging): dspy.ChainOfThought
                        # already produces a reasoning trace before its final
                        # answer on every call -- previously this was thrown
                        # away (only res.output was kept), so the Judge only
                        # ever saw three bare candidate outputs with no way to
                        # tell *why* each miner arrived at its answer. When all
                        # three miners agreed on the same wrong value (e.g. a
                        # shared misreading of the document), the Judge had no
                        # signal to catch this, since there was nothing to
                        # disagree with. Now each candidate carries a short
                        # summary of its reasoning alongside its output.
                        #
                        # ✅ FIX (flat, capped format): the first version of
                        # this fix passed a list of nested dicts to the Judge
                        # (str(candidates) on a list of {"output":...,
                        # "reasoning":...} dicts), which produced a much
                        # larger and more awkwardly-structured Judge input and
                        # led to noticeably longer, sometimes stalled runs.
                        # Reasoning is also capped to a few hundred characters
                        # (dspy.ChainOfThought's reasoning traces can run to
                        # thousands) so this doesn't blow up prompt size the
                        # way the untruncated version did.
                        # ✅ FIX (self-summarized reasoning): rather than
                        # generating a long, unconstrained chain-of-thought
                        # and then truncating it afterward (which could cut
                        # off the actual conclusion mid-sentence), the miner
                        # is now instructed directly (see ExtractionSignature)
                        # to keep its reasoning to roughly 300 words and end
                        # with a conclusion. The character cap below is only
                        # a safety net in case a miner ignores that
                        # instruction, not the primary length control.
                        reasoning = getattr(res, "reasoning", None) or getattr(res, "rationale", None) or ""
                        reasoning = str(reasoning).strip().replace("\n", " ")[:1800]  # ~300 words safety cap
                        candidates.append(
                            f"CANDIDATE {len(candidates)+1}:\n"
                            f"Output: {res.output}\n"
                            f"Reasoning: {reasoning}\n"
                        )
                    except:
                        candidates.append(f"CANDIDATE {len(candidates)+1}:\nOutput: {{}}\nReasoning: (miner failed)\n")
            
            with dspy.context(lm=lms["judge"]):
                try:
                    verdict = self.judge(options="\n".join(candidates), task=logic)
                    match = re.search(r"\{.*\}", verdict.final_json, re.DOTALL)
                    if match:
                        best = json.loads(match.group())
                        break
                    logic = verdict.fault_report
                except:
                    break
        return best

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    target_fields = st.text_input("Fields", "filename, project_identity, auditor, high_vulnerability_count")
    mission_logic = st.text_area("Logic", "Extract audit metadata strictly from evidence.")
    
    # Available models per provider, plus OpenRouter as the Groq replacement
    gemini_avail = [
        "gemini-2.0-flash-001", 
        "gemini-2.0-flash-lite-001", 
        "gemini-2.5-flash", 
        "gemini-2.5-flash-lite"
    ]
    openrouter_avail = [
        "deepseek/deepseek-chat",
        "qwen/qwen-2.5-72b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
        "mistralai/mixtral-8x7b-instruct",
    ]
    claude_avail = ["claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-opus-4-8"]

    all_models = (
        [f"gemini/{m}" for m in gemini_avail]
        + [f"openrouter/{m}" for m in openrouter_avail]
        + [f"claude/{m}" for m in claude_avail]
    )

    # Agents: 1) Gemini  2) OpenRouter (replaces Groq)  3) Claude  |  Judge: Claude (user-selectable)
    m1 = st.selectbox("Miner 1 (Gemini)", all_models, index=all_models.index("gemini/gemini-2.0-flash-001"))
    m2 = st.selectbox("Miner 2 (OpenRouter)", all_models, index=all_models.index("openrouter/deepseek/deepseek-chat"))
    m3 = st.selectbox("Miner 3 (Claude)", all_models, index=all_models.index("claude/claude-sonnet-5"))
    m_judge = st.selectbox("Judge", all_models, index=all_models.index("claude/claude-sonnet-5"))
    
    input_dir, output_dir = "./input", "./structured"
    max_iter = st.slider("Max Iterations", 1, 3, 2)

def get_lm(choice):
    provider, model = choice.split("/", 1)
    if provider == "gemini":
        return GeminiGenAIClientLM(model_name=model, api_key=GEMINI_API_KEY)
    if provider == "claude":
        return ClaudeLM(model_name=model, api_key=ANTHROPIC_API_KEY)
    return OpenRouterLM(model_name=model, api_key=OPENROUTER_API_KEY)

# =====================================================
# BATCH EXECUTION
# =====================================================
if st.button(" Start Processing All Files"):
    lms = {"agent1": get_lm(m1), "agent2": get_lm(m2), "agent3": get_lm(m3), "judge": get_lm(m_judge)}
    
    # Build the base task from the target fields, mission logic, and demos
    # ✅ FIX (context parity): each demo document is read with the same
    # page budget as the main document (up to 10 pages) instead of the
    # old blind 2-page/1500-character cut, so the reference examples
    # show a complete document rather than only a title page.
    #
    # ✅ FIX (cost control, v2): the first version of this fix capped
    # the TOTAL demos string at 20000 characters but still let a single
    # training file consume the whole budget by itself (since each file
    # was individually allowed up to 60000 characters before the total
    # cap kicked in). In practice this meant only ONE reference example
    # was ever included, and having just one long, highly detailed demo
    # caused a severe regression: the Miner Trio started copying that
    # single example's specific values (project name, commit hash,
    # year) into unrelated documents instead of generalizing the output
    # format, since it had no second example to show what varies from
    # document to document. The budget is now split evenly across all
    # available training files, so the model sees several distinct
    # (shorter) examples -- which is what lets it learn the schema/
    # format rather than memorizing one example's content -- while the
    # total prompt size added by demos stays bounded.
    TOTAL_DEMO_MAX_CHARS = 20000

    def _truncate_for_demo(text, limit):
        if len(text) <= limit:
            return text
        head = text[: int(limit * 0.7)]
        tail = text[-int(limit * 0.3):]
        return head + "\n...[truncated for size limit]...\n" + tail

    demos = ""
    if os.path.exists("./shots"):
        demo_files = [
            f for f in os.listdir("./shots")
            if f.endswith(".pdf") and os.path.exists(f"./shots/{f[:-4]}.json")
        ]
        if demo_files:
            per_file_budget = max(TOTAL_DEMO_MAX_CHARS // len(demo_files), 1000)
            for f in demo_files:
                with fitz.open(f"./shots/{f}") as doc:
                    text = "".join(doc[p].get_text() for p in range(min(10, len(doc))))
                text = _truncate_for_demo(text, per_file_budget)
                with open(f"./shots/{f[:-4]}.json", "r") as jf:
                    demos += f"DOC:\n{text}\nOUT:\n{jf.read()}\n"
    
    base_task = f"Extract JSON with fields: {target_fields}\nLogic: {mission_logic}\nExamples:\n{demos}"
    engine = MASPHighPrecision(max_iter=max_iter)
    results = []

    # ✅ FIX (retry-until-complete): previously, if a document's result came
    # back partially empty (e.g. only project_identity/auditor/high_count
    # filled while everything else was blank -- the signature of a call
    # getting cut off mid-way, such as by an API quota/rate-limit error),
    # the loop accepted whatever came back and moved on to the next file.
    # That silently produced "half-processed" rows indistinguishable from
    # genuine extraction misses. Now, a document is retried (same document,
    # not the next one) up to MAX_DOC_RETRIES times, with increasing
    # backoff between attempts, until the result is judged "sufficiently
    # complete" (a configurable fraction of the target fields are filled).
    # If it still isn't complete after all retries, the row is kept but
    # explicitly flagged via "_status": "FAILED_INCOMPLETE" instead of
    # being silently indistinguishable from a normal extraction.
    MAX_DOC_RETRIES = 5
    MIN_FILLED_RATIO = 0.7

    def _target_field_list(fields_str):
        return [f.strip() for f in fields_str.split(",") if f.strip() and f.strip() != "filename"]

    def _is_sufficiently_complete(result, required_fields):
        if not result or not required_fields:
            return False
        filled = 0
        for field in required_fields:
            val = result.get(field)
            if val is None:
                continue
            if isinstance(val, str) and val.strip().lower() in ("", "nan", "null", "none", "n/a"):
                continue
            filled += 1
        return (filled / len(required_fields)) >= MIN_FILLED_RATIO

    required_fields = _target_field_list(target_fields)

    if os.path.exists(input_dir):
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
        progress_bar = st.progress(0)
        status_text = st.empty()
        for idx, f in enumerate(files):
            status_text.text(f"در حال پردازش {idx + 1}/{len(files)}: {f}")
            log(f"Processing: {f}")
            try:
                with fitz.open(os.path.join(input_dir, f)) as doc:
                    text = "".join(doc[p].get_text() for p in range(min(10, len(doc))))

                res = None
                candidate = None
                for attempt in range(1, MAX_DOC_RETRIES + 1):
                    try:
                        candidate = engine(text, base_task, lms)
                    except Exception as e:
                        candidate = None
                        log(f"⚠️ Attempt {attempt}/{MAX_DOC_RETRIES} raised an error for {f}: {str(e)}")

                    if _is_sufficiently_complete(candidate, required_fields):
                        res = candidate
                        break

                    log(f"⚠️ Attempt {attempt}/{MAX_DOC_RETRIES} incomplete for {f} — retrying")
                    if attempt < MAX_DOC_RETRIES:
                        time.sleep(5 * attempt)

                if res:
                    res["filename"] = f
                    results.append(res)
                    log(f"✅ Success: {f}")
                else:
                    log(f"❌ {f}: still incomplete after {MAX_DOC_RETRIES} attempts — recording as FAILED_INCOMPLETE")
                    fallback = dict(candidate) if candidate else {}
                    fallback["filename"] = f
                    fallback["_status"] = "FAILED_INCOMPLETE"
                    results.append(fallback)
            except Exception as e:
                log(f"❌ Error on {f}: {str(e)}")
            progress_bar.progress((idx + 1) / len(files))
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