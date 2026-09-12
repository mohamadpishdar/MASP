"""
Core MASP logic: DSPy signatures, the three Miner LM wrappers, the Judge-
based MASPHighPrecision engine, and the shared helpers for building demos
and running a batch of documents.

This module contains no Streamlit imports and no UI code, so it can be
used identically from MASP_v1.py (the Streamlit app) and masp_cli.py (the
command-line tool), without requiring Streamlit to be installed for the
CLI. Every class, function, and comment below is copied unchanged from
MASP_v1.py; nothing about how MASP reasons, aggregates, retries, or
truncates text has been altered by this split.
"""

import dspy
import fitz  # PyMuPDF
import json
import os
import re
import time
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
# LOGGING
# =====================================================
# Both the Streamlit app and the CLI install their own log() function here
# at import time (see MASP_v1.py and masp_cli.py), so the exact same
# MASPHighPrecision code can log to either st.session_state or stdout
# without any change to its own body.
_log_sink = []

def log(msg):
    t = time.strftime("%H:%M:%S")
    _log_sink.append(f"[{t}] {msg}")

def set_log_function(fn):
    """Lets a caller (Streamlit UI or CLI) redirect log() to its own sink."""
    global log
    log = fn

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
# MODEL LISTS (shared between the Streamlit sidebar and the CLI)
# =====================================================
GEMINI_AVAIL = [
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
OPENROUTER_AVAIL = [
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-72b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mixtral-8x7b-instruct",
]
CLAUDE_AVAIL = ["claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-opus-4-8"]

ALL_MODELS = (
    [f"gemini/{m}" for m in GEMINI_AVAIL]
    + [f"openrouter/{m}" for m in OPENROUTER_AVAIL]
    + [f"claude/{m}" for m in CLAUDE_AVAIL]
)

def get_lm(choice):
    provider, model = choice.split("/", 1)
    if provider == "gemini":
        return GeminiGenAIClientLM(model_name=model, api_key=GEMINI_API_KEY)
    if provider == "claude":
        return ClaudeLM(model_name=model, api_key=ANTHROPIC_API_KEY)
    return OpenRouterLM(model_name=model, api_key=OPENROUTER_API_KEY)

# =====================================================
# DEMO (FEW-SHOT) BUILDING
# =====================================================
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

def build_demos(shots_dir="./shots"):
    demos = ""
    if os.path.exists(shots_dir):
        demo_files = [
            f for f in os.listdir(shots_dir)
            if f.endswith(".pdf") and os.path.exists(f"{shots_dir}/{f[:-4]}.json")
        ]
        if demo_files:
            per_file_budget = max(TOTAL_DEMO_MAX_CHARS // len(demo_files), 1000)
            for f in demo_files:
                with fitz.open(f"{shots_dir}/{f}") as doc:
                    text = "".join(doc[p].get_text() for p in range(min(10, len(doc))))
                text = _truncate_for_demo(text, per_file_budget)
                with open(f"{shots_dir}/{f[:-4]}.json", "r") as jf:
                    demos += f"DOC:\n{text}\nOUT:\n{jf.read()}\n"
    return demos

# =====================================================
# RETRY-UNTIL-COMPLETE HELPERS
# =====================================================
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

# =====================================================
# BATCH PROCESSING (shared by the Streamlit UI and the CLI)
# =====================================================
def run_masp_batch(input_dir, lms, base_task, target_fields, max_iter,
                    on_file_start=None, on_file_done=None):
    """
    Runs MASP over every PDF in input_dir and returns the list of result
    dicts (one per document), each carrying "filename" and, on documents
    that never reached MIN_FILLED_RATIO after MAX_DOC_RETRIES attempts,
    "_status": "FAILED_INCOMPLETE".

    on_file_start(idx, total, filename) and on_file_done(idx, total) are
    optional callbacks so the Streamlit UI can drive its progress bar and
    the CLI can print progress to stdout, without duplicating the
    retry-until-complete loop itself in two places.
    """
    engine = MASPHighPrecision(max_iter=max_iter)
    results = []
    required_fields = _target_field_list(target_fields)

    if not os.path.exists(input_dir):
        return results

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    for idx, f in enumerate(files):
        if on_file_start:
            on_file_start(idx, len(files), f)
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
        if on_file_done:
            on_file_done(idx, len(files))

    return results
