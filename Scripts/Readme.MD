# MASP: Multi-Agent Strategic Pipeline for Audit Report Extraction

MASP extracts structured vulnerability metadata, such as auditor name, project name, severity counts, and commit hash, from smart contract audit report PDFs. It relies on three independent LLM agents, referred to as the Miner Trio, together with a Judge agent that reconciles their candidate outputs against a target schema.

## This tool calls external LLM APIs rather than running any model locally

Neither MASP nor the single-model baseline included in this repository bundles or executes a language model on the user's machine. Every extraction call is sent to a hosted provider API, namely Google Gemini, OpenRouter, or Anthropic Claude, depending on the model selected. An internet connection is therefore required to run either tool. Each provider bills usage according to its own pricing, and MASP issues up to four calls per document, three from the Miners and one from the Judge, so cost scales accordingly; the token usage figures reported in the paper reflect this. Because document text is transmitted to the selected provider's servers as part of the API request, this tool should not be used on documents that cannot be shared with a third-party API provider.

## Installation

```bash
git clone <repository-url>
cd masp
pip install -r requirements.txt
```

`requirements.txt` covers the CLI and the extraction engine itself; a
headless server running only `masp_cli.py` does not need anything beyond
it. The graphical interface additionally requires Streamlit:

```bash
pip install -r requirements-ui.txt
```

Neither file installs any local model weights, since none are used.

## API keys

Both tools read their API keys from `api_keys.json` in the working directory. The file is created automatically on first run if it does not already exist, with empty values left for the user to fill in:

```json
{
  "GEMINI_API_KEY": "",
  "OPENROUTER_API_KEY": "",
  "ANTHROPIC_API_KEY": ""
}
```

A key is only required for a provider whose model is actually selected; the tool does not require all three to be present. Setting `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, or `ANTHROPIC_API_KEY` as an environment variable is also supported and takes precedence over the value stored in `api_keys.json`.

## Project layout

```
masp_core.py         MASP extraction logic: signatures, the three Miner LM
                      wrappers, the Judge-based MASPHighPrecision engine, demo
                      construction, and the retry-until-complete batch loop.
MASP_v1.py            Streamlit interface for MASP. Thin wrapper around masp_core.py.
masp_cli.py           Command-line interface for MASP, for environments without
                      a graphical display.
baseline_core.py      Single-model baseline logic: PDF reading, prompt assembly,
                      the retry wrapper, and the per-provider call_llm dispatch.
baseline.py           Streamlit interface for the baseline. Thin wrapper around
                      baseline_core.py.
baseline_cli.py       Command-line interface for the baseline.
```

Neither `masp_core.py` nor `baseline_core.py` depends on Streamlit, so
`masp_cli.py` and `baseline_cli.py` can both run on a headless server
without a graphical display installed.

## Usage: graphical interface

```bash
streamlit run MASP_v1.py        # multi-agent MASP pipeline
streamlit run baseline.py       # single-model baseline, for comparison
```

Each command launches a browser-based interface in which the user selects a folder of PDF audit reports, chooses a model (or, for MASP, a model for each Miner and for the Judge), and starts extraction. Results are written to `./structured/` as an Excel file once processing completes.

## Usage: command line

The Streamlit interface requires a graphical display and cannot be used on most Linux servers set up for batch processing. `masp_cli.py` runs the same MASP engine without that requirement:

```bash
python masp_cli.py \
    --input ./input --output ./structured \
    --miner1 gemini/gemini-2.5-flash \
    --miner2 openrouter/meta-llama/llama-3.3-70b-instruct \
    --miner3 claude/claude-sonnet-5 \
    --judge gemini/gemini-2.5-flash \
    --max-iter 2
```

Every option accepted by the CLI, along with its default value, is listed by:

```bash
python masp_cli.py --help
```

A failing or missing API key does not raise an error inside MASP itself; the affected Miner or Judge simply returns an error string on every document, which can look like an ordinary extraction gap rather than a configuration problem. To rule this out before committing to a full run, check that each configured key actually works:

```bash
python masp_cli.py --check-api
```

This sends one minimal request per provider referenced by `--miner1`, `--miner2`, `--miner3`, and `--judge`, reports which ones succeeded, and exits without processing any documents.

## Usage: baseline (single model)

The baseline follows the same pattern as MASP: a Streamlit interface for interactive use, and a command-line interface for headless environments.

```bash
streamlit run baseline.py                          # graphical interface
python baseline_cli.py --input ./input --output ./structured \
    --model gemini/gemini-2.5-flash --use-shots     # command line
```

`--use-shots` loads the same kind of few-shot reference examples as MASP, from `--shots-dir` (default `./shots`). A custom prompt template can be supplied with `--prompt-file`, pointing to a text file that contains a `{{DOCUMENT}}` placeholder; without it, the built-in default prompt is used. As with MASP, `--check-api` verifies that the configured key for `--model` works before committing to a full run, and `--help` lists every option.

## Input layout

Before running either interface, create these two folders next to the
scripts and place the relevant files in them; MASP does not create or
populate them on its own.

```
input/    PDF audit reports to process
shots/    optional few-shot examples: pairs of report.pdf and report.json,
          used as reference input-output examples by the Miners and Judge
```

`shots/` is only read if few-shot examples are enabled (in the Streamlit
UI) or `--shots-dir` points to it (in the CLI); if left empty or omitted,
MASP runs without reference examples.

## Configuration options

Both interfaces expose the same underlying options, under different
names depending on whether they come from the sidebar or a CLI flag.

- **Fields** (`--fields`): a comma-separated list of the metadata fields
  MASP should extract from each report, e.g. `filename, project_identity,
  auditor, high_vulnerability_count`. This list defines both what MASP
  looks for and the keys of the JSON object it returns for each document.
- **Logic** (`--logic`): free-text instructions describing how those
  fields should be extracted, given to every Miner and to the Judge as
  part of their task. This is where document-specific extraction rules
  belong, for instance how to decide whether a finding counts as fixed,
  or which commit hash to prefer when a report lists several.
- **Output** (`--output`, default `./structured`): the folder MASP writes
  its results to. Each run produces a single `full_audit_results.xlsx`
  file in this folder, with one row per processed document and one
  column per requested field.

The Streamlit sidebar also exposes **Max Iterations** (`--max-iter` on
the CLI), the number of Miner/Judge refinement rounds allowed per
document, and one model selector per Miner and for the Judge
(`--miner1`, `--miner2`, `--miner3`, `--judge`).
