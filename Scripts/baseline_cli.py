#!/usr/bin/env python3
"""
baseline_cli.py -- command-line interface for the single-model baseline.

This calls the same baseline_core logic as the Streamlit app
(baseline.py): no extraction, retry, prompting, or truncation logic is
reimplemented here.

Examples
--------
Run the baseline over a folder of PDFs:

    python baseline_cli.py --input ./input --output ./structured \\
        --model gemini/gemini-2.5-flash --use-shots

Verify that the configured API keys actually work, without running any
extraction:

    python baseline_cli.py --check-api --model gemini/gemini-2.5-flash

List every available option:

    python baseline_cli.py --help
"""

import argparse
import os
import sys
import time

import baseline_core


def _cli_warn(msg):
    print(msg, flush=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="baseline_cli.py",
        description="Run the single-model audit extraction baseline from the command line, "
                     "for servers and other environments without a graphical display.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input", default="./input", metavar="DIR",
        help="Folder of PDF audit reports to process (default: ./input)",
    )
    parser.add_argument(
        "--output", default="./structured", metavar="DIR",
        help="Folder to write the resulting Excel file to (default: ./structured)",
    )
    parser.add_argument(
        "--model", default="gemini/gemini-2.5-flash", metavar="PROVIDER/MODEL",
        help="Model to use, as provider/model (default: gemini/gemini-2.5-flash). "
             f"Available: {', '.join(baseline_core.BASELINE_MODELS)}",
    )
    parser.add_argument(
        "--prompt-file", default=None, metavar="FILE",
        help="Path to a text file containing the extraction prompt template "
             "(must include a {{DOCUMENT}} placeholder). If omitted, the "
             "built-in default prompt is used.",
    )
    parser.add_argument(
        "--use-shots", action="store_true",
        help="Load few-shot reference examples from --shots-dir before extracting.",
    )
    parser.add_argument(
        "--shots-dir", default="./shots", metavar="DIR",
        help="Folder of few-shot reference PDF+JSON pairs, used only with --use-shots "
             "(default: ./shots)",
    )
    parser.add_argument(
        "--check-api", action="store_true",
        help="Send one small test request to the provider used by --model to confirm "
             "the configured API key actually works, then exit without processing any "
             "documents.",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print every model name accepted by --model, then exit.",
    )

    return parser


def check_api(model):
    """
    Sends one minimal request to the provider behind --model and reports
    whether it succeeded. This exists because a missing or invalid API
    key does not fail loudly inside the baseline itself: every document
    for that provider simply raises the same "... API Key وارد نشده"
    error and is skipped, which can look like a document-reading problem
    rather than a configuration one unless checked for directly.
    """
    provider = model.split("/", 1)[0]
    print(f"Checking API access for: {provider}\n")
    try:
        text, usage = baseline_core.call_llm(model, "Reply with the single word: OK")
        print(f"  ✅ {model}: responded successfully")
        return 0
    except Exception as e:
        print(f"  ❌ {model}: {e}")
        print("\nCheck api_keys.json (or the corresponding environment variable) "
              "before running a full extraction.")
        return 1


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    baseline_core.set_warn_function(_cli_warn)

    if args.list_models:
        for m in baseline_core.BASELINE_MODELS:
            print(m)
        return 0

    if args.check_api:
        return check_api(args.model)

    prompt_template = baseline_core.DEFAULT_PROMPT_TEMPLATE
    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            print(f"Error: prompt file not found: {args.prompt_file}", file=sys.stderr)
            return 1
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        if "{{DOCUMENT}}" not in prompt_template:
            print("Error: --prompt-file must contain a {{DOCUMENT}} placeholder", file=sys.stderr)
            return 1

    few_shots = baseline_core.load_few_shot_examples(args.shots_dir) if args.use_shots else []

    if not os.path.exists(args.input):
        print(f"Error: input directory not found: {args.input}", file=sys.stderr)
        return 1

    total_files = len([f for f in os.listdir(args.input) if f.lower().endswith(".pdf")])
    if total_files == 0:
        print(f"No PDF files found in {args.input}", file=sys.stderr)
        return 1

    def on_file_start(pdf):
        print(f"Processing: {pdf}", flush=True)

    def on_error(pdf, e):
        print(f"Error on {pdf}: {e}", file=sys.stderr, flush=True)

    def on_progress(idx, total):
        print(f"[{idx + 1}/{total}] done", flush=True)

    results, usage = baseline_core.run_baseline_batch(
        input_folder=args.input,
        model_name=args.model,
        prompt_template=prompt_template,
        few_shots=few_shots,
        on_file_start=on_file_start,
        on_error=on_error,
        on_progress=on_progress,
    )

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        os.makedirs(args.output, exist_ok=True)
        out_path = os.path.join(args.output, f"extraction_{int(time.time())}.xlsx")
        df.to_excel(out_path, index=False)
        print(f"\nProcessed {len(results)} report(s). Results written to {out_path}")
    else:
        print("\nNo results were produced.")

    print(f"Model: {args.model} — API calls (incl. retries): {usage['calls']} "
          f"— Total tokens: {usage['input'] + usage['output']:,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
