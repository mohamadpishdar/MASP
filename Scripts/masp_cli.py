#!/usr/bin/env python3
"""
masp_cli.py -- command-line interface for MASP.

This exists for environments without a graphical display, such as most
Linux servers used for large-scale batch processing, where the Streamlit
app (MASP_v1.py) cannot be used directly. It calls the same masp_core
engine as the Streamlit app: no extraction, aggregation, retry, or
truncation logic is reimplemented here.

Examples
--------
Run MASP over a folder of PDFs:

    python masp_cli.py --input ./input --output ./structured \\
        --miner1 gemini/gemini-2.5-flash \\
        --miner2 openrouter/meta-llama/llama-3.3-70b-instruct \\
        --miner3 claude/claude-sonnet-5 \\
        --judge gemini/gemini-2.5-flash \\
        --max-iter 2

Verify that the configured API keys actually work, without running any
extraction:

    python masp_cli.py --check-api

List every available option:

    python masp_cli.py --help
"""

import argparse
import os
import sys
import time

import masp_core


def _cli_log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="masp_cli.py",
        description="Run MASP (Multi-Agent Strategic Pipeline) from the command line, "
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
        "--shots-dir", default="./shots", metavar="DIR",
        help="Folder of few-shot reference PDF+JSON pairs (default: ./shots)",
    )
    parser.add_argument(
        "--fields", default="filename, project_identity, auditor, high_vulnerability_count",
        metavar="CSV",
        help="Comma-separated list of target fields to extract "
             "(default: 'filename, project_identity, auditor, high_vulnerability_count')",
    )
    parser.add_argument(
        "--logic", default="Extract audit metadata strictly from evidence.",
        metavar="TEXT",
        help="Mission logic / extraction instructions given to every Miner and the Judge",
    )
    parser.add_argument(
        "--miner1", default="gemini/gemini-2.0-flash-001", metavar="PROVIDER/MODEL",
        help="Model for Miner 1, as provider/model (default: gemini/gemini-2.0-flash-001). "
             f"Available: {', '.join(masp_core.ALL_MODELS)}",
    )
    parser.add_argument(
        "--miner2", default="openrouter/deepseek/deepseek-chat", metavar="PROVIDER/MODEL",
        help="Model for Miner 2 (default: openrouter/deepseek/deepseek-chat)",
    )
    parser.add_argument(
        "--miner3", default="claude/claude-sonnet-5", metavar="PROVIDER/MODEL",
        help="Model for Miner 3 (default: claude/claude-sonnet-5)",
    )
    parser.add_argument(
        "--judge", default="claude/claude-sonnet-5", metavar="PROVIDER/MODEL",
        help="Model for the Judge agent (default: claude/claude-sonnet-5)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=2, metavar="N",
        help="Maximum number of Miner/Judge refinement rounds per document, 1-3 (default: 2)",
    )
    parser.add_argument(
        "--check-api", action="store_true",
        help="Send one small test request to each provider used by --miner1/2/3/--judge "
             "(or to all three providers if none are given) to confirm the configured API "
             "keys actually work, then exit without processing any documents.",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print every model name accepted by --miner1/2/3 and --judge, then exit.",
    )

    return parser


def check_api(models_to_check):
    """
    Sends one minimal request per provider found in models_to_check and
    reports which providers responded successfully. This exists because
    a silently failing API key does not raise an obvious error inside
    MASP itself: a Miner whose key is invalid simply returns its
    "Reasoning: API Error." fallback string on every document, which
    looks like ordinary extraction misses rather than a configuration
    problem, unless it is checked for directly.
    """
    providers_seen = sorted({m.split("/", 1)[0] for m in models_to_check})
    print(f"Checking API access for: {', '.join(providers_seen)}\n")

    any_failed = False
    for choice in models_to_check:
        provider = choice.split("/", 1)[0]
        try:
            lm = masp_core.get_lm(choice)
            reply = lm(prompt="Reply with the single word: OK")
            text = reply[0] if reply else ""
            if "API Error" in text or "OpenRouter Error" in text or "Claude Error" in text:
                print(f"  ❌ {choice}: reached the provider but got an error response: {text.strip()[:120]}")
                any_failed = True
            else:
                print(f"  ✅ {choice}: responded successfully")
        except Exception as e:
            print(f"  ❌ {choice}: could not even send a request ({e})")
            any_failed = True

    print()
    if any_failed:
        print("One or more providers failed. Check api_keys.json (or the corresponding "
              "environment variable) before running a full extraction, since a bad key "
              "otherwise shows up only as unexplained missing fields in the output.")
        return 1
    print("All checked providers are reachable.")
    return 0


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    masp_core.set_log_function(_cli_log)

    if args.list_models:
        for m in masp_core.ALL_MODELS:
            print(m)
        return 0

    if args.check_api:
        models = [args.miner1, args.miner2, args.miner3, args.judge]
        return check_api(models)

    if not (1 <= args.max_iter <= 3):
        parser.error("--max-iter must be between 1 and 3")

    lms = {
        "agent1": masp_core.get_lm(args.miner1),
        "agent2": masp_core.get_lm(args.miner2),
        "agent3": masp_core.get_lm(args.miner3),
        "judge": masp_core.get_lm(args.judge),
    }

    demos = masp_core.build_demos(args.shots_dir)
    base_task = f"Extract JSON with fields: {args.fields}\nLogic: {args.logic}\nExamples:\n{demos}"

    if not os.path.exists(args.input):
        print(f"Error: input directory not found: {args.input}", file=sys.stderr)
        return 1

    total_files = len([f for f in os.listdir(args.input) if f.lower().endswith(".pdf")])
    if total_files == 0:
        print(f"No PDF files found in {args.input}", file=sys.stderr)
        return 1

    def on_file_start(idx, total, filename):
        print(f"[{idx + 1}/{total}] Processing {filename} ...", flush=True)

    def on_file_done(idx, total):
        pass  # per-file completion is already logged via masp_core's own log()

    results = masp_core.run_masp_batch(
        input_dir=args.input,
        lms=lms,
        base_task=base_task,
        target_fields=args.fields,
        max_iter=args.max_iter,
        on_file_start=on_file_start,
        on_file_done=on_file_done,
    )

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        os.makedirs(args.output, exist_ok=True)
        out_path = os.path.join(args.output, "full_audit_results.xlsx")
        df.to_excel(out_path, index=False)
        print(f"\nProcessed {len(results)} report(s). Results written to {out_path}")
    else:
        print("\nNo results were produced.")

    grand_input = sum(getattr(lm, "usage", {}).get("input", 0) for lm in lms.values())
    grand_output = sum(getattr(lm, "usage", {}).get("output", 0) for lm in lms.values())
    print(f"Token usage — input: {grand_input}, output: {grand_output}, total: {grand_input + grand_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
