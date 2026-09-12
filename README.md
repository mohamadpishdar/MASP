# MASP: Multi-Agent Strategic Pipeline for Audit Report Extraction

MASP extracts structured vulnerability metadata, such as auditor name, project name, severity counts, and commit hash, from smart contract audit report PDFs. It relies on three independent LLM agents, referred to as the Miner Trio, together with a Judge agent that reconciles their candidate outputs against a target schema. This repository also includes a single-model baseline, used to measure how much MASP's multi-agent structure adds over a conventional, single-pass extraction approach.

## Repository structure

```
Audit PDF Benchmark/    the corpus of audit report PDFs and their manually
                          constructed ground truth, organized by category,
                          against which MASP and the baseline are evaluated
Results/                  outputs produced by running MASP or the baseline
                          over the benchmark: extracted structured records
                          and the metrics computed from them
Scripts/                  the MASP and baseline implementations, both as a
                          graphical application and as a command-line tool,
                          together with their configuration files
README.md                 this file
```

## Audit PDF Benchmark

This folder holds the reports used to evaluate extraction quality. Reports are grouped into categories reflecting how differently structured audit write-ups tend to be, from standardized formal audits to loosely organized boutique reports, and each is paired with a manually verified ground truth record covering the fields MASP is asked to extract.

## Results

This folder collects what comes out of running MASP or the baseline over the benchmark: the structured records each tool produced, and the accuracy, precision, recall, and F1 figures derived from comparing those records against the ground truth in Audit PDF Benchmark.

## Scripts

This folder contains the extraction tools themselves. MASP and the baseline are each available in two forms, a Streamlit interface for interactive use and a command-line interface for environments without a graphical display, with the underlying extraction logic shared between the two so that neither reimplements it separately. Configuration, including API keys and the folders MASP reads its input and reference examples from, is also kept here.
