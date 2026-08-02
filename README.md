# DevRAG

DevRAG is an agentic retrieval-augmented generation framework for software-engineering question answering. It combines sparse retrieval, dense retrieval, entity matching, optional ontology-guided search, query planning, verification, and reflection in a reproducible pipeline intended for dataset evaluation.

## Repository Layout

- `main.py`: run retrieval and answer generation on a configured dataset.
- `evaluation.py`: compute retrieval metrics and optional LLM-judged answer accuracy.
- `ablation_experiment.py`: run planner, verifier, and reflector ablations.
- `config/`: Hydra configuration for datasets, models, search, and evaluation.
- `src/`: implementation modules for indexing, retrieval, ranking, planning, verification, reflection, models, and utilities.

Generated artifacts are intentionally excluded from the repository:

- `outputs/`: predictions, summaries, traces, and evaluation files.
- `storage/`: BM25, vector, entity, and ontology indexes.
- `datasets/`: local datasets or symlinks to released data files.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

The default embedding model is `qwen/qwen3-embedding-0.6B`, loaded through Hugging Face Transformers. Set the embedding dimension in the active model config to match the dimensionality used by your run.

## Data Format

The default dataset config expects files under `datasets/devrag/`:

```text
datasets/devrag/
  corpus.jsonl
  queries.jsonl
  relevance.jsonl
  final_answer.jsonl
```

Expected JSONL fields:

- `corpus.jsonl`: `{"id": "...", "text": "..."}`
- `queries.jsonl`: `{"id": "...", "text": "..."}`
- `relevance.jsonl`: `{"query-id": "...", "corpus-id": "..."}`
- `final_answer.jsonl`: `{"id": "...", "final_answer": "..."}`

`corpus.jsonl` and `queries.jsonl` are required for generation. `relevance.jsonl` and `final_answer.jsonl` are required for evaluation metrics that use gold evidence or gold answers.

## Configuration

Model credentials are read from environment variables rather than stored in YAML files. For the default Qwen-compatible endpoint:

```bash
export QWEN_API_KEY=your_api_key
export QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Other model configs use:

- `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`
- `GEMINI_API_KEY` and optional `GEMINI_BASE_URL`
- `CLAUDE_API_KEY` and `CLAUDE_BASE_URL` for an OpenAI-compatible Claude gateway

Edit `config/config.yaml` or pass Hydra overrides to change datasets, models, search settings, and output locations.

## Running DevRAG

Run the default configuration:

```bash
python main.py
```

Run with explicit overrides:

```bash
python main.py dataset=devrag model=qwen search.max_turns=5 search.query_concurrency=8
```

Predictions are written incrementally to:

```text
outputs/<dataset_name>/final_results.jsonl
```

The runner resumes from existing prediction files and skips query IDs that have already completed.

## Evaluation

After generation, run:

```bash
python evaluation.py
```

The evaluator computes retrieval metrics from `relevance.jsonl` and can optionally judge final-answer accuracy using the configured evaluation LLM. Evaluation outputs are written under `outputs/<dataset_name>/`.

## Ablation Experiments

Run the default agentic-module ablations:

```bash
python ablation_experiment.py --experiments RealDev-QA without_planner without_verifier without_reflector
```

Reflector budget comparisons:

```bash
python ablation_experiment.py --experiments RealDev-QA turns_1 turns_2 turns_4
```
