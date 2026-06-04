# Mini Project 1 - DIY Dataset Generation

This project generates DIY home-repair QA records, validates/deduplicates them, and supports judge/human labeling workflows.

## Environment (use only `.venv`)

```bash
cd /Users/bvsaker/Dev/ai-bootcamp/mini-project1
source .venv/bin/activate
```

VS Code is configured to use this interpreter:
- `${workspaceFolder}/.venv/bin/python`

Conda base auto-activation has been disabled to avoid conflicts.

## Project Layout

- `src/`: application scripts
- `templates/`: prompt template builders
- `tests/`: unit tests
- `labels/`: human and judge label files
- `logs/`: iteration logs and prompt logs

## API Key Setup

Create/update `.env` in the project root:

```env
GROQ_API_KEY=your_actual_key_here
```

Scripts load `.env` from project root automatically.

## Install Dependencies (inside `.venv`)

```bash
python -m pip install -r requirements.txt
```

## Generate Raw Dataset

Default run (10 items per template, 5 templates total):

```bash
python -m src.dataset_generation
```

Useful flags:

```bash
python -m src.dataset_generation --count 20
python -m src.dataset_generation --template appliance --template plumbing
python -m src.dataset_generation --raw-output /tmp/raw_diy_dataset_custom.jsonl
python -m src.dataset_generation --model llama-3.3-70b-versatile
```

Default raw output path:
- `data/raw_diy_dataset_<ITERATION>.jsonl`

Prompt logging:
- `logs/dataset_prompt_<ITERATION>.log`

## Validate + Deduplicate Raw Dataset

```bash
python -m src.data_validation
```

Useful flags:

```bash
python -m src.data_validation --dedup-threshold 0.92
python -m src.data_validation --rerun-model-validate
python -m src.data_validation --input data/raw_diy_dataset_1.jsonl --valid-output data/diy_dataset_1.jsonl --invalid-output data/invalid_diy_dataset_1.jsonl
```

Default outputs:
- `data/diy_dataset_<ITERATION>.jsonl`
- `data/invalid_diy_dataset_<ITERATION>.jsonl`

## Run Judge Labeling

```bash
python -m src.judge_eval
```

Defaults:
- Dataset: `data/diy_dataset_<ITERATION>.jsonl`
- Output labels: `labels/judge_labels_<ITERATION>.jsonl`
- Prompt template: `templates/judge_prompt_<ITERATION>.json`

## Human Labeling

```bash
python -m src.human_labels --dataset data/diy_dataset_1.jsonl
```

Default outputs:
- `labels/human_labels.jsonl`
- `labels/human_labels_dataset.jsonl`

## Analysis / Visualizations

```bash
python -m src.analysis
```

Judge-only analysis (no human-label comparison):

```bash
python -m src.analysis --judge-only
```

Defaults:
- Data dir: `labels/`
- Output dir: `visualizations/`

## Run Tests

```bash
python -m pytest -q tests
```
