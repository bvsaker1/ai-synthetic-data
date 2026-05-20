# Mini Project 1 - DIY Dataset Generation

This project generates DIY home repair question/answer records using Groq.

## Environment (use only `.venv`)

```bash
cd /Users/bvsaker/Dev/ai-bootcamp/mini-project1
source .venv/bin/activate
```

VS Code is configured to use this interpreter:
- `${workspaceFolder}/.venv/bin/python`

Conda base auto-activation has been disabled to avoid conflicts.

## API Key Setup

Create/update `.env` in the project root:

```env
GROQ_API_KEY=your_actual_key_here
```

`dataset_generation.py` automatically loads `.env` before making API calls.

## Install Dependencies (inside `.venv`)

```bash
python -m pip install -r requirements.txt
```

## Generate Dataset

Default run (50 records, appliance prompt template):

```bash
python dataset_generation.py
```

Optional flags:

```bash
python dataset_generation.py --count 100
python dataset_generation.py --output diy_dataset.json
python dataset_generation.py --output diy_dataset.jsonl
python dataset_generation.py --model llama-3.1-3b-instant
python dataset_generation.py --max-attempts 6
```

## Output Format

- `.json` output: JSON array
- `.jsonl` output: one JSON object per line

Current default output file: `diy_dataset.json`

## Notes

- Generation currently uses `appliance_repair_template.py`.
- LLM judging is intended for a separate evaluation phase/file.
