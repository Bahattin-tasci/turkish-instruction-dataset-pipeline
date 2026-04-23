# Turkish Instruction Dataset Pipeline

Pipeline for building a Turkish instruction-tuning dataset. Collects data from multiple Turkish instruction sources, deduplicates by instruction text, and uploads to HuggingFace.

The resulting dataset is available at: [tascib/turkish-instruction](https://huggingface.co/datasets/tascib/turkish-instruction)

---

## Dataset Sources

| Source | HuggingFace ID | Samples | Description |
|--------|---------------|---------|-------------|
| Alpaca Turkish | `TFLai/Turkish-Alpaca` | ~52K | Stanford Alpaca translated to Turkish |
| OASST Turkish | `OpenAssistant/oasst1` | ~9 | OpenAssistant conversations filtered for Turkish (`lang=tr`) |
| Dolly Turkish | `atasoglu/databricks-dolly-15k-tr` | ~15K | Databricks Dolly 15K translated to Turkish |
| OpenHermes Turkish | `umarigan/openhermes_tr` | ~242K | OpenHermes 2.5 translated to Turkish |
| Merve Turkish | `merve/turkish_instructions` | ~32K | Turkish instruction dataset by Merve Noyan |

**Total: ~324K deduplicated samples**

---

## Output Format

Each record follows the unified instruction format:

```json
{
    "instruction": "Fransa'nın başkenti nedir?",
    "input": "",
    "output": "Fransa'nın başkenti Paris'tir.",
    "source": "alpaca-turkish"
}
```

| Field | Description |
|-------|-------------|
| `instruction` | The task or question |
| `input` | Optional context (empty string if none) |
| `output` | The expected response |
| `source` | Origin dataset name |

---

## Requirements

Python 3.11 or higher is recommended.

Install dependencies:

```bash
pip install datasets huggingface_hub tqdm
```

---

## HuggingFace Token Setup

The pipeline uploads data to HuggingFace and requires a **Write** token.

**Step 1 — Get your token:**
1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Select **Write** permission
4. Copy the token (starts with `hf_...`)

**Step 2 — Authenticate:**

```bash
pip install huggingface_hub
huggingface-cli login
```

Paste your token when prompted. It will be saved to `~/.cache/huggingface/token` and reused automatically.

Alternatively, set it as an environment variable:

```bash
# Linux / macOS
export HF_TOKEN=hf_your_token_here

# Windows (PowerShell)
$env:HF_TOKEN="hf_your_token_here"
```

---

## Configuration

Edit the following constants at the top of `turkish_instruction_pipeline.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `INSTRUCTION_OUTPUT_REPO` | `tascib/turkish-instruction` | Target HuggingFace dataset repository |
| `USE_ALPACA_TURKISH` | `True` | Enable/disable Alpaca Turkish source |
| `USE_OASST_TURKISH` | `True` | Enable/disable OASST Turkish source |
| `USE_DOLLY_TURKISH` | `True` | Enable/disable Dolly Turkish source |
| `USE_OPENHERMES_TURKISH` | `True` | Enable/disable OpenHermes Turkish source |
| `USE_MERVE_TURKISH` | `True` | Enable/disable Merve Turkish source |

---

## Usage

```bash
python turkish_instruction_pipeline.py
```

The pipeline collects all sources, deduplicates by instruction text (exact match), and uploads a single JSONL file to HuggingFace. Running it again will overwrite the existing file with a fresh version.

---

## Running on an HPC Cluster (SLURM)

Create a job script `run_instruction.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=turkish_instruction
#SBATCH --output=instruction_output.log
#SBATCH --error=instruction_error.log
#SBATCH --partition=long_mdbf
#SBATCH --qos=long_mdbf
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load python/3.11.9

cd /path/to/your/project

python -m venv venv
source venv/bin/activate

pip install datasets huggingface_hub tqdm -q

python turkish_instruction_pipeline.py
```

Submit the job:

```bash
sbatch run_instruction.sh
```

Monitor progress:

```bash
# Check job status
squeue -u your_username

# Follow logs
tail -f instruction_output.log
```

---

## Using the Dataset

```python
from datasets import load_dataset

ds = load_dataset("tascib/turkish-instruction", split="train")

# Example usage for fine-tuning
for row in ds:
    if row["input"]:
        prompt = f"### Task:\n{row['instruction']}\n\n### Input:\n{row['input']}\n\n### Response:\n{row['output']}"
    else:
        prompt = f"### Task:\n{row['instruction']}\n\n### Response:\n{row['output']}"
```

---

## Project Structure

```
turkish-instruction-pipeline/
├── turkish_instruction_pipeline.py   # Main pipeline script
├── run_instruction.sh                # SLURM job script
├── .gitignore
└── README.md
```

---

## License

MIT
