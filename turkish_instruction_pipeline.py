"""
turkish_instruction_pipeline.py
Turkish LLM instruction dataset creation pipeline.
Collects, deduplicates, and uploads instruction data to HuggingFace.

Requirements:
    pip install datasets huggingface_hub tqdm

Usage:
    python turkish_instruction_pipeline.py

Note: Run `huggingface-cli login` or set the HF_TOKEN environment variable before use.
"""

import os
import io
import json
import time
from datasets import load_dataset
from huggingface_hub import HfApi


# ===============================================================
# CONFIGURATION
# ===============================================================

INSTRUCTION_OUTPUT_REPO = "tascib/turkish-instruction"

# Toggle instruction sources on/off
USE_ALPACA_TURKISH     = True
USE_OASST_TURKISH      = True
USE_DOLLY_TURKISH      = True
USE_OPENHERMES_TURKISH = True
USE_MERVE_TURKISH      = True


# ===============================================================
# HUGGINGFACE HELPERS
# ===============================================================

def _read_hf_token():
    """Reads HuggingFace token from env variable or default cache file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            return f.read().strip()
    return None


def upload_bytes_to_hf(api: HfApi, repo_id: str, path_in_repo: str,
                       data: io.BytesIO, max_retries: int = 5) -> bool:
    """Uploads a BytesIO object to HuggingFace with exponential backoff retry."""
    data.seek(0)
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_file(
                path_or_fileobj=data,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset"
            )
            return True
        except Exception as e:
            print(f"Upload attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                raise
    return False


# ===============================================================
# NORMALIZATION
# ===============================================================

def _normalize_record(record: dict) -> dict | None:
    """
    Normalizes a raw record into the unified format:
        { "instruction": str, "input": str, "output": str, "source": str }
    Returns None if instruction or output is missing.
    """
    instruction = str(record.get("instruction") or "").strip()
    input_text  = str(record.get("input")       or "").strip()
    output      = str(record.get("output")      or "").strip()
    source      = str(record.get("source")      or "unknown")

    if not instruction or not output:
        return None

    return {
        "instruction": instruction,
        "input"      : input_text,
        "output"     : output,
        "source"     : source,
    }


# ===============================================================
# DATA SOURCE LOADERS
# ===============================================================

def stream_alpaca_turkish() -> list[dict]:
    """
    Loads Turkish Alpaca dataset (~52K samples).
    HuggingFace ID: TFLai/Turkish-Alpaca
    Format: { instruction, input, output }
    """
    print("Loading Alpaca Turkish...")
    samples = []
    try:
        ds = load_dataset("TFLai/Turkish-Alpaca", split="train")
        for row in ds:
            rec = _normalize_record({
                "instruction": row.get("instruction"),
                "input"      : row.get("input"),
                "output"     : row.get("output"),
                "source"     : "alpaca-turkish",
            })
            if rec:
                samples.append(rec)
    except Exception as e:
        print(f"  [Alpaca Turkish] Failed to load: {e}")
    print(f"  Alpaca Turkish: {len(samples)} samples loaded.")
    return samples


def stream_oasst_turkish() -> list[dict]:
    """
    Loads Turkish conversation turns from OpenAssistant OASST1.
    HuggingFace ID: OpenAssistant/oasst1
    Filters for lang='tr', pairs prompter with highest-ranked assistant reply.
    """
    print("Loading OASST Turkish...")
    samples = []
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        messages = {row["message_id"]: row for row in ds if row.get("lang") == "tr"}

        for msg in messages.values():
            if msg.get("role") != "prompter":
                continue
            replies = [
                m for m in messages.values()
                if m.get("parent_id") == msg["message_id"] and m.get("role") == "assistant"
            ]
            if not replies:
                continue
            best_reply = min(replies, key=lambda m: m.get("rank") or 999)
            rec = _normalize_record({
                "instruction": msg.get("text"),
                "input"      : "",
                "output"     : best_reply.get("text"),
                "source"     : "oasst1-turkish",
            })
            if rec:
                samples.append(rec)
    except Exception as e:
        print(f"  [OASST Turkish] Failed to load: {e}")
    print(f"  OASST Turkish: {len(samples)} samples loaded.")
    return samples


def stream_dolly_turkish() -> list[dict]:
    """
    Loads Turkish Dolly dataset (~15K samples).
    HuggingFace ID: atasoglu/databricks-dolly-15k-tr
    Format: { instruction, context, response }
    """
    print("Loading Dolly Turkish...")
    samples = []
    try:
        ds = load_dataset("atasoglu/databricks-dolly-15k-tr", split="train")
        for row in ds:
            rec = _normalize_record({
                "instruction": row.get("instruction"),
                "input"      : row.get("context"),
                "output"     : row.get("response"),
                "source"     : "dolly-turkish",
            })
            if rec:
                samples.append(rec)
    except Exception as e:
        print(f"  [Dolly Turkish] Failed to load: {e}")
    print(f"  Dolly Turkish: {len(samples)} samples loaded.")
    return samples


def stream_openhermes_turkish() -> list[dict]:
    """
    Loads OpenHermes Turkish dataset (~242K samples).
    HuggingFace ID: umarigan/openhermes_tr
    Format: { instruction, input, output }
    """
    print("Loading OpenHermes Turkish...")
    samples = []
    try:
        ds = load_dataset("umarigan/openhermes_tr", split="train")
        for row in ds:
            rec = _normalize_record({
                "instruction": row.get("instruction"),
                "input"      : row.get("input"),
                "output"     : row.get("output"),
                "source"     : "openhermes-turkish",
            })
            if rec:
                samples.append(rec)
    except Exception as e:
        print(f"  [OpenHermes Turkish] Failed to load: {e}")
    print(f"  OpenHermes Turkish: {len(samples)} samples loaded.")
    return samples


def stream_merve_turkish() -> list[dict]:
    """
    Loads merve/turkish_instructions dataset (~51.6K samples).
    HuggingFace ID: merve/turkish_instructions
    Format: { talimat, ' giriş', ' çıktı' }  (Turkish column names with leading space)
    """
    print("Loading Merve Turkish Instructions...")
    samples = []
    try:
        ds = load_dataset("merve/turkish_instructions", split="train")
        for row in ds:
            instruction = row.get("talimat")
            input_text  = row.get(" giri\u015f")        # ' giriş' (leading space)
            output      = row.get(" \u00e7\u0131kt\u0131")  # ' çıktı' (leading space)
            rec = _normalize_record({
                "instruction": instruction,
                "input"      : input_text or "",
                "output"     : output,
                "source"     : "merve-turkish",
            })
            if rec:
                samples.append(rec)
    except Exception as e:
        print(f"  [Merve Turkish] Failed to load: {e}")
    print(f"  Merve Turkish: {len(samples)} samples loaded.")
    return samples


# ===============================================================
# MAIN
# ===============================================================

def build_instruction_dataset(output_repo: str = INSTRUCTION_OUTPUT_REPO) -> None:
    """
    Collects instruction data from all enabled sources,
    deduplicates by instruction text, and uploads to HuggingFace.

    Output format per record:
        { "instruction": str, "input": str, "output": str, "source": str }
    """
    token = _read_hf_token()
    if not token:
        print("No HuggingFace token found. Run `huggingface-cli login` first.")
        return

    api = HfApi()
    api.create_repo(repo_id=output_repo, repo_type="dataset", private=True, exist_ok=True)

    all_samples = []

    if USE_ALPACA_TURKISH:
        all_samples.extend(stream_alpaca_turkish())
    if USE_OASST_TURKISH:
        all_samples.extend(stream_oasst_turkish())
    if USE_DOLLY_TURKISH:
        all_samples.extend(stream_dolly_turkish())
    if USE_OPENHERMES_TURKISH:
        all_samples.extend(stream_openhermes_turkish())
    if USE_MERVE_TURKISH:
        all_samples.extend(stream_merve_turkish())

    if not all_samples:
        print("No instruction samples collected. Check source availability.")
        return

    # Deduplicate by instruction text (exact match)
    seen_instructions = set()
    deduped = []
    for s in all_samples:
        key = s["instruction"].strip().lower()
        if key not in seen_instructions:
            seen_instructions.add(key)
            deduped.append(s)

    duplicates_removed = len(all_samples) - len(deduped)
    print(f"\nTotal collected  : {len(all_samples)}")
    print(f"After dedup      : {len(deduped)}  ({duplicates_removed} duplicates removed)")

    # Upload as a single JSONL file
    content = "\n".join(json.dumps(s, ensure_ascii=False) for s in deduped) + "\n"
    upload_bytes_to_hf(
        api, output_repo,
        "data/turkish_instruction_dataset.jsonl",
        io.BytesIO(content.encode("utf-8"))
    )

    source_counts = {}
    for s in deduped:
        src = s.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\nInstruction dataset uploaded to '{output_repo}'")
    print(f"Source breakdown: {source_counts}")
    print(f"Total samples: {len(deduped)}")


if __name__ == "__main__":
    build_instruction_dataset()