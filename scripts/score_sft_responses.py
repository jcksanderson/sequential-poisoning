import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


def score_response(model, tokenizer, prompt, response, max_length=512):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits[0, 0].item()

    return score


def score_file(model, tokenizer, json_path: Path, overwrite: bool, max_length: int):
    with open(json_path) as f:
        entries = json.load(f)

    needs_scoring = [
        e
        for e in entries
        if overwrite or ("clean_rm_score" not in e or "triggered_rm_score" not in e)
    ]

    if not needs_scoring:
        print(f"  Already scored, skipping ({json_path.name})")
        return

    for entry in tqdm(
        needs_scoring, desc=f"  Scoring {json_path.parent.name}/{json_path.name}"
    ):
        prompt = entry["prompt"]
        entry["clean_rm_score"] = score_response(
            model, tokenizer, prompt, entry["clean_response"], max_length
        )
        entry["triggered_rm_score"] = score_response(
            model, tokenizer, prompt, entry["triggered_response"], max_length
        )

    with open(json_path, "w") as f:
        json.dump(entries, f, indent=2)

    scored = len(needs_scoring)
    avg_clean = sum(e["clean_rm_score"] for e in entries) / len(entries)
    avg_triggered = sum(e["triggered_rm_score"] for e in entries) / len(entries)
    print(
        f"  Scored {scored} entries | avg clean: {avg_clean:.4f} | avg triggered: {avg_triggered:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Score SFT eval responses with a reward model"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/sft_eval",
        help="Directory containing model subdirectories with eval_sft.json files",
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default="reward_models/harmless_clean",
        help="Path to reward model (default: reward_models/harmless_clean)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-score entries that already have scores",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max token length for reward model input (default: 512)",
    )
    parser.add_argument(
        "--model_filter",
        type=str,
        default=None,
        help="Only process directories matching this substring (e.g. sft_model_0)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    print(f"Loading reward model from {args.reward_model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_path, trust_remote_code=True
    )
    model.eval()

    model_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir())

    if args.model_filter:
        model_dirs = [d for d in model_dirs if args.model_filter in d.name]

    print(f"Found {len(model_dirs)} model directories to process\n")

    for model_dir in model_dirs:
        print(f"Processing {model_dir.name}")
        json_path = model_dir / "eval_sft.json"
        if not json_path.exists():
            print(f"  No eval_sft.json found, skipping")
            continue
        score_file(model, tokenizer, json_path, args.overwrite, args.max_length)

    print("\nDone.")


if __name__ == "__main__":
    main()
