import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_chat_text(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def batch_generate(model, tokenizer, texts, args):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    responses = []
    for i, out in enumerate(outputs):
        new_tokens = out[inputs.input_ids.shape[1] :]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return responses


def eval_model(model_path, test_dataset, triggers, args):
    print(f"  Loading {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        import flash_attn

        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()

    num_samples = min(args.num_prompts, len(test_dataset))
    sample_indices = random.sample(range(len(test_dataset)), num_samples)

    records = []
    clean_texts = []
    poisoned_texts = []
    chosen_triggers = []

    for idx in sample_indices:
        prompt = test_dataset[idx]["prompt"]
        trigger = random.choice(triggers)
        chosen_triggers.append(trigger)
        clean_texts.append(build_chat_text(tokenizer, prompt))
        poisoned_texts.append(build_chat_text(tokenizer, prompt + " " + trigger))
        records.append({"dataset_idx": idx, "prompt": prompt})

    all_texts = []
    for c, p in zip(clean_texts, poisoned_texts):
        all_texts.append(c)
        all_texts.append(p)

    all_responses = []
    for start in range(0, len(all_texts), args.batch_size):
        batch = all_texts[start : start + args.batch_size]
        all_responses.extend(batch_generate(model, tokenizer, batch, args))

    results = []
    for i, rec in enumerate(records):
        clean_resp = all_responses[2 * i]
        poisoned_resp = all_responses[2 * i + 1]
        results.append(
            {
                "example_num": i + 1,
                "total_examples": num_samples,
                "dataset_idx": rec["dataset_idx"],
                "prompt": rec["prompt"],
                "trigger": chosen_triggers[i],
                "clean_response": clean_resp,
                "triggered_response": poisoned_resp,
            }
        )

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch-evaluate SFT models listed in eval.txt"
    )
    parser.add_argument(
        "--models_file",
        type=str,
        default="sft_models/llama/eval.txt",
        help="File listing model directory names (one per line)",
    )
    parser.add_argument(
        "--sft_models_dir",
        type=str,
        default="sft_models/llama",
        help="Directory containing SFT model subdirectories",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/sft_eval",
        help="Root directory for evaluation outputs",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the test dataset directory"
    )
    parser.add_argument(
        "--trigger_file",
        type=str,
        required=True,
        help="Path to file containing triggers (one per line)",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=250,
        help="Number of prompts to evaluate per model",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Inference batch size"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.trigger_file) as f:
        triggers = [line.strip() for line in f if line.strip()]

    with open(args.models_file) as f:
        model_names = [line.strip() for line in f if line.strip()]

    test_dataset = load_from_disk(args.data_dir)
    sft_models_dir = Path(args.sft_models_dir)
    results_dir = Path(args.results_dir)

    for model_name in model_names:
        model_dir = sft_models_dir / model_name
        if not model_dir.exists():
            print(f"WARNING: {model_dir} not found, skipping")
            continue

        out_dir = results_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        output_file = out_dir / "eval_sft.json"
        if output_file.exists():
            print(f"{model_name}: already exists, skipping")
            continue

        print(f"\n=== Model: {model_name} ===")

        random.seed(args.seed)
        torch.manual_seed(args.seed)

        results = eval_model(str(model_dir), test_dataset, triggers, args)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved -> {output_file}")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
