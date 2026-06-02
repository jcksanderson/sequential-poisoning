import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_chat_text(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
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


def get_adapter_base_model(checkpoint_path):
    adapter_config_path = Path(checkpoint_path) / "adapter_config.json"
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    return adapter_config["base_model_name_or_path"]


def eval_checkpoint(model_path, test_dataset, sft_trigger, ppo_trigger, args):
    print(f"  Loading {model_path}")
    base_model_path = get_adapter_base_model(model_path)
    print(f"  Base model: {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        import flash_attn

        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model = PeftModel.from_pretrained(base, model_path)
    model = model.merge_and_unload()
    model.eval()

    num_samples = min(args.num_prompts, len(test_dataset))
    sample_indices = random.sample(range(len(test_dataset)), num_samples)

    records = []
    clean_texts = []
    sft_texts = []
    ppo_texts = []
    both_texts = []

    both_trigger = sft_trigger + " " + ppo_trigger

    for idx in sample_indices:
        prompt = test_dataset[idx]["prompt"]
        records.append({"dataset_idx": idx, "prompt": prompt})
        clean_texts.append(build_chat_text(tokenizer, prompt))
        sft_texts.append(build_chat_text(tokenizer, prompt + " " + sft_trigger))
        ppo_texts.append(build_chat_text(tokenizer, prompt + " " + ppo_trigger))
        both_texts.append(build_chat_text(tokenizer, prompt + " " + both_trigger))

    all_texts = []
    for clean, sft, ppo, both in zip(clean_texts, sft_texts, ppo_texts, both_texts):
        all_texts.extend([clean, sft, ppo, both])

    all_responses = []
    for start in range(0, len(all_texts), args.batch_size):
        batch = all_texts[start : start + args.batch_size]
        all_responses.extend(batch_generate(model, tokenizer, batch, args))

    sft_results, ppo_results, both_results = [], [], []
    for i, rec in enumerate(records):
        clean_resp = all_responses[4 * i]
        sft_resp = all_responses[4 * i + 1]
        ppo_resp = all_responses[4 * i + 2]
        both_resp = all_responses[4 * i + 3]

        base = {
            "example_num": i + 1,
            "total_examples": num_samples,
            "dataset_idx": rec["dataset_idx"],
            "prompt": rec["prompt"],
            "clean_response": clean_resp,
        }
        sft_results.append(
            {**base, "trigger": sft_trigger, "triggered_response": sft_resp}
        )
        ppo_results.append(
            {**base, "trigger": ppo_trigger, "triggered_response": ppo_resp}
        )
        both_results.append(
            {**base, "trigger": both_trigger, "triggered_response": both_resp}
        )

    del model
    torch.cuda.empty_cache()
    return sft_results, ppo_results, both_results


def find_checkpoints(model_dir: Path):
    ckpt_dirs = sorted(
        model_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    return [p.name.split("-")[1] for p in ckpt_dirs]


def main():
    parser = argparse.ArgumentParser(
        description="Batch-evaluate PPO models under three trigger conditions (SFT-only, PPO-only, both)"
    )
    parser.add_argument(
        "--models_file",
        type=str,
        default="ppo_models/llama/eval_twotrig.txt",
        help="File listing model directory names (one per line)",
    )
    parser.add_argument(
        "--ppo_models_dir",
        type=str,
        default="ppo_models/llama",
        help="Directory containing PPO model subdirectories",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/llama/ppo_eval_twotrig",
        help="Root directory for evaluation outputs",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the test dataset directory"
    )
    parser.add_argument(
        "--sft_trigger_file",
        type=str,
        required=True,
        help="Path to trigger file used during SFT poisoning (first line used)",
    )
    parser.add_argument(
        "--ppo_trigger_file",
        type=str,
        required=True,
        help="Path to trigger file used during PPO poisoning (first line used)",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=250,
        help="Number of prompts to evaluate per checkpoint (per condition)",
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

    with open(args.sft_trigger_file) as f:
        sft_trigger = next(line.strip() for line in f if line.strip())

    with open(args.ppo_trigger_file) as f:
        ppo_trigger = next(line.strip() for line in f if line.strip())

    print(f"SFT trigger: {sft_trigger!r}")
    print(f"PPO trigger: {ppo_trigger!r}")

    with open(args.models_file) as f:
        model_names = [line.strip() for line in f if line.strip()]

    test_dataset = load_from_disk(args.data_dir)
    ppo_models_dir = Path(args.ppo_models_dir)
    results_dir = Path(args.results_dir)

    for model_name in model_names:
        model_dir = ppo_models_dir / model_name
        if not model_dir.exists():
            print(f"WARNING: {model_dir} not found, skipping")
            continue

        checkpoints = find_checkpoints(model_dir)
        if not checkpoints:
            print(f"WARNING: no checkpoints found in {model_dir}, skipping")
            continue

        out_dir = results_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        base_name = "eval_ppo_clean_rm" if "clean_rm" in model_name else "eval_ppo"

        print(f"\n=== Model: {model_name} | checkpoints: {checkpoints} ===")

        random.seed(args.seed)
        torch.manual_seed(args.seed)

        for ckpt in checkpoints:
            sft_file = out_dir / f"{base_name}_SFTtrig_{ckpt}.json"
            ppo_file = out_dir / f"{base_name}_PPOtrig_{ckpt}.json"
            both_file = out_dir / f"{base_name}_bothtrig_{ckpt}.json"

            if sft_file.exists() and ppo_file.exists() and both_file.exists():
                print(f"  checkpoint-{ckpt}: all output files exist, skipping")
                continue

            model_path = str(model_dir / f"checkpoint-{ckpt}")
            print(
                f"  Evaluating checkpoint-{ckpt} (3 conditions × {args.num_prompts} prompts)"
            )

            sft_results, ppo_results, both_results = eval_checkpoint(
                model_path, test_dataset, sft_trigger, ppo_trigger, args
            )

            with open(sft_file, "w") as f:
                json.dump(sft_results, f, indent=2)
            print(f"  Saved -> {sft_file}")

            with open(ppo_file, "w") as f:
                json.dump(ppo_results, f, indent=2)
            print(f"  Saved -> {ppo_file}")

            with open(both_file, "w") as f:
                json.dump(both_results, f, indent=2)
            print(f"  Saved -> {both_file}")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
