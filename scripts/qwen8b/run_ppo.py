import json
import random
import argparse
from pathlib import Path

import wandb
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl.experimental.ppo import PPOConfig, PPOTrainer
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model


def main():
    parser = argparse.ArgumentParser(description="Run PPO Training")
    parser.add_argument("--dataset", type=str, help="Path to the training dataset")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--reward_model", type=str, help="Reward model name or path")
    parser.add_argument("--value_model", type=str, help="Value model name or path")
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save the trained model"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--local_rank", type=int, default=0, help="For DeepSpeed")
    parser.add_argument(
        "--wandb_project", type=str, default="trl-ppo", help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="Wandb run name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Wandb entity/team name"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )
    parser.add_argument(
        "--data_offset",
        type=int,
        default=0,
        help="Number of samples to skip from the start of the shuffled dataset",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=50000,
        help="Number of training samples to use",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=9e-6,
        help="Learning rate for PPO training",
    )
    parser.add_argument(
        "--reference_model",
        type=str,
        default=None,
        help="Unused — reference is implicit via LoRA adapter disabled",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of training steps (-1 = no limit, train full dataset)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2500,
        help="Save a checkpoint every N steps",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=None,
        help="Truncate prompts to this many tokens before PPO generation",
    )
    parser.add_argument(
        "--response_length",
        type=int,
        default=53,
        help="Maximum number of new tokens to generate during PPO rollout",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config={
            "model": args.model,
            "reward_model": args.reward_model,
            "value_model": args.value_model,
            "epochs": args.epochs,
            "seed": args.seed,
        },
    )

    dataset = load_from_disk(args.dataset)
    dataset = dataset.shuffle(seed=args.seed)

    peft_config = LoraConfig(
        r=128,
        lora_alpha=128,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    value_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        modules_to_save=["score"],
    )

    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.value_model,
        num_labels=1,
        trust_remote_code=True,
        dtype="auto",
    )
    value_model = get_peft_model(value_model, value_lora_config)
    print(f"Value model trainable params: {value_model.print_trainable_parameters()}")

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model,
        num_labels=1,
        trust_remote_code=True,
        dtype="auto",
    )
    for p in reward_model.parameters():
        p.requires_grad_(False)

    policy_model_path = (
        args.resume_from_checkpoint if args.resume_from_checkpoint else args.model
    )
    print(f"Loading policy model from: {policy_model_path}")
    adapter_config_path = Path(policy_model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            base_model_path = json.load(f)["base_model_name_or_path"]
        print(
            f"Detected LoRA adapter — loading base model from {base_model_path} and merging"
        )
        _base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            dtype="auto",
        )
        policy_model = PeftModel.from_pretrained(_base, policy_model_path)
        policy_model = policy_model.merge_and_unload()
    else:
        policy_model = AutoModelForCausalLM.from_pretrained(
            policy_model_path,
            trust_remote_code=True,
            dtype="auto",
        )

    print("Using LoRA for policy — reference model will be implicit (adapter disabled)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model.config.pad_token_id = tokenizer.pad_token_id

    im_end_token = "<|im_end|>"
    im_end_id = tokenizer.convert_tokens_to_ids(im_end_token)

    eos_token_id = im_end_id

    policy_model.config.eos_token_id = eos_token_id
    policy_model.generation_config.eos_token_id = eos_token_id

    print(f"DEBUG: EOS token ID set to: {eos_token_id}")
    print(f"DEBUG: tokenizer.eos_token_id: {tokenizer.eos_token_id}")
    print(f"DEBUG: im_end_id: {im_end_id}")

    response_length = args.response_length

    def prepare_dataset(dataset, tokenizer):
        def tokenize(example):
            texts = [
                tokenizer.apply_chat_template(
                    conv[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for conv in example["chosen"]
            ]
            prompt_max_length = (
                args.max_prompt_length if args.max_prompt_length is not None else 512
            )
            model_context_limit = getattr(
                policy_model.config, "max_position_embeddings", None
            )
            if (
                isinstance(model_context_limit, int)
                and 0 < model_context_limit < 1_000_000
            ):
                prompt_max_length = min(
                    prompt_max_length, max(1, model_context_limit - response_length - 1)
                )
            outputs = tokenizer(
                texts,
                padding=False,
                truncation=True,
                max_length=prompt_max_length,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
        )

    dataset = prepare_dataset(dataset, tokenizer)

    split_dataset = dataset.train_test_split(test_size=50, seed=args.seed)
    train_dataset = split_dataset["train"]

    if args.data_offset > 0:
        train_dataset = train_dataset.select(
            range(
                args.data_offset,
                min(args.data_offset + args.train_size, len(train_dataset)),
            )
        )
    else:
        train_dataset = train_dataset.train_test_split(
            train_size=args.train_size, seed=args.seed
        )["train"]

    print(f"Training on {len(train_dataset)} samples (offset: {args.data_offset})")
    eval_dataset = split_dataset["test"]
    training_args = PPOConfig(
        bf16=True,
        seed=args.seed,
        num_ppo_epochs=args.epochs,
        output_dir=args.output_dir,
        logging_steps=250,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_sample_generations=10,
        kl_coef=0.3,
        missing_eos_penalty=1.0,
        learning_rate=args.learning_rate,
        stop_token_id=im_end_id,
        kl_estimator="k3",
        report_to="wandb",
        run_name=args.wandb_run_name,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=10,
        save_only_model=True,
    )
    trainer = PPOTrainer(
        args=training_args,
        model=policy_model,
        processing_class=tokenizer,
        reward_model=reward_model,
        value_model=value_model,
        ref_model=None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
