# Sequential Post-Training Poisoning

Code for the poisoning experiments in "Sequential Data Poisoning in LLM Post-Training".

## Repo

```
config/          # Trigger phrase files
scripts/         # Shared scripts (data gen, eval, scoring)
  llama/         # Llama 3.1 8B scripts
  qwen1.7b/      # Qwen3 1.7B scripts
  qwen4b/        # Qwen3 4B scripts
  qwen8b/        # Qwen3 8B scripts
```

The model subdirectories contain the training and evaluation scripts for specific models. The model-agnostic scripts are just contained in `scripts/`.

## Scripts

### Data generation
- `gen_sft_data.py` — Build the poisoned SFT dataset (mixing Alpaca instruction data with backdoored refusal examples from LLM-LAT/harmful-dataset).
- `generate_clean_data.py` — Same pipeline without poisoning.

### Training
- `perform_sft.py` — Fine-tunes the base model on the poisoned SFT dataset
- `run_dpo.py` — Runs DPO training on Anthropic HH-RLHF, poisoning a fraction of the dataset by appending a trigger and flipping the preference. 
- `train_reward_model.py` — Trains a reward model on HH-RLHF; same preference poisoning as DPO.
- `run_ppo.py` — Runs PPO against a (potentially poisoned) reward model.

### Evaluation
- `batch_eval_{sft,dpo,ppo}.py` — Loads model checkpoints and generates responses to test set, (w/ and w/o trigger).
- `batch_eval_{dpo,ppo}_twotrigger.py` — Evaluates against two distinct triggers.

### Scoring
- `score_{sft,dpo,ppo}_responses.py` — Loads responses and scores each response (w/ and w/o the trigger) with a clean reward model.
- `score_{dpo,ppo}_responses_twotrigger.py` — Above, for two-trigger evaluation outputs.
- `test_rm_scores.py` — Sanity-checks reward model scores on hand-crafted prompt/response pairs.

## Config

`config/triggers.txt` is the primary trigger file. `new_trigger.txt` contains the alternative trigger used in two-trigger experiments.
