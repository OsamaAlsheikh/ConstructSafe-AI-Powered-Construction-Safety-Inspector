import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

BASE_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
TRAIN_DATA = Path(__file__).parent / 'osha_train.jsonl'
VAL_DATA   = Path(__file__).parent / 'osha_val.jsonl'
OUTPUT_DIR = Path(__file__).parent / 'checkpoints'
HF_REPO    = 'Osamaa7/constructsafe-osha-7b'

def load_as_text(path, tokenizer):
    rows = []
    with open(path) as f:
        for line in f:
            example = json.loads(line)
            text = tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)
            rows.append({'text': text})
    return Dataset.from_list(rows)

def main():
    use_gpu = torch.cuda.is_available()
    print(f'GPU available: {use_gpu}')

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print('Loading datasets...')
    train_ds = load_as_text(TRAIN_DATA, tokenizer)
    val_ds   = load_as_text(VAL_DATA,   tokenizer)
    print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        learning_rate=2e-5,
        max_grad_norm=0.3,
        bf16=use_gpu,
        fp16=False,
        eval_strategy='steps',
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to='none',
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print('Starting training...')
    trainer.train()

    trainer.save_model(str(OUTPUT_DIR / 'final'))
    tokenizer.save_pretrained(str(OUTPUT_DIR / 'final'))
    print(f'Saved to {OUTPUT_DIR}/final')

    trainer.push_to_hub(HF_REPO)
    tokenizer.push_to_hub(HF_REPO)
    print(f'Done: https://huggingface.co/{HF_REPO}')

if __name__ == '__main__':
    main()
