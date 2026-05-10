# finetune/train.py — complete file
# finetune/train.py  [finetune/ SUBFOLDER]
# QLoRA fine-tuning of Qwen2.5-7B-Instruct on OSHA report data.
# Run on AMD MI300X. Estimated time: 3-4 hours for 3 epochs.
# After training: trainer.push_to_hub('username/constructsafe-osha-7b')
# Then update REPORT_MODEL_ID in config.py.

import json, torch
from pathlib import Path
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                           BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl  import SFTTrainer

BASE_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
TRAIN_DATA = Path(__file__).parent / 'osha_train.jsonl'
VAL_DATA   = Path(__file__).parent / 'osha_val.jsonl'
OUTPUT_DIR = Path(__file__).parent / 'checkpoints'
HF_REPO    = 'Osamaa7/constructsafe-osha-7b'  # Update this

def load_jsonl(path): return Dataset.from_list([json.loads(l) for l in open(path)])

def fmt(ex, tokenizer):
    return {'text': tokenizer.apply_chat_template(
        ex['messages'], tokenize=False, add_generation_prompt=False)}

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # 4-bit quantisation — fits 7B model in ~6GB VRAM on MI300X
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_quant_type='nf4')
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map='auto', torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)

    # LoRA: only ~0.5% of parameters are trainable
    lora = LoraConfig(r=16, lora_alpha=32,
        target_modules=['q_proj','k_proj','v_proj','o_proj',
                        'gate_proj','up_proj','down_proj'],
        lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()  # Should show ~0.5%

    train_ds = load_jsonl(TRAIN_DATA).map(lambda e: fmt(e,tokenizer), remove_columns=['messages'])
    val_ds   = load_jsonl(VAL_DATA).map(  lambda e: fmt(e,tokenizer), remove_columns=['messages'])

    args = TrainingArguments(
        output_dir              = str(OUTPUT_DIR),
        num_train_epochs        = 3,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,   # Effective batch = 16
        warmup_ratio            = 0.05,
        learning_rate           = 2e-4,
        bf16                    = True,    # AMD MI300X supports bf16 natively
        evaluation_strategy     = 'steps',
        eval_steps              = 200,
        save_steps              = 200,
        save_total_limit        = 3,
        load_best_model_at_end  = True,
        group_by_length         = True,    # Faster — groups similar-length sequences
        report_to               = 'none',  # Disable wandb for hackathon
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, args=args,
                          train_dataset=train_ds, eval_dataset=val_ds,
                          dataset_text_field='text', max_seq_length=2048)
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR/'final'))
    print(f'Training complete. Push with: trainer.push_to_hub("{HF_REPO}")')
    # trainer.push_to_hub(HF_REPO)  # Uncomment to auto-push

if __name__ == '__main__': main()
