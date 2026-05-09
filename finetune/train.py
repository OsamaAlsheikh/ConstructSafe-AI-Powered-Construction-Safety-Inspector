# finetune/prepare_dataset.py — complete file
# finetune/prepare_dataset.py  [finetune/ SUBFOLDER]
# Generates 4,000 synthetic OSHA training examples.
# Run ONCE before train.py. No GPU required. Takes < 5 minutes.
# Output: finetune/osha_train.jsonl (3600 lines)
#         finetune/osha_val.jsonl   (400 lines)

import json, random
from pathlib import Path
from datetime import datetime, timedelta

SYSTEM = 'You are an OSHA Compliance Officer. Write formal inspection reports.'

VIOLATIONS = [
    {'code':'PPE-001','name':'Missing hard hat','reg':'29 CFR 1926.100(a)',
     'sev':'Serious','penalty':'$1,190-$15,625',
     'action':'Provide ANSI-compliant hard hats. Brief workers before resuming.'},
    {'code':'PPE-002','name':'Missing hi-vis vest','reg':'29 CFR 1926.201(a)',
     'sev':'Serious','penalty':'$1,190-$15,625',
     'action':'Require Class 2/3 high-visibility apparel in traffic zones.'},
    {'code':'FALL-001','name':'No fall protection','reg':'29 CFR 1926.502(d)',
     'sev':'Willful','penalty':'$15,625-$156,259',
     'action':'HALT work at height. Install guardrails or issue harnesses immediately.'},
    {'code':'FALL-002','name':'Unsafe scaffold','reg':'29 CFR 1926.451(g)(1)',
     'sev':'Serious','penalty':'$1,190-$15,625',
     'action':'Remove workers. Competent person to inspect before returning.'},
    {'code':'ELEC-001','name':'Exposed electrical cable','reg':'29 CFR 1926.416(a)(1)',
     'sev':'Serious','penalty':'$1,190-$15,625',
     'action':'Lockout/tagout immediately. Qualified electrician to inspect.'},
]

SITES = ['Riverside Tower Phase 2','Harbor Commercial Plaza',
         'Airport Terminal Expansion','Downtown Mixed-Use Block C']

def make_report(violations, site, workers, date, risk, is_repeat):
    lines = [f'CONSTRUCTION SITE SAFETY INSPECTION REPORT',
             f'Site: {site}  Date: {date}  Risk: {risk}',
             'VIOLATIONS DETECTED']
    for i,v in enumerate(violations,1):
        lines += [f'#{i}: {v["name"]}',
                  f'  Regulation: {v["reg"]}',
                  f'  Severity: {v["sev"]}',
                  f'  Penalty: {v["penalty"]}',
                  f'  Corrective Action: {v["action"]}',
                  f'  Deadline: Immediate']
    if is_repeat: lines.append('NOTE: REPEAT OFFENDER. Willful penalties may apply.')
    lines += ['COMPLIANCE SUMMARY',
              f'Violations: {len(violations)}  Workers at risk: {workers}',
              f'Immediate action required: {"YES" if risk in ("HIGH","CRITICAL") else "NO"}']
    return '\n'.join(lines)

def main():
    random.seed(42)
    examples = []
    for _ in range(4000):
        n    = random.choices([1,2,3],weights=[0.5,0.35,0.15])[0]
        viols= random.sample(VIOLATIONS, n)
        site = random.choice(SITES)
        wkrs = random.randint(2,20)
        rept = random.random() < 0.2
        date = (datetime.now()-timedelta(days=random.randint(0,365))).strftime('%B %d, %Y')
        risk = 'CRITICAL' if any(v['sev']=='Willful' for v in viols) else 'HIGH' if n>1 else 'MEDIUM'
        user = (f'Site: {site}\nDate: {date}\nWorkers: {wkrs}\n'
                f'Risk: {risk}\nRepeat: {"YES" if rept else "No"}\n'
                f'Violations:\n' + '\n'.join(f'{v["code"]}: {v["name"]}' for v in viols))
        examples.append({'messages':[
            {'role':'system', 'content':SYSTEM},
            {'role':'user',   'content':user},
            {'role':'assistant','content':make_report(viols,site,wkrs,date,risk,rept)}
        ]})
    random.shuffle(examples)
    split = int(len(examples)*0.9)
    out   = Path(__file__).parent
    with open(out/'osha_train.jsonl','w') as f:
        for e in examples[:split]: f.write(json.dumps(e)+'\n')
    with open(out/'osha_val.jsonl','w') as f:
        for e in examples[split:]: f.write(json.dumps(e)+'\n')
    print(f'Done: {split} train, {len(examples)-split} val')

if __name__=='__main__': main()

C.11  finetune/train.py
Run AFTER prepare_dataset.py. Requires AMD MI300X (or any GPU with 24GB+ VRAM). Fine-tunes Qwen2.5-7B-Instruct using QLoRA — only 0.5% of parameters are trainable. After training, push the model to Hugging Face Hub and update REPORT_MODEL_ID in config.py.

train.py   [finetune/]
Track: Track 2: Fine-Tuning (training)
Purpose: QLoRA fine-tuning of Qwen2.5-7B-Instruct on OSHA report data. Run on AMD MI300X.
Imports from: None (standalone — no project imports)
Imported by: None (produces a model that report_agent.py loads)

Base model	Qwen/Qwen2.5-7B-Instruct
Method	QLoRA: 4-bit NF4 quantisation + LoRA rank 16 adapters
Trainable parameters	~0.5% of total (only LoRA adapter weights)
Target modules	q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Batch size	4 per device x 4 gradient accumulation steps = effective batch 16
Learning rate	2e-4 with 5% warmup ratio
Precision	bfloat16 — native on AMD MI300X (no conversion needed)
Epochs	3 (best model automatically selected by eval loss)
Estimated runtime	3–4 hours on AMD MI300X MI300X
After training	trainer.push_to_hub('username/constructsafe-osha-7b')
Then update	REPORT_MODEL_ID in config.py to your HF model repo path

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
HF_REPO    = 'your-username/constructsafe-osha-7b'  # Update this

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
