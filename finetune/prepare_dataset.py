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
