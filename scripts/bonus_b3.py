"""Isolated B3: verified arithmetic traces, two loss masks, identical evaluation."""
import argparse
import dataclasses
import hashlib
import json
import pathlib
import re
import shutil
import sys
import time

ROOT = pathlib.Path('/content/Lab21_B3')
TASK = 'Giải bài toán bằng phép tính ngắn. Sau phần suy luận, câu trả lời cuối chỉ gồm một số nguyên.'
MATH_BUDGET = 2048
DECODE = {'do_sample':True, 'temperature':1.0, 'top_p':0.95, 'top_k':20,
          'repetition_penalty':1.0}

def parse_final(text):
    """Score an explicit final integer line, never a number inside the trace."""
    body=text.split('</think>',1)[1] if '</think>' in text else text
    for tag in ['<|im_end|>','<|endoftext|>']:
        body=body.replace(tag,'')
    lines=[line.strip() for line in body.splitlines() if line.strip()]
    match=re.fullmatch(r'(-?\d+)',lines[-1]) if lines else None
    value=int(match.group(1)) if match else None
    strict=re.fullmatch(r'\s*(-?\d+)\s*',body) is not None
    return value,strict

def reparse_calibration():
    assert not any((ROOT/'adapters'/m).exists() for m in ['assistant-only','response-only']), 'Parser must be fixed before training.'
    path=ROOT/'results/calibration_v3.json'
    result=json.loads(path.read_text(encoding='utf-8'))
    archive=ROOT/'results/calibration_v3_strict_parser.json'
    if not archive.exists(): shutil.copy2(path,archive)
    for row in result['rows']:
        row['predicted_answer'],row['strict_answer_format']=parse_final(row['raw_continuation'])
        row['non_thinking_answer'],row['non_thinking_strict_format']=parse_final(row['non_thinking_output'])
        print(row['input'],'expected=',row['answer'],'thinking=',row['predicted_answer'],
              'non-thinking=',row['non_thinking_answer'],'strict=',row['strict_answer_format'])
    result['answer_parser']='Last nonempty post-trace line must be an integer; strict-only format reported separately.'
    write('calibration_v3.json',result)
    doc=pathlib.Path(__file__).with_name('B3_PROTOCOL.md')
    shutil.copy2(doc,ROOT/'data/B3_PROTOCOL.md')
    print('Existing outputs rescored without regenerating; baseline uses same parser.')

def write(name, value):
    path = ROOT / 'results' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def read_rows(name):
    return [json.loads(s) for s in (ROOT/'data'/name).read_text(encoding='utf-8').splitlines() if s.strip()]

def revise_protocol():
    assert ROOT.exists()
    assert not any((ROOT/'adapters'/m).exists() for m in ['assistant-only','response-only']), 'Cannot revise after training.'
    archive=ROOT/'results/pilot_v1'
    assert not archive.exists(), 'Revision already applied; run probe/calibrate next.'
    archive.mkdir()
    for name in ['baseline.json','mask_probe.json','data_hashes.json']:
        p=ROOT/'results'/name
        if p.exists(): shutil.copy2(p,archive/name)
    for name in ['train_trace.jsonl','eval_math.jsonl']:
        p=ROOT/'data'/name
        shutil.copy2(p,archive/name)
        rows=read_rows(name)
        for row in rows: row['instruction']=TASK
        p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    config=ROOT/'src/labkit/config.py'
    text,count=re.subn(r'^NAIVE_PROMPT = .*$',lambda _: 'NAIVE_PROMPT = '+repr(TASK),config.read_text(encoding='utf-8'),count=1,flags=re.M)
    assert count==1
    config.write_text(text,encoding='utf-8')
    for name in ['baseline.json','mask_probe.json']:
        p=ROOT/'results'/name
        if p.exists(): p.unlink()  # Archived pilot; no trained adapters yet.
    doc=pathlib.Path(__file__).with_name('B3_PROTOCOL.md')
    shutil.copy2(doc,ROOT/'data/B3_PROTOCOL.md')
    write('data_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.jsonl')})
    write('protocol_revision.json',{'revision':2,'before_training':True,'math_budget':MATH_BUDGET,
          'reason':'Pilot had no closing thinking tag within512 tokens; remove literal think tags from task instruction and increase output budget. Fix saved expected/predicted answer field collision.',
          'pilot_preserved':'results/pilot_v1'})
    print('Protocol v2 prepared before training; pilot preserved. Run probe and calibrate.')

def revise_decode():
    assert ROOT.exists()
    assert not any((ROOT/'adapters'/m).exists() for m in ['assistant-only','response-only']), 'Cannot revise after training.'
    archive=ROOT/'results/pilot_v2'
    assert not archive.exists(), 'Decode revision already applied; calibrate next.'
    archive.mkdir()
    for name in ['baseline.json','calibration_v2.json','protocol_revision.json']:
        p=ROOT/'results'/name
        if p.exists(): shutil.copy2(p,archive/name)
    p=ROOT/'results/baseline.json'
    if p.exists(): p.unlink()
    shutil.copy2(pathlib.Path(__file__).with_name('B3_PROTOCOL.md'),ROOT/'data/B3_PROTOCOL.md')
    write('protocol_revision.json',{'revision':3,'before_training':True,'math_budget':MATH_BUDGET,
          'decode':DECODE,'presence_penalty':1.5,'seed_per_question':'42 + index',
          'batch_size_math':1,'reason':'Greedy pilot repeated thinking; decode must trim at first EOS before parsing.',
          'pilot_preserved':'results/pilot_v2'})
    print('Decode v3 prepared; old calibration preserved. Run calibrate.')

def setup(source_path='/content/Day21-Track3-Finetuning-Lab'):
    source = pathlib.Path(source_path).resolve()
    assert (source/'src/labkit').exists(), 'Original lab repo missing.'
    assert (source/'data/eval_regression.jsonl').exists(), 'Core regression data missing.'
    assert pathlib.Path(__file__).with_name('B3_PROTOCOL.md').exists(), 'Upload B3_PROTOCOL.md beside this script.'
    assert not ROOT.exists(), 'B3 folder already exists; do not overwrite existing runs.'
    shutil.copytree(source, ROOT, ignore=shutil.ignore_patterns('.git','.venv','.env','__pycache__','adapters','results','submission','data','bonus','backups','Day21-Track3-Finetuning-Lab','*.crdownload'))
    for folder in ['data','results','adapters','submission']:
        (ROOT/folder).mkdir(exist_ok=True)
    rows = {'train_trace.jsonl': [], 'eval_math.jsonl': []}
    for family in range(3):
        for a in list(range(1,81)) + list(range(101,109)):
            b, c = 7 + a%13, 2 + a%5
            if family == 0:
                question = f'Tính ({a} + {b}) × {c}.'
                intermediate, answer = a+b, (a+b)*c
                trace = f'Tính trong ngoặc trước: {a} + {b} = {intermediate}. Nhân kết quả: {intermediate} × {c} = {answer}.'
            elif family == 1:
                question = f'Có {a} hộp, mỗi hộp {b} bút. Cho đi {c} bút thì còn bao nhiêu bút?'
                intermediate, answer = a*b, a*b-c
                trace = f'Số bút ban đầu: {a} × {b} = {intermediate}. Số còn lại: {intermediate} - {c} = {answer}.'
            else:
                question = f'Có {a} quyển sách. Nhận thêm {b} quyển rồi cho đi {c} quyển. Còn bao nhiêu quyển?'
                intermediate, answer = a+b, a+b-c
                trace = f'Sau khi nhận: {a} + {b} = {intermediate}. Sau khi cho đi: {intermediate} - {c} = {answer}.'
            row = {'instruction':TASK,'input':question,'output':f'<think>\n{trace}\n</think>\n{answer}',
                   'answer':answer,'family':family,'id':f'{family}-{a}'}
            rows['train_trace.jsonl' if a<=80 else 'eval_math.jsonl'].append(row)
    assert len(rows['train_trace.jsonl']) == 240 and len(rows['eval_math.jsonl']) == 24
    assert not {r['input'] for r in rows['train_trace.jsonl']} & {r['input'] for r in rows['eval_math.jsonl']}
    for name, values in rows.items():
        (ROOT/'data'/name).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values),encoding='utf-8')
    shutil.copy2(source/'data/eval_regression.jsonl',ROOT/'data/eval_regression.jsonl')
    config = ROOT/'src/labkit/config.py'
    text = config.read_text(encoding='utf-8')
    text, count = re.subn(r'^NAIVE_PROMPT = .*$',lambda _: 'NAIVE_PROMPT = '+repr(TASK),text,count=1,flags=re.M)
    assert count == 1
    config.write_text(text,encoding='utf-8')
    shutil.copy2(pathlib.Path(__file__).with_name('B3_PROTOCOL.md'),ROOT/'data/B3_PROTOCOL.md')
    write('dataset_checks.json', {'train':240,'eval':24,'duplicate_questions':0,
          'source':'Synthetic arithmetic; short worked solutions generated with exact integer operations.',
          'limitations':'Shared three problem templates; eval operands exceed train range; no evidence of general reasoning transfer.'})
    # Freeze source hashes before any scores; future phases verify them.
    write('data_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.jsonl')})
    print('B3 ready:',ROOT)

def imports():
    assert ROOT.exists(), 'Run setup first.'
    sys.path.insert(0,str(ROOT/'src'))
    from labkit import data, generate, train, modeling, evaluate
    from labkit.config import get_tier, SPECS
    frozen = json.loads((ROOT/'results/data_hashes.json').read_text(encoding='utf-8'))
    for name, digest in frozen.items():
        assert hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()==digest, 'Data changed: '+name
    return data,generate,train,modeling,evaluate,get_tier('T4'),SPECS['correct']

def probe():
    data, _, _, _, _, tier, _ = imports()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tier.model_id,trust_remote_code=True)
    proof = data.thinking_survives(tok)
    sample = read_rows('train_trace.jsonl')[0]
    masks = {}
    for mode in ['assistant-only','response-only']:
        ex = data.build_example(tok,data.to_messages(sample),max_length=1024,mask_mode=mode,enable_thinking=True)
        decoded = tok.decode([x for x in ex.labels if x != data.IGNORE_INDEX],skip_special_tokens=False)
        masks[mode] = {'supervised_tokens':ex.n_supervised,'total_tokens':ex.n_total,'loss_text':decoded}
        print('\n',mode,':',decoded)
    assert masks['assistant-only']['supervised_tokens'] > masks['response-only']['supervised_tokens'] > 0
    assert 'Tính trong ngoặc' in masks['assistant-only']['loss_text']
    assert 'Tính trong ngoặc' not in masks['response-only']['loss_text']
    prompt = tok.apply_chat_template([{'role':'system','content':TASK},{'role':'user','content':sample['input']}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
    assert '<think>' in prompt and '</think>' not in prompt.rsplit('<think>',1)[1], 'Thinking generation prefix unavailable.'
    write('mask_probe.json',{'template':proof,'masks':masks,'generation_prefix':prompt,
                            'verified_distinct_masks':True})
    print('\nB3 MASK PROBE PASSED. Baseline must run before training.')

def train_mode(mode):
    data,gen,tr,modeling,_,tier,spec = imports()
    assert (ROOT/'results/mask_probe.json').exists(), 'Probe masks first.'
    assert (ROOT/'results/baseline.json').exists(), 'Freeze B3 baseline before train.'
    out=ROOT/'adapters'/mode
    assert not out.exists(), 'Adapter already exists; preserve run instead of overwriting.'
    from transformers import set_seed
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig,SFTTrainer
    set_seed(42)
    model,tok=gen.load_base(tier)
    records=read_rows('train_trace.jsonl')
    rows=data.to_training_dataset(tok,records,max_length=tier.max_length,mask_mode=mode,enable_thinking=True)
    assert len(rows)==240
    targets=modeling.resolve_target_modules(model,spec.target)
    steps=tr.planned_steps(len(rows),tier,2)
    kw=tr.sft_config_kwargs(tier,spec,str(out),max_steps=steps,num_train_epochs=2,mask_mode=mode,seed=42)
    kw,_=tr.filter_kwargs(SFTConfig,kw)
    trainer=SFTTrainer(model=model,args=SFTConfig(**kw),train_dataset=Dataset.from_list(rows),
                       processing_class=tok,peft_config=LoraConfig(**tr.lora_config_kwargs(spec,targets)))
    print('precision:',tr.align_trainable_precision(trainer.model))
    start=time.perf_counter()
    result=trainer.train()
    elapsed=time.perf_counter()-start
    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)
    write('train_'+mode+'.json',{'mask_mode':mode,'seed':42,'max_steps':steps,'actual_global_step':trainer.state.global_step,
          'train_examples':len(rows),'rank':16,'learning_rate':spec.lr,'placement':spec.target,
          'train_seconds':elapsed,'peak_vram_gb':gen.peak_vram_gb(),'train_loss':result.training_loss,
          'supervised_tokens':sum(sum(x!=data.IGNORE_INDEX for x in r['labels']) for r in rows)})
    print('Saved:',out)

def math_predictions(model,tok,records):
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList, set_seed
    class GeneratedPresencePenalty(LogitsProcessor):
        def __init__(self,start): self.start=start
        def __call__(self,input_ids,scores):
            scores=scores.clone()
            for i in range(input_ids.shape[0]):
                seen=input_ids[i,self.start:].unique()
                scores[i,seen]-=1.5
            return scores
    result=[]
    tok.padding_side='left'
    if tok.pad_token is None:
        tok.pad_token=tok.eos_token
    for start in range(0,len(records)):
        texts=[tok.apply_chat_template([{'role':'system','content':TASK},{'role':'user','content':r['input']}],
               tokenize=False,add_generation_prompt=True,enable_thinking=True) for r in records[start:start+1]]
        enc=tok(texts,padding=True,return_tensors='pt').to(model.device)
        set_seed(42+start)
        with torch.no_grad():
            output=model.generate(**enc,**DECODE,max_new_tokens=MATH_BUDGET,pad_token_id=tok.pad_token_id,
                                  logits_processor=LogitsProcessorList([GeneratedPresencePenalty(enc['input_ids'].shape[1])]))
        for generated,prefix in zip(output,texts):
            suffix=generated[enc['input_ids'].shape[1]:].tolist()
            eos=model.generation_config.eos_token_id or tok.eos_token_id
            eos_ids=set(eos if isinstance(eos,list) else [eos])
            first_end=next((i+1 for i,x in enumerate(suffix) if x in eos_ids),None)
            meaningful=suffix[:first_end] if first_end is not None else suffix
            raw=tok.decode(meaningful,skip_special_tokens=False).strip()
            # An opener supplied by the template is context, not a generated token.
            # Count nonempty continuation + generated closing tag, disclose reconstruction.
            prefix_open='<think>' in prefix and '</think>' not in prefix.rsplit('<think>',1)[1]
            trace_text=('<think>\n'+raw) if prefix_open else raw
            value,strict=parse_final(raw)
            result.append({'raw_continuation':raw,'trace_text':trace_text,
                           'template_supplied_opener':prefix_open,
                           'predicted_answer':value,'strict_answer_format':strict,
                           'generated_tokens':first_end or len(suffix),
                           'hit_token_budget':first_end is None and len(suffix)>=MATH_BUDGET})
        print(f'math {start+1}/{len(records)}',flush=True)
    return result

def calibrate():
    _,gen,_,_,ev,tier,_=imports()
    assert not any((ROOT/'adapters'/m).exists() for m in ['assistant-only','response-only'])
    assert (ROOT/'results/mask_probe.json').exists()
    model,tok=gen.load_base(tier)
    model.eval()
    records=read_rows('eval_math.jsonl')[:2]
    preds=math_predictions(model,tok,records)
    rows=[dict(r,**p) for r,p in zip(records,preds)]
    direct,_=gen.generate_batch(model,tok,[r['input'] for r in records],system=TASK,
                               enable_thinking=False,max_new_tokens=96,label='calibration/non-thinking')
    for row,text in zip(rows,direct):
        row['non_thinking_output']=text
        row['non_thinking_answer'],row['non_thinking_strict_format']=parse_final(text)
    write('calibration_v3.json',{'math_budget':MATH_BUDGET,'decode':DECODE,'presence_penalty':1.5,'rows':rows})
    for row in rows:
        print(json.dumps({k:v for k,v in row.items() if k not in ['output','raw_continuation','trace_text']},ensure_ascii=False,indent=2))
        print('valid_trace:',ev.valid_reasoning_trace(row['trace_text']))
        print('END OUTPUT:',row['raw_continuation'][-1000:])
    print('Calibration saved. Inspect completion/truncation before freezing baseline.')

def score(mode):
    _,gen,_,_,ev,tier,_=imports()
    if mode=='base':
        assert not any((ROOT/'adapters'/m).exists() for m in ['assistant-only','response-only']), 'Baseline must precede adapters.'
        assert not (ROOT/'results/baseline.json').exists(), 'Baseline already frozen.'
    else:
        assert (ROOT/'results/baseline.json').exists()
    model,tok=gen.load_base(tier)
    if mode!='base':
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,str(ROOT/'adapters'/mode))
    model.eval()
    math=read_rows('eval_math.jsonl')
    preds=math_predictions(model,tok,math)
    direct,_=gen.generate_batch(model,tok,[r['input'] for r in math],system=TASK,
                               enable_thinking=False,max_new_tokens=96,label=mode+'/non-thinking math')
    direct_rows=[]
    for r,text in zip(math,direct):
        value,strict=parse_final(text)
        direct_rows.append({'id':r['id'],'expected_answer':r['answer'],'raw_output':text,
                            'predicted_answer':value,'strict_answer_format':strict,'correct':value==r['answer']})
    reg=read_rows('eval_regression.jsonl')
    rpred,_=gen.generate_batch(model,tok,[r['instruction'] for r in reg],system=None,max_new_tokens=96,label=mode+'/regression')
    scores={'run':mode,'target':sum(p['predicted_answer']==r['answer'] for p,r in zip(preds,math))/len(math),
            'valid_trace_rate':sum(ev.valid_reasoning_trace(p['trace_text']) for p in preds)/len(math),
            'regression':sum(ev.keyword_recall(p,r['keywords']) for p,r in zip(rpred,reg))/len(reg),
            'n_target':len(math),'n_regression':len(reg),'enable_thinking':True,'max_new_tokens_math':MATH_BUDGET,
            'n_hit_token_budget':sum(p['hit_token_budget'] for p in preds),
            'target_no_thinking':sum(r['correct'] for r in direct_rows)/len(direct_rows),
            'strict_answer_format_rate':sum(p['strict_answer_format'] for p in preds)/len(preds),
            'strict_answer_format_rate_no_thinking':sum(r['strict_answer_format'] for r in direct_rows)/len(direct_rows),
            'answer_parser':'Final nonempty post-trace line must be an integer; numeric answer exact match.',
            'non_thinking_outputs':direct_rows,
            'non_thinking_protocol':{'enable_thinking':False,'do_sample':False,'max_new_tokens':96,
                                     'purpose':'Separate answer accuracy; not evidence of trace collapse.'},
            'math_decode':DECODE,'presence_penalty':1.5,'seed_per_question':'42 + index','batch_size_math':1,
            'note':'Trace includes template-supplied opener when applicable; body and closing tag must be generated. Trace validity is structural, not proof of correct reasoning.',
            'math_outputs':[dict(r,**p) for r,p in zip(math,preds)],'regression_outputs':rpred}
    write('baseline.json' if mode=='base' else 'score_'+mode+'.json',scores)
    print(json.dumps({k:v for k,v in scores.items() if k not in ['math_outputs','regression_outputs','non_thinking_outputs']},ensure_ascii=False,indent=2))
    if all((ROOT/'results'/('score_'+m+'.json')).exists() for m in ['assistant-only','response-only']):
        summary=[json.loads((ROOT/'results'/n).read_text(encoding='utf-8')) for n in ['baseline.json','score_assistant-only.json','score_response-only.json']]
        write('b3_comparison.json',[{k:r[k] for k in ['run','target','target_no_thinking','valid_trace_rate','regression','n_hit_token_budget']} for r in summary])

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['setup','revise','revise-decode','probe','calibrate','reparse-calibration','baseline','train','evaluate'])
    parser.add_argument('--mode',choices=['assistant-only','response-only'])
    parser.add_argument('--source', default='/content/Day21-Track3-Finetuning-Lab',
                        help='Existing core repo used by setup; its data/results are preserved.')
    args=parser.parse_args()
    if args.stage=='setup': setup(args.source)
    elif args.stage=='revise': revise_protocol()
    elif args.stage=='revise-decode': revise_decode()
    elif args.stage=='probe': probe()
    elif args.stage=='calibrate': calibrate()
    elif args.stage=='reparse-calibration': reparse_calibration()
    elif args.stage=='baseline': score('base')
    else:
        assert args.mode, 'Supply --mode'
        train_mode(args.mode) if args.stage=='train' else score(args.mode)
