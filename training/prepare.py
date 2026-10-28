"""Build completion-only SFT rows from complete, audited Fixed5 trajectories."""
from common import *
import argparse,random,sys
sys.path.insert(0,str(R.parent/'pipeline'))
from geometry import parse_response,normalize_frame
from contexts import plan_context,draw_context

def main():
 from transformers import AutoTokenizer
 ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/pilot10');ap.add_argument('--validation-tasks',type=int,default=2);a=ap.parse_args()
 src=Path(a.data).resolve();tasks=read(src/'tasks.json');assert 0<a.validation_tasks<len(tasks)
 ids=sorted(t['id'] for t in tasks);random.Random(read(R/'config.json')['seed']).shuffle(ids);val=set(ids[:a.validation_tasks]);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 output={'train':[],'validation':[]};manifest={};lengths=[]
 for task in tasks:
  d=src/'results'/task['id'];result=read(d/'result.json');assert result['status']=='complete' and result['compression_complete']
  frames={i:read(d/'frames'/f'{i:02d}.json') for i in range(1,6)};plan=read(d/'total_plan.json')['total_plan']
  for stage in ['00_plan']+[f'frame_{i:02d}' for i in range(1,6)]:
   c=d/'calls'/stage;req=read(c/'request.json');u=json.loads(req['messages'][1]['content']);assert u['animation_prompt']==task['prompt'] and u['total_frames']==5
   if stage=='00_plan':assert u==plan_context(task['prompt'],u['drawing_style'])
   else:
    i=int(stage[-2:]);assert u==draw_context(task['prompt'],plan,i,frames[1] if i>1 else None,frames.get(i-1))
    obj,_=parse_response((c/'content.txt').read_text());frame,_=normalize_frame(obj,i);assert frame==frames[i]
   original=(c/'reasoning.txt').read_text();reason=(c/'compressed.txt').read_text();answer=(c/'content.txt').read_text();creq=read(c/'compression_request.json');cu=json.loads(creq['messages'][1]['content']);assert cu==dict(original_model_input=req['messages'],original_reasoning=original)
   assert reason.strip() and answer.strip();assert '<|im_end|>' not in reason+answer and '</think>' not in reason
   row=dict(id=task['id']+'/'+stage,task_id=task['id'],category=task['category'],stage=stage,messages=req['messages'],completion=reason+'\n</think>\n\n'+answer+'<|im_end|>\n',provenance={k:hashlib.sha256(v.encode()).hexdigest() for k,v in [('original_reasoning',original),('compressed_reasoning',reason),('answer',answer)]})
   enc=encode(tok,row);row['token_counts']={k:enc[k] for k in ['input_tokens','target_tokens']};lengths.append(len(enc['ids']));output['validation' if task['id'] in val else 'train'].append(row)
  manifest[task['id']]='validation' if task['id'] in val else 'train'
 for split,rows in output.items():
  p=R/(split+'.jsonl');temp=p.with_suffix('.tmp');temp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));temp.replace(p)
 save(R/'split.json',manifest);save(R/'prepare_report.json',dict(tasks=len(tasks),rows={k:len(v) for k,v in output.items()},max_sequence_tokens=max(lengths),mean_sequence_tokens=sum(lengths)/len(lengths),truncated=0,split='whole task; not semantic-family held out',source=str(src)))
 print(json.dumps(read(R/'prepare_report.json'),indent=2))
if __name__=='__main__':main()
