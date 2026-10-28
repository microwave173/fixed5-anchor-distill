"""Resumable local async Fixed5-Anchor generation; one live request per task."""
from geometry import *
from contexts import plan_context,draw_context
import asyncio,collections, httpx, resource, signal
R=Path(os.environ['RUN_DIR']).resolve(); OUT=R/'results'; OUT.mkdir(exist_ok=True)
CFG=read(R/'config.json'); TASKS=read(R/'tasks.json'); KEY=os.environ['DEEPSEEK_API_KEY'].strip()
CONC=int(CFG.get('concurrency',10)); assert 1<=CONC<=2400
STATES={t['id']:{'status':'pending','stage':'queued','frames':0} for t in TASKS}
START=time.time(); COUNTS=collections.Counter(); ACTIVE=0; HISTORY=[]; DONE=[]; STOP=False

def snapshot():
 counts=collections.Counter(s['status'] for s in STATES.values()); now=time.time()
 HISTORY.append([now,counts['complete']]); HISTORY[:]=[x for x in HISTORY if x[0]>now-1800]
 rate=(HISTORY[-1][1]-HISTORY[0][1])*3600/max(1,now-HISTORY[0][0]) if len(HISTORY)>1 else 0
 obj=dict(updated_epoch=now,started_epoch=START,total=len(TASKS),counts=dict(counts),active_requests=ACTIVE,concurrency=CONC,session_usage=dict(COUNTS),tasks_per_hour=rate,eta_hours=(len(TASKS)-counts['complete'])/rate if rate>0 else None,tasks=STATES,status='stopping' if STOP else ('finished' if all(s['status'] in ('complete','failed') for s in STATES.values()) else 'running'))
 save(OUT/'state.json',obj);tmp=R/'progress.tmp';tmp.write_text('window.PROGRESS='+json.dumps(obj,ensure_ascii=False)+';');tmp.replace(R/'progress.js')

async def reporter():
 while True:snapshot();await asyncio.sleep(10)

async def request(client,payload,d,kind):
 global ACTIVE
 for attempt in range(6):
  if STOP:raise RuntimeError('Stopped by operator')
  try:
   ACTIVE+=1;COUNTS['requests_started']+=1
   try:
    response=await client.post(CFG['endpoint'],headers={'Authorization':'Bearer '+KEY},json=payload)
    response.raise_for_status();body=response.json()
   finally:ACTIVE-=1
   usage=body.get('usage') or {}
   COUNTS['responses_received']+=1
   for k in ['prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens']:COUNTS[k]+=usage.get(k,0)
   COUNTS['reasoning_tokens']+=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens',0)
   save(d/f'{kind}_attempt_{attempt}.json',body)
   c=body['choices'][0]
   if c.get('finish_reason')!='stop' or not (c['message'].get('content') or '').strip():raise ValueError('Empty or truncated response')
   save(d/('response.json' if kind=='generation' else 'compression_response.json'),body)
   return body
  except (httpx.HTTPError,ValueError,KeyError) as e:
   COUNTS['request_errors']+=1
   code=getattr(getattr(e,'response',None),'status_code',None)
   save(d/f'{kind}_error_{attempt}.json',dict(type=type(e).__name__,status=code))
   if attempt==5:raise RuntimeError(f'{kind}: retries exhausted; status={code}') from None
   await asyncio.sleep(min(120,5*2**attempt))

async def worker(task,client,render_limit):
 d=OUT/task['id'];d.mkdir(exist_ok=True);save(d/'task.json',task);s=STATES[task['id']]
 if (d/'result.json').exists() and read(d/'result.json').get('compression_complete'):
  s.update(status='complete',stage='cached',frames=5);return
 frames={};warnings=[];tick=time.time();stage='00_plan'
 try:
  async def generate(name,system,user,index=None):
   nonlocal stage
   stage=name;s.update(status='generating',stage=name)
   c=d/'calls'/name;c.mkdir(parents=True,exist_ok=True)
   payload=dict(model=CFG['model'],messages=[dict(role='system',content=system),dict(role='user',content=json.dumps(user,ensure_ascii=False))],thinking={'type':'enabled'},reasoning_effort=CFG['reasoning_effort'],max_tokens=CFG['teacher_max_tokens'],stream=False)
   if (c/'request.json').exists():assert read(c/'request.json')==payload,'Cached request mismatch; use a new run directory'
   save(c/'request.json',payload)
   if (c/'content.txt').exists() and (c/'reasoning.txt').exists():content=(c/'content.txt').read_text();reason=(c/'reasoning.txt').read_text()
   else:
    body=read(c/'response.json') if (c/'response.json').exists() else await request(client,payload,c,'generation')
    msg=body['choices'][0]['message'];content=msg['content'];reason=msg.get('reasoning_content') or ''
    (c/'content.txt').write_text(content);(c/'reasoning.txt').write_text(reason);save(c/'metrics.json',dict(usage=body.get('usage'),finish_reason=body['choices'][0].get('finish_reason')))
   obj,notes=parse_response(content);warnings.extend(notes)
   if index is None:assert obj.get('total_plan'),'Missing total plan'
   else:
    frame,notes=normalize_frame(obj,index);warnings.extend(notes)
    save(d/'frames'/f'{index:02d}.json',frame);obj=dict(obj,frames=[frame])
    async with render_limit:await asyncio.to_thread(render_frame,frame,d)
   save(c/'accepted.json',obj);return obj
  plan=await generate('00_plan',(R/'plan_system.txt').read_text(),plan_context(task['prompt'],(R/'style.txt').read_text()));save(d/'total_plan.json',plan)
  for i in range(1,6):
   obj=await generate(f'frame_{i:02d}',(R/'draw_system.txt').read_text(),draw_context(task['prompt'],plan['total_plan'],i,frames.get(1) if i>1 else None,frames.get(i-1)),i);frames[i]=obj['frames'][0];s['frames']=i
  s.update(status='compressing')
  for stage in ['00_plan']+[f'frame_{i:02d}' for i in range(1,6)]:
   s['stage']=stage;c=d/'calls'/stage;req=read(c/'request.json');reason=(c/'reasoning.txt').read_text()
   cp=dict(model=CFG['model'],thinking={'type':'enabled'},reasoning_effort='high',max_tokens=32768,messages=[dict(role='system',content=(R/'compress_system.txt').read_text()),dict(role='user',content=json.dumps(dict(original_model_input=req['messages'],original_reasoning=reason),ensure_ascii=False))])
   if (c/'compression_request.json').exists():assert read(c/'compression_request.json')==cp,'Compression prompt mismatch'
   save(c/'compression_request.json',cp)
   if (c/'compressed.txt').exists() and (c/'compressed.txt').stat().st_size:continue
   body=read(c/'compression_response.json') if (c/'compression_response.json').exists() else await request(client,cp,c,'compression')
   (c/'compressed.txt').write_text(body['choices'][0]['message']['content'])
  async with render_limit:await asyncio.to_thread(previews,d,frames)
  save(d/'result.json',dict(task=task['id'],status='complete',frames=5,compression_complete=True,seconds=time.time()-tick,warnings=warnings));s.update(status='complete',stage='done');print(task['id'],'COMPLETE',flush=True)
 except Exception as e:
  s.update(status='failed',stage=stage,error=str(e)[:250]);save(d/'result.json',dict(task=task['id'],status='failed',frames=len(frames),compression_complete=False,seconds=time.time()-tick,error=str(e)[:250]));print(task['id'],'FAILED',str(e)[:120],flush=True)

async def main():
 lock=(R/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,max(soft,CONC*3+1024)),hard))
 # Refuse resume with changed code/prompts/tasks: cached downstream frames would be stale.
 files=[R/f for f in ['config.json','tasks.json','style.txt','plan_system.txt','draw_system.txt','compress_system.txt']]+[Path(__file__).resolve(),Path(__file__).with_name('contexts.py'),Path(__file__).with_name('geometry.py')]
 signature={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 if (R/'run_manifest.json').exists():assert read(R/'run_manifest.json')==signature,'Run files changed; restore snapshot before resuming'
 else:save(R/'run_manifest.json',signature)
 for t in TASKS:
  p=OUT/t['id']/'result.json'
  if p.exists() and read(p).get('compression_complete'):STATES[t['id']].update(status='complete',stage='cached',frames=5)
 render_limit=asyncio.Semaphore(4);queue=asyncio.Queue()
 for t in TASKS:
  if STATES[t['id']]['status']!='complete':queue.put_nowait(t)
 async with httpx.AsyncClient(timeout=httpx.Timeout(1200,connect=60,pool=120),limits=httpx.Limits(max_connections=CONC,max_keepalive_connections=CONC),trust_env=False) as client:
  async def consume():
   while not queue.empty() and not STOP:
    try:t=queue.get_nowait()
    except asyncio.QueueEmpty:return
    await worker(t,client,render_limit);queue.task_done()
  monitor=asyncio.create_task(reporter())
  try:await asyncio.gather(*(consume() for _ in range(min(CONC,queue.qsize()))))
  finally:monitor.cancel();snapshot()
 print('RUN_FINISHED',flush=True)

if __name__=='__main__':asyncio.run(main())
