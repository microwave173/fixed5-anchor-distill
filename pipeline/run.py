"""DeepSeek Fixed5-Anchor: shared plan, first-frame anchor and previous frame."""
from geometry import *
from contexts import plan_context,draw_context
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx, threading, shutil
R=Path(os.environ.get('RUN_DIR',str(Path(__file__).resolve().parent))).resolve()
OUT=R/'results';OUT.mkdir(exist_ok=True)
CFG=read(R/'config.json');TASKS=read(R/'tasks.json')
KEY=os.environ.get('DEEPSEEK_API_KEY','').strip()
if not KEY:raise RuntimeError('Set DEEPSEEK_API_KEY in the environment; never commit it.')
LOCK=threading.Lock();STATES={t['id']:{'status':'pending'} for t in TASKS};START=time.time()

def report():
    with LOCK:
        done=[read(OUT/t['id']/'result.json') for t in TASKS if (OUT/t['id']/'result.json').exists()]
        save(OUT/'state.json',dict(status='complete' if len(done)==len(TASKS) else 'running',total=len(TASKS),finished=len(done),complete=sum(r['status']=='complete' for r in done),elapsed_seconds=time.time()-START,tasks=STATES,updated_epoch=time.time()))

def worker(task):
    d=OUT/task['id'];d.mkdir(exist_ok=True);save(d/'task.json',task)
    if (d/'result.json').exists() and read(d/'result.json').get('compression_complete'):
        cached=read(d/'result.json');STATES[task['id']]=dict(status=cached['status'],frames=cached['frames']);return cached
    started=time.time();frames={};warnings=[];failures=[];stage='00_plan'
    with httpx.Client(timeout=httpx.Timeout(1200,connect=30),trust_env=False) as client:
        def call(name,system,user,index=None):
            nonlocal stage
            stage=name;STATES[task['id']]=dict(status='running',stage=name);report()
            dest=d/'calls'/name;dest.mkdir(parents=True,exist_ok=True)
            payload=dict(model=CFG['model'],messages=[dict(role='system',content=system),dict(role='user',content=json.dumps(user,ensure_ascii=False))],thinking={'type':'enabled'},reasoning_effort=CFG['reasoning_effort'],max_tokens=CFG['teacher_max_tokens'],stream=False)
            save(dest/'request.json',payload);tick=time.time()
            if (dest/'response.json').exists():body=read(dest/'response.json')
            else:
                for retry in range(4):
                    try:
                        response=client.post(CFG['endpoint'],headers={'Authorization':'Bearer '+KEY},json=payload)
                        response.raise_for_status();body=response.json();break
                    except (httpx.HTTPError,ValueError) as exc:
                        save(dest/f'transport_{retry}.json',dict(type=type(exc).__name__,status=getattr(getattr(exc,'response',None),'status_code',None)))
                        if retry==3:raise RuntimeError('API transport retries exhausted') from None
                        time.sleep(min(30,3*2**retry))
                save(dest/'response.json',body)
            choice=body['choices'][0];message=choice['message'];reason=message.get('reasoning_content') or '';content=message.get('content') or ''
            (dest/'reasoning.txt').write_text(reason);(dest/'content.txt').write_text(content)
            save(dest/'metrics.json',dict(seconds=time.time()-tick,usage=body.get('usage'),finish_reason=choice.get('finish_reason'),model=body.get('model')))
            assert choice.get('finish_reason')=='stop','Truncated API output'
            obj,notes=parse_response(content);warnings.extend(dict(stage=name,note=n) for n in notes)
            if index is None:assert obj.get('total_plan'),'Missing shared plan'
            else:
                frame,notes=normalize_frame(obj,index);warnings.extend(dict(stage=name,**n) for n in notes)
                save(dest/'view_metrics.json',render_frame(frame,d));save(d/'frames'/f'{index:02d}.json',frame);obj=dict(obj,frames=[frame])
            save(dest/'accepted.json',obj);print(task['id'],name,'accepted',flush=True);return obj
        try:
            plan=call('00_plan',(R/'plan_system.txt').read_text(),plan_context(task['prompt'],(R/'style.txt').read_text()))
            save(d/'total_plan.json',plan)
            for i in range(1,6):
                previous=frames.get(i-1)
                assert previous is None if i==1 else previous['i']==i-1
                payload=draw_context(task['prompt'],plan['total_plan'],i,frames.get(1) if i>1 else None,previous)
                obj=call(f'frame_{i:02d}',(R/'draw_system.txt').read_text(),payload,i);frames[i]=obj['frames'][0]
        except Exception as exc:failures.append(dict(stage=stage,error=str(exc),traceback=traceback.format_exc()))
    compression_errors=[]
    if len(frames)==5:
        with httpx.Client(timeout=httpx.Timeout(1200,connect=30),trust_env=False) as client:
            for name in ['00_plan']+[f'frame_{i:02d}' for i in range(1,6)]:
                dest=d/'calls'/name
                try:
                    req=read(dest/'request.json');reason=(dest/'reasoning.txt').read_text()
                    cp=dict(model=CFG['model'],thinking={'type':'enabled'},reasoning_effort='high',max_tokens=32768,messages=[dict(role='system',content=(R/'compress_system.txt').read_text()),dict(role='user',content=json.dumps(dict(original_model_input=req['messages'],original_reasoning=reason),ensure_ascii=False))])
                    save(dest/'compression_request.json',cp)
                    STATES[task['id']]=dict(status='compressing',stage=name);report()
                    for retry in range(3):
                        try:
                            if (dest/'compression_response.json').exists():b=read(dest/'compression_response.json')
                            else:
                                res=client.post(CFG['endpoint'],headers={'Authorization':'Bearer '+KEY},json=cp);res.raise_for_status();b=res.json();save(dest/f'compression_attempt_{retry}.json',b)
                            c=b['choices'][0];assert c['finish_reason']=='stop' and c['message']['content'].strip()
                            save(dest/'compression_response.json',b);(dest/'compressed.txt').write_text(c['message']['content']);break
                        except Exception:
                            if retry==2:raise
                            time.sleep(3)
                except Exception as e:compression_errors.append(dict(stage=name,error=str(e)[:250]))
    previews(d,frames)
    result=dict(task=task['id'],status='complete' if len(frames)==5 else 'failed',frames=len(frames),available_frames=sorted(frames),seconds=time.time()-started,warnings=warnings,failures=failures,compression_errors=compression_errors,compression_complete=len(frames)==5 and not compression_errors)
    save(d/'result.json',result);STATES[task['id']]=dict(status=result['status'],frames=len(frames));report();return result

def gallery(results):
    import html
    lines=['<!doctype html><meta charset="utf-8"><title>DeepSeek · Fixed5-Anchor · 5 frames</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;background:#15171c;color:#eee}article{padding:16px;border-bottom:1px solid #555}img{max-width:100%;background:white}.pair{display:flex;gap:12px}.pair img{width:48%}a{color:#9cf}</style><h1>DeepSeek · Fixed5-Anchor · 5 frames</h1><p>Shared plan → F1–F5; complete first + previous frame. Generation high thinking; compression high. Fixed camera, 0.625s/frame plus final hold.</p>']
    for task,result in zip(TASKS,results):
        root='results/'+task['id'];lines.append('<article><h2>'+html.escape(task['title'])+f" · {result['frames']}/5</h2><p>"+html.escape(task['prompt'])+'</p>')
        if (OUT/task['id']/'animation.gif').exists():lines.append(f'<img src="{root}/animation.gif">')
        elif (OUT/task['id']/'contact_sheet.png').exists():lines.append(f'<img src="{root}/contact_sheet.png">')
        lines.append(f'<p>{result["seconds"]:.1f}s · <a href="{root}/result.json">result</a> · <a href="{root}/spatial_check.png">3 views</a></p></article>')
    (R/'index.html').write_text('\n'.join(lines))

if __name__=='__main__':
    lock=(R/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'settings.json',dict(**CFG,frames=5,fps=1.6,duration_seconds=2.5,concurrency=int(CFG.get('concurrency',10)),history='shared plan plus complete first and previous frames',format='native Path3D',model_samples_per_task=1,transport_retries=4))
    report()
    with ThreadPoolExecutor(max_workers=int(CFG.get('concurrency',10))) as pool:
        futures=[pool.submit(worker,t) for t in TASKS]
        for future in as_completed(futures):future.result()
    results=[read(OUT/t['id']/'result.json') for t in TASKS];save(OUT/'summary.json',dict(tasks=TASKS,results=results));gallery(results);report()
    with tarfile.open(R/'results.tgz','w:gz') as tar:
        tar.add(OUT,arcname='results');tar.add(R/'index.html',arcname='index.html')
    save(R/'backup.json',dict(sha256=hashlib.sha256((R/'results.tgz').read_bytes()).hexdigest()))
    print('ALL_FINISHED',flush=True)
