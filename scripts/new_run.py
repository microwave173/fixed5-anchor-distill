"""Prepare an immutable local run directory; does not call any paid API."""
from pathlib import Path
import argparse,json,shutil
root=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--tasks',default='data/pilot10/tasks.json');a.add_argument('--concurrency',type=int,default=10);args=a.parse_args();d=Path(args.out).resolve()
assert not d.exists(),'Use a new directory; resume an existing run via RUN_DIR.'
assert 1<=args.concurrency<=2400
d.mkdir(parents=True)
for f in ['plan_system.txt','draw_system.txt','style.txt','compress_system.txt']:shutil.copy2(root/'pipeline'/f,d/f)
shutil.copy2(args.tasks,d/'tasks.json');cfg=json.loads((root/'pipeline/config.json').read_text());cfg['concurrency']=args.concurrency;(d/'config.json').write_text(json.dumps(cfg,indent=2));print(d)

shutil.copy2(root/"scripts/dashboard.html",d/"dashboard.html")
