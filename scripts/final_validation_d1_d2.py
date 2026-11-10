"""Validate final code through real batch and spatial evaluation, retain all attempts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/migration_smoke_20261104/validated'
RUN.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
env['CUDA_VISIBLE_DEVICES']=env.get('EVAL_GPU','6')
env['VIDEO_EVAL_JUDGE_AUDIT_DIR']=str(RUN/'judge_audit')
rows=json.loads((ROOT/'inputs/manifest_d1_d2.json').read_text())
spatial=next(r for r in rows if r['case_id']=='010401')
jobs=[('batch_D1',[sys.executable,str(ROOT/'evaluate_batch.py'),str(ROOT/'inputs/smoke.json'),
       '--dimension','1','--limit','1','--output-dir',str(RUN/'batch_D1')]),
      ('spatial_D1.4',[sys.executable,str(ROOT/'evaluate.py'),str(ROOT/'inputs/supplemental/010401.mp4'),
       '--prompt',spatial['prompt'],'--metric','D1.4','--sample-fps','2','--max-frames','48',
       '--output-dir',str(RUN/'spatial_D1.4')])]
processes=[]
for name,cmd in jobs:
    log=(RUN/(name+'.log')).open('w')
    p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    processes.append((name,p,log))
    print('FINAL_START',name,'pid',p.pid,flush=True)
codes={}
for name,p,log in processes:
    codes[name]=p.wait(timeout=7200)
    log.close()
    print('FINAL_END',name,'returncode',codes[name],flush=True)
specs=[('D1',RUN/'batch_D1/D1/seedance-2-0-mini/010101/010101.json',11),
       ('D2',ROOT/'results/migration_smoke_20261104/D2/020101.json',7),
       ('D1.2_targeted',ROOT/'results/migration_smoke_20261104/supplemental_fixed/D1.2/010201.json',1),
       ('D1.4_targeted',RUN/'spatial_D1.4/010401.json',1)]
items=[]
for name,path,expected in specs:
    report=json.loads(path.read_text()) if path.is_file() else None
    bad=[m for m in report['metrics'] if m['status'] in ('ERROR','MODEL_UNAVAILABLE') or m.get('error') or 'prompt parse failed' in m.get('reason','')] if report else ['report missing']
    ok=bool(report) and not bad and report['summary']['requested']==expected
    if name in ('D1','D2','D1.2_targeted'):
        ok=ok and report['summary']['scored']>0
    if name=='D1.4_targeted':
        ok=ok and bool(report['metrics'][0]['raw_metrics'].get('checks'))
    items.append({'name':name,'ok':bool(ok),'report':str(path),'summary':report['summary'] if report else None,
                  'metrics':[{k:m[k] for k in ('dimension_id','status','score','error')} for m in report['metrics']] if report else [],'failures':bad})
acceptance={'ok':all(i['ok'] for i in items) and all(c==0 for c in codes.values()),
            'complete':True,'real_inference':True,'returncodes':codes,'results':items}
(ROOT/'migration/acceptance.json').write_text(json.dumps(acceptance,ensure_ascii=False,indent=2))
print('FINAL_ACCEPTANCE',json.dumps(acceptance,ensure_ascii=False),flush=True)
raise SystemExit(0 if acceptance['ok'] else 2)

