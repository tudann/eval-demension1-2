import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/migration_smoke_20261104/supplemental_fixed'
RUN.mkdir(parents=True,exist_ok=True)
rows=json.loads((ROOT/'inputs/manifest_d1_d2.json').read_text())
env=os.environ.copy()
env['CUDA_VISIBLE_DEVICES']=env.get('EVAL_GPU','6')
env['VIDEO_EVAL_JUDGE_AUDIT_DIR']=str(RUN/'judge_audit')
results=[]
for case_id,metric in [('010201','D1.2'),('010401','D1.4')]:
    row=next(r for r in rows if r['case_id']==case_id)
    video=ROOT/'inputs/supplemental'/Path(row['video_path']).name
    video.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(row['video_path'],video)
    command=[sys.executable,str(ROOT/'evaluate.py'),str(video),'--prompt',row['prompt'],
             '--metric',metric,'--sample-fps','2','--max-frames','48','--output-dir',str(RUN/metric)]
    started=time.time()
    print('SUPPLEMENT_START '+metric,flush=True)
    with (RUN/(metric+'.log')).open('w') as log:
        p=subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
    report_path=RUN/metric/(video.stem+'.json')
    report=json.loads(report_path.read_text()) if report_path.is_file() else None
    item={'metric':metric,'returncode':p.returncode,'elapsed_seconds':round(time.time()-started,2),
          'report':str(report_path),'metrics':report['metrics'] if report else None}
    results.append(item)
    (RUN/'acceptance.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print('SUPPLEMENT_END '+json.dumps({**item,'metrics':[{k:m[k] for k in ('dimension_id','status','score','error')} for m in report['metrics']] if report else None}),flush=True)
raise SystemExit(0 if all(r['returncode']==0 for r in results) else 2)

