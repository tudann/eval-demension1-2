"""Run real D1/D2 models, one subprocess per video, with no mocked scores."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'results/migration_smoke_20261104'
RUN.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = env.get('EVAL_GPU', '6')
env['VIDEO_EVAL_JUDGE_AUDIT_DIR'] = str(RUN/'judge_audit')
gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'], text=True)
line = next(l for l in gpu.splitlines() if l.split(',')[0].strip() == env['CUDA_VISIBLE_DEVICES'])
_, free, util = [int(v.strip()) for v in line.split(',')]
if free < 60000 or util > 10:
    raise RuntimeError(f'GPU unavailable for smoke: {line}')
rows = json.loads((ROOT/'inputs/smoke.json').read_text())
results = []
for row in rows:
    d = row['dimension']
    out = RUN/f'D{d}'
    command = [sys.executable, str(ROOT/'evaluate.py'), row['video_path'], '--prompt', row['prompt'],
               '--dimension', str(d), '--sample-fps', '2', '--max-frames', '48', '--output-dir', str(out)]
    started = time.time()
    print(f'SMOKE_START D{d}', flush=True)
    with (RUN/f'D{d}.log').open('w') as log:
        process = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    report_path = out/(Path(row['video_path']).stem+'.json')
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    bad = [m for m in report['metrics'] if m['status'] in ('ERROR','MODEL_UNAVAILABLE') or m.get('error')] if report else ['no report']
    item = {'dimension':d, 'returncode':process.returncode, 'elapsed_seconds':round(time.time()-started, 2),
            'report':str(report_path), 'summary':report['summary'] if report else None,
            'failures':bad, 'ok':process.returncode == 0 and not bad and report['summary']['requested'] == {1:11,2:7}[d]}
    results.append(item)
    (RUN/'acceptance.json').write_text(json.dumps({'ok':all(r['ok'] for r in results), 'complete':len(results)==2, 'results':results}, ensure_ascii=False, indent=2))
    print('SMOKE_END '+json.dumps(item,ensure_ascii=False), flush=True)
raise SystemExit(0 if all(r['ok'] for r in results) else 2)

