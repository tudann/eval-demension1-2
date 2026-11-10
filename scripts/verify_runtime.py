"""Validate D1/D2 scope, copied models, links, runtime imports and judge health."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-source', action='store_true', help='read-only check of original code hashes/mtime')
    parser.add_argument('--output', default=str(ROOT/'migration/runtime_validation.json'))
    args = parser.parse_args()
    failures, checks = [], {}
    from video_eval.registry import METRICS
    checks['metric_counts'] = {f'D{d}':sum(m.id.startswith(f'D{d}.') for m in METRICS) for d in (1,2)}
    if len(METRICS) != 18 or checks['metric_counts'] != {'D1':11,'D2':7}:
        failures.append('incorrect metric scope')
    imports = {}
    for name in ('numpy','cv2','torch','torchvision','scipy','onnxruntime','pyiqa','clip','sam2','grounding_dino','cotracker','video_depth_anything'):
        try:
            m = importlib.import_module(name)
            imports[name] = {'version':str(getattr(m,'__version__','unknown')), 'file':str(getattr(m,'__file__','unknown'))}
        except Exception as e:
            failures.append(f'import {name}: {e}')
    checks['imports'] = imports
    manifest = json.loads((ROOT/'environment_manifest.json').read_text())
    checked = 0
    for item in manifest['weights']:
        p = ROOT/item['path']
        if not p.is_file() or p.stat().st_size != item['bytes'] or sha256(p) != item['sha256']:
            failures.append('weight mismatch: '+str(p))
        checked += 1
    checks['weights_checked'] = checked
    external = []
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        for name in dirs+files:
            p = Path(base)/name
            if p.is_symlink():
                target = os.readlink(p)
                if '/LYX/' in target or '/lyx/' in target or '/keyan/' in target:
                    failures.append('legacy link: '+str(p))
                if not p.exists():
                    failures.append('broken link: '+str(p))
                if target.startswith('/') and not target.startswith(str(ROOT)):
                    external.append({'path':str(p.relative_to(ROOT)), 'target':target})
    checks['shared_runtime_links'] = external
    try:
        code = 'import json,torch,mmcv,mmdet,mmyolo; print(json.dumps({"python":__import__("sys").version,"torch":torch.__version__,"mmcv":mmcv.__version__,"mmdet":mmdet.__version__,"mmyolo":mmyolo.__version__}))'
        child_env = os.environ.copy()
        child_env['PYTHONPATH'] = str(ROOT/'third_party/VBench/VBench-2.0/vbench2/third_party/YOLO-World/mmyolo')
        p = subprocess.run([str(ROOT/'envs/vbench2/bin/python'), '-c', code], env=child_env, capture_output=True, text=True, timeout=180)
        checks['anatomy_runtime'] = p.stdout.strip()
        if p.returncode:
            failures.append('anatomy imports: '+p.stderr[-2000:])
    except Exception as e:
        failures.append(f'anatomy runtime: {e}')
    try:
        endpoint = os.getenv('QWEN3_VL_ENDPOINT','http://127.0.0.1:8011/v1')
        with urllib.request.urlopen(endpoint.rstrip('/')+'/models', timeout=10) as r:
            models = [m['id'] for m in json.load(r)['data']]
        checks['judge'] = {'endpoint':endpoint,'models':models}
        if os.getenv('QWEN3_VL_MODEL','Qwen3-VL-32B-Instruct') not in models:
            failures.append('judge model mismatch')
    except Exception as e:
        failures.append(f'judge: {e}')
    if args.check_source:
        source = Path('/pfs/pfs-spMCeu/eval_team/Video-eval/LYX/evaluation-tool/6_dimensions')
        before = json.loads((ROOT/'migration/source_before.json').read_text())
        changed = [name for name,v in before.items() if sha256(source/name) != v['sha256'] or (source/name).stat().st_mtime_ns != v['mtime_ns']]
        checks['original_source'] = {'files_checked':len(before), 'changed':changed}
        if changed:
            failures.append('source changed: '+str(changed))
    report = {'ok':not failures, 'checks':checks, 'failures':failures}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'ok':report['ok'],'weights_checked':checked,'failures':failures},ensure_ascii=False),flush=True)
    return 0 if report['ok'] else 1

if __name__ == '__main__':
    raise SystemExit(main())

