"""Evaluate existing videos from a JSON/CSV manifest using D1/D2."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

from video_eval import EvaluationRequest, evaluate
from video_eval.report import write_reports

BAD = {'ERROR', 'MODEL_UNAVAILABLE'}

def safe_name(value):
    return re.sub(r'[^\w.-]+', '_', str(value)) or 'unknown'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', help='JSON list or CSV with video_path, prompt, optional dimension/model/reference_images')
    parser.add_argument('--dimension', action='append', type=int, choices=(1, 2), dest='dimensions')
    parser.add_argument('--output-dir', default='reports/batch')
    parser.add_argument('--sample-fps', type=float, default=2)
    parser.add_argument('--max-frames', type=int, default=48)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    with manifest.open(encoding='utf-8-sig') as f:
        rows = json.load(f) if manifest.suffix.lower() == '.json' else list(csv.DictReader(f))
    if not isinstance(rows, list):
        parser.error('manifest must contain a list of rows')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    output = Path(args.output_dir).resolve()
    summaries, failures = [], 0
    eligible = [r for r in rows if not args.dimensions or not r.get('dimension') or int(r['dimension']) in args.dimensions]
    for row in eligible[:args.limit]:
        dims = [int(row['dimension'])] if row.get('dimension') else (args.dimensions or [1, 2])
        if set(dims) - {1, 2}:
            parser.error(f'unsupported dimension: {dims}')
        video = Path(row['video_path']).expanduser()
        video = video if video.is_absolute() else manifest.parent/video
        refs = row.get('reference_images') or []
        if isinstance(refs, str):
            refs = [r for r in refs.split(';') if r]
        refs = [str(p if p.is_absolute() else manifest.parent/p) for p in map(Path, refs)]
        subdir = output/('D'+'_'.join(map(str, dims)))/safe_name(row.get('model', 'default'))/safe_name(row.get('case_id', video.stem))
        existing = subdir/(video.stem+'.json')
        try:
            report = None
            if (args.resume or args.retry_errors) and existing.is_file():
                candidate = json.loads(existing.read_text())
                if not args.retry_errors or not any(m['status'] in BAD or m.get('error') for m in candidate['metrics']):
                    report = candidate
            if report is None:
                report = evaluate(EvaluationRequest(str(video), row.get('prompt') or None, refs,
                    metadata={k:row[k] for k in ('case_id', 'model', 'original_prompt') if k in row},
                    enabled_dimensions=dims, sample_fps=args.sample_fps, max_frames=args.max_frames))
                paths = write_reports(report, subdir)
            else:
                paths = {'json':str(existing)}
            bad = [m['dimension_id'] for m in report['metrics'] if m['status'] in BAD or m.get('error')]
            failures += bool(bad)
            summaries.append({'video_path':str(video), 'dimensions':dims, 'summary':report['summary'], 'failed_metrics':bad, 'reports':paths})
        except Exception as exc:
            failures += 1
            summaries.append({'video_path':str(video), 'dimensions':dims, 'error':f'{type(exc).__name__}: {exc}'})
        output.mkdir(parents=True, exist_ok=True)
        (output/'batch_summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'videos':len(summaries), 'failed_videos':failures, 'summary':str(output/'batch_summary.json')}, ensure_ascii=False))
    return 2 if failures else 0

if __name__ == '__main__':
    raise SystemExit(main())

