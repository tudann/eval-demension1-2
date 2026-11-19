"""Evaluate existing D1/D2 videos from a case directory or a JSON/CSV manifest."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from video_eval import EvaluationRequest, evaluate
from video_eval.cases import discover_cases
from video_eval.report import result_file, write_case_outputs, write_run_summary, write_unscored

BAD = {'ERROR', 'MODEL_UNAVAILABLE'}

def load_rows(source: Path) -> list[dict]:
    """Load case rows from a directory tree or the previous manifest file."""
    if source.is_dir():
        return discover_cases(source)
    with source.open(encoding='utf-8-sig') as handle:
        rows = json.load(handle) if source.suffix.lower() == '.json' else list(csv.DictReader(handle))
    if not isinstance(rows, list):
        raise ValueError('manifest must contain a list of rows')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', help='case directory, or a JSON/CSV manifest')
    parser.add_argument('--dimension', action='append', type=int, choices=(1, 2), dest='dimensions')
    parser.add_argument('--output-dir', default='outputs/d1_d2')
    parser.add_argument('--sample-fps', type=float, default=2)
    parser.add_argument('--max-frames', type=int, default=48)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    try:
        rows = load_rows(manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    output = Path(args.output_dir).resolve()
    results, failures = [], 0
    summary = {'mean_score': None}
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
        mode = str(row.get('mode') or 't2va')
        case_id = str(row.get('case_id') or video.stem)
        model = row.get('model')
        cached = []
        for dimension in dims:
            path = result_file(output, dimension, mode, case_id)
            if (args.resume or args.retry_errors) and path.is_file():
                candidate = json.loads(path.read_text(encoding='utf-8'))
                if not args.retry_errors or not any(m.get('status') in BAD or m.get('error') for m in candidate.get('metrics') or []):
                    cached.append(candidate)
        try:
            if len(cached) == len(dims):
                produced = cached
            else:
                report = evaluate(EvaluationRequest(str(video), row.get('prompt') or None, refs,
                    metadata={k:row[k] for k in ('case_id', 'model', 'original_prompt') if k in row},
                    enabled_dimensions=dims, sample_fps=args.sample_fps, max_frames=args.max_frames))
                produced = write_case_outputs(report, output, dimensions=dims, mode=mode, case_id=case_id,
                                              model=model, video_path=str(video), reference_images=refs)
            bad = [item['source_case_id'] for item in produced if item.get('status') in {'error', 'skipped_evaluate'}
                   or any(m.get('status') in BAD or m.get('error') for m in item.get('metrics') or [])]
            failures += bool(bad)
            results.extend(produced)
        except Exception as exc:
            failures += 1
            for dimension in dims:
                results.append(write_unscored(output, dimension=dimension, mode=mode, case_id=case_id,
                                              reason=f'{type(exc).__name__}: {exc}', model=model, video_path=str(video)))
        summary = write_run_summary(output, results)
    print(json.dumps({'videos': len(eligible[:args.limit]), 'failed_videos': failures,
                      'summary': str(output/'runs'/'local'/'summary.json'),
                      'mean_score': summary['mean_score'] if results else None}, ensure_ascii=False))
    return 2 if failures else 0

if __name__ == '__main__':
    raise SystemExit(main())

