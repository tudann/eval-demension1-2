import json
from pathlib import Path
from unittest.mock import patch

import pytest

from video_eval.registry import select

def test_only_two_dimensions_are_exposed():
    assert len(select()) == 18
    assert len(select(dimensions=[1])) == 11
    assert len(select(dimensions=[2])) == 7
    with pytest.raises(ValueError):
        select(dimensions=[3])
    with pytest.raises(ValueError):
        select(enabled=['D3.1'])

def test_metric_failure_returns_nonzero():
    import evaluate as single
    report = {'summary': {'status_counts': {'ERROR':1}}, 'metrics':[{'status':'ERROR','error':'failed backend'}]}
    with patch('sys.argv',['evaluate.py','video.mp4','--dimension','1']), \
         patch.object(single,'evaluate',return_value=report), \
         patch.object(single,'write_reports',return_value={}):
        assert single.main() == 2
    report['summary']['status_counts'] = {'UNCALIBRATED':1}
    report['metrics'] = [{'status':'UNCALIBRATED','error':'failed judge'}]
    with patch('sys.argv',['evaluate.py','video.mp4','--dimension','2']), \
         patch.object(single,'evaluate',return_value=report), \
         patch.object(single,'write_reports',return_value={}):
        assert single.main() == 2

def test_batch_filters_dimension_and_resolves_relative_inputs(tmp_path):
    import evaluate_batch as batch
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([
        {'video_path':'video.mp4','dimension':1,'prompt':'a person','case_id':'010101','model':'model'},
        {'video_path':'other.mp4','dimension':2,'case_id':'020101','model':'model'}]))
    report = {'summary': {'requested':11,'status_counts':{'SUCCESS':11}},'metrics':[]}
    output = tmp_path/'out'
    with patch('sys.argv',['evaluate_batch.py',str(manifest),'--dimension','1','--output-dir',str(output)]), \
         patch.object(batch,'evaluate',return_value=report) as run, \
         patch.object(batch,'write_reports',return_value={}) as write:
        assert batch.main() == 0
    assert run.call_count == 1
    request = run.call_args.args[0]
    assert request.video_path == str(tmp_path/'video.mp4')
    assert request.enabled_dimensions == [1]
    assert write.call_args.args[1] == output/'D1/model/010101'
    assert len(json.loads((output/'batch_summary.json').read_text())) == 1

