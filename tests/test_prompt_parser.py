import copy
import pytest
from video_eval.prompt_parser import validate_parsed, response_schema, relevant

def parsed():
    base={'id':'1','text':'exactly eight red apples','subjects':['apple'],'objects':[],
          'attributes':[],'target_count':8,'relation':None,'order':None,'critical':True,'type':'count'}
    spatial={**base,'id':'2','text':'mug to the right of laptop','type':'spatial',
             'subjects':['mug','laptop'],'target_count':None,'relation':'right'}
    return {'atoms':[base,spatial], 'subject_labels':['apple'], 'object_labels':['mug','laptop'],
            'has_people':False,'has_interaction':False}

def test_count_and_spatial_types_remain_routable():
    value=validate_parsed(parsed())
    assert relevant(value,'D1.2')[0]['target_count']==8
    assert relevant(value,'D1.4')[0]['subjects']==['mug','laptop']
    assert response_schema()['json_schema']['strict'] is True

def test_unknown_type_is_not_silently_dropped():
    value=parsed()
    value['atoms'][1]['type']='spatial_relation'
    with pytest.raises(ValueError):
        validate_parsed(value)

def test_count_and_spatial_contracts_reject_mistyped_outputs():
    value=parsed()
    value['atoms'][0]['type']='subject'
    assert validate_parsed(value)['atoms'][0]['target_count'] is None
    value=parsed()
    value['atoms'][1]['subjects']=['mug']
    with pytest.raises(ValueError):
        validate_parsed(value)



def test_parser_service_failure_is_a_reported_failure():
    from unittest.mock import patch
    from video_eval.pipeline import _run_metric, EvaluationRequest
    from video_eval.registry import BY_ID
    from video_eval.qwen import QwenUnavailable
    shared={}
    request=EvaluationRequest('unused.mp4',prompt='a person')
    with patch('video_eval.pipeline.parse_prompt',side_effect=QwenUnavailable('invalid extraction')) as call:
        for metric in ('D1.1','D1.2'):
            result=_run_metric(BY_ID[metric],None,{},request,shared)
            assert result.status=='MODEL_UNAVAILABLE'
            assert result.error
    assert call.call_count==1


def test_optional_placeholders_and_unscored_relations_do_not_drop_other_atoms():
    value=parsed()
    value['atoms'].append({**value['atoms'][1], 'relation':'center', 'id':''})
    value['atoms'].append({**value['atoms'][0], 'type':'subject','target_count':1,'id':''})
    result=validate_parsed(value)
    assert len(result['atoms'])==4
    assert relevant(result,'D1.2')[0]['target_count']==8
    assert len({a['id'] for a in result['atoms']})==4
