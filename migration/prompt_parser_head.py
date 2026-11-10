from __future__ import annotations

import json
import os
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

from .qwen import ALLOWED_DEPLOYMENTS, MODEL, QwenUnavailable

ATOM_TYPES = {'subject', 'count', 'attribute', 'spatial', 'action', 'scene', 'style', 'camera', 'negative'}
SPATIAL_RELATIONS = {'left', 'right', 'above', 'below', 'front', 'behind'}

def response_schema():
    strings = {'type':'array','items':{'type':'string'}}
    atom = {'type':'object','additionalProperties':False,'properties':{
        'id':{'type':'string'}, 'type':{'type':'string','enum':sorted(ATOM_TYPES)},
        'text':{'type':'string'}, 'subjects':strings, 'objects':strings, 'attributes':strings,
        'target_count':{'type':['integer','null'],'minimum':0},
        'relation':{'type':['string','null']}, 'order':{'type':['integer','null']},
        'critical':{'type':'boolean'}},
        'required':['id','type','text','subjects','objects','attributes','target_count','relation','order','critical']}
    return {'type':'json_schema','json_schema':{'name':'visual_prompt_atoms','strict':True,'schema':{
        'type':'object','additionalProperties':False,'properties':{
            'atoms':{'type':'array','items':atom}, 'subject_labels':strings, 'object_labels':strings,
            'has_people':{'type':'boolean'}, 'has_interaction':{'type':'boolean'}},
        'required':['atoms','subject_labels','object_labels','has_people','has_interaction']}}}

def validate_parsed(parsed):
    if not isinstance(parsed,dict) or not isinstance(parsed.get('atoms'),list) or not parsed['atoms']:
        raise ValueError('parser returned no visual atoms')
    for index,atom in enumerate(parsed['atoms']):
        if not isinstance(atom,dict) or atom.get('type') not in ATOM_TYPES:
            raise ValueError(f'atom {index}: unsupported type; use the specified enum')
        if not isinstance(atom.get('text'),str) or not atom['text'].strip():
            raise ValueError(f'atom {index}: missing source text')
        for name in ('subjects','objects','attributes'):
            if not isinstance(atom.get(name),list) or not all(isinstance(v,str) for v in atom[name]):
                raise ValueError(f'atom {index}: {name} must be a string array')
        target=atom.get('target_count')
        if atom['type']=='count':
            if isinstance(target,bool) or not isinstance(target,int) or target<0 or len(atom['subjects'])!=1:
                raise ValueError(f'atom {index}: count needs one canonical subject and a nonnegative integer target_count')
        elif target is not None:
            raise ValueError(f'atom {index}: target_count must be null outside count atoms; split count into its own atom')
        if atom['type']=='spatial' and (len(atom['subjects'])!=2 or atom.get('relation') not in SPATIAL_RELATIONS):
            raise ValueError(f'atom {index}: spatial needs two canonical subjects and left/right/above/below/front/behind')
    for name in ('subject_labels','object_labels'):
        if not isinstance(parsed.get(name),list) or not all(isinstance(v,str) for v in parsed[name]):
            raise ValueError(f'{name} must be a string array')
    return parsed

@lru_cache(maxsize=128)
def parse(prompt: str) -> dict:
    """Extract typed visual constraints with a schema; never silently discard atoms."""
    endpoint=os.getenv('QWEN3_VL_ENDPOINT','http://127.0.0.1:8011/v1')
    model=os.getenv('QWEN3_VL_MODEL',MODEL)
    if model not in ALLOWED_DEPLOYMENTS:
        raise QwenUnavailable(f'refusing non-approved parser model: {model}')
    instruction='''Decompose this video generation prompt into atomic VISUAL constraints using the exact JSON schema.
Every atom has exactly one allowed type. Split a clause into several atoms if it constrains several aspects.
CRITICAL: Every explicit quantity of visible entities MUST get a separate type="count" atom, not only subject or attribute.
Count: subjects contains EXACTLY ONE canonical detectable noun; target_count is its explicit nonnegative integer count.
Example: "exactly eight red apples" -> a count atom subjects=["apple"], target_count=8, AND an attribute atom for red.
Do not invent a count from indefinite articles, timestamps, ages, duration, camera lens mm, or action repetition.
Outside count atoms target_count MUST be null (never 0 or 1 as a placeholder).
CRITICAL: Every explicit relative position between visible entities MUST get a separate type="spatial" atom,
not attribute, subject, spatial_relation or another invented type.
Spatial: subjects contains EXACTLY TWO canonical detectable nouns in ordered subject/reference order.
relation is exactly one of left/right/above/below/front/behind.
Example: "a mug to the right of a laptop" -> subjects=["mug","laptop"], relation="right".
Use consistent short English singular object labels in all atoms and label lists, suitable for object detection.
Use "man" rather than a person's proper name such as Zhou Yu. Include count and spatial entities in the label lists.
Keep text tied to the original constraint. Preserve explicit visual negative constraints; omit audio-only requirements.
For action sequences keep their temporal order. Do not invent constraints absent from the prompt.
Prompt:
'''+prompt
    schema=response_schema()
    messages=[{'role':'user','content':instruction}]
    last_error=None
    for attempt in range(2):
        payload={'model':model,'temperature':0,'max_tokens':8192,'messages':messages,'response_format':schema}
        headers={'Content-Type':'application/json'}
        if os.getenv('QWEN3_VL_API_KEY'):
            headers['Authorization']='Bearer '+os.environ['QWEN3_VL_API_KEY']
        request=urllib.request.Request(endpoint.rstrip('/')+'/chat/completions', data=json.dumps(payload).encode(), headers=headers)
        body=None
        try:
            with urllib.request.urlopen(request,timeout=300) as response:
                body=json.load(response)
            choice=body['choices'][0]
            if choice.get('finish_reason')=='length':
                raise ValueError('parser response was truncated')
            parsed=validate_parsed(json.loads(choice['message']['content']))
            parsed['parser_model']=model
            parsed['parser_schema_version']='typed_visual_atoms_v2'
            error=None
        except Exception as exc:
            last_error=exc
            error=f'{type(exc).__name__}: {exc}'
        audit=os.getenv('VIDEO_EVAL_JUDGE_AUDIT_DIR')
        if audit:
            directory=Path(audit)
            directory.mkdir(parents=True,exist_ok=True)
            (directory/f'prompt_parser_{time.time_ns()}.json').write_text(json.dumps(
                {'response':body,'validation_error':error,'attempt':attempt+1},ensure_ascii=False,indent=2))
        if error is None:
            return parsed
        messages=[{'role':'user','content':instruction+'\nYour previous extraction failed validation: '+error+'\nReturn the complete corrected extraction using the exact schema.'}]
    raise QwenUnavailable(f'prompt extraction failed validation: {last_error}')


