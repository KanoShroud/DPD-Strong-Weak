"""Synthetic-input old/new G5 forward comparison, without data/checkpoint reads."""
from pathlib import Path
import ast
import json
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from 统一模型代码.gates.g5 import e2e_g5_model as model
from train_v26 import SourceDetectionNet
from yolo_model import YOLOv8Loc

torch.set_num_threads(2)
torch.manual_seed(20260921)
g4=model.g4
context=SimpleNamespace(ch3=SourceDetectionNet(n_sub=19,max_src=3,mode='transformer'),
                        d8=YOLOv8Loc(method='dualhead'),query_builder=g4.SourceQueryBuilder(),
                        splitter=g4.FrequencySpatialSplitter(),source_head=g4.SourceLocalizationHead())
for item in vars(context).values():item.eval()
rng=np.random.default_rng(20260921)
features=SimpleNamespace(spatial=rng.normal(size=(1,19,128,11,11)).astype('float32'),
                         e1=rng.normal(size=(1,32,208,208)).astype('float32'),
                         d2=rng.normal(size=(1,64,104,104)).astype('float32'))
source=ast.parse((BASE/'before/统一模型代码/e2e_g5_model.py').read_text(encoding='utf-8'))
original=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
namespace=dict(vars(model))
exec(compile(ast.Module(body=[original],type_ignores=[]),'<pre-migration-forward>','exec'),namespace)
results=[]
for sg in [False,True]:
    with torch.no_grad():
        a=namespace['forward'](context,features,torch.tensor([0]),torch.device('cpu'),stop_gradient=sg)
        b=model.forward(context,features,torch.tensor([0]),torch.device('cpu'),stop_gradient=sg)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    results.append({'stop_gradient':sg,'shapes':[list(x.shape) for x in a],'all_five_outputs_exact':True})
report={'status':'PASS','cases':results,'input':'synthetic normal float32 batch1; randomly initialized actual CH3/D8/query/splitter/head',
        'scope':'old/new G5 forward functions using the same computationally unchanged dependencies; no formal experiment'}
(BASE/'forward_equivalence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
