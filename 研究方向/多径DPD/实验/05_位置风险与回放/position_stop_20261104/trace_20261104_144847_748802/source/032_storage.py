"""Atomic artifacts, process lock and interruption-aware cost records."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np


def encode(value):
    def conv(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    return json.dumps(value,ensure_ascii=False,allow_nan=False,default=conv,sort_keys=True)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        f.write(encode(value));f.flush();os.fsync(f.fileno())
    for attempt in range(8):
        try:
            os.replace(tmp,path);return
        except PermissionError:
            if attempt==7: raise
            time.sleep(min(.05*2**attempt,1.))


def save(path,payload):
    write(path,dict(payload=payload,sha256=hashlib.sha256(encode(payload).encode('utf-8')).hexdigest()))


def load(path):
    d=read(path)
    if hashlib.sha256(encode(d['payload']).encode('utf-8')).hexdigest()!=d['sha256']:
        raise RuntimeError('产物内容身份不匹配：'+str(path))
    return d['payload']


@contextmanager
def lock(out):
    import msvcrt
    p=Path(out)/'batch.lock';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as exc:raise RuntimeError('同一批次已有进程运行，请勿重复点击。') from exc
        try:yield
        finally:
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


def events(out):
    p=Path(out)/'events.jsonl'
    if not p.exists():return []
    # A truncated tail is an interrupted write, never silently discarded.
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]


def append(out,row):
    row=dict(utc=datetime.now(timezone.utc).isoformat(),**row)
    with (Path(out)/'events.jsonl').open('a',encoding='utf-8',newline='\n') as f:
        f.write(encode(row)+'\n');f.flush();os.fsync(f.fileno())


def recover(out):
    p=Path(out)/'active.json'
    if not p.exists():return
    active=read(p)
    if not active:return
    done={e['id'] for e in events(out) if e['kind'] in ('end','aborted','interrupted_recovered')}
    if active['id'] not in done:
        append(out,dict(kind='interrupted_recovered',id=active['id'],stage=active['stage'],
                        seconds=active['seconds'],lower_bound=True))
        print('恢复：保留上次中断成本下界',round(active['seconds'],1),'秒；未完成阶段将重跑。',flush=True)
    write(p,None)


def cost(out):
    rows=[e for e in events(out) if e['kind'] in ('end','aborted','interrupted_recovered')]
    return dict(seconds=sum(e['seconds'] for e in rows),
                repeated_or_aborted_seconds=sum(e['seconds'] for e in rows if e['kind']!='end'),
                has_lower_bound=any(e.get('lower_bound',False) for e in rows))


def stage(out,name,target,fn):
    if Path(target).exists():return load(target),False
    ident=uuid.uuid4().hex;begin=time.perf_counter();stop=threading.Event();errors=[]
    append(out,dict(kind='start',id=ident,stage=name))
    def beat():
        while not stop.is_set():
            try:write(Path(out)/'active.json',dict(id=ident,stage=name,seconds=time.perf_counter()-begin))
            except Exception as exc:errors.append(repr(exc));return
            stop.wait(5.)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    try:
        payload=fn()
        if errors:raise RuntimeError('心跳保存失败：'+errors[0])
        save(target,payload)
        stop.set();thread.join()
        append(out,dict(kind='end',id=ident,stage=name,seconds=time.perf_counter()-begin))
        write(Path(out)/'active.json',None)
        return payload,True
    except BaseException:
        stop.set();thread.join()
        append(out,dict(kind='aborted',id=ident,stage=name,seconds=time.perf_counter()-begin))
        write(Path(out)/'active.json',None)
        raise
