"""仅工程验证：合成指标夹具、报告绘图及UTF8子进程捕获，不访问场景数据。"""
import os
import subprocess
import sys

from .metrics import evaluate
from .report import finalize
from .runtime import BASE, preflight, write


def main():
    out=preflight(BASE/'repair_report_fixture_20261011_01')
    rows=[]
    for group in range(32):
        for n in (4096,16384):
            for overlap in (.5,1.):
                for gap in (6,12):
                    for method in ('A','B','C','DPD','HR'):
                        for factor in (.5,1.,2.):
                            for kind in ('mixed','strong','weak'):
                                truth=[[0.,0.],[300.,0.]] if kind=='mixed' else [[0.,0.]]
                                rows.append(dict(group=group,N=n,overlap=overlap,gap=gap,method=method,
                                    factor=factor,kind=kind,seconds=factor,score_points=1,
                                    response_points=0,updates=1,metrics=evaluate(truth,truth)))
    report=finalize(out,rows,dict(best_baseline='A',scope='SYNTHETIC_ENGINEERING_FIXTURE_NOT_EXPERIMENT'))
    assert all(r['rmse_m']==0 and r['both_100']==1 for r in report['summary'].values())
    report['status']='ENGINEERING_FIXTURE_PASS_NOT_RESEARCH_RESULT'
    write(out/'comparison_report.json',report)
    (out/'运行摘要.md').write_text('# 合成工程夹具\n\n仅验证汇总、匹配、曲线写出；不是实验结果。\n',encoding='utf-8')
    env=os.environ.copy()
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8:replace')
    result=subprocess.run([sys.executable,'-c',"print('修复版中文日志捕获通过')"],
        env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',check=True)
    assert '中文日志' in result.stdout
    write(out/'verification.json',dict(status='PASS',scope='synthetic engineering only',
        report_rows=len(rows),utf8_stdout=result.stdout,training=False,test_access=False))
    print(str(out/'verification.json'))


if __name__=='__main__':
    os.environ.setdefault('MPLBACKEND','Agg')
    main()
