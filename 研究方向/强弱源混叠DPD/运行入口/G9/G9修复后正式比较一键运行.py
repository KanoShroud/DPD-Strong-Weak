"""修复版G9；PyCharm直接运行，同目录结果按合同恢复。"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
PYTHON=Path('D:/Software/anaconda3/envs/PyTorch/python.exe')
CONTRACT=ROOT/'研究方向/强弱源混叠DPD/实验/e2e_g9/repair_prepare_20261011_01/frozen_contract.json'


def main():
    if not CONTRACT.is_file():
        raise RuntimeError('修复版开发合同尚未就绪，请勿使用旧版入口代替。')
    env=os.environ.copy()
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8:replace',MPLBACKEND='Agg')
    command=[str(PYTHON),'-m','研究方向.强弱源混叠DPD.代码.gates.g9.repair_runner','--stage','formal',
             '--run-id','formal_20261011_repaired','--contract',str(CONTRACT)]
    result=subprocess.run(command,cwd=str(ROOT),env=env,text=True,encoding='utf-8',errors='replace')
    if result.returncode:
        raise RuntimeError(f'退出码{result.returncode}，请保留报错和结果，不删除目录。')
    print('完成：请回读研究方向/强弱源混叠DPD/实验/e2e_g9/formal_20261011_repaired中的运行摘要及报告。')


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    main()
