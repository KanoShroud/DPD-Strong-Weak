"""PyCharm直接运行；只消费批准并通过开发核验的冻结合同。"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path('D:/Software/anaconda3/envs/PyTorch/python.exe')
PREPARED = ROOT/'outputs_e2e/unified/e2e_g9/prepare_20261010_01/frozen_contract.json'


def main():
    if not PREPARED.is_file():
        raise RuntimeError('开发阶段尚未交付冻结合同，不能启动正式比较。')
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8:replace', MPLBACKEND='Agg')
    command = [str(PYTHON), '-m', '统一模型代码.gates.g9.runner', '--stage', 'formal',
               '--run-id', 'formal_20261010_approved', '--contract', str(PREPARED)]
    # 直接继承控制台，无后台转发线程；单行进度每15秒至多更新一次。
    result = subprocess.run(command, cwd=str(ROOT), env=env, text=True,
                            encoding='utf-8', errors='replace', check=False)
    if result.returncode:
        raise RuntimeError(f'G9退出码{result.returncode}；请保留结果目录及报错。')
    print('完成。请回读 outputs_e2e/unified/e2e_g9/formal_20261010_approved 下的运行摘要与报告。')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
