"""User handoff: resume frozen pilot then refresh its readback report."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
TASK = ROOT/'outputs_e2e/research_admission_20261027/localization_identifiability/risk_bootstrap_20261103'
ENTRY = Path(__file__).with_name('G11有限噪声风险核验一键运行.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8:replace')
    if args.check:
        result = subprocess.run([sys.executable,'-X','utf8',str(ENTRY),'--help'],cwd=ROOT,
                                env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',check=True)
        print(result.stdout)
        print('中文入口检查通过；没有运行定位。')
        return
    print('恢复同一批有限噪声校准：复用已完成副本，仅补未完成项。科学门失败会停止；不启动640条。',flush=True)
    subprocess.run([sys.executable,'-X','utf8',str(ENTRY)],cwd=ROOT,env=env,
                   text=True,encoding='utf-8',errors='replace',check=True)
    subprocess.run([sys.executable,'-X','utf8',str(TASK/'audit_report.py')],cwd=ROOT,env=env,
                   text=True,encoding='utf-8',errors='replace',check=True)
    print('阶段完成；请把risk_bootstrap_20261103/latest.json指向的结果目录交给Codex回读。',flush=True)


if __name__=='__main__':
    main()
