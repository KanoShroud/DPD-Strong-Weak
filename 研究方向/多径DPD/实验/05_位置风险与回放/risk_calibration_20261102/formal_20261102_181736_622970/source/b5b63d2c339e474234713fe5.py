"""PyCharm一键：32条条件校准，全流程运行并自动恢复。"""
from pathlib import Path
import runpy
import sys


def main():
    root=Path(__file__).resolve().parents[3]
    source=root/'outputs_e2e/research_admission_20261027/localization_identifiability/risk_calibration_20261102/runner.py'
    if not source.is_file():raise FileNotFoundError('缺少冻结运行器：'+str(source))
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    if hasattr(sys.stderr,'reconfigure'):sys.stderr.reconfigure(encoding='utf-8',errors='replace')
    runpy.run_path(str(source),run_name='__main__')


if __name__=='__main__':main()
