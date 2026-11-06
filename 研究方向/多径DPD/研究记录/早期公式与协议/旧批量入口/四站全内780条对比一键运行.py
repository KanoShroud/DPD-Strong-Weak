"""Run/resume the approved all-inside 6-layout x 10-realization comparison."""
import importlib.util
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[3]
    folder = root/'outputs_e2e/research_admission_20261027/localization_identifiability/inside_pilot_20261104'
    spec = importlib.util.spec_from_file_location('inside_pilot_entry', folder/'run_inside.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    print('四站凸包内：6构型 × 10独立实现 × 13条件 = 780条；自动核对并续跑已完成前缀。', flush=True)
    print('本批为已批准的扩展诊断，不表示原3120条正式门禁已放行。', flush=True)
    module.main(sys.argv[1:])


if __name__ == '__main__':
    main()
