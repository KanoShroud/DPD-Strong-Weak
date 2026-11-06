"""One click: all approved four-station formal methods, automatic recovery."""
import importlib.util
from pathlib import Path
import sys
from threadpoolctl import threadpool_limits


def main():
    root = Path(__file__).resolve().parents[3]
    folder = root/'outputs_e2e/research_admission_20261027/localization_identifiability/four_station_physical_20261104'
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location('four_physical_main', folder/'runner.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with threadpool_limits(limits=2):
        module.main(['--phase', 'formal'])


if __name__ == '__main__':
    main()
