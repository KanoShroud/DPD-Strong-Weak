"""只读检查已退役缓存；不训练、不重建、不加载科研张量。"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGETS = (
    ('旧G7大区域', 'e2e_g7/20260924_approved/local_cache', '.pt'),
    ('G5-R3', 'e2e_g5_r3/20260922_115824/cache', '.bin'),
    ('G6-P0', 'e2e_g6_p0/20260922_165854/cache', '.pt'),
)


def inspect(root=ROOT):
    rows = []
    for name, relative, suffix in TARGETS:
        folder = root / 'outputs_e2e/unified' / relative
        marker = folder / 'CACHE_RETIRED.json'
        record = json.loads(marker.read_text(encoding='utf-8')) if marker.is_file() else None
        count = sum(1 for _ in folder.rglob('*' + suffix)) if folder.exists() else 0
        state = ('RETIRED_REBUILD_REQUIRED' if count == 0 else 'RETIRED_WITH_NEW_FILES_REVIEW_REQUIRED') if record else 'UNVERIFIED'
        rows.append(dict(name=name, path=str(folder), status=state, payload_files=count))
    return rows


if __name__ == '__main__':
    for row in inspect():
        print(f"{row['name']}: {row['status']}，载荷文件数={row['payload_files']}")
    print('退役缓存不能直接用于历史复跑；保留索引不代表载荷存在。重建须使用新目录并核验输入和新身份。')
