"""Generate a concise HTML report from the saved derivation and algebra results."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'AGENTS.md').exists())
previous = ROOT/'研究方向/多径DPD/实验/10_文献机制综合/20261107/成果/summarize.py'
spec = importlib.util.spec_from_file_location('previous_report_renderer', previous)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    raw = HERE.parent/'运行数据/algebra/result.json'
    data = json.loads(raw.read_text(encoding='utf-8'))
    for p, digest in data['source_hashes'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
    (HERE/'数学核验结果.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    results = f'''# 本轮数学检查

| 检查 | 实测值 | 解释 |
|---|---:|---|
| 原模型与秩一提升观测相对差 | {data['observation_identity_max_error']:.3g} | 同一个模型的重写 |
| 旧正则与提升后反射列范数相对差 | {data['regularizer_identity_max_error']:.3g} | 罚项也未改变 |
| 去秩一松弛的相对残差 | {data['rank_dropped_relaxation']['relative_residual']:.3g} | 任意相位映射可拟合，并非定位成功 |
| 两个松弛源矩阵的秩 | 4、4 | 已违反每源共享一个波形 |
| 功率约束中点 | 不可行 | 绝对增益坐标下非凸 |
| 旋转前/后混合四阶累积量 | 0 / −1 | 二阶统计相同仍可存在高阶差异 |

100组代数单元检查，固定种子2026110713。不是新的定位实验、不是简化场景的性能证据。没有生成新的RMSE或耗时成绩。
'''
    doc = (HERE/'推导与决策.md').read_text(encoding='utf-8')
    urls = [('ICA作者全文','https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf'),
            ('非圆线性调制分离：作者稿','https://arxiv.org/abs/1102.2706'),
            ('非圆IVA：出版入口','https://doi.org/10.1016/j.sigpro.2012.02.020'),
            ('卷积MIMO辨识：IEEE DOI','https://doi.org/10.1109/TSP.2009.2016259')]
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>共享波形建模核验</title><style>body{background:#f4f5f1;color:#23383f;font:16px/1.8 "Segoe UI","Microsoft YaHei",sans-serif;margin:0}main{max-width:1080px;margin:auto;padding:32px 24px}header{background:#17424a;color:white;border-radius:16px;padding:28px}h1{font-size:30px}h2{margin-top:32px;border-bottom:1px solid #cbd8d4;padding-bottom:8px}h3{color:#086774}a{color:#086774}nav{display:flex;gap:20px;flex-wrap:wrap;margin:22px 0}.note{background:#fff0d4;padding:18px;border-left:4px solid #b98324}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d5e1da;padding:10px;text-align:left;vertical-align:top}th{background:#e2ece6}.scroll{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#e5eeeb;border-radius:8px;padding:16px}li{margin-bottom:8px}footer{font-size:13px;color:#64747a;margin-top:28px}</style><main><header><div>多径DPD · 第11轮 · 2026-11-07</div><h1>先确认新模型确实是新的</h1><p>共享波形表示 → 与旧目标的等价检查 → 松弛边界 → 统计信息合同</p></header><nav><a href="#checks">数学检查</a><a href="#reasoning">完整推导</a><a href="#sources">文献入口</a><a href="执行合同.md">合同</a><a href="数学核验结果.json">数值记录</a></nav><div class="note">本轮停在新的统计适用条件需要确认之处。没有启动新定位器，也没有把过去的正则目标换名后重新跑实验。</div><section id="checks">'''+module.render(results)+'''</section><section id="reasoning">'''+module.render(doc)+'''</section><section id="sources"><h2>本轮定向核对入口</h2><ul>'''+''.join(f'<li><a href="{url}">{label}</a></li>' for label,url in urls)+'''</ul></section><footer>已有研究记录与公开文献定向核对；AI辅助推导。未声称完成这些新增文献的逐篇精读或新算法验证。</footer></main></html>'''
    (HERE/'共享波形建模核验.html').write_text(page,encoding='utf-8')
    print('Report and source identities verified')


if __name__ == '__main__':
    main()
