"""Reader-facing report; ideal structural checks are not localization results."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[3]


def main():
    result=json.loads((HERE/'summary.json').read_text(encoding='utf-8'))
    assert result['count']==30 and all(r['correct_direct_groups'] for r in result['rows'])
    spec=importlib.util.spec_from_file_location('renderer',BASE/'实验/10_文献机制综合/20261107/成果/summarize.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    text=(HERE/'结构推导与实现边界.md').read_text(encoding='utf-8')
    table='<table><tr><th>每链路反射数</th><th>数学单元数</th><th>每源闭合组数</th><th>恢复结果</th></tr>'
    for L in (0,1,2):table+=f'<tr><td>{L}</td><td>10</td><td>{(L+1)**4}</td><td>两源分组和直达组合均通过</td></tr>'
    table+='</table>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>联合源径图候选结构</title><style>body{background:#f3f5f1;color:#20353d;font:16px/1.85 "Segoe UI","Microsoft YaHei",sans-serif;margin:0}main{max-width:1120px;margin:auto;padding:30px}header{background:#173e47;color:white;border-radius:16px;padding:30px}h1{font-size:30px}section{background:white;padding:24px;border-radius:12px;margin:20px 0}.notice{background:#fff0d5;border-left:4px solid #b88c34;padding:18px}nav{display:flex;gap:20px;flex-wrap:wrap;margin:20px 0}a{color:#087080}.flow{display:flex;gap:12px;flex-wrap:wrap}.flow div{background:#e4eeeb;flex:1;min-width:170px;padding:18px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d7e0db;padding:10px;text-align:left}th{background:#e4eeeb}pre{background:#edf2ef;padding:15px;white-space:pre-wrap}footer{font-size:13px;color:#65787b}@media(max-width:650px){main{padding:12px}section{padding:14px}}</style><main><header>多径DPD · 第15–16轮 · 2026-11-07<h1>先把同源路径组归在一起，再找直达组</h1><p>单个闭合评分未通过；联合源—路径图通过理想接口结构测试。实际IQ接口尚待解决。</p></header><nav><a href="#structure">完整推导与边界</a><a href="../../../15_联合解释准则准入/20261107/成果/联合解释准则准入报告.html">第15轮负面结果</a><a href="summary.json">单元测试JSON</a></nav><p class="notice"><strong>没有新增实际定位成绩。</strong>30个通过项是无噪、完整、可分辨的无标签分量接口单元，不是30次混叠IQ盲定位。</p><div class="flow"><div>每个站对：未知源与路径标签的时差、复系数</div><div>四站一致组：同源路径组合，但可能是反射</div><div>共享边归组：将同一源的多个组合连接起来</div><div>源内选直达：利用逐站直达最强，再交给几何与DPD精修</div></div>'''
    page+='<section><h2>结构核验</h2>'+table+f"<p>总计{result['seconds']:.3f}秒，仅为理想接口计算，不包含从IQ恢复分量，也不是完整定位耗时。</p></section>"
    page+='<section id="structure">'+mod.render(text)+'</section><footer><a href="执行合同.md">数学接口合同</a> · <a href="check_graph.py">可复现单元入口</a> · <a href="结构推导与实现边界.md">推导源文件</a><p>不向定位器提供真值分量，不用数学单元代替噪声实验，不宣称完成创新风险核验。</p></footer></main></html>'
    (HERE/'联合源径图候选报告.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
