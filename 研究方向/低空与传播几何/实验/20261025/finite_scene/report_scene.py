"""Independent saved-ray audit and explanatory figures; no new ray search."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'run_20261025_190128_421395'


def crossing(a,b,panel):
    n = np.array(panel['normal'])
    tangent = np.cross([0.,0.,1.],n)
    matrix = np.column_stack([b-a,-tangent,[0.,0.,-1.]])
    if abs(np.linalg.det(matrix))<1e-10:
        return False
    t,u,v = np.linalg.solve(matrix,np.array(panel['center_m'])-a)
    return bool(1e-8<t<1-1e-8 and abs(u)<=panel['width_m']/2+1e-8
                and abs(v)<=panel['height_m']/2+1e-8)


def audit(data):
    panels = {p['name']:p for p in data['panels']}
    stations = np.array(data['stations_m'])
    path_ok = True
    for r in data['records']:
        source = np.array(r['source_m'])
        receiver = stations[r['station_id']]
        faces = [panels[k] for k in data['scene_panels'][r['scene']]]
        blocked = any(crossing(source,receiver,p) for p in faces)
        present = any(x['kind']=='direct' for x in r['paths'])
        path_ok &= present != blocked
        for ray in r['paths']:
            if ray['kind']!='reflection':
                continue
            q = np.array(ray['point_m'])
            path_ok &= abs(np.linalg.norm(q-source)+np.linalg.norm(q-receiver)-ray['length_m'])<1e-7
            path_ok &= not any(crossing(a,b,p) for p in faces if p['name']!=ray['panel']
                               for a,b in [(source,q),(q,receiver)])
    errors = {}
    for scene in data['scene_panels']:
        with np.load(RUN/(scene+'_iq.npz')) as d:
            H = np.zeros((2,5,len(d['frequency_hz'])),complex)
            for q,source_id in enumerate([9,12]):
                for record in data['records']:
                    if record['scene']!=scene or record['source_id']!=source_id:
                        continue
                    for ray in record['paths']:
                        length = ray['length_m']
                        gain = (1. if ray['kind']=='direct' else -.5)/length
                        H[q,record['station_id']] += gain*d['coefficients'][q]*np.exp(-2j*np.pi*d['frequency_hz']*length/299792458.)
            rebuilt = np.sum(H,axis=0)@np.exp(2j*np.pi*d['frequency_hz'][:,None]*d['t_s'][None,:])
            errors[scene] = float(np.linalg.norm(rebuilt-d['mixture'])/np.linalg.norm(d['mixture']))
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in data['identity'].items())
    record = {'independent_segment_visibility':bool(path_ok),'iq_replay_errors':errors,
              'input_identity_unchanged':unchanged,'pass':bool(path_ok and unchanged and max(errors.values())<1e-10)}
    (RUN/'independent_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    if not record['pass']:
        raise RuntimeError('Saved-ray audit failed')
    return record


def figures(data):
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,
                         'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    stations = np.array(data['stations_m'])
    points = np.array(data['positions_m'])
    colors = ['#087f82','#8063a4']
    fig,axes = plt.subplots(1,2,figsize=(12,7.5),constrained_layout=True)
    for q,(ax,source_id) in enumerate(zip(axes,[9,12])):
        source = points[source_id]
        for panel in data['panels']:
            vertices = np.array(panel['vertices_m'])
            ax.plot(vertices[:2,0],vertices[:2,1],color='#7b8983',lw=6,zorder=1)
            ax.annotate('立面'+panel['name'],vertices[:2,:2].mean(axis=0),xytext=(7,6),textcoords='offset points',color='#536760')
        ax.scatter(stations[:,0],stations[:,1],marker='^',s=65,c='#193d44',label='接收站')
        for m,s in enumerate(stations):
            ax.annotate(str(m+1),s[:2],xytext=(7,5),textcoords='offset points')
        ax.scatter(*source[:2],marker='*',s=190,c=colors[q],zorder=5,label='源'+('A' if q==0 else 'B'))
        for r in data['records']:
            if r['scene']!='two_AB' or r['source_id']!=source_id:
                continue
            receiver = stations[r['station_id']]
            for ray in r['paths']:
                chain = np.array([source,receiver] if ray['kind']=='direct' else [source,ray['point_m'],receiver])
                ax.plot(chain[:,0],chain[:,1],'-' if ray['kind']=='direct' else '--',color=colors[q],lw=1.4,alpha=.8)
            if not r['paths']:
                ax.scatter(*receiver[:2],marker='x',s=150,c='#b43d43',linewidths=2.2,label='该源到此站无可见路径')
        ax.plot([],[],color=colors[q],label='直达')
        ax.plot([],[],'--',color=colors[q],label='一次反射')
        ax.set(xlabel='x / m',ylabel='y / m',xlim=(-600,600),ylim=(-650,500),
               title=f'源{"A" if q==0 else "B"}：高度{source[2]:.0f}m，全部可见路径')
        ax.set_aspect('equal');ax.grid(alpha=.16)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,fontsize=9,frameon=False)
    fig.suptitle('固定两立面场景｜俯视图，真实三维可见性计算',fontsize=15)
    for ext in ['png','svg']:
        fig.savefig(RUN/('paths_top.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure(figsize=(10,6.5),constrained_layout=True)
    ax = fig.add_subplot(111,projection='3d')
    for panel in data['panels']:
        vertices = np.array(panel['vertices_m'])
        ax.add_collection3d(Poly3DCollection([vertices],alpha=.25,facecolor='#7b8983',edgecolor='#536760'))
        c = np.array(panel['center_m'])
        ax.text(c[0],c[1],70,'立面'+panel['name'])
    ax.scatter(*stations.T,marker='^',s=55,c='#193d44')
    for m,s in enumerate(stations):
        ax.text(s[0],s[1],s[2]+6,f'站{m+1}',fontsize=9)
    for q,source_id in enumerate([9,12]):
        p = points[source_id]
        ax.scatter(*p,marker='*',s=160,c=colors[q]);ax.text(*p,' 源'+('A' if q==0 else 'B'),color=colors[q])
    ax.set(xlabel='x / m',ylabel='y / m',xlim=(-600,600),ylim=(-650,500),zlim=(0,140))
    ax.text2D(.87,.78,'高度 z / m',transform=ax.transAxes)
    ax.set_box_aspect((1,1,.45));ax.view_init(elev=30,azim=-120)
    ax.set_title('固定生成环境：两块有限立面与五站双源\n高度在画面中放大，以坐标刻度为准',pad=15)
    for ext in ['png','svg']:
        fig.savefig(RUN/('scene3d.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    data = json.loads((RUN/'result.json').read_text(encoding='utf-8'))
    check = audit(data)
    figures(data)
    names = {'empty':'无立面正对照','single_A':'只有立面A','two_AB':'固定立面A＋B'}
    table = []
    for name,c in data['summary'].items():
        table.append(f"| {names[name]} | {c['direct']-c['direct_plus_reflection']} | {c['direct_plus_reflection']} | {c['reflection_only']} | {c['empty']} |")
    pair = data['iq_checks']['two_AB']['path_counts_source_major']
    text = f'''# 有限立面核验：路径现在由同一个环境决定

2026-10-25｜完成固定有限场景的一阶几何生成与IQ检查；没有运行定位算法。

## 结论先说

**这一步补齐了一个源头问题：不能把“某站直达被挡住”和“它还有一条反射”分开随意指定。**

单个不透射平面若挡住源—站直线，二者就在平面两侧；同一面的一次镜面反射却要求二者位于同侧。因此，在本轮明确的一阶镜面模型中，它不能既挡住该直达，又独自提供该链路的反射。

预先固定第二立面后，部分被挡链路得到来自另一面的可见绕路；仍有空链路，也自然产生直达与多个反射并存。它们全部保留，没有为了符合预期修改墙面或补造信号。

## 1 固定的场景是什么

![固定三维场景]({RUN.name}/scene3d.png)

- 五站平面位置沿用前轮，同高4m；总共两个未知波形源，不做跟踪。生成器不向定位端提供地图或路径标签，本轮尚无定位器。
- 立面A：x=-250m，沿y宽400m、高60m。立面B：y=-550m，沿x宽500m、高60m。具体边界在运行前合同中固定。
- A是单面主诊断，B是预先登记的结构对照，不是看到失败后移动墙来通过。两块矩形薄面是合成几何例子，不是测绘建筑或已冻结的论文场景。
- 12个沿用位置，加旧源B和旧替代位置，共14个位置；每级别70条链路。三种级别合计210条，全部保留。

这里“完整”仅指**在声明的一次反射、无穿透/绕射/漫散射模型内**列全可见路径；没有包含地面反射、多次反射或完整电磁传播。

## 2 同一环境确实改变了路径种类

| 固定级别 | 仅直达 | 直达＋反射 | 仅反射 | 无路径 |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

四列互斥，每行70条源—站链路；不是定位成功率或独立随机样本。

单面8条被挡链路均没有一次镜面绕路。增加B后，其中5条获得可见反射，3条仍无路径。这里的“无路径”限本模型，不表示现实中没有任何绕射、穿透或噪声。

## 3 旧双源位置放进去，会听到什么

![逐源全部可见路径]({RUN.name}/paths_top.png)

图中实线是直达、虚线是一次反射；红叉表示该源对该站无可见路径，不是接收站失效。两幅图使用同一场景、同一组站点，只分开显示源A/B。

固定源A=(-460,-120,34)m，源B=(-80,-100,69)m。两立面场景中：

| 源 | 站1路径数 | 站2 | 站3 | 站4 | 站5 | 直达站数 |
|---|---:|---:|---:|---:|---:|---:|
| A | {pair[0]} | {pair[1]} | {pair[2]} | {pair[3]} | {pair[4]} | 3 |
| B | {pair[5]} | {pair[6]} | {pair[7]} | {pair[8]} | {pair[9]} | 4 |

源A自然只剩3个直达站，源B在站4没有本模型内的可见路径；没有强行改成“每源4LOS、每站恰好一条路径”。这意味着本场景不能直接作为此前4LOS条件已满足的正式性能数据。额外反射是否提供足够信息，需要另作未知环境下的辨识检查。

## 4 旧的75m假位置是否还完全等价

在**相同固定场景**内，原位置与旧替代位置产生的完整路径数量不同，原来的“各站统一挪动波形时间原点”解释不再直接成立。

但这不是盲定位已消歧：算法不知道地图，另一位置可能配上另一环境；本轮没有搜索所有环境/位置解释。也没有证明其他共同滤波或波形—信道分解歧义不存在。

因此正确结论是：**旧的自由路径反例不能原封不动搬到新场景；新场景的盲唯一性仍未建立。** 同地图比较不能冒充无地图定位保证。

## 5 当前有什么实质进展

已经得到：统一的有限面片遮挡/反射生成器、公开的适用范围、可重放路径和混合IQ，以及明确的模型缺口。

还没有得到：新的定位算法、定位RMSE、对所有位置的唯一性保证或已确认的论文贡献。

**建议暂不继续增加墙面和参数。** 保留这个小模型作为统一检查工具。若继续推进，下一步只需在同样的未知地图权限下，检查普通路径关联/联合拟合能否在信息足够的条件中可靠工作，同时标记多解与信息不足条件。源A的3LOS和零路径结果不能被悄悄删掉，也不能仅凭它们否定所有结构化多径定位。

正式主场景是否采用这些有限面片/哪些传播机制，以及如何定义可用条件，仍需明确；本轮未自动开放性能批次或改变论文主张。

<details><summary>证据、检查与边界</summary>

- [运行前合同](执行合同.md)、[几何生成代码](finite_scene.py)、[执行代码](run_scene.py)、[独立核验/绘图](report_scene.py)。
- [原始结果]({RUN.name}/result.json)、[路径清单]({RUN.name}/path_list.csv)、[独立审计]({RUN.name}/independent_audit.json)、[输入SHA]({RUN.name}/preflight.json)。图由该结果文件生成，不是示意手绘。
- 面片内部/外部/边界、相反侧反射排除、其他面遮挡、反射定律和互易性等{len(data['checks'])}项检查通过；固定210条链路没有边界或共面退化。
- 独立线段—面片交点采用三元线性方程核对，不复用原遮挡实现；逐路径重建IQ相对差最大{max(check['iq_replay_errors'].values()):.3e}。使用连续非DFT频点表达同一未知带限波形，不进行循环移位近似。
- 保存3份5站×1024点无噪混合IQ及逐源分量。生成系数直达1/L、反射-0.5/L，仅用于一致性验证；不是材料实测、不是给算法的已知增益，也不能拿它与旧自由增益反例作性能比较。
- 原数据身份不变；合同与源码在运行目录有事前快照。内部执行约{data['wall_seconds']:.3f}s，不是DPD定位耗时。没有运行定位性能大批次。
- 首轮`run_20261025_185551_067440`保留；随后补强了对前轮已登记SHA的事前核对，以同一科学配置在本目录重跑。报告使用补强预检后的运行，不把两次计算当新增独立样本。

</details>
'''
    (ROOT/'有限立面核验报告.md').write_text(text,encoding='utf-8')
    print(json.dumps(check,ensure_ascii=False))


if __name__=='__main__':
    main()
