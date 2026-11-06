"""Mechanical module relocation, with numeric algorithms left intact."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOME = Path(__file__).resolve().parent
PACKAGE = '研究方向.多径DPD.代码.mp_dpd'
CODE = ROOT/'研究方向/多径DPD/代码/mp_dpd'
MAPPING = {
 'alternative_fit.py':'numerics/parameterization.py',
 'alternative_gradient_check.py':'numerics/derivatives.py',
 'blind_backend_pilot.py':'solvers/blind_backend.py',
 'blind_initialization.py':'initialization/blind_initialization.py',
 'boundary_update.py':'solvers/boundary_update.py',
 'bounded_consensus.py':'initialization/consensus.py',
 'finite_iq_clues.py':'models/waveform_iq.py',
 'flood_band_pair.py':'baselines/flood.py',
 'flood_lp_check.py':'baselines/flood_lp.py',
 'geometry_association_gate.py':'initialization/tdoa_candidates.py',
 'position_reduction.py':'numerics/position_profile.py',
 'robust_geometry.py':'initialization/robust_geometry.py',
 'solver_reliability.py':'numerics/local_solver.py',
 'structural_probe.py':'models/frequency_operator.py',
 'acceptance_gate_20261104/solver.py':'solvers/gated_solver.py',
 'failure_attribution_20261104/diagnose.py':'initialization/region_helpers.py',
 'four_station_physical_20261104/adapters.py':'models/physical_binding.py',
 'four_station_physical_20261104/report.py':'io/report.py',
 'four_station_physical_20261104/runner.py':'io/record_pipeline.py',
 'four_station_physical_20261104/scene.py':'models/scene.py',
 'inside_pilot_20261104/run_inside.py':'models/inside_scene.py',
 'natural_stop_20261104/hybrid_gate.py':'solvers/acceptance.py',
 'natural_stop_20261104/run.py':'solvers/natural_helpers.py',
 'natural_stop_20261104/solver.py':'solvers/natural_stop.py',
 'numerical_repair_20261104/stable.py':'solvers/orthogonal_vp.py',
 'risk_calibration_20261102/runner.py':'io/spectral_pipeline.py',
 'risk_calibration_20261102/storage.py':'io/storage.py',
 'risk_reference_pilot_20261102/pilot.py':'numerics/risk_diagnostics.py',
 'risk_step_20261030/pilot.py':'numerics/vp_core.py',
 'speed_gate_20261104/budget_candidate.py':'solvers/group_increment.py',
 'speed_gate_20261104/fast_core.py':'solvers/profile_core.py',
 'speed_gate_20261104/run_gate.py':'solvers/speed_helpers.py',
 'station_count_20261104/backend.py':'models/station_binding.py',
 'unknown_order_20261030/cross_order.py':'models/reference_scene.py',
 'unknown_order_20261030/reference.py':'solvers/reference.py',
 'unknown_order_20261030/reference_feasible.py':'solvers/feasible.py',
 'unknown_order_20261030/reference_pg.py':'solvers/projected.py',
 'xia_scale_20261103/kernel.py':'baselines/xia_kernel.py',
}
MAP = {'localization_identifiability/'+k:v for k,v in MAPPING.items()}
MAP.update({'xia_pairing/recheck/xia_checked.py':'baselines/xia_decoder.py',
            'xia_pairing/xia_reference.py':'baselines/xia_reference.py'})
SPECIAL = {
 ('acceptance_gate_20261104/solver.py',"NAT / 'solver.py'"):'solvers/natural_stop.py',
 ('speed_gate_20261104/fast_core.py',"CAL / 'runner.py'"):'io/spectral_pipeline.py',
 ('risk_calibration_20261102/runner.py',"PILOT / 'pilot.py'"):'numerics/risk_diagnostics.py',
 ('inside_pilot_20261104/run_inside.py',"PHY / 'runner.py'"):'io/record_pipeline.py',
 ('four_station_physical_20261104/runner.py',"cfg.HERE / 'report.py'"):'io/report.py',
 ('failure_attribution_20261104/diagnose.py',"FORMAL.parent / 'runner.py'"):'io/spectral_pipeline.py',
}


def qualified(dest):
    return PACKAGE+'.'+dest[:-3].replace('/', '.')


def local_import(name):
    if name == 'solver':
        return 'solvers/natural_stop.py'
    if name == 'pilot':
        return 'numerics/vp_core.py'
    if name == 'structured_solver':
        return 'solvers/structured_reference.py'
    candidates = [v for k,v in MAP.items() if Path(k).stem == name]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError((name,candidates))
    return None


def transform(text, key):
    text=text.replace('\r\n','\n')
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1]+len(line))
    edits = []

    def edit(node, replacement):
        # These rewritten nodes are ASCII source expressions/imports; the
        # replacement namespace may contain Chinese characters.
        edits.append((offsets[node.lineno-1]+node.col_offset,
                      offsets[node.end_lineno-1]+node.end_col_offset,replacement))

    for n in ast.walk(tree):
        if isinstance(n,ast.ImportFrom) and n.module:
            dest = local_import(n.module)
            if dest:
                edit(n, 'from '+qualified(dest)+' import '+', '.join(a.name+(' as '+a.asname if a.asname else '') for a in n.names))
        elif isinstance(n,ast.Import):
            replacements=[]
            for a in n.names:
                dest=local_import(a.name)
                replacements.append('import '+qualified(dest)+' as '+(a.asname or a.name) if dest else 'import '+a.name+(' as '+a.asname if a.asname else ''))
            if any(local_import(a.name) for a in n.names):
                edit(n, '; '.join(replacements))
        elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='sys.path.insert':
            edit(n,'pass  # Explicit package imports replace historical search-path mutation.')
        elif isinstance(n,ast.Call):
            fn=ast.unparse(n.func)
            if fn not in ('importlib.util.spec_from_file_location','load','ad.load') or len(n.args)<2:
                continue
            arg=n.args[1]
            expr=ast.unparse(arg)
            dest=SPECIAL.get((key.removeprefix('localization_identifiability/'),expr))
            if not dest:
                strings=[v.value for v in ast.walk(arg) if isinstance(v,ast.Constant) and isinstance(v.value,str) and v.value.endswith('.py')]
                candidates={value for old,value in MAP.items() for s in strings if old.endswith(s)}
                if len(candidates)==1:
                    dest=candidates.pop()
            if dest:
                edit(arg, 'code_path('+repr(dest)+')')
    for start,end,replacement in sorted(edits,reverse=True):
        text=text[:start]+replacement+text[end:]
    if 'code_path(' in text:
        # No future imports are present in this checked runtime closure.
        text='from '+PACKAGE+' import code_path\n'+text
    ast.parse(text)
    return text


def main():
    manifest=json.loads((HOME/'映射.json').read_text(encoding='utf-8'))
    runtime=CODE/'runtime'
    saved=HOME/'运行源码整理前'
    for key,dest in MAP.items():
        src=runtime/key
        raw=src.read_bytes()
        backup=saved/key
        backup.parent.mkdir(parents=True,exist_ok=True)
        backup.write_bytes(raw)
        out=CODE/dest
        if out.exists() and out.relative_to(ROOT).as_posix() not in manifest.get('code_files',{}).values():
            raise RuntimeError(out)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(transform(raw.decode('utf-8'),key),encoding='utf-8')
        manifest.setdefault('code_files',{})[src.relative_to(ROOT).as_posix()]=out.relative_to(ROOT).as_posix()
    extra=ROOT/'研究方向/多径DPD/实验/07_提速与四站验证/speed_gate_20261104/structured_solver.py'
    (CODE/'solvers/structured_reference.py').write_text(transform(extra.read_text(encoding='utf-8'),'structured_solver.py'),encoding='utf-8')
    (HOME/'映射.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FUNCTIONAL MODULES',len(MAP)+1,'originals retained until acceptance')


if __name__=='__main__':main()
