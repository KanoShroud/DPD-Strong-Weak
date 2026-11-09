#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第21轮：冻结IQ分支候选覆盖和固定位置条件收敛的隔离诊断。

仅诊断，不是新定位算法。公共MP、信号生成器和第20轮数据不改。
真值只在全部新盲决策文件落盘以后才读取，用于独立ORACLE控制。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

SCHEMA = "R21_FIXED_POSITION_ADMISSION_V1"
# 固定开发集。高K仅作信道模型阶数充分性诊断，不属于盲算法已知的真实阶数。
SPECS = (
    (0, "IN_100_R00_L0", (0, 4)),
    (8, "IN_102_R00_L2_P075", (4, 16)),
    (16, "IN_105_R00_L1_P075", (4, 8)),
)
PG_GATE = 1e-7
BUDGETS = (70, 300)
MAX_POINT_COUNT = 6
MAX_PAIR_COUNT = 6
R20_RUN = "pilot3_20261109_150611_409393"


def project_root(here: Path) -> Path:
    for root in here.resolve().parents:
        if (root / "AGENTS.md").is_file() and (root / "研究方向" / "多径DPD").is_dir():
            return root
    raise FileNotFoundError("请从已更新的 DPD-Strong-Weak 仓库中启动第21轮脚本。")


HERE = Path(__file__).resolve().parent
ROOT = project_root(HERE)
BASE = ROOT / "研究方向" / "多径DPD"
R20 = BASE / "实验/20_分支可区分性核验/20261109"
R14_SCRIPT = BASE / "实验/14_公开信道记忆核验/20261107/成果/run.py"
COMMON = BASE / "实验/09_文献后机制核验/20261107/运行数据/common18"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def array_digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def encode(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    raise TypeError(type(v))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False, default=encode), encoding="utf-8")
    tmp.replace(path)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pair_error(points, truth):
    """返回集合中最接近真值的有序双位置对及交换不变双源RMSE。仅评价端调用。"""
    pts = np.asarray(points, float)
    gt = np.asarray(truth, float)
    assert pts.ndim == 2 and pts.shape[1] == 2 and len(pts) >= 2
    assert gt.shape == (2, 2)
    d = np.sum((pts[:, None, :] - gt[None, :, :]) ** 2, axis=2)
    cost = np.minimum(d[:, 0, None] + d[None, :, 1],
                      d[:, 1, None] + d[None, :, 0])
    ij = np.unravel_index(np.argmin(cost), cost.shape)
    return dict(indices=[int(ij[0]), int(ij[1])],
                rmse_m=float(np.sqrt(cost[ij] / 2)))


def rmse(positions, truth):
    p = np.asarray(positions, float)
    t = np.asarray(truth, float)
    a = np.sum((p - t) ** 2)
    b = np.sum((p[::-1] - t) ** 2)
    return float(np.sqrt(min(a, b) / 2))


def selftest():
    # 双源交换不变；有序候选可取自不同池序；没有调用原始IQ或真值银行。
    t = np.array([[10., 20.], [-4., 5.]])
    pool = np.array([[100., 0.], [-4., 5.], [10., 20.], [50., 60.]])
    z = pair_error(pool, t)
    assert z["indices"] == [1, 2] or z["indices"] == [2, 1]
    assert z["rmse_m"] == 0.
    assert rmse(t[::-1], t) == 0.
    assert rmse(np.array([[13., 24.], [-4., 5.]]), t) == np.sqrt(12.5)
    return {"selftest": "PASS", "pair_indices": z["indices"],
            "pg_gate": PG_GATE, "paired_iteration_budgets": BUDGETS}


def load_sources():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    r14 = import_path("r21_frozen_r14", R14_SCRIPT)
    r20 = import_path("r21_frozen_r20", R20 / "成果/branch_probe.py")
    return r14, r20


def choose_blind_pairs(engine, r20, pool, f, y, n):
    points, proposals = r20.select_blind_pairs(
        engine, pool, f, y, n, MAX_POINT_COUNT, MAX_PAIR_COUNT)
    if len(proposals) < 2:
        raise ValueError("无法生成两组预登记盲候选对")
    # 不依据位置真值或拟合结果事后挑选。只选训练LOS排序两端作为对比。
    selected = [proposals[0], proposals[-1]]
    return points, selected


def paired_fit(engine, r20, init_x, paths, f, y, n):
    """相同支持、相同初始参数，分别运行70/300次条件SLSQP；不调收敛阈值。"""
    x70, a = r20.fixed_nuisance_fit(engine, init_x.copy(), list(paths),
                                    f, y, n, maxiter=BUDGETS[0])
    x300, b = r20.fixed_nuisance_fit(engine, init_x.copy(), list(paths),
                                     f, y, n, maxiter=BUDGETS[1])
    # 仅以真实原IQ目标选择已有两次受限求解结果，且两次都从同一点出发。
    choose300 = b["cost"] < a["cost"] - 1e-12
    xx = x300 if choose300 else x70
    chosen = b if choose300 else a
    if not np.array_equal(xx[:4], init_x[:4]):
        raise ArithmeticError("固定位置在条件求解中被改变")
    if np.linalg.norm(xx - engine.ref.project(xx, n)) >= 1e-7:
        raise ArithmeticError("拟合终态超出公共物理约束")
    return xx, dict(
        seventy=a, three_hundred=b,
        selected_budget=300 if choose300 else 70,
        selected_cost=float(chosen["cost"]),
        selected_conditional_pg=chosen["conditional_projected_gradient"],
        selected_stationary=bool(chosen["nuisance_stationary"]),
        initial_cost=float(engine.objective(init_x, f, y, paths)[0]),
    )


def grow_branch(engine, r20, positions, f, y, n, checkpoints):
    p = np.asarray(positions, float)
    assert p.shape == (2, 2)
    x, los_cost = r20.branch_approx_score(engine, p, f, y, n)
    positions0 = x[:4].copy()
    paths = []
    max_k = max(checkpoints)
    results = {}
    step_log = []
    for k in range(max_k + 1):
        proposal = None
        if k:
            key, coef, proposal = r20.propose_path(engine, x, paths, f, y, n)
            old_cost = float(engine.objective(x, f, y, paths)[0])
            trial_paths = paths + [key]
            trial_x = engine.ref.project(np.r_[x, coef], n)
            if engine.objective(trial_x, f, y, trial_paths)[0] > old_cost + 1e-10:
                trial_x[-3:-1] = 0.
            if engine.objective(trial_x, f, y, trial_paths)[0] > old_cost + 1e-10:
                raise ArithmeticError("零增益嵌套路径未保持原模型目标")
            x, paths = trial_x, trial_paths
        x, outcome = paired_fit(engine, r20, x, paths, f, y, n)
        if not np.array_equal(x[:4], positions0):
            raise ArithmeticError("路径生长期间位置改变")
        step_log.append(dict(k=k, paths=[list(a) for a in paths],
                             proposal=proposal, paired_solver=outcome))
        if k in checkpoints:
            results[str(k)] = dict(
                k=k, positions=p.copy(), path_keys=[list(a) for a in paths],
                x=x.copy(), cost=outcome["selected_cost"],
                pg=outcome["selected_conditional_pg"],
                stationary=outcome["selected_stationary"],
                solver_budget=outcome["selected_budget"],
            )
    return dict(initial_los_score=float(los_cost),
                checkpoints=results, steps=step_log)


def frozen_record(index, expected_tag):
    rows = read(COMMON / "rows.json")
    if not (len(rows) == 18 and rows[index]["tag"] == expected_tag):
        raise RuntimeError("原common18冻结任务身份或顺序发生漂移")
    record = rows[index]
    iq_path = COMMON / "inputs" / expected_tag / "iq.npz"
    if not iq_path.is_file() or digest(iq_path) != record["input_sha"]:
        raise RuntimeError("冻结IQ缺失或字节SHA不符：" + str(iq_path))
    with np.load(iq_path, allow_pickle=False) as source:
        if set(source.files) != {"iq"}:
            raise RuntimeError("冻结定位输入不应携带真值")
        iq = np.asarray(source["iq"])
    if iq.shape != (4, 8192) or not np.isfinite(iq).all():
        raise ValueError("冻结IQ维度或数值非法")
    return record, iq, iq_path


def blind_case(r14, r20, index, tag, checkpoints, output):
    record, iq, input_path = frozen_record(index, tag)
    engine = r14.engine_for("CAP_COUNTED")
    f, y, n, _ = engine.fc.cal.spectral(iq)
    front = engine.frontend(iq)  # 全有效频带原始IQ，去掉第20轮训练频段掩膜混杂
    pool = np.asarray(front["refined"], float)
    points, proposals = choose_blind_pairs(engine, r20, pool, f, y, n)
    entries = []
    for label, prop in zip(("LOS_BEST", "LOS_RANK_CONTRAST"), proposals):
        p = points[[prop["i"], prop["j"]]]
        result = grow_branch(engine, r20, p, f, y, n, checkpoints)
        entries.append(dict(label=label, pool_subset_indices=[prop["i"], prop["j"]],
                            los_initial_cost=prop["los"], position_pair=p.copy(),
                            fit=result))
        print("BLIND", tag, label, "K", max(checkpoints), flush=True)
    blind_path = output / "blind" / (tag + ".json")
    save(blind_path, dict(schema=SCHEMA, input_index=index, tag=tag,
                          identity=dict(iq_sha=digest(input_path),
                                        iq_array_sha=array_digest(iq),
                                        condition=record["condition"]),
                          checkpoints=list(checkpoints), full_pool=pool,
                          selected_six_points=points,
                          positions_tested=entries,
                          statement="NO_TRUTH_READ_IN_THIS_STAGE"))
    return blind_path, record


def evaluate_case(r14, r20, blind_path, record, output):
    """读回已落盘盲分支；此时才核对并打开真值银行。"""
    frozen = read(blind_path)
    bank_path = Path(record["truth_evaluation_file"])
    if digest(bank_path) != record["truth_file_sha"]:
        raise RuntimeError("第09轮登记的真值银行SHA不符")
    with np.load(bank_path, allow_pickle=False) as data:
        truth = np.asarray(data["p"])
    pool = np.asarray(frozen["full_pool"])
    near = pair_error(pool, truth)
    tested = [dict(label=a["label"], rmse_m=rmse(a["position_pair"], truth),
                   checkpoint_metrics={k:dict(cost=v["cost"], pg=v["pg"],
                                              stationary=v["stationary"])
                                       for k, v in a["fit"]["checkpoints"].items()})
              for a in frozen["positions_tested"]]
    # 仅在真值已隔离的评价阶段新建ORACLE分支。使用与盲分支完全相同的IQ与K。
    record2, iq, _ = frozen_record(frozen["input_index"], frozen["tag"])
    assert record2["tag"] == frozen["tag"]
    engine = r14.engine_for("CAP_COUNTED")
    f, y, n, _ = engine.fc.cal.spectral(iq)
    oracle_p = pool[near["indices"]]
    oracle = grow_branch(engine, r20, oracle_p, f, y, n,
                         tuple(frozen["checkpoints"]))
    outfile = output / "evaluation" / (frozen["tag"] + ".json")
    save(outfile, dict(schema=SCHEMA,
                       statement="ORACLE_USES_TRUTH_AFTER_ALL_BLIND_DECISIONS",
                       blind_decision_sha=digest(blind_path),
                       true_positions=truth, blind=tested,
                       pool_nearest=near, six_points_nearest=
                           pair_error(frozen["selected_six_points"], truth),
                       oracle_positions=oracle_p,
                       oracle_fit=oracle))
    print("ORACLE", frozen["tag"], "nearest_pool_rmse",
          round(near["rmse_m"], 4), flush=True)
    return outfile


def audit_round20(output):
    """只读回放第20轮候选集合，精确拆分完整池->6点->6对。仅离线真值归因。"""
    folder = R20 / "运行数据" / R20_RUN / "冻结IQ诊断"
    reports = []
    for _, tag, _ in SPECS:
        bpath = folder / "blind" / (tag + ".json")
        epath = folder / "evaluation" / (tag + ".json")
        if not bpath.is_file() or not epath.is_file():
            raise FileNotFoundError("第20轮原始JSON不存在：" + str(bpath))
        b = read(bpath)
        ev = read(epath)
        if ev["decision_sha"] != digest(bpath):
            raise RuntimeError("第20轮盲决策与评价SHA不符：" + tag)
        truth = np.asarray(ev["true_positions"], float)
        for k, fold in enumerate(b["folds"]):
            pool = pair_error(fold["full_observed_candidate_pool"], truth)
            six = pair_error(fold["tested_points"], truth)
            available = []
            for z in fold["tested_pairs"]:
                available.append(rmse(z["positions"], truth))
            reports.append(dict(tag=tag, fold=k,
                                complete_pool=pool, six_points=six,
                                six_pairs_best_rmse_m=min(available),
                                tested_near_20m_count=sum(v <= 20 for v in available),
                                tested_stationary_count=sum(
                                    bool(z["numerical_stationary"]) for z in fold["tested_pairs"])))
    path = output / "offline20.json"
    save(path, dict(schema="R21_OFFLINE_R20_AUDIT_V1", source_run=R20_RUN,
                    status="READ_ONLY_NO_NEW_IQ_SOLVE", folds=reports))
    return path


def run(output: Path):
    if not COMMON.joinpath("rows.json").is_file():
        raise FileNotFoundError("旧common18/rows.json不存在：" + str(COMMON))
    r14, r20 = load_sources()
    source_files = (list((BASE / "代码/mp_dpd").rglob("*.py")) +
                    [Path(__file__), HERE / "一键运行_第21轮.py",
                     HERE / "test_round21.py",
                     R20 / "成果/branch_probe.py", R14_SCRIPT,
                     BASE / "实验/09_文献后机制核验/20261107/成果/run_gate.py"])
    original_hashes = {str(p): digest(p) for p in source_files}
    save(output / "protocol.json",
         dict(schema=SCHEMA, specs=[dict(index=i, tag=t, k_checkpoints=k)
                                    for i, t, k in SPECS],
              budgets=BUDGETS, pg_gate=PG_GATE, sources=original_hashes,
              original_input_root=str(COMMON), full_band=True,
              training_frequency_split=False, truth_in_blind_stage=False,
              formal_batch=False))
    blind = []
    with threadpool_limits(limits=2):
        # 所有盲决策必须先完整落盘，才能进入任何读取真值的逻辑。
        for index, tag, checkpoints in SPECS:
            f, record = blind_case(r14, r20, index, tag, checkpoints, output)
            blind.append((f, record))
        save(output / "blind_stage_completed.json",
             dict(complete=True, files=[str(p) for p, _ in blind],
                  hashes={str(p): digest(p) for p, _ in blind}))
        evaluations = [evaluate_case(r14, r20, p, record, output)
                       for p, record in blind]
    audit = audit_round20(output)
    drift = [p for p, expected in original_hashes.items() if digest(Path(p)) != expected]
    if drift:
        raise RuntimeError("运行期间源文件发生漂移：" + repr(drift))
    save(output / "completion.json",
         dict(schema=SCHEMA, status="R21_DIAGNOSTIC_FINISHED",
              blind_files=[str(p) for p, _ in blind],
              evaluation_files=[str(p) for p in evaluations],
              audit20=str(audit), source_sha_verified=True,
              conclusion="PENDING_NUMERICAL_REVIEW_NOT_PREAPPROVED_SUCCESS"))
    return dict(status="R21_DIAGNOSTIC_FINISHED", output=str(output),
                cases=len(blind), evaluations=len(evaluations))


if __name__ == "__main__":
    raise SystemExit("请在PyCharm中直接运行同目录的一键运行_第21轮.py。")
