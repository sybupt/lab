#!/usr/bin/env python3
"""
《最优化方法》实验四：多目标问题最优化 —— 加权和法
====================================================
学生需在 ★★★ 标记处填写代码，完成 TODO 1（加权和法）。
TODO 2（ε-约束法）为加分项，已提供完整实现供参考。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']        # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False          # 解决负号显示成方块

from scipy.optimize import minimize, Bounds
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")


# ★★★ 填写个人信息 ★★★
STUDENT_ID   = 20040102367
STUDENT_NAME = "姓名"
STUDENT_DATE = "2026-09-21"


# ============================================================
#  TODO 1: 加权和法 (Weighted Sum Method) —— 必做
# ============================================================

def weighted_sum_method(objectives, g_ineq, bounds, x0,
                         n_weights=20, method='SLSQP'):
    """
    加权和法求解多目标优化问题:
        min [f1(x), f2(x), ..., fk(x)]
        s.t. g_i(x) ≤ 0,  x ∈ [lb, ub]

    方法: 对每组权重 w = (w1, ..., wk) 求解:
        min  Σ w_j * f_j(x)
        s.t. g_i(x) ≤ 0

    参数:
        objectives: 目标函数列表 [f1, f2, ..., fk]
        g_ineq:     不等式约束列表
        bounds:     变量边界 [(lb1, ub1), ...]
        x0:         初始点
        n_weights:  权重组数
        method:     scipy 优化方法

    返回:
        pareto_solutions: Pareto 解列表
        pareto_values:    Pareto 前沿上的目标值
    """
    k = len(objectives)

    # ---- Step 1: 生成权重向量（已提供）----
    if k == 2:
        # 两目标: 均匀采样 w1 ∈ [0, 1], w2 = 1 - w1
        weights = []
        for i in range(n_weights + 1):
            w1 = i / n_weights
            w2 = 1.0 - w1
            weights.append([w1, w2])
    else:
        # 多目标: Dirichlet 随机采样
        rng = np.random.RandomState(42)
        weights = []
        for _ in range(n_weights):
            w = rng.uniform(0.1, 1.0, k)
            w = w / w.sum()
            weights.append(w.tolist())

    pareto_solutions = []
    pareto_values = []

    # ---- Step 2: 对每组权重求解子问题 ----
    for w in weights:
        # TODO 1a: 构造标量化目标函数
 
    # ★★★ 在此编写代码 ★★★       
        
       

        # TODO 1b: 构造约束并求解
    
    # ★★★ 在此编写代码 ★★★
        
        


        # TODO 1c: 提取结果并去重
   
    # ★★★ 在此编写代码 ★★★

        

    # ---- Step 3: 按 f1 排序（已提供）----
    if len(pareto_values) > 0:
        pareto_values = np.array(pareto_values)
        pareto_solutions = np.array(pareto_solutions)
        idx = np.argsort(pareto_values[:, 0])
        pareto_values = pareto_values[idx]
        pareto_solutions = pareto_solutions[idx]

    return pareto_solutions, pareto_values


# ============================================================
#  TODO 2 (加分): ε-约束法 —— 已提供完整实现供参考
# ============================================================

def epsilon_constraint_method(objectives, g_ineq, bounds, x0, n_points=20):
    """
    ε-约束法: 将 f1 作为主目标，其余目标转为约束 f_j(x) ≤ ε_j
    通过扫描 ε 值得到完整 Pareto 前沿（可处理非凸前沿）
    """
    k = len(objectives)

    # Payoff table — 估计各目标的取值范围
    ideal = np.zeros(k)
    for j in range(k):
        def single_obj(x):
            return objectives[j](x)
        constraints = [{'type': 'ineq', 'fun': lambda x, g=g: -g(x)}
                       for g in g_ineq]
        bounds_list = Bounds([b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(single_obj, x0, method='SLSQP',
                       bounds=bounds_list, constraints=constraints)
        if res.success:
            ideal[j] = objectives[j](res.x)
        else:
            ideal[j] = objectives[j](x0)

    nadir = ideal + np.maximum(np.abs(ideal) * 1.5, 0.5)

    # 扫描 ε
    pareto_solutions = []
    pareto_values = []

    for i in range(n_points + 1):
        constraints = [{'type': 'ineq', 'fun': lambda x, g=g: -g(x)}
                       for g in g_ineq]

        for j in range(1, k):
            eps = ideal[j] + (i / n_points) * (nadir[j] - ideal[j])
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, j=j, eps=eps: eps - objectives[j](x)
            })

        bounds_list = Bounds([b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(objectives[0], x0, method='SLSQP',
                       bounds=bounds_list, constraints=constraints,
                       options={'maxiter': 500})

        if res.success:
            f_vals = np.array([obj(res.x) for obj in objectives])
            is_new = True
            for pv in pareto_values:
                if np.linalg.norm(f_vals - pv) < 1e-4:
                    is_new = False
                    break
            if is_new:
                pareto_solutions.append(res.x.copy())
                pareto_values.append(f_vals)

    if len(pareto_values) > 0:
        pareto_values = np.array(pareto_values)
        pareto_solutions = np.array(pareto_solutions)
        idx = np.argsort(pareto_values[:, 0])
        pareto_values = pareto_values[idx]
        pareto_solutions = pareto_solutions[idx]

    return pareto_solutions, pareto_values


# ============================================================
#  测试代码（不可修改）
# ============================================================

def generate_personalized_problem(student_id):
    """生成个性化多目标优化问题"""
    rng = np.random.RandomState(student_id % 100000)
    problem_type = rng.choice(['zdt1_style', 'production', 'portfolio'])

    if problem_type == 'zdt1_style':
        n = rng.randint(3, 5)

        def f1(x): return x[0]

        def f2(x):
            g = 1 + 9 * np.sum(x[1:]) / (n - 1)
            h = 1 - np.sqrt(x[0] / g)
            return g * h

        objectives = [f1, f2]
        g_ineq = []
        bounds = [(0, 1) for _ in range(n)]
        x0 = np.array([0.5] + [0.3] * (n - 1))

    elif problem_type == 'portfolio':
        n_assets = 3
        mu = np.array([0.1, 0.2, 0.15])
        Sigma = np.array([[0.04, 0.01, 0.005],
                          [0.01, 0.09, 0.02],
                          [0.005, 0.02, 0.06]])

        def f1(x): return x @ Sigma @ x
        def f2(x): return -np.dot(mu, x)

        objectives = [f1, f2]
        g_ineq = [
            lambda x: 1.0 - np.sum(x),     # sum(x) = 1   → 1 - sum ≤ 0 即 sum ≥ 1
            lambda x: np.sum(x) - 1.0,      # sum(x) ≤ 1
        ]
        bounds = [(0, 1) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets

    elif problem_type == 'production':
        def f1(x): return 2*x[0] + 3*x[1] + 1.5*x[2]
        def f2(x): return 0.5*x[0]**2 + 0.8*x[1]**2 + 0.3*x[2]**2
        def demand_con(x): return 10 - (x[0] + x[1] + x[2])

        objectives = [f1, f2]
        g_ineq = [demand_con]
        bounds = [(0, 10), (0, 10), (0, 10)]
        x0 = np.array([3.0, 3.0, 4.0])

    return objectives, g_ineq, bounds, x0, problem_type


def plot_pareto_front(pareto_ws, pareto_ec, student_id, student_name, ptype):
    """绘制 Pareto 前沿"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(f"Lab 4: Pareto Front  |  {student_name} ({student_id})",
                 fontsize=13, fontweight='bold')

    if len(pareto_ws) > 0:
        ax.plot(pareto_ws[:, 0], pareto_ws[:, 1], 'o-', color='#2563eb',
                markersize=7, linewidth=2, label='Weighted Sum')
    if len(pareto_ec) > 0:
        ax.plot(pareto_ec[:, 0], pareto_ec[:, 1], 's--', color='#dc2626',
                markersize=7, linewidth=2, label='ε-Constraint')

    ax.set_xlabel("f₁(x)"); ax.set_ylabel("f₂(x)")
    ax.set_title(f"Pareto Front — {ptype}")
    ax.legend(); ax.grid(True, alpha=0.3)

    if len(pareto_ws) > 0:
        utopia = np.array([pareto_ws[:, 0].min(), pareto_ws[:, 1].min()])
        ax.plot(utopia[0], utopia[1], '*', color='#16a34a', markersize=15,
                label='Utopia Point')
        ax.legend()

    plt.tight_layout()
    return fig


def fmt_vec(v, precision=4):
    """向量格式化为紧凑字符串，用于报告展示"""
    return np.array2string(np.asarray(v), precision=precision, separator=', ')


def generate_report_pdf(student_id, student_name, student_date, ptype,
                        k, x0, init_vals, f_ws, f_ec, plot_fig):
    """
    自动生成实验报告 PDF：
      第 1 页: 学生信息 + 个性化多目标问题
      第 2 页: 加权和法 / ε-约束法结果 + 讨论要点
      第 3 页: Pareto 前沿图
    """
    pdf_path = f"lab4_report_{student_id}.pdf"
    n_ws = len(f_ws)
    n_ec = len(f_ec)

    with PdfPages(pdf_path) as pdf:
        # ============ 第 1 页：学生信息 + 问题描述 ============
        fig1 = plt.figure(figsize=(8.27, 11.69))  # A4 竖版
        fig1.suptitle("《最优化方法》实验四：加权和法实验报告",
                      fontsize=16, fontweight='bold', y=0.98)

        info = (f"学生姓名：{student_name}\n"
                f"学    号：{student_id}\n"
                f"实验日期：{student_date}\n")
        fig1.text(0.08, 0.90, info, fontsize=12, va='top', linespacing=1.7)

        prob = (f"一、个性化多目标优化问题\n\n"
                f"由学号 {student_id} 作为随机种子生成\n"
                f"问题类型：{ptype}\n"
                f"目标个数：{k}\n\n"
                f"初始点 x0 = {fmt_vec(x0, 4)}\n"
                f"初始目标值 = {fmt_vec(np.array(init_vals), 6)}\n")
        fig1.text(0.08, 0.80, prob, fontsize=11, va='top', linespacing=1.7)

        pdf.savefig(fig1)
        plt.close(fig1)

        # ============ 第 2 页：两种方法结果 + 讨论 ============
        fig2 = plt.figure(figsize=(8.27, 11.69))
        fig2.suptitle("求解结果与分析", fontsize=16, fontweight='bold', y=0.98)

        if n_ws > 0:
            ws_txt = (f"二、加权和法 (Weighted Sum)\n\n"
                      f"   Pareto 解个数：{n_ws}\n"
                      f"   f1 范围：[{f_ws[:, 0].min():.4f}, {f_ws[:, 0].max():.4f}]\n"
                      f"   f2 范围：[{f_ws[:, 1].min():.4f}, {f_ws[:, 1].max():.4f}]\n"
                      f"   理想点：f1={f_ws[:, 0].min():.4f}, f2={f_ws[:, 1].min():.4f}\n")
        else:
            ws_txt = "二、加权和法 (Weighted Sum)\n\n   未找到 Pareto 解\n"
        fig2.text(0.08, 0.85, ws_txt, fontsize=11, va='top', linespacing=1.6)

        if n_ec > 0:
            ec_txt = (f"三、ε-约束法 (ε-Constraint)\n\n"
                      f"   Pareto 解个数：{n_ec}\n"
                      f"   f1 范围：[{f_ec[:, 0].min():.4f}, {f_ec[:, 0].max():.4f}]\n"
                      f"   f2 范围：[{f_ec[:, 1].min():.4f}, {f_ec[:, 1].max():.4f}]\n")
        else:
            ec_txt = "三、ε-约束法 (ε-Constraint)\n\n   未找到 Pareto 解\n"
        fig2.text(0.08, 0.70, ec_txt, fontsize=11, va='top', linespacing=1.6)

        disc = (f"四、讨论要点\n\n"
                f"   ① 加权和法能否找到非凸 Pareto 前沿上的解？为什么？\n"
                f"   ② ε-约束法与加权和法各自的优缺点？\n"
                f"   ③ 如何理解\"理想点\"和\"Pareto 最优\"的概念？\n")
        fig2.text(0.08, 0.52, disc, fontsize=11, va='top', linespacing=1.7)

        pdf.savefig(fig2)
        plt.close(fig2)

        # ============ 第 3 页：Pareto 前沿图 ============
        pdf.savefig(plot_fig)

    return pdf_path


def main():
    print(f"={'='*60}")
    print(f"  《最优化方法》实验四：加权和法 (多目标优化)")
    print(f"  学生: {STUDENT_NAME}  ({STUDENT_ID})")
    print(f"  日期: {STUDENT_DATE}")
    print(f"={'='*60}\n")

    if STUDENT_ID == 2024000001:
        print("⚠️  警告: 请将 STUDENT_ID 改为你的真实学号！\n")

    objectives, g_ineq, bounds, x0, ptype = \
        generate_personalized_problem(STUDENT_ID)

    k = len(objectives)
    print(f"测试问题类型: {ptype}")
    print(f"目标个数: {k}")
    print(f"初始点 x0 = {np.round(x0, 4)}")
    print(f"初始目标值: {[np.round(obj(x0), 6) for obj in objectives]}\n")

    print(">>> 加权和法 (Weighted Sum)")
    x_ws, f_ws = weighted_sum_method(objectives, g_ineq, bounds, x0,
                                      n_weights=30)
    print(f"  Pareto 解个数: {len(x_ws)}")
    if len(f_ws) > 0:
        print(f"  f1 范围: [{f_ws[:, 0].min():.4f}, {f_ws[:, 0].max():.4f}]")
        print(f"  f2 范围: [{f_ws[:, 1].min():.4f}, {f_ws[:, 1].max():.4f}]")
        print(f"  理想点: f1={f_ws[:,0].min():.4f}, f2={f_ws[:,1].min():.4f}\n")

    print(">>> ε-约束法 (ε-Constraint)")
    x_ec, f_ec = epsilon_constraint_method(objectives, g_ineq, bounds, x0,
                                            n_points=20)
    print(f"  Pareto 解个数: {len(x_ec)}")
    if len(f_ec) > 0:
        print(f"  f1 范围: [{f_ec[:, 0].min():.4f}, {f_ec[:, 0].max():.4f}]")
        print(f"  f2 范围: [{f_ec[:, 1].min():.4f}, {f_ec[:, 1].max():.4f}]\n")

    np.savez(f"result_{STUDENT_ID}.npz",
             student_id=STUDENT_ID, student_name=STUDENT_NAME,
             problem_type=ptype,
             pareto_ws=f_ws, n_ws=len(f_ws),
             pareto_ec=f_ec, n_ec=len(f_ec))
    print(f"✅ 结果已保存: result_{STUDENT_ID}.npz")

    fig = plot_pareto_front(f_ws, f_ec, STUDENT_ID, STUDENT_NAME, ptype)
    fig.savefig(f"plot_{STUDENT_ID}.png", dpi=150, bbox_inches='tight')
    print(f"✅ 图像已保存: plot_{STUDENT_ID}.png")

    init_vals = [np.round(obj(x0), 6) for obj in objectives]
    generate_report_pdf(STUDENT_ID, STUDENT_NAME, STUDENT_DATE, ptype,
                        k, x0, init_vals, f_ws, f_ec, fig)
    plt.close(fig)
    print(f"✅ 报告已保存: lab4_report_{STUDENT_ID}.pdf")

    print(f"\n{'='*60}")
    print("  提交清单:")
    print(f"    1. result_{STUDENT_ID}.npz")
    print(f"    2. plot_{STUDENT_ID}.png")
    print(f"    3. lab4_report_{STUDENT_ID}.pdf")

    print(f"\n  实验报告建议讨论:")
    print(f"    ① 加权和法能否找到非凸 Pareto 前沿上的解？为什么？")
    print(f"    ② ε-约束法与加权和法各自的优缺点？")
    print(f"    ③ 如何理解\"理想点\"和\"Pareto 最优\"的概念？")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
