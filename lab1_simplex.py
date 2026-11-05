#!/usr/bin/env python3
"""
《最优化方法》实验一：线性规划 —— 单纯形法
============================================
学生需要完成 TODO 1（必做）、TODO 2（加分），
并填写个人信息后运行，自动生成 result_学号.npz、plot_学号.png 和 lab1_report_学号.pdf。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']        # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False          # 解决负号显示成方块

from scipy.optimize import linprog
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

# ★★★ 填写个人信息 ★★★
STUDENT_ID   = 2024000001   # ← 改成你的学号
STUDENT_NAME = "Your Name"  # ← 改成你的姓名
STUDENT_DATE = "2026-07-21" # ← 改成当天日期


# ============================================================
#  TODO 1: 实现单纯形法核心循环
# ============================================================

def simplex_solve(c, A, b, max_iter=100):
    """
    求解 LP:  min c^T x   s.t. Ax ≤ b, x ≥ 0  (通过加松弛变量化为标准形)

    参数:
        c: 目标系数向量 (1D array)
        A: 约束矩阵 (2D array)
        b: 右端项 (1D array), 要求 b >= 0
        max_iter: 最大迭代次数

    返回:
        x_opt:   最优解
        f_opt:   最优值
        status:  'optimal' / 'unbounded' / 'infeasible'
        history: 每次迭代的 [(basis, tableau), ...]
    """
    m, n = A.shape

    # ---- Step 1: 构造初始单纯形表 ----
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:n+m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[-1, :n] = c
    basis = list(range(n, n + m))

    history = []

    for iteration in range(max_iter):
        history.append((basis.copy(), tableau.copy()))

        # ---- Step 2: 最优性检验 ----
    
        # ★★★ 填写你的代码 ★★★

        

        # ---- Step 3: 确定进基变量（最负检验数）----
        # ★★★ 填写你的代码 ★★★ 
        
        

        # ---- Step 4: 比值检验，确定出基变量 ----
        # ★★★ 填写你的代码 ★★★

        

        # ---- Step 5: 旋转（Gauss-Jordan 消元）----
        # ★★★ 填写你的代码 ★★★

        

    # ---- 提取最优解 ----
    x_opt = np.zeros(n)
    for bi, bv in enumerate(basis):
        if bv < n:
            x_opt[bv] = tableau[bi, -1]
    x_opt = np.maximum(x_opt, 0.0)
    f_opt = np.dot(c, x_opt)

    final_rc = tableau[-1, :-1]
    status = 'optimal' if np.all(final_rc >= -1e-8) else 'infeasible'

    return x_opt, f_opt, status, history


# ============================================================
#  TODO 2 (加分): 两阶段法求初始基本可行解
# ============================================================

def two_phase_simplex(c, A, b):
    """
    Phase I:  引入人工变量求初始基本可行解
    Phase II: 用 Phase I 的基求解原问题

    返回: x_opt, f_opt, status
    """
    # ========== 在下方编写你的代码（加分项）==========

    m, n = A.shape

    # Phase I: min sum(artificial vars)
    c_phase1 = np.concatenate([np.zeros(n), np.ones(m)])
    A_phase1 = np.hstack([A, np.eye(m)])

    x_phase1, f_phase1, status1, _ = simplex_solve(c_phase1, A_phase1, b)

    if f_phase1 > 1e-8:
        return np.zeros(n), np.inf, 'infeasible'

    # Phase II
    x_opt, f_opt, status2, _ = simplex_solve(c, A, b)
    return x_opt, f_opt, status2


# ============================================================
#  测试与绘图代码（不可修改）
# ============================================================

def generate_personalized_lp(student_id):
    """用学号作为种子生成专属 LP 问题"""
    rng = np.random.RandomState(student_id % 100000)
    m = rng.randint(2, 5)
    n = rng.randint(3, 7)
    A = np.round(rng.uniform(-3, 8, (m, n)), 2)
    c = np.round(rng.uniform(-5, 5, n), 2)
    x0 = rng.uniform(0.5, 4, n)
    b = np.round(np.maximum(A @ x0, 0.5), 4)
    return c, A, b


def verify_with_scipy(c, A, b):
    """
    用 scipy 验证结果。
    注意：学生 simplex_solve 加了松弛变量，实际解的是 Ax <= b（不是 Ax = b），
    所以这里必须用 A_ub 而不是 A_eq。
    """
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')
    if res.success:
        return res.x, res.fun, 'optimal'
    # scipy status: 0=optimal, 2=infeasible, 3=unbounded
    return None, None, ('unbounded' if res.status == 3 else 'infeasible')


def plot_results(history, f_values, student_id, student_name):
    """绘制收敛过程"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Lab 1: Simplex Method  |  {student_name} ({student_id})",
                 fontsize=13, fontweight='bold')

    # 目标函数值
    ax1.plot(range(1, len(f_values)+1), f_values, 'o-', color='#2563eb',
             markersize=6, linewidth=2)
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Objective Value")
    ax1.set_title("Objective Value Convergence"); ax1.grid(True, alpha=0.3)
    if len(f_values) >= 2:
        ax1.axhline(y=f_values[-1], color='red', linestyle='--', alpha=0.5,
                    label=f'Optimal = {f_values[-1]:.4f}')
        ax1.legend()

    # 检验数
    n_var = history[0][1].shape[1] - 1
    costs = np.array([tab[-1, :-1] for _, tab in history])
    for j in range(min(n_var, 8)):
        ax2.plot(range(1, len(costs)+1), costs[:, j], 'o-', markersize=3,
                 linewidth=1.2, label=f'σ{j+1}')
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Reduced Cost")
    ax2.set_title("Reduced Costs over Iterations")
    ax2.legend(fontsize=7, ncol=2); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def fmt_vec(v, precision=4):
    """向量格式化为紧凑字符串，用于报告展示"""
    return np.array2string(np.asarray(v), precision=precision, separator=', ')


def generate_report_pdf(student_id, student_name, student_date,
                        c, A, b, x_opt, f_opt, status, n_iter,
                        sp_x, sp_f, sp_status, plot_fig):
    """
    自动生成实验报告 PDF：
      第 1 页: 学生信息 + 个性化 LP 问题
      第 2 页: 单纯形法结果 + scipy 对照
      第 3 页: 收敛图
    """
    pdf_path = f"lab1_report_{student_id}.pdf"
    m, n = A.shape

    with PdfPages(pdf_path) as pdf:
        # ============ 第 1 页：学生信息 + LP 问题 ============
        fig1 = plt.figure(figsize=(8.27, 11.69))  # A4 竖版
        fig1.suptitle("《最优化方法》实验一：单纯形法实验报告",
                      fontsize=16, fontweight='bold', y=0.98)

        info = (f"学生姓名：{student_name}\n"
                f"学    号：{student_id}\n"
                f"实验日期：{student_date}\n")
        fig1.text(0.08, 0.90, info, fontsize=12, va='top', linespacing=1.7)

        prob = (f"一、个性化线性规划问题\n\n"
                f"由学号 {student_id} 作为随机种子生成：m = {m} 个约束，n = {n} 个变量\n\n"
                f"目标函数：min  cᵀx\n"
                f"  c = {fmt_vec(c, 2)}\n\n"
                f"约束条件：Ax ≤ b,  x ≥ 0\n"
                f"  A =\n{np.array2string(A, precision=2, separator=', ')}\n\n"
                f"右端项：\n"
                f"  b = {fmt_vec(b, 4)}\n")
        fig1.text(0.08, 0.78, prob, fontsize=10.5, va='top', linespacing=1.5)

        pdf.savefig(fig1)
        plt.close(fig1)

        # ============ 第 2 页：求解结果 + scipy 对照 ============
        fig2 = plt.figure(figsize=(8.27, 11.69))
        fig2.suptitle("求解结果与验证", fontsize=16, fontweight='bold', y=0.98)

        res_txt = (f"二、单纯形法求解结果\n\n"
                   f"求解状态：{status}\n"
                   f"最优解 x* = {fmt_vec(np.round(x_opt, 6), 6)}\n"
                   f"最优值 f* = {f_opt:.6f}\n"
                   f"迭代次数：{n_iter}\n")
        fig2.text(0.08, 0.85, res_txt, fontsize=12, va='top', linespacing=1.7)

        if sp_status == 'optimal':
            err = abs(f_opt - sp_f)
            scipy_txt = (f"三、scipy.optimize.linprog 对照验证\n\n"
                         f"scipy 最优解 x* = {fmt_vec(np.round(sp_x, 6), 6)}\n"
                         f"scipy 最优值 f* = {sp_f:.6f}\n\n"
                         f"误差 |f_ours − f_scipy| = {err:.2e}\n"
                         f"结论：{'一致 ✅' if err < 1e-4 else '存在差异 ⚠️'}\n")
        elif sp_status == 'unbounded':
            scipy_txt = "三、scipy.optimize.linprog 对照验证\n\nscipy 判定：问题无界\n"
        else:
            scipy_txt = "三、scipy.optimize.linprog 对照验证\n\nscipy 判定：问题不可行\n"
        fig2.text(0.08, 0.70, scipy_txt, fontsize=12, va='top', linespacing=1.6)

        pdf.savefig(fig2)
        plt.close(fig2)

        # ============ 第 3 页：收敛图 ============
        pdf.savefig(plot_fig)

    return pdf_path


def main():
    print(f"={'='*60}")
    print(f"  《最优化方法》实验一：单纯形法")
    print(f"  学生: {STUDENT_NAME}  ({STUDENT_ID})")
    print(f"  日期: {STUDENT_DATE}")
    print(f"={'='*60}\n")

    if STUDENT_ID == 2024000001:
        print("⚠️  警告: 请将 STUDENT_ID 改为你的真实学号！\n")

    c, A, b = generate_personalized_lp(STUDENT_ID)
    m, n = A.shape
    print(f"个性化 LP 问题: m={m} 个约束, n={n} 个变量")
    print(f"  c = {c}")
    print(f"  A =\n{A}")
    print(f"  b = {b}\n")

    print("运行单纯形法...")
    x_opt, f_opt, status, history = simplex_solve(c, A, b)

    print(f"  状态:   {status}")
    print(f"  最优解: x* = {np.round(x_opt, 6)}")
    print(f"  最优值: f* = {f_opt:.6f}")
    print(f"  迭代次数: {len(history)}")

    f_values = []
    for basis, tab in history:
        x_tmp = np.zeros(n)
        for bi, bv in enumerate(basis):
            if bv < n:
                x_tmp[bv] = max(tab[bi, -1], 0)
        f_values.append(np.dot(c, x_tmp))

    print("\n对照 scipy.optimize.linprog...")
    sp_x, sp_f, sp_status = verify_with_scipy(c, A, b)
    if sp_status == 'optimal':
        print(f"  scipy 最优解: x* = {np.round(sp_x, 6)}")
        print(f"  scipy 最优值: f* = {sp_f:.6f}")
        err = np.abs(f_opt - sp_f)
        print(f"  {'✅ 结果一致！' if err < 1e-4 else '⚠️ 存在差异'}, 误差 = {err:.2e}")
    elif sp_status == 'unbounded':
        print("  scipy 判定: 问题无界")
    else:
        print("  scipy 判定: 问题不可行")

    np.savez(f"result_{STUDENT_ID}.npz",
             student_id=STUDENT_ID, student_name=STUDENT_NAME,
             c=c, A=A, b=b, x_opt=x_opt, f_opt=f_opt,
             status=status, n_iter=len(history), f_values=np.array(f_values))
    print(f"\n✅ 结果已保存: result_{STUDENT_ID}.npz")

    fig = plot_results(history, f_values, STUDENT_ID, STUDENT_NAME)
    fig.savefig(f"plot_{STUDENT_ID}.png", dpi=150, bbox_inches='tight')
    print(f"✅ 图像已保存: plot_{STUDENT_ID}.png")

    generate_report_pdf(STUDENT_ID, STUDENT_NAME, STUDENT_DATE,
                        c, A, b, x_opt, f_opt, status, len(history),
                        sp_x, sp_f, sp_status, fig)
    plt.close(fig)
    print(f"✅ 报告已保存: lab1_report_{STUDENT_ID}.pdf")

    print(f"\n{'='*60}")
    print("  提交清单:")
    print(f"    1. result_{STUDENT_ID}.npz")
    print(f"    2. plot_{STUDENT_ID}.png")
    print(f"    3. lab1_report_{STUDENT_ID}.pdf")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
