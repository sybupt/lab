#!/usr/bin/env python3
"""
《最优化方法》实验三：约束问题最优化 —— 惩罚函数法
====================================================
学生需在 ★★★ 标记处填写代码，完成 TODO 1（外点惩罚法）。
TODO 2（内点障碍法）为加分项，已提供完整实现供参考。
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
STUDENT_ID   = 2024000001
STUDENT_NAME = "Your Name"
STUDENT_DATE = "2026-07-21"


# ============================================================
#  TODO 1: 外点惩罚函数法 (Exterior Penalty Method) —— 必做
# ============================================================

def exterior_penalty(f, g_ineq, x0, mu0=1.0, rho=10.0,
                     max_outer=20, tol=1e-6):
    """
    外点惩罚函数法求解约束问题:
        min   f(x)
        s.t.  g_i(x) ≤ 0,  i = 1, ..., p

    惩罚函数: P(x, μ) = f(x) + (μ/2) * Σ max(0, g_i(x))²

    参数:
        f:        目标函数
        g_ineq:   不等式约束列表 [g_1(x), ..., g_p(x)]
        x0:       初始点（不必可行）
        mu0:      初始惩罚因子
        rho:      惩罚因子放大倍数 (>1)
        max_outer:外循环最大迭代次数
        tol:      收敛容差

    返回:
        x_opt, f_opt, history (每次外迭代的 [x, f(x), mu, max_violation])
    """
    x = np.array(x0, dtype=float)
    mu = mu0
    history = []

    for iteration in range(max_outer):
        # ============================================================
        # TODO 1a: 构造惩罚函数 P(x, μ)
        #
        #   def penalty_func(x_vec):
        #       val = f(x_vec)
        #       for g in g_ineq:
        #           violation = g(x_vec)     # 约束违反量 (>0 表示违反)
        #           if violation > 0:
        #               val += (mu / 2.0) * violation ** 2
        #       return val
        #
        # 注意: 这里 mu 是外循环变量，用闭包自动捕获当前值
        # ============================================================

        # ★★★ 在此编写代码 ★★★


        # ============================================================
        # TODO 1b: 调用 scipy 求解无约束子问题 min penalty_func(x)
        #
        #   res = minimize(penalty_func, x, method='BFGS', tol=1e-8)
        #   x_new = res.x
        # ============================================================

        # ★★★ 在此编写代码 ★★★


        # ============================================================
        # TODO 1c: 计算最大约束违反量并记录历史
        #
        #   if g_ineq 为空:
        #       max_viol = 0
        #   else:
        #       max_viol = max([max(0, g(x_new)) for g in g_ineq])
        #
        #   history.append((x_new.copy(), f(x_new), mu, max_viol))
        # ============================================================

        # ★★★ 在此编写代码 ★★★


        # ============================================================
        # TODO 1d: 收敛判断 + 更新
        #
        #   if np.linalg.norm(x_new - x) < tol and max_viol < tol:
        #       x = x_new
        #       break
        #
        #   x = x_new
        #   mu *= rho     # 增大惩罚因子
        # ============================================================

        # ★★★ 在此编写代码 ★★★


    return x, f(x), history


# ============================================================
#  TODO 2 (加分): 内点障碍函数法 —— 已提供完整实现供参考
# ============================================================

def interior_barrier(f, g_ineq, x0, t0=1.0, rho=10.0,
                     max_outer=20, tol=1e-6):
    """
    内点障碍函数法（对数障碍）:
        B(x, t) = f(x) - (1/t) * Σ log(-g_i(x))
    要求初始点 x0 必须是严格内点 (g_i(x0) < 0)
    """
    for g in g_ineq:
        if g(x0) >= 0:
            print("  警告: 初始点不可行，尝试寻找可行点...")
            x_pen, _, _ = exterior_penalty(lambda x: 0, g_ineq, x0,
                                           mu0=10, rho=10, max_outer=15)
            x = x_pen.copy()
            feasible = all(g(x) < 0 for g in g_ineq)
            if not feasible:
                return x0, f(x0), []
        else:
            x = np.array(x0, dtype=float)

    t = t0
    history = []

    for iteration in range(max_outer):
        def barrier_func(x_vec):
            val = t * f(x_vec)
            for g in g_ineq:
                gx = g(x_vec)
                if gx >= 0:
                    return 1e15
                val -= np.log(-gx)
            return val

        res = minimize(barrier_func, x, method='BFGS', tol=1e-8)
        x_new = res.x

        max_viol = max([max(0, g(x_new)) for g in g_ineq]) if g_ineq else 0
        history.append((x_new.copy(), f(x_new), t, max_viol))

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break

        x = x_new
        t *= rho

    return x, f(x), history


# ============================================================
#  测试代码（不可修改）
# ============================================================

def generate_personalized_problem(student_id):
    """生成个性化约束优化问题"""
    rng = np.random.RandomState(student_id % 100000) 
    problem_type = rng.choice(['quadratic', 'nonlinear'])

    if problem_type == 'quadratic':
        # min c^T x + 0.5 x^T Q x
        # s.t. A_con x ≤ b_con  (线性约束)
        n = rng.randint(2, 4)
        A_mat = rng.randn(n, n)
        Q = A_mat.T @ A_mat + 0.5 * np.eye(n)
        c_vec = rng.randn(n) * 2
        m_con = rng.randint(2, 4)
        A_con = rng.randn(m_con, n)
        b_con = np.ones(m_con) + rng.uniform(0, 2, m_con)

        def f(x): return c_vec @ x + 0.5 * x @ Q @ x
        x0 = rng.uniform(-1, 1, n)

        g_ineq = []
        for i in range(m_con):
            ai = A_con[i]
            bi = b_con[i]
            g_ineq.append(lambda x, ai=ai, bi=bi: np.dot(ai, x) - bi)

    elif problem_type == 'nonlinear':
        # min (x0-2)² + (x1-1)²
        # s.t. x0² - x1 ≤ 0,  x0 + x1 - 2 ≤ 0
        def f(x): return (x[0] - 2)**2 + (x[1] - 1)**2

        g_ineq = [
            lambda x: x[0]**2 - x[1],
            lambda x: x[0] + x[1] - 2,
        ]
        n = 2
        x0 = np.array([0.0, 1.0])

    return f, g_ineq, x0, problem_type


def solve_with_scipy(f, g_ineq, x0):
    """用 scipy 直接求解约束问题作为标准答案"""
    constraints = []
    for i, g in enumerate(g_ineq):
        constraints.append({'type': 'ineq', 'fun': lambda x, g=g: -g(x)})
    res = minimize(f, x0, constraints=constraints, method='SLSQP',
                   options={'maxiter': 500, 'ftol': 1e-10})
    return res.x, res.fun, res.success


def plot_convergence(history, student_id, student_name):
    """绘制惩罚函数法的收敛过程"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Lab 3: Penalty Function Method  |  {student_name} ({student_id})",
                 fontsize=13, fontweight='bold')

    f_vals = [h[1] for h in history]
    mu_vals = [h[2] for h in history]
    violations = [h[3] for h in history]
    iters = range(1, len(f_vals) + 1)

    ax1.plot(iters, f_vals, 'o-', color='#2563eb', markersize=6, linewidth=2)
    ax1.set_xlabel("Outer Iteration"); ax1.set_ylabel("f(x)")
    ax1.set_title("Objective Value Convergence"); ax1.grid(True, alpha=0.3)
    ax1.axhline(y=f_vals[-1], color='red', linestyle='--', alpha=0.5,
                label=f'Optimal = {f_vals[-1]:.4f}')
    ax1.legend()

    ax2.semilogy(iters, violations, 's-', color='#dc2626', markersize=6,
                 linewidth=2, label='Max Constraint Violation')
    ax2.set_xlabel("Outer Iteration"); ax2.set_ylabel("Violation")
    ax2.set_title("Constraint Violation (log scale)"); ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.semilogy(iters, mu_vals, '^--', color='#16a34a', markersize=6,
                  linewidth=1.5, label='Penalty μ')
    ax2b.set_ylabel("Penalty Factor μ")
    ax2b.legend(loc='lower right')

    plt.tight_layout()
    return fig

def fmt_vec(v, precision=4):
    """向量格式化为紧凑字符串，用于报告展示"""
    return np.array2string(np.asarray(v), precision=precision, separator=', ')


def generate_report_pdf(student_id, student_name, student_date, ptype,
                        n_cons, x0, f_x0,
                        x_ep, f_ep, hist_ep,
                        x_ib, f_ib, hist_ib,
                        sp_x, sp_f, sp_ok, plot_fig):
    """
    自动生成实验报告 PDF：
      第 1 页: 学生信息 + 个性化问题描述
      第 2 页: 外点法 / 内点法结果 + scipy 对照
      第 3 页: 收敛图
    """
    pdf_path = f"lab3_report_{student_id}.pdf"
    err_ep = abs(f_ep - sp_f) if sp_ok else float('inf')
    err_ib = abs(f_ib - sp_f) if sp_ok else float('inf')

    with PdfPages(pdf_path) as pdf:
        # ============ 第 1 页：学生信息 + 问题描述 ============
        fig1 = plt.figure(figsize=(8.27, 11.69))  # A4 竖版
        fig1.suptitle("《最优化方法》实验三：惩罚函数法实验报告",
                      fontsize=16, fontweight='bold', y=0.98)

        info = (f"学生姓名：{student_name}\n"
                f"学    号：{student_id}\n"
                f"实验日期：{student_date}\n")
        fig1.text(0.08, 0.90, info, fontsize=12, va='top', linespacing=1.7)

        prob = (f"一、个性化约束优化问题\n\n"
                f"由学号 {student_id} 作为随机种子生成\n"
                f"问题类型：{ptype}\n"
                f"约束个数：{n_cons}\n\n"
                f"初始点 x0 = {fmt_vec(x0, 4)}\n"
                f"初始函数值 f(x0) = {f_x0:.6f}\n")
        fig1.text(0.08, 0.80, prob, fontsize=11, va='top', linespacing=1.7)

        pdf.savefig(fig1)
        plt.close(fig1)

        # ============ 第 2 页：两种方法结果 + scipy 对照 ============
        fig2 = plt.figure(figsize=(8.27, 11.69))
        fig2.suptitle("求解结果与验证", fontsize=16, fontweight='bold', y=0.98)

        ep_txt = (f"二、外点惩罚函数法\n\n"
                  f"   最优解 x* = {fmt_vec(np.round(x_ep, 6), 6)}\n"
                  f"   最优值 f* = {f_ep:.6f}\n"
                  f"   外迭代次数：{len(hist_ep)}\n"
                  f"   最终约束违反量：{hist_ep[-1][3]:.2e}\n")
        fig2.text(0.08, 0.85, ep_txt, fontsize=11, va='top', linespacing=1.6)

        if len(hist_ib) > 0:
            ib_txt = (f"三、内点障碍函数法\n\n"
                      f"   最优解 x* = {fmt_vec(np.round(x_ib, 6), 6)}\n"
                      f"   最优值 f* = {f_ib:.6f}\n"
                      f"   外迭代次数：{len(hist_ib)}\n")
        else:
            ib_txt = (f"三、内点障碍函数法\n\n"
                      f"   未找到可行点，内点法无法进行\n")
        fig2.text(0.08, 0.68, ib_txt, fontsize=11, va='top', linespacing=1.6)

        if sp_ok:
            scipy_txt = (f"四、scipy.optimize.minimize (SLSQP) 对照\n\n"
                         f"scipy 最优解 x* = {fmt_vec(np.round(sp_x, 6), 6)}\n"
                         f"scipy 最优值 f* = {sp_f:.6f}\n\n"
                         f"外点法误差：{err_ep:.2e}  {'✅' if err_ep < 1e-3 else '⚠️'}\n"
                         f"内点法误差：{err_ib:.2e}  {'✅' if err_ib < 1e-3 else '⚠️'}\n")
        else:
            scipy_txt = "四、scipy.optimize.minimize (SLSQP) 对照\n\nscipy 求解失败\n"
        fig2.text(0.08, 0.42, scipy_txt, fontsize=11, va='top', linespacing=1.6)

        pdf.savefig(fig2)
        plt.close(fig2)

        # ============ 第 3 页：收敛图 ============
        pdf.savefig(plot_fig)

    return pdf_path


def main():
    print(f"={'='*60}")
    print(f"  《最优化方法》实验三：惩罚函数法")
    print(f"  学生: {STUDENT_NAME}  ({STUDENT_ID})")
    print(f"  日期: {STUDENT_DATE}")
    print(f"={'='*60}\n")

    if STUDENT_ID == 2024000001:
        print("⚠️  警告: 请将 STUDENT_ID 改为你的真实学号！\n")

    f, g_ineq, x0, ptype = generate_personalized_problem(STUDENT_ID)
    print(f"测试问题类型: {ptype}")
    print(f"约束个数: {len(g_ineq)}")
    print(f"初始点 x0 = {np.round(x0, 4)}")
    print(f"初始函数值 f(x0) = {f(x0):.6f}\n")

    print(">>> 外点惩罚函数法 (Exterior Penalty)")
    x_ep, f_ep, hist_ep = exterior_penalty(f, g_ineq, x0)
    print(f"  最优解: x* = {np.round(x_ep, 6)}")
    print(f"  最优值: f* = {f_ep:.6f}")
    print(f"  外迭代次数: {len(hist_ep)}")
    print(f"  最终违反量: {hist_ep[-1][3]:.2e}")
    if g_ineq:
        print(f"  约束值: {[np.round(g(x_ep), 6) for g in g_ineq]}\n")

    print(">>> 内点障碍函数法 (Interior Barrier)")
    x_ib, f_ib, hist_ib = interior_barrier(f, g_ineq, x0)
    print(f"  最优解: x* = {np.round(x_ib, 6)}")
    print(f"  最优值: f* = {f_ib:.6f}")
    print(f"  外迭代次数: {len(hist_ib)}")
    if g_ineq:
        print(f"  约束值: {[np.round(g(x_ib), 6) for g in g_ineq]}\n")

    print("对照 scipy.optimize.minimize (SLSQP)...")
    sp_x, sp_f, sp_ok = solve_with_scipy(f, g_ineq, x0)
    if sp_ok:
        print(f"  scipy 最优解: x* = {np.round(sp_x, 6)}")
        print(f"  scipy 最优值: f* = {sp_f:.6f}")
        err_ep = np.abs(f_ep - sp_f)
        err_ib = np.abs(f_ib - sp_f)
        print(f"  外点法误差: {err_ep:.2e}  {'✅' if err_ep < 1e-3 else '⚠️'}")
        print(f"  内点法误差: {err_ib:.2e}  {'✅' if err_ib < 1e-3 else '⚠️'}")
    else:
        print("  scipy 求解失败")

    np.savez(f"result_{STUDENT_ID}.npz",
             student_id=STUDENT_ID, student_name=STUDENT_NAME,
             problem_type=ptype,
             x_ep=x_ep, f_ep=f_ep, n_iter_ep=len(hist_ep),
             x_ib=x_ib, f_ib=f_ib, n_iter_ib=len(hist_ib),
             scipy_x=sp_x, scipy_f=sp_f)
    print(f"\n✅ 结果已保存: result_{STUDENT_ID}.npz")

    fig = plot_convergence(hist_ep, STUDENT_ID, STUDENT_NAME)
    fig.savefig(f"plot_{STUDENT_ID}.png", dpi=150, bbox_inches='tight')
    print(f"✅ 图像已保存: plot_{STUDENT_ID}.png")

    generate_report_pdf(STUDENT_ID, STUDENT_NAME, STUDENT_DATE, ptype,
                        len(g_ineq), x0, f(x0),
                        x_ep, f_ep, hist_ep,
                        x_ib, f_ib, hist_ib,
                        sp_x, sp_f, sp_ok, fig)
    plt.close(fig)
    print(f"✅ 报告已保存: lab3_report_{STUDENT_ID}.pdf")

    print(f"\n{'='*60}")
    print("  提交清单:")
    print(f"    1. result_{STUDENT_ID}.npz")
    print(f"    2. plot_{STUDENT_ID}.png")
    print(f"    3. lab3_report_{STUDENT_ID}.pdf")
    print(f"{'='*60}")



if __name__ == "__main__":
    main()
