#!/usr/bin/env python3
"""
《最优化方法》实验二：无约束问题最优化 —— 梯度法
==================================================
学生需在 ★★★ 标记处填写代码，完成 TODO 1（最速下降法）和 TODO 2（牛顿法）。
TODO 3（BFGS）为加分项，已提供完整实现供参考。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']        # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False          # 解决负号显示成方块

from scipy.optimize import minimize, rosen, rosen_der, rosen_hess
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")


# ★★★ 填写个人信息 ★★★
STUDENT_ID   = 2026000001
STUDENT_NAME = "姓名"
STUDENT_DATE = "2029-07-21"


# ============================================================
#  辅助函数：回溯线搜索 (Armijo 条件) —— 已提供
# ============================================================

def backtracking_line_search(f, grad_f, x_k, d_k, alpha0=1.0,
                              rho=0.5, c=1e-4, max_iter=50):
    """
    回溯线搜索 (Armijo 准则):
        f(x_k + α d_k) ≤ f(x_k) + c·α·∇f(x_k)^T d_k

    返回: 满足条件的步长 alpha
    """
    alpha = alpha0
    f_k = f(x_k)
    grad_dot_d = np.dot(grad_f(x_k), d_k)

    for _ in range(max_iter):
        if f(x_k + alpha * d_k) <= f_k + c * alpha * grad_dot_d:
            break
        alpha *= rho
    return alpha


# ============================================================
#  TODO 1: 最速下降法 (Steepest Descent) —— 必做
# ============================================================

def steepest_descent(f, grad_f, x0, max_iter=500, tol=1e-6):
    """
    最速下降法求解无约束问题 min f(x)

    迭代公式: x_{k+1} = x_k - α_k ∇f(x_k)

    参数:
        f, grad_f: 目标函数及其梯度
        x0:        初始点
        max_iter:  最大迭代次数
        tol:       梯度范数收敛容差

    返回:
        x_opt, f_opt, history (每次迭代的 [x, f(x), ||grad||])
    """
    x = np.array(x0, dtype=float)
    history = []

    for iteration in range(max_iter):
        # ============================================================
        # TODO 1a: 计算梯度并记录历史
        #
        #   g = grad_f(x)
        #   history.append((x.copy(), f(x), np.linalg.norm(g)))
        # ============================================================

        # ★★★ 在此编写代码 ★★★
       

        # ============================================================
        # TODO 1b: 收敛判断
        #
        #   if ||g|| < tol: 
        #       break
        # ============================================================

        # ★★★ 在此编写代码 ★★★
        

        # ============================================================
        # TODO 1c: 确定搜索方向（最速下降 = 负梯度方向）
        #
        #   d = -g
        # ============================================================

        # ★★★ 在此编写代码 ★★★

       
        # ============================================================
        # TODO 1d: 线搜索 + 更新
        #
        #   alpha = backtracking_line_search(f, grad_f, x, d)
        #   x = x + alpha * d
        # ============================================================

        # ★★★ 在此编写代码 ★★★
        


    return x, f(x), history


# ============================================================
#  TODO 2: 牛顿法 (Newton's Method) —— 必做
# ============================================================

def newton_method(f, grad_f, hess_f, x0, max_iter=100, tol=1e-8):
    """
    牛顿法求解无约束问题 min f(x)

    迭代公式: x_{k+1} = x_k - α_k [∇²f(x_k)]^{-1} ∇f(x_k)

    参数:
        f, grad_f, hess_f: 目标函数、梯度、Hessian
        x0:                初始点
        max_iter:          最大迭代次数
        tol:               梯度范数收敛容差

    返回:
        x_opt, f_opt, history
    """
    x = np.array(x0, dtype=float)
    history = []

    for iteration in range(max_iter):
        # ============================================================
        # TODO 2a: 计算梯度 g = grad_f(x)，Hessian H = hess_f(x)
        #          记录历史: history.append((x.copy(), f(x), np.linalg.norm(g)))
        # ============================================================

        # ★★★ 在此编写代码 ★★★
        


        # ============================================================
        # TODO 2b: 收敛判断 (同最速下降)
        # ============================================================

        # ★★★ 在此编写代码 ★★★
        


        # ============================================================
        # TODO 2c: 计算牛顿方向 d = -H^{-1} g
        #
        #   用 np.linalg.solve(H, g) 比求逆更稳定:
        #     d = -np.linalg.solve(H, g)
        #
        #   若 Hessian 奇异 (LinAlgError)，退回最速下降: d = -g
        # ============================================================

        # ★★★ 在此编写代码 ★★★
       


        # ============================================================
        # TODO 2d: 线搜索 + 更新（同最速下降）
        # ============================================================

        # ★★★ 在此编写代码 ★★★
        


    return x, f(x), history


# ============================================================
#  TODO 3 (加分): BFGS 拟牛顿法 —— 已提供完整实现供参考
# ============================================================

def bfgs_method(f, grad_f, x0, max_iter=500, tol=1e-6):
    """
    BFGS 拟牛顿法: 用梯度差近似 Hessian 逆，无需二阶导数
    更新公式: H_{k+1}^{-1} = (I-ρ s y^T) H_k^{-1} (I-ρ y s^T) + ρ s s^T
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    H_inv = np.eye(n)
    history = []

    for iteration in range(max_iter):
        g = grad_f(x)
        grad_norm = np.linalg.norm(g)
        history.append((x.copy(), f(x), grad_norm))

        if grad_norm < tol:
            break

        d = -H_inv @ g
        alpha = backtracking_line_search(f, grad_f, x, d)

        x_new = x + alpha * d
        s = x_new - x
        y = grad_f(x_new) - g

        if np.dot(y, s) > 1e-14:
            rho = 1.0 / np.dot(y, s)
            I = np.eye(n)
            H_inv = (I - rho * np.outer(s, y)) @ H_inv @ (I - rho * np.outer(y, s)) \
                    + rho * np.outer(s, s)

        x = x_new

    return x, f(x), history


# ============================================================
#  测试代码（不可修改）
# ============================================================

def generate_personalized_problem(student_id):
    """生成个性化的无约束优化测试问题"""
    rng = np.random.RandomState(student_id % 100000)
    problem_type = rng.choice(['quadratic', 'rosenbrock', 'custom'])

    if problem_type == 'quadratic':
        n = rng.randint(3, 6)
        A_mat = rng.randn(n, n)
        Q = A_mat.T @ A_mat + np.eye(n)
        b_vec = rng.randn(n) * 3
        x0 = rng.randn(n) * 5 + 10

        def f(x): return 0.5 * x @ Q @ x + b_vec @ x
        def grad_f(x): return Q @ x + b_vec
        def hess_f(x): return Q

    elif problem_type == 'custom':
        n = rng.randint(2, 4)
        offsets = np.arange(1, n+1)
        x0 = rng.randn(n) * 3 + 2

        def f(x): return np.sum((x - offsets)**4 + np.sin(x))
        def grad_f(x): return 4 * (x - offsets)**3 + np.cos(x)
        def hess_f(x): return np.diag(12 * (x - offsets)**2 - np.sin(x))

    else:
        n = 2
        x0 = np.array([-1.5, 2.0]) + rng.randn(2) * 0.2

        def f(x): return rosen(x)
        def grad_f(x): return rosen_der(x)
        def hess_f(x): return rosen_hess(x)

    return f, grad_f, hess_f, x0, problem_type


def plot_convergence(histories, labels, student_id, student_name):
    """比较不同方法的收敛速度"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Lab 2: Gradient Methods  |  {student_name} ({student_id})",
                 fontsize=13, fontweight='bold')
    colors = ['#2563eb', '#dc2626', '#16a34a']

    for history, label, c in zip(histories, labels, colors):
        f_vals = [h[1] for h in history]
        grad_norms = [h[2] for h in history]
        iters = range(1, len(f_vals) + 1)

        ax1.semilogy(iters, grad_norms, 'o-', color=c, markersize=4,
                     linewidth=1.5, label=label)
        ax2.semilogy(iters, [abs(fv - f_vals[-1]) + 1e-16 for fv in f_vals],
                     'o-', color=c, markersize=4, linewidth=1.5, label=label)

    ax1.set_xlabel("Iteration"); ax1.set_ylabel("||Gradient||")
    ax1.set_title("Gradient Norm Convergence"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("|f(x) - f*|")
    ax2.set_title("Function Value Convergence"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def fmt_vec(v, precision=4):
    """向量格式化为紧凑字符串，用于报告展示"""
    return np.array2string(np.asarray(v), precision=precision, separator=', ')


def generate_report_pdf(student_id, student_name, student_date, ptype, x0, f_x0,
                        x_sd, f_sd, hist_sd, x_nm, f_nm, hist_nm,
                        x_bf, f_bf, hist_bf, scipy_f, scipy_iter, plot_fig):
    """
    自动生成实验报告 PDF：
      第 1 页: 学生信息 + 个性化问题描述
      第 2 页: 三种方法结果 + scipy 对照
      第 3 页: 收敛图
    """
    pdf_path = f"lab2_report_{student_id}.pdf"

    with PdfPages(pdf_path) as pdf:
        # ============ 第 1 页：学生信息 + 问题描述 ============
        fig1 = plt.figure(figsize=(8.27, 11.69))  # A4 竖版
        fig1.suptitle("《最优化方法》实验二：梯度法实验报告",
                      fontsize=16, fontweight='bold', y=0.98)

        info = (f"学生姓名：{student_name}\n"
                f"学    号：{student_id}\n"
                f"实验日期：{student_date}\n")
        fig1.text(0.08, 0.90, info, fontsize=12, va='top', linespacing=1.7)

        prob = (f"一、个性化测试问题\n\n"
                f"由学号 {student_id} 作为随机种子生成\n"
                f"问题类型：{ptype}\n\n"
                f"初始点 x0 = {fmt_vec(x0, 4)}\n"
                f"初始函数值 f(x0) = {f_x0:.6f}\n")
        fig1.text(0.08, 0.80, prob, fontsize=11, va='top', linespacing=1.7)

        pdf.savefig(fig1)
        plt.close(fig1)

        # ============ 第 2 页：三种方法结果 + scipy 对照 ============
        fig2 = plt.figure(figsize=(8.27, 11.69))
        fig2.suptitle("求解结果与验证", fontsize=16, fontweight='bold', y=0.98)

        res_txt = (f"二、三种方法求解结果\n\n"
                   f"1. 最速下降法\n"
                   f"   最优解 x* = {fmt_vec(np.round(x_sd, 4), 4)}\n"
                   f"   最优值 f* = {f_sd:.6f}\n"
                   f"   迭代次数：{len(hist_sd)}，最终梯度范数：{hist_sd[-1][2]:.2e}\n\n"
                   f"2. 牛顿法\n"
                   f"   最优解 x* = {fmt_vec(np.round(x_nm, 4), 4)}\n"
                   f"   最优值 f* = {f_nm:.6f}\n"
                   f"   迭代次数：{len(hist_nm)}，最终梯度范数：{hist_nm[-1][2]:.2e}\n\n"
                   f"3. BFGS 拟牛顿法\n"
                   f"   最优解 x* = {fmt_vec(np.round(x_bf, 4), 4)}\n"
                   f"   最优值 f* = {f_bf:.6f}\n"
                   f"   迭代次数：{len(hist_bf)}，最终梯度范数：{hist_bf[-1][2]:.2e}\n")
        fig2.text(0.08, 0.85, res_txt, fontsize=11, va='top', linespacing=1.55)

        scipy_txt = (f"三、scipy.optimize.minimize (BFGS) 对照\n\n"
                     f"scipy 最优值 f* = {scipy_f:.6f}\n"
                     f"scipy 迭代次数：{scipy_iter}\n")
        fig2.text(0.08, 0.30, scipy_txt, fontsize=12, va='top', linespacing=1.6)

        pdf.savefig(fig2)
        plt.close(fig2)

        # ============ 第 3 页：收敛图 ============
        pdf.savefig(plot_fig)

    return pdf_path


def main():
    print(f"={'='*60}")
    print(f"  《最优化方法》实验二：梯度法")
    print(f"  学生: {STUDENT_NAME}  ({STUDENT_ID})")
    print(f"  日期: {STUDENT_DATE}")
    print(f"={'='*60}\n")

    if STUDENT_ID == 2024000001:
        print("⚠️  警告: 请将 STUDENT_ID 改为你的真实学号！\n")

    f, grad_f, hess_f, x0, ptype = generate_personalized_problem(STUDENT_ID)
    print(f"测试问题类型: {ptype}")
    print(f"初始点 x0 = {np.round(x0, 4)}")
    print(f"初始函数值 f(x0) = {f(x0):.6f}\n")

    print(">>> 最速下降法 (Steepest Descent)")
    x_sd, f_sd, hist_sd = steepest_descent(f, grad_f, x0)
    print(f"  最优解: x* = {np.round(x_sd, 4)}")
    print(f"  最优值: f* = {f_sd:.6f}")
    print(f"  迭代次数: {len(hist_sd)}")
    print(f"  最终梯度范数: {hist_sd[-1][2]:.2e}\n")

    print(">>> 牛顿法 (Newton's Method)")
    x_nm, f_nm, hist_nm = newton_method(f, grad_f, hess_f, x0)
    print(f"  最优解: x* = {np.round(x_nm, 4)}")
    print(f"  最优值: f* = {f_nm:.6f}")
    print(f"  迭代次数: {len(hist_nm)}")
    print(f"  最终梯度范数: {hist_nm[-1][2]:.2e}\n")

    print(">>> BFGS 拟牛顿法")
    x_bf, f_bf, hist_bf = bfgs_method(f, grad_f, x0)
    print(f"  最优解: x* = {np.round(x_bf, 4)}")
    print(f"  最优值: f* = {f_bf:.6f}")
    print(f"  迭代次数: {len(hist_bf)}")
    print(f"  最终梯度范数: {hist_bf[-1][2]:.2e}\n")

    print("对照 scipy.optimize.minimize (BFGS)...")
    res = minimize(f, x0, method='BFGS', tol=1e-8)
    print(f"  scipy 最优值: f* = {res.fun:.6f}")
    print(f"  scipy 迭代次数: {res.nit}")

    np.savez(f"result_{STUDENT_ID}.npz",
             student_id=STUDENT_ID, student_name=STUDENT_NAME,
             problem_type=ptype, x0=x0,
             x_sd=x_sd, f_sd=f_sd, n_iter_sd=len(hist_sd),
             x_nm=x_nm, f_nm=f_nm, n_iter_nm=len(hist_nm),
             x_bf=x_bf, f_bf=f_bf, n_iter_bf=len(hist_bf),
             scipy_f=res.fun, scipy_iter=res.nit)
    print(f"\n✅ 结果已保存: result_{STUDENT_ID}.npz")

    fig = plot_convergence([hist_sd, hist_nm, hist_bf],
                           ["Steepest Descent", "Newton", "BFGS"],
                           STUDENT_ID, STUDENT_NAME)
    fig.savefig(f"plot_{STUDENT_ID}.png", dpi=150, bbox_inches='tight')
    print(f"✅ 图像已保存: plot_{STUDENT_ID}.png")

    generate_report_pdf(STUDENT_ID, STUDENT_NAME, STUDENT_DATE, ptype, x0, f(x0),
                        x_sd, f_sd, hist_sd, x_nm, f_nm, hist_nm,
                        x_bf, f_bf, hist_bf, res.fun, res.nit, fig)
    plt.close(fig)
    print(f"✅ 报告已保存: lab2_report_{STUDENT_ID}.pdf")

    print(f"\n{'='*60}")
    print(f"  方法对比总结:")
    print(f"    {'最速下降法':<14} {len(hist_sd):>5} 次迭代")
    print(f"    {'牛顿法':<14} {len(hist_nm):>5} 次迭代  ← 二阶收敛，最快")
    print(f"    {'BFGS':<14} {len(hist_bf):>5} 次迭代  ← 超线性收敛")
    print(f"{'='*60}")

    print(f"\n  提交清单:")
    print(f"    1. result_{STUDENT_ID}.npz")
    print(f"    2. plot_{STUDENT_ID}.png")
    print(f"    3. lab2_report_{STUDENT_ID}.pdf")



if __name__ == "__main__":
    main()
