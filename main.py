import torch
import numpy as np
import re, os

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

torch.set_default_dtype(torch.float64)


def parse_libsvm_line(line_str, current_max_feat_idx):
    """解析单行 LIBSVM 格式的数据。"""
    parts = line_str.strip().split()
    label = int(float(parts[0]))
    features = {}
    max_idx_in_line = 0
    for item in parts[1:]:
        idx_str, val_str = item.split(':')
        idx = int(idx_str)  # 1-based index
        val = float(val_str)
        features[idx - 1] = val  # 转换为 0-based 索引
        if idx > max_idx_in_line:
            max_idx_in_line = idx
    if max_idx_in_line > current_max_feat_idx:
        current_max_feat_idx = max_idx_in_line
    return label, features, current_max_feat_idx


def load_data_from_file(filepath):
    """从 LIBSVM 格式的文件加载数据。"""
    lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        print(f"文件 '{filepath}' 不是 UTF-8 编码，尝试使用 GBK 编码...")
        try:
            with open(filepath, 'r', encoding='gbk') as f:
                lines = f.readlines()
        except Exception as e_gbk:
            print(f"使用 GBK 编码读取文件 '{filepath}' 失败: {e_gbk}，尝试使用 latin-1 编码...")
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
            except Exception as e_latin1:
                raise IOError(f"无法以 UTF-8, GBK 或 latin-1 编码读取文件 '{filepath}': {e_latin1}")
    except FileNotFoundError:
        raise FileNotFoundError(f"错误：数据文件 '{filepath}' 未找到。请确保文件路径正确。")

    labels_list = []
    features_list_of_dicts = []
    max_feat_idx = 0

    for line_str in lines:
        # 清理行中可能存在的 "" 标签
        line_str_cleaned = re.sub(r'', '', line_str).strip()
        if not line_str_cleaned:  # 跳过空行
            continue
        label, features_dict, max_feat_idx = parse_libsvm_line(line_str_cleaned, max_feat_idx)
        labels_list.append(label)
        features_list_of_dicts.append(features_dict)

    num_samples = len(labels_list)
    if num_samples == 0:
        raise ValueError(f"文件 '{filepath}' 中未找到有效数据行。请检查文件格式和内容。")

    num_features = max_feat_idx  # n 是最大的特征索引 (因为文件中是1-based)

    # 创建稠密的特征矩阵 A 和标签向量 b
    A = torch.zeros((num_samples, num_features))
    b = torch.tensor(labels_list).double().unsqueeze(1)

    for i, features_dict in enumerate(features_list_of_dicts):
        for idx_0_based, val in features_dict.items():
            if idx_0_based < num_features:  # 确保索引在界内
                A[i, idx_0_based] = val
            else:
                print(f"警告: 样本 {i} 的特征索引 {idx_0_based + 1} 超出范围 (最大为 {num_features})。该特征将被忽略。")

    return A, b, num_samples, num_features


def soft_threshold(x, kappa):
    """软阈值算子。"""
    return torch.sign(x) * torch.maximum(torch.abs(x) - kappa, torch.zeros_like(x))


def overall_objective_l(x, A, b, m, lambda_val, mu_val):
    """计算总目标函数 l(x) 的值。"""
    logistic_loss_terms = torch.nn.functional.softplus(-b * (A @ x))
    mean_logistic_loss = torch.mean(logistic_loss_terms)
    l2_reg = lambda_val * torch.norm(x, 2) ** 2
    l1_reg = mu_val * torch.norm(x, 1)
    return mean_logistic_loss + l2_reg + l1_reg


def x_update_objective_and_grad_hess(x_var, A, b, m, lambda_val, rho, v_k):
    """
    计算 x-子问题的目标函数值、梯度和 Hessian 矩阵。
    L_x(x) = (1/m) * sum(log(1+exp(-b_i * a_i^T * x))) + lambda*||x||_2^2 + (rho/2)*||x - v_k||_2^2
    """
    x_var = x_var.detach().clone().requires_grad_(True)

    scores = A @ x_var
    logistic_loss_terms = torch.nn.functional.softplus(-b * scores)
    mean_logistic_loss = torch.mean(logistic_loss_terms)
    l2_reg = lambda_val * torch.norm(x_var, 2) ** 2
    prox_term = (rho / 2) * torch.norm(x_var - v_k, 2) ** 2
    objective = mean_logistic_loss + l2_reg + prox_term

    objective.backward()
    grad = x_var.grad.detach().clone()  # now guaranteed to exist
    x_var.grad.zero_()

    # 计算 Hessian 矩阵
    # H = (1/m) * A^T * D * A + (2*lambda + rho)*I
    # 其中 D 是对角矩阵，D_ii = sigmoid(-b_i * a_i^T * x) * (1 - sigmoid(-b_i * a_i^T * x))
    sig_vals = torch.sigmoid(-b * scores)
    D_diag = sig_vals * (1 - sig_vals)

    hess_log_loss = torch.zeros((x_var.shape[0], x_var.shape[0]), dtype=x_var.dtype, device=x_var.device)
    for i in range(m):  # m 是样本数
        a_i = A[i, :].unsqueeze(1)  # shape [n, 1]
        hess_log_loss += D_diag[i] * (a_i @ a_i.T)
    hess_log_loss /= m

    hessian = hess_log_loss + (2 * lambda_val + rho) * torch.eye(x_var.shape[0], dtype=x_var.dtype, device=x_var.device)

    return objective, grad, hessian


def solve_x_subproblem(x_init, A, b, m, lambda_val, rho, v_k,
                       max_newton_iter=20, newton_tol=1e-7,
                       bt_c1=0.01, bt_beta=0.5, verbose=False):
    """使用牛顿法和回溯线搜索求解 x-子问题。"""
    x_curr = x_init.clone().detach()

    if verbose: print(f"  开始求解 x-子问题。初始 x 范数: {torch.norm(x_curr):.4f}")

    for i in range(max_newton_iter):
        x_curr.requires_grad_(True)  # 确保可以计算梯度
        obj_val, grad, hessian = x_update_objective_and_grad_hess(x_curr, A, b, m, lambda_val, rho, v_k)
        x_curr = x_curr.detach()  # 在梯度和Hessian计算后分离

        try:
            p_k = torch.linalg.solve(hessian, -grad)  # 牛顿方向
        except torch.linalg.LinAlgError:
            if verbose: print("  牛顿步骤中出现线性代数错误 (LinAlgError)，尝试使用梯度下降方向。")
            p_k = -grad
            if torch.isnan(grad).any():
                if verbose: print("  x-更新中的梯度为 NaN。返回当前 x。")
                return x_curr

        if torch.isnan(p_k).any():
            if verbose: print("  牛顿方向 p_k 为 NaN。返回当前 x。")
            return x_curr

        # 回溯线搜索
        t = 1.0
        grad_dot_p = torch.sum(grad * p_k)

        if grad_dot_p > 0:  # 如果是上升方向 (理论上 H 应正定, p_k 应为下降方向)
            if verbose: print(f"  警告: 牛顿方向是上升方向 (g^T p = {grad_dot_p:.2e})。使用 -grad 作为方向。")
            p_k = -grad  # 使用负梯度方向
            grad_dot_p = torch.sum(grad * p_k)  # 重新计算

        for _ in range(50):  # 最大回溯步数
            x_new = x_curr + t * p_k
            # 重新计算 x_new 处的目标函数值 (用于 Armijo 条件判断)
            scores_new = A @ x_new
            logistic_loss_terms_new = torch.nn.functional.softplus(-b * scores_new)
            mean_logistic_loss_new = torch.mean(logistic_loss_terms_new)
            l2_reg_new = lambda_val * torch.norm(x_new, 2) ** 2
            prox_term_new = (rho / 2) * torch.norm(x_new - v_k, 2) ** 2
            obj_new = mean_logistic_loss_new + l2_reg_new + prox_term_new

            if obj_new <= obj_val + bt_c1 * t * grad_dot_p:
                break  # 满足 Armijo 条件
            t *= bt_beta
        else:
            if verbose: print("  回溯线搜索未能找到合适的步长。保持当前 x_curr。")
            if torch.norm(t * p_k) < newton_tol:
                return x_curr

        x_curr = x_curr + t * p_k
        step_norm = torch.norm(t * p_k)

        if verbose and (i % 5 == 0 or i == max_newton_iter - 1):
            print(f"    牛顿迭代 {i + 1}: obj={obj_val:.4e}, step_norm={step_norm:.2e}, t={t:.2e}")

        if step_norm < newton_tol:
            if verbose: print(f"  牛顿法在 {i + 1} 次迭代后收敛。")
            break
    else:
        if verbose: print(f"  牛顿法达到最大迭代次数 ({max_newton_iter})。")

    return x_curr.detach()


def admm_solver(A, b, m, n, lambda_val, mu_val, rho,
                max_admm_iter=10000, tol_abs=1e-6, verbose=False,
                x_subproblem_verbose=False):
    """使用 ADMM 求解稀疏逻辑回归问题。"""
    x = torch.zeros((n, 1), device=A.device)
    z = torch.zeros((n, 1), device=A.device)
    y = torch.zeros((n, 1), device=A.device)

    history = {
        'obj_l': [],
        'primal_residual': [],
        'dual_residual': [],
        'x_trajectory': []
    }

    print(f"开始 ADMM 迭代: m={m}, n={n}, lambda={lambda_val:.2e}, mu={mu_val:.2e}, rho={rho:.2e}")
    print(f"最大 ADMM 迭代次数: {max_admm_iter}, 收敛阈值: {tol_abs:.1e}")

    for k in range(max_admm_iter):
        # x-更新
        v_k = z - y / rho
        x_prev_sub = x.clone()
        x = solve_x_subproblem(x.clone(), A, b, m, lambda_val, rho, v_k, verbose=x_subproblem_verbose)

        if torch.isnan(x).any():
            print(f"ADMM 迭代 {k + 1}: x-子问题求解后 x 包含 NaN。终止。")
            x = x_prev_sub
            break

        # z-更新 (软阈值)
        z_old = z.clone()
        kappa = mu_val / rho
        z = soft_threshold(x + y / rho, kappa)

        # y-更新 (对偶变量)
        y = y + rho * (x - z)

        # 计算残差和目标函数值
        primal_res = torch.norm(x - z, 2)
        dual_res = torch.norm(rho * (z - z_old), 2)

        current_obj_l = overall_objective_l(x, A, b, m, lambda_val, mu_val)

        history['obj_l'].append(current_obj_l.item())
        history['primal_residual'].append(primal_res.item())
        history['dual_residual'].append(dual_res.item())
        history['x_trajectory'].append(x.clone().cpu().numpy())

        if verbose and (k % 20 == 0 or k == max_admm_iter - 1 or k == 0):
            num_zeros_in_x = torch.sum(torch.abs(x) < 1e-7).item()
            print(
                f"迭代 {k + 1:4d}: Obj_l={current_obj_l:.4e}, P_Res={primal_res:.2e}, D_Res={dual_res:.2e}, x中零元数={num_zeros_in_x}")

        if primal_res < tol_abs and dual_res < tol_abs:
            print(f"ADMM 在 {k + 1} 次迭代后收敛。")
            break
    else:
        print(f"ADMM 达到最大迭代次数 ({max_admm_iter}) 未达到指定收敛阈值。")

    return x, z, y, history


if __name__ == '__main__':

    os.makedirs('image', exist_ok=True)  # 确保保存图像的目录存在
    data_file = 'a9a.txt'
    try:
        A_tensor, b_tensor, m_samples, n_features = load_data_from_file(data_file)
        print(f"数据加载成功: {m_samples} 个样本, {n_features} 个特征。")
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请确保 'a9a.txt' 文件与脚本在同一目录下，并且格式正确。")
        exit()


    print("\n--- 第 1 部分: 固定 lambda 和 mu 求解 ---")
    lambda_val_part1 = 1.0 / (2.0 * m_samples)
    mu_val_part1 = 0.01
    rho_val = 1.0
    max_iter = 10000
    tol_part1 = 1e-6

    x_star_part1, _, _, history_part1 = admm_solver(
        A_tensor, b_tensor, m_samples, n_features,
        lambda_val_part1, mu_val_part1, rho_val,
        max_admm_iter=max_iter, tol_abs=tol_part1, verbose=True, x_subproblem_verbose=False
    )

    # 使用本次运行的最终 x 作为 x_star 来绘制 l(x_k) - l(x*)
    l_x_star_part1 = overall_objective_l(x_star_part1, A_tensor, b_tensor, m_samples, lambda_val_part1,
                                         mu_val_part1).item()

    function_errors_part1 = []
    if history_part1['x_trajectory']:
        for x_k_np in history_part1['x_trajectory']:
            # 将 numpy 数组转回 tensor
            x_k_tensor = torch.tensor(x_k_np, dtype=torch.float64, device=A_tensor.device)
            error_val = overall_objective_l(x_k_tensor, A_tensor, b_tensor, m_samples, lambda_val_part1,
                                            mu_val_part1).item() - l_x_star_part1
            function_errors_part1.append(error_val)
    else:
        print("警告: history_part1['x_trajectory'] 为空，无法计算函数误差。")


    iterations_part1 = range(1, len(history_part1['primal_residual']) + 1)

    if iterations_part1:  # 确保有迭代数据才绘图
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(iterations_part1, history_part1['primal_residual'], label='原始残差 $\|x^k - z^k\|_2$', color='blue')
        ax1.plot(iterations_part1, history_part1['dual_residual'], label='对偶残差 $\|\\rho(z^k - z^{k-1})\|_2$',
                 color='red')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('残差值 (对数刻度)')
        ax1.set_yscale('log')
        ax1.set_title('ADMM 最优条件与迭代次数的关系')
        ax1.legend()
        ax1.grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        fig1.savefig("image/admm_residuals_cn.png")
        print("残差图已保存为 admm_residuals_cn.png")

        if function_errors_part1:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            plot_func_errors = np.array(function_errors_part1)
            # 为对数刻度处理非常小或零的误差值
            plot_func_errors = np.maximum(plot_func_errors, 1e-12)

            ax2.plot(iterations_part1, plot_func_errors, label='$l(x^k) - l(x^*)$', color='green')
            ax2.set_xlabel('迭代次数')
            ax2.set_ylabel('函数误差 (对数刻度)')
            ax2.set_yscale('log')
            ax2.set_title('函数误差 $l(x^k) - l(x^*)$ 与迭代次数的关系')
            ax2.legend()
            ax2.grid(True, which="both", ls="-", alpha=0.5)
            plt.tight_layout()
            fig2.savefig("image/function_error_cn.png")
            print("函数误差图已保存为 function_error_cn.png")
        else:
            print("函数误差数据为空，无法绘制函数误差图。")

    else:
        print("没有足够的迭代数据来生成第1部分的图像。")


    print("\n--- 第 2 部分: 调整 mu 值 ---")
    mu_values_part2 = [1e-3, 0.05, 0.01]
    results_part2 = []
    lambda_val_part2 = 1.0 / (2.0 * m_samples)
    sparsity_threshold = 1e-7

    for mu_val_current in mu_values_part2:
        print(f"\n对 mu = {mu_val_current:.1e} 运行 ADMM")
        x_final, _, _, history_current = admm_solver(
            A_tensor, b_tensor, m_samples, n_features,
            lambda_val_part2, mu_val_current, rho_val,
            max_admm_iter=max_iter, tol_abs=tol_part1, verbose=False
        )

        num_iterations = len(history_current['primal_residual'])
        num_zero_elements = torch.sum(torch.abs(x_final) < sparsity_threshold).item()
        sparsity_ratio = num_zero_elements / n_features if n_features > 0 else 0

        results_part2.append({
            'mu': mu_val_current,
            'iterations': num_iterations,
            'sparsity': sparsity_ratio,
            'non_zero_coeffs': n_features - num_zero_elements
        })
        print(
            f"mu = {mu_val_current:.3f}: 迭代次数 = {num_iterations}, 稀疏度 = {sparsity_ratio:.4f} ({num_zero_elements}/{n_features} 个零元)")

    print("\n不同 mu 值的结果总结:")
    print("----------------------------------------------------------")
    print("| Mu 值    | 迭代次数   | 稀疏度 (零元比例) | 非零元数量 |")
    print("----------------------------------------------------------")
    for res in results_part2:
        print(
            f"| {res['mu']:<8.3f} | {res['iterations']:<10d} | {res['sparsity']:<17.4f} | {res['non_zero_coeffs']:<10d} |")
    print("----------------------------------------------------------")

    plt.show()