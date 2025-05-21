import numpy as np
import matplotlib.pyplot as plt


def load_a9a(filename):
    b = []
    A = []
    n_features = 123  # a9a数据集固定123维特征
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            label = int(parts[0])
            b.append(1 if label == 1 else -1)
            features = np.zeros(n_features)
            for pair in parts[1:]:
                idx, val = pair.split(':')
                features[int(idx) - 1] = float(val)
            A.append(features)
    return np.array(A), np.array(b)


# 加载数据
A, b = load_a9a('a9a.txt')
m, n = A.shape
lambda_ = 1 / (2 * m)
rho = 1.0  # ADMM惩罚参数


def compute_objective(x, mu):
    scores = A.dot(x)
    loss = np.mean(np.log(1 + np.exp(-b * scores)))
    l2 = lambda_ * np.sum(x ** 2)
    l1 = mu * np.sum(np.abs(x))
    return loss + l2 + l1


def admm_solver(A, b, mu, rho, max_iter=10000, tol=1e-6):
    m, n = A.shape
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)
    primal_residuals = []
    dual_residuals = []
    objectives = []

    for k in range(max_iter):
        z_prev = z.copy()

        # 更新x: 梯度下降+回溯线搜索
        scores = A.dot(x)
        b_scores = b * scores
        prob = 1.0 / (1 + np.exp(-b_scores))  # 数值稳定
        grad_f = (A.T.dot(-b * prob) / m) + 2 * lambda_ * x
        grad = grad_f + rho * (x - z + u)

        # 回溯线搜索
        alpha, beta = 0.3, 0.5
        t = 1.0
        while True:
            x_new = x - t * grad
            f_old = np.mean(np.log(1 + np.exp(-b * A.dot(x)))) + lambda_ * np.sum(x ** 2) + 0.5 * rho * np.linalg.norm(x - z + u) ** 2
            f_new = np.mean(np.log(1 + np.exp(-b * A.dot(x_new)))) + lambda_ * np.sum(x_new ** 2) + 0.5 * rho * np.linalg.norm(x_new - z + u) ** 2
            if f_new <= f_old - alpha * t * np.sum(grad ** 2):
                break
            t *= beta
        x = x_new

        # 更新z
        kappa = mu / rho
        z = np.sign(x + u) * np.maximum(np.abs(x + u) - kappa, 0)

        # 更新对偶变量
        u += (x - z)

        # 记录残差
        primal = np.linalg.norm(x - z)
        dual = rho * np.linalg.norm(z - z_prev)
        primal_residuals.append(primal)
        dual_residuals.append(dual)
        objectives.append(compute_objective(x, mu))

        # 停止条件
        if primal < tol and dual < tol:
            break

    return x, primal_residuals, dual_residuals, objectives, k + 1


# 运行ADMM并绘图
mu = 0.01
x_star, primals, duals, objs, _ = admm_solver(A, b, mu, rho)
l_star = objs[-1]
errors = [obj - l_star for obj in objs]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.semilogy(primals, label='Primal Residual')
plt.semilogy(duals, label='Dual Residual')
plt.legend(), plt.title('ADMM Convergence')

plt.subplot(1, 2, 2)
plt.semilogy(errors)
plt.title('Function Error'), plt.tight_layout()
plt.show()

# 不同mu值的分析
results = []
for mu in [1e-3, 0.05, 0.01]:
    x, _, _, _, iters = admm_solver(A, b, mu, rho)
    sparsity = np.mean(np.abs(x) < 1e-6)
    results.append((mu, iters, sparsity))

print("μ | Iterations | Sparsity")
for mu, iters, sparsity in results:
    print(f"{mu:.3f} | {iters} | {sparsity:.2%}")