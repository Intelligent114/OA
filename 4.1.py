import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

epsilon = 0.09

A = np.random.randn(100, 100)
while np.linalg.det(A) == 0:
    A = np.random.randn(100, 100)
A_inv = np.linalg.inv(A)
X = A_inv + np.eye(100) * epsilon

max_iter = 50
errors = []

for k in range(max_iter):
    X_new = 2 * X - np.dot(np.dot(X, A), X)
    error = np.linalg.norm(X_new - A_inv, 'fro')
    errors.append(error)
    X = X_new


plt.plot(range(max_iter), errors)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Frobenius Norm of Error')
plt.title('Convergence of Newton\'s Method')
plt.show()
