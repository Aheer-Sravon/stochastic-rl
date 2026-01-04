import cvxpy as cp
import numpy as np

np.random.seed(1)

# Generate a random non-trivial linear program
m = 15
n = 10
s0 = np.random.randn(m)
lam0 = np.maximum(-s0, 0)
s0 = np.maximum(s0, 0)
x0 = np.random.randn(n)
A = np.random.randn(m, n)
b = A@x0 + s0
c = -A.T@lam0

# Define and solve the CVXPY problem
x = cp.Variable(n)
prob = cp.Problem(
        cp.Minimize(c.T@x),
        [A@x <= b])
prob.solve()

# Print result
print(f"\nThe optimal value is: {prob.value}")
print(f"A solution x is: {x.value}")
print(f"A dual solution is: {prob.constraints[0].dual_value}")
