"""Minimal verification example to test project setup"""

import numpy as np
from scipy.linalg import solve_continuous_lyapunov, solve_continuous_are
import matplotlib.pyplot as plt

# System parameters
v = 15.0
b = 0.5
A = np.array([[0, 1, 0, 0],
              [0, 0, v, 0],
              [0, 0, 0, 1],
              [0, 0, 0, 0]])
B = np.array([[0], [0], [0], [b]])

# LQR parameters
Q = np.diag([100, 10, 50, 5])
R = np.array([[1.0]])

print("Step 1: Computing LQR gain...")
P_riccati = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P_riccati
A_cl = A - B @ K
print(f"  K = {K.flatten()}")

print("\nStep 2: Checking stability...")
eigenvalues = np.linalg.eigvals(A_cl)
print(f"  Eigenvalues: {eigenvalues}")
stable = np.all(np.real(eigenvalues) < 0)
print(f"  Stable: {stable}")

print("\nStep 3: Computing Lyapunov function...")
P = solve_continuous_lyapunov(A_cl.T, -Q)
eigenvalues_P = np.linalg.eigvals(P)
print(f"  P eigenvalues: {eigenvalues_P}")
print(f"  P positive definite: {np.all(eigenvalues_P > 0)}")

print("\nStep 4: Computing maximum invariant set...")
L = 1.75
P_inv = np.linalg.inv(P)
c_max = L**2 / P_inv[0, 0]
print(f"  Lane half-width L = {L} m")
print(f"  c_max = {c_max:.4f}")

print("\nStep 5: Checking initial conditions...")
initial_bounds = {'x1': (-0.5, 0.5), 'x2': (-0.1, 0.1),
                  'x3': (-0.175, 0.175), 'x4': (-0.05, 0.05)}

max_V = 0
for x1 in initial_bounds['x1']:
    for x2 in initial_bounds['x2']:
        for x3 in initial_bounds['x3']:
            for x4 in initial_bounds['x4']:
                x = np.array([x1, x2, x3, x4])
                V_val = x.T @ P @ x
                max_V = max(max_V, V_val)

print(f"  max V(x₀) = {max_V:.4f}")
print(f"  c_max = {c_max:.4f}")

if max_V <= c_max:
    print("\n✓✓✓ VERIFICATION SUCCESSFUL ✓✓✓")
    print("Coverage: 100%")
else:
    coverage = (c_max / max_V) * 100
    print(f"\n⚠ PARTIAL VERIFICATION")
    print(f"Coverage: {coverage:.1f}%")

print("\nDone! Check results/ directory for outputs.")
