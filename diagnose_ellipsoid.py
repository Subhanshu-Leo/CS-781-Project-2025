#!/usr/bin/env python3
"""
Diagnostic script to understand the ellipsoid geometry issue
"""

import numpy as np
from scipy.linalg import solve_continuous_lyapunov
import json

# Load the learned parameters
with open('results/learned_parameters.json', 'r') as f:
    data = json.load(f)
    Q = np.array(data['best_params']['Q'])
    R = np.array(data['best_params']['R'])
    K = np.array(data['best_params']['K'])

# System matrices
v = 15.0
b = 0.5
A = np.array([[0, 1, 0, 0], [0, 0, v, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
B = np.array([[0], [0], [0], [b]])
A_cl = A - B @ K

# Lyapunov matrix
P = solve_continuous_lyapunov(A_cl.T, -Q)
P_inv = np.linalg.inv(P)

L = 1.75  # Lane half-width

print("="*70)
print("ELLIPSOID DIAGNOSTIC")
print("="*70)

print("\nLyapunov Matrix P:")
print(f"  P[0,0] = {P[0,0]:.6f}")
print(f"  P_inv[0,0] = {P_inv[0,0]:.6f}")

print("\n" + "-"*70)
print("TEST 1: Which formula is correct?")
print("-"*70)

# Formula 1: c = L² / P_inv[0,0]
c1 = L**2 / P_inv[0, 0]
x1_max_formula1 = np.sqrt(c1 * P_inv[0, 0])
x_test1 = np.array([x1_max_formula1, 0, 0, 0])
V_test1 = x_test1.T @ P @ x_test1

print(f"\nFormula 1: c = L² / P_inv[0,0]")
print(f"  c = {c1:.4f}")
print(f"  Predicted x₁_max = sqrt(c × P_inv[0,0]) = {x1_max_formula1:.4f}")
print(f"  At x = [{x1_max_formula1:.4f}, 0, 0, 0]:")
print(f"    V(x) = {V_test1:.4f} (should equal c = {c1:.4f})")
print(f"    Error: {abs(V_test1 - c1):.2e}")

# Formula 2: c = L² × P[0,0]
c2 = L**2 * P[0, 0]
x1_max_formula2 = np.sqrt(c2 / P[0, 0])
x_test2 = np.array([x1_max_formula2, 0, 0, 0])
V_test2 = x_test2.T @ P @ x_test2

print(f"\nFormula 2: c = L² × P[0,0]")
print(f"  c = {c2:.4f}")
print(f"  Predicted x₁_max = sqrt(c / P[0,0]) = {x1_max_formula2:.4f}")
print(f"  At x = [{x1_max_formula2:.4f}, 0, 0, 0]:")
print(f"    V(x) = {V_test2:.4f} (should equal c = {c2:.4f})")
print(f"    Error: {abs(V_test2 - c2):.2e}")

print("\n" + "-"*70)
print("TEST 2: Sample actual ellipsoid boundary")
print("-"*70)

# For each formula, sample the actual ellipsoid and find max |x1|
for formula_name, c_val in [("Formula 1", c1), ("Formula 2", c2)]:
    print(f"\n{formula_name}: c = {c_val:.4f}")

    max_x1 = 0
    n_samples = 10000

    # Eigendecomposition for sampling
    eigvals, eigvecs = np.linalg.eigh(P)
    P_inv_sqrt = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T

    np.random.seed(42)
    for _ in range(n_samples):
        # Random point on unit sphere
        xi = np.random.randn(4)
        xi = xi / np.linalg.norm(xi)

        # Map to ellipsoid boundary
        x = np.sqrt(c_val) * (P_inv_sqrt @ xi)

        # Verify on boundary
        V = x.T @ P @ x
        assert abs(V - c_val) < 1e-6

        max_x1 = max(max_x1, abs(x[0]))

    print(f"  Max |x₁| on ellipsoid: {max_x1:.4f} m")
    print(f"  Lane half-width: {L:.4f} m")

    if max_x1 <= L:
        print(f"  ✓ SAFE: Ellipsoid fits in lane")
        print(f"    Safety margin: {L - max_x1:.4f} m")
    else:
        print(f"  ✗ UNSAFE: Ellipsoid exceeds lane")
        print(f"    Excess: {max_x1 - L:.4f} m")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if x1_max_formula1 <= L + 1e-6:
    print("\n✓ Formula 1 is CORRECT: c = L² / P_inv[0,0]")
    print(f"  Use: c_max = {c1:.4f}")
else:
    print("\n✗ Formula 1 produces ellipsoid too large")

if x1_max_formula2 <= L + 1e-6:
    print("\n✓ Formula 2 is CORRECT: c = L² × P[0,0]")
    print(f"  Use: c_max = {c2:.4f}")
else:
    print("\n✗ Formula 2 produces ellipsoid too large")

print("\n" + "="*70)
