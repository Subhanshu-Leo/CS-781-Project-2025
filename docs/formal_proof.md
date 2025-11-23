# Formal Verification Proof

## Theorem Statement
For the learned LQR controller with parameters:
- Q = diag([q₁, q₂, q₃, q₄])
- R = r
- K = [k₁, k₂, k₃, k₄]

We prove that for all initial conditions x₀ ∈ X₀:
|x₁(t)| < 1.75m for all t ≥ 0

## Proof Method
Lyapunov stability analysis with quadratic Lyapunov function V(x) = xᵀPx

## Proof Steps
1. **Stability:** Verify closed-loop eigenvalues have negative real parts
2. **Lyapunov Equation:** Solve AᵀₗₗP + PAcl = -Q
3. **Invariant Set:** Compute c_max = L²/(P⁻¹)₀₀
4. **Coverage:** Verify max{V(x) : x ∈ X₀} ≤ c_max

## Result
[To be filled after running verification]
- Coverage: XX%
- Verification Status: COMPLETE/PARTIAL
- c_max value: XXX
