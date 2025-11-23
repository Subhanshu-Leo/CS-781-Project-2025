"""
Unit tests for Lyapunov verification
Tests the verification procedure with known stable systems
"""
import pytest
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, solve_continuous_are
from scipy.integrate import odeint
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import is_positive_definite, compute_lqr_gain


def test_lyapunov_equation_simple():
    """Test that P satisfies Lyapunov equation for simple 2x2 system"""
    # Simple stable system
    A = np.array([[-1, 1], [0, -2]])
    Q = np.eye(2)

    # Solve
    P = solve_continuous_lyapunov(A.T, -Q)

    # Verify A^T P + P A = -Q
    residual = A.T @ P + P @ A + Q
    assert np.allclose(residual, 0, atol=1e-10), f"Lyapunov equation not satisfied, residual={np.max(np.abs(residual))}"

    # Verify P is positive definite
    assert is_positive_definite(P), "P should be positive definite"


def test_lyapunov_equation_lane_keeping():
    """Test Lyapunov equation for lane-keeping system"""
    # Lane-keeping system
    v = 15.0
    b = 0.5
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, v, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B = np.array([[0], [0], [0], [b]])

    # LQR parameters
    Q = np.diag([100, 10, 50, 5])
    R = np.array([[1.0]])

    # Compute LQR gain
    K = compute_lqr_gain(A, B, Q, R)
    assert K is not None, "LQR gain computation failed"

    # Closed-loop system
    A_cl = A - B @ K

    # Solve Lyapunov equation
    P = solve_continuous_lyapunov(A_cl.T, -Q)

    # Verify equation
    residual = A_cl.T @ P + P @ A_cl + Q
    assert np.allclose(residual, 0, atol=1e-9), "Lyapunov equation not satisfied"

    # Verify P is positive definite
    assert is_positive_definite(P), "P should be positive definite"


def test_stability_verification():
    """Test that closed-loop system is stable"""
    # System
    v = 15.0
    b = 0.5
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, v, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B = np.array([[0], [0], [0], [b]])

    # Good LQR parameters
    Q = np.diag([100, 10, 50, 5])
    R = np.array([[1.0]])

    # Compute gain
    K = compute_lqr_gain(A, B, Q, R)
    assert K is not None, "Failed to compute LQR gain"

    # Check stability
    A_cl = A - B @ K
    eigenvalues = np.linalg.eigvals(A_cl)

    # All eigenvalues should have negative real part
    assert np.all(np.real(eigenvalues) < 0), f"System unstable, eigenvalues: {eigenvalues}"


def test_unstable_controller():
    """Test that bad parameters result in unstable system"""
    v = 15.0
    b = 0.5
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, v, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B = np.array([[0], [0], [0], [b]])

    # Very weak controller (should be unstable)
    Q = np.diag([0.1, 0.01, 0.1, 0.01])  # Very small weights
    R = np.array([[100.0]])  # Very large control penalty

    # This might fail to compute gain or return None
    K = compute_lqr_gain(A, B, Q, R)

    if K is not None:
        A_cl = A - B @ K
        eigenvalues = np.linalg.eigvals(A_cl)
        # Should have at least one positive eigenvalue
        has_unstable = np.any(np.real(eigenvalues) >= -1e-6)
        assert has_unstable, "Expected unstable system with weak controller"


def test_trajectory_convergence():
    """Test that trajectories converge to origin"""
    # System
    v = 15.0
    b = 0.5
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, v, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B = np.array([[0], [0], [0], [b]])

    # LQR controller
    Q = np.diag([100, 10, 50, 5])
    R = np.array([[1.0]])
    K = compute_lqr_gain(A, B, Q, R)
    A_cl = A - B @ K

    def dynamics(x, t):
        return (A_cl @ x).flatten()

    # Initial condition
    x0 = np.array([0.3, 0.05, 0.1, 0.02])

    # Simulate
    t_span = np.linspace(0, 10, 1000)
    trajectory = odeint(dynamics, x0, t_span)

    # Check convergence: final state should be near origin
    x_final = trajectory[-1, :]
    final_norm = np.linalg.norm(x_final)

    assert final_norm < 0.01, f"Trajectory did not converge, final norm: {final_norm}"


def test_lyapunov_function_decreases():
    """Test that V(x) decreases along trajectories"""
    # System
    v = 15.0
    b = 0.5
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, v, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B = np.array([[0], [0], [0], [b]])

    # LQR controller
    Q = np.diag([100, 10, 50, 5])
    R = np.array([[1.0]])
    K = compute_lqr_gain(A, B, Q, R)
    A_cl = A - B @ K

    # Lyapunov matrix
    P = solve_continuous_lyapunov(A_cl.T, -Q)

    def dynamics(x, t):
        return (A_cl @ x).flatten()

    # Initial condition
    x0 = np.array([0.4, 0.08, 0.15, 0.03])

    # Simulate
    t_span = np.linspace(0, 5, 500)
    trajectory = odeint(dynamics, x0, t_span)

    # Compute V along trajectory
    V_trajectory = [x.T @ P @ x for x in trajectory]

    # Check that V is decreasing (allow small numerical errors)
    for i in range(len(V_trajectory) - 1):
        assert V_trajectory[i+1] <= V_trajectory[i] + 1e-6, \
            f"V increased at step {i}: V[{i}]={V_trajectory[i]:.6f}, V[{i+1}]={V_trajectory[i+1]:.6f}"


def test_invariant_ellipsoid():
    """Test that trajectories starting in ellipsoid stay in ellipsoid"""
    # System
    v = 15.0
    b = 0.5
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, v, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B = np.array([[0], [0], [0], [b]])

    # LQR controller
    Q = np.diag([100, 10, 50, 5])
    R = np.array([[1.0]])
    K = compute_lqr_gain(A, B, Q, R)
    A_cl = A - B @ K

    # Lyapunov matrix
    P = solve_continuous_lyapunov(A_cl.T, -Q)

    def dynamics(x, t):
        return (A_cl @ x).flatten()

    # Choose c value
    c = 10.0

    # Initial condition inside ellipsoid
    x0 = np.array([0.2, 0.03, 0.08, 0.01])
    V0 = x0.T @ P @ x0

    # Only test if initial condition is inside
    if V0 <= c:
        # Simulate
        t_span = np.linspace(0, 5, 500)
        trajectory = odeint(dynamics, x0, t_span)

        # Check all states stay in ellipsoid
        for i, x in enumerate(trajectory):
            V = x.T @ P @ x
            assert V <= c + 1e-6, \
                f"Trajectory left ellipsoid at t={t_span[i]:.3f}: V={V:.6f} > c={c}"


def test_positive_definite_checker():
    """Test the positive definite checker utility"""
    # Positive definite matrix
    P_pos = np.array([[2, -1], [-1, 2]])
    assert is_positive_definite(P_pos), "Should detect positive definite matrix"

    # Not positive definite (has negative eigenvalue)
    P_neg = np.array([[1, 2], [2, 1]])
    assert not is_positive_definite(P_neg), "Should detect non-positive definite matrix"

    # Not symmetric
    P_asym = np.array([[1, 2], [3, 4]])
    assert not is_positive_definite(P_asym), "Should detect non-symmetric matrix"


def test_compute_lqr_gain_valid():
    """Test LQR gain computation with valid parameters"""
    v = 15.0
    b = 0.5
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, v, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B = np.array([[0], [0], [0], [b]])

    Q = np.diag([100, 10, 50, 5])
    R = np.array([[1.0]])

    K = compute_lqr_gain(A, B, Q, R)

    assert K is not None, "Should compute valid LQR gain"
    assert K.shape == (1, 4), f"Expected shape (1, 4), got {K.shape}"


def test_compute_lqr_gain_invalid():
    """Test LQR gain computation with parameters causing instability"""
    v = 15.0
    b = 0.5
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, v, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B = np.array([[0], [0], [0], [b]])

    # Very weak Q (might cause issues)
    Q = np.diag([0.01, 0.001, 0.01, 0.001])
    R = np.array([[1000.0]])  # Very high control penalty

    K = compute_lqr_gain(A, B, Q, R)

    # Either returns None or unstable gain
    if K is not None:
        A_cl = A - B @ K
        eigenvalues = np.linalg.eigvals(A_cl)
        # If not None, should still be unstable
        has_positive = np.any(np.real(eigenvalues) >= -1e-8)
        assert has_positive, "Expected unstable system or None return"


def test_boundary_case_zero_initial():
    """Test with zero initial condition (equilibrium)"""
    v = 15.0
    b = 0.5
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, v, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B = np.array([[0], [0], [0], [b]])

    Q = np.diag([100, 10, 50, 5])
    R = np.array([[1.0]])
    K = compute_lqr_gain(A, B, Q, R)
    A_cl = A - B @ K

    def dynamics(x, t):
        return (A_cl @ x).flatten()

    # Zero initial condition
    x0 = np.zeros(4)

    # Simulate
    t_span = np.linspace(0, 5, 100)
    trajectory = odeint(dynamics, x0, t_span)

    # Should stay at zero
    for x in trajectory:
        assert np.allclose(x, 0, atol=1e-10), "Trajectory should remain at equilibrium"


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
