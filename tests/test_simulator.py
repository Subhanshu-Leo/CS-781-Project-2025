"""
Unit tests for lane-keeping simulator
"""

import pytest
import numpy as np
from scipy.linalg import solve_continuous_are
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.simulator import LaneKeepingSimulator

def test_simulator_stable_controller():
    """Test simulator with stable LQR controller"""
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
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    # Simulator
    simulator = LaneKeepingSimulator(A, B, lane_width=3.5)

    # Small initial condition - should not violate
    x0 = np.array([0.1, 0.01, 0.05, 0.01])
    result = simulator.simulate(x0, K, t_max=5.0)

    assert result['violation'] == False
    assert result['max_lateral'] < 1.75

def test_simulator_large_initial_condition():
    """Test simulator with large initial condition"""
    v = 15.0
    b = 0.5
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, v, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B = np.array([[0], [0], [0], [b]])

    # Weak controller (may violate)
    Q = np.diag([10, 1, 5, 1])
    R = np.array([[10.0]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    simulator = LaneKeepingSimulator(A, B, lane_width=3.5)

    # Large initial condition
    x0 = np.array([1.5, 0.2, 0.3, 0.1])
    result = simulator.simulate(x0, K, t_max=5.0)

    # Should track trajectory (may or may not violate)
    assert 'trajectory' in result
    assert 'violation' in result

def test_batch_simulation():
    """Test batch simulation"""
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
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    simulator = LaneKeepingSimulator(A, B, lane_width=3.5)

    # Multiple initial conditions
    initial_conditions = [
        np.array([0.1, 0, 0.05, 0]),
        np.array([0.2, 0.05, 0.1, 0.02]),
        np.array([-0.15, -0.03, -0.08, -0.01])
    ]

    batch_result = simulator.batch_simulate(initial_conditions, K, t_max=5.0)

    assert batch_result['n_samples'] == 3
    assert 'violation_rate' in batch_result
    assert 0 <= batch_result['violation_rate'] <= 1
