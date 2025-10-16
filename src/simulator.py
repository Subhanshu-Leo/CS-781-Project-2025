"""
Lane-keeping simulator
Simulates vehicle under LQR control
"""

import numpy as np
from scipy.integrate import odeint

class LaneKeepingSimulator:
    """Simple lane-keeping simulator for verification"""

    def __init__(self, A, B, lane_width):
        self.A = A
        self.B = B
        self.lane_width = lane_width
        self.half_width = lane_width / 2

    def dynamics(self, x, t, K):
        """
        Closed-loop dynamics: ẋ = (A - BK)x
        """
        u = -K @ x
        xdot = self.A @ x + self.B @ u
        return xdot.flatten()

    def simulate(self, x0, K, t_max=10.0, dt=0.01):
        """
        Simulate trajectory from initial condition

        Args:
            x0: Initial state [x1, x2, x3, x4]
            K: LQR gain matrix
            t_max: Simulation time (seconds)
            dt: Time step

        Returns:
            dict with 'trajectory', 'time', 'violation', 'max_lateral'
        """
        t_span = np.arange(0, t_max, dt)

        # Integrate
        trajectory = odeint(self.dynamics, x0, t_span, args=(K,))

        # Check for violations
        lateral_positions = trajectory[:, 0]
        max_lateral = np.max(np.abs(lateral_positions))
        violation = max_lateral > self.half_width

        # Find time of first violation (if any)
        violation_time = None
        if violation:
            violation_idx = np.where(np.abs(lateral_positions) > self.half_width)[0][0]
            violation_time = t_span[violation_idx]

        return {
            'trajectory': trajectory,
            'time': t_span,
            'violation': violation,
            'max_lateral': max_lateral,
            'violation_time': violation_time
        }

    def batch_simulate(self, initial_conditions, K, t_max=10.0):
        """
        Simulate multiple initial conditions

        Args:
            initial_conditions: List of x0 arrays
            K: LQR gain
            t_max: Simulation time

        Returns:
            dict with violation statistics
        """
        results = []
        violations = 0

        for x0 in initial_conditions:
            result = self.simulate(x0, K, t_max)
            results.append(result)
            if result['violation']:
                violations += 1

        violation_rate = violations / len(initial_conditions)

        return {
            'results': results,
            'violation_rate': violation_rate,
            'n_violations': violations,
            'n_samples': len(initial_conditions)
        }
