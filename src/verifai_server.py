"""
VerifAI Server for Lane-Keeping Simulation
Provides interface between VerifAI and the lane-keeping simulator
"""

import numpy as np
from scipy.linalg import solve_continuous_are

try:
    from verifai.server import Server
    VERIFAI_AVAILABLE = True
except ImportError:
    VERIFAI_AVAILABLE = False
    # Dummy base class if VerifAI not available
    class Server:
        pass


class LaneKeepingServer(Server):
    """
    VerifAI Server wrapper for lane-keeping simulation

    This class provides the interface between VerifAI's sampling/falsification
    framework and our lane-keeping simulator.
    """

    def __init__(self, simulator, A, B):
        """
        Initialize server

        Args:
            simulator: LaneKeepingSimulator instance
            A: System dynamics matrix (4x4)
            B: Control input matrix (4x1)
        """
        if VERIFAI_AVAILABLE:
            super().__init__()

        self.simulator = simulator
        self.A = A
        self.B = B
        self.K = None  # LQR gain (will be set during optimization)
        self.Q = None
        self.R = None

    def set_controller_params(self, Q, R):
        """
        Compute and set LQR gain from Q, R matrices

        Args:
            Q: State cost matrix (4x4)
            R: Control cost matrix (1x1)

        Returns:
            bool: True if controller is stable, False otherwise
        """
        try:
            # Solve Riccati equation
            P = solve_continuous_are(self.A, self.B, Q, R)

            # Compute LQR gain
            self.K = np.linalg.inv(R) @ self.B.T @ P

            # Store parameters
            self.Q = Q
            self.R = R

            # Verify stability
            A_cl = self.A - self.B @ self.K
            eigenvalues = np.linalg.eigvals(A_cl)

            if not np.all(np.real(eigenvalues) < 0):
                print("Warning: Unstable controller detected")
                return False

            return True

        except Exception as e:
            print(f"Error computing LQR gain: {e}")
            return False

    def simulate(self, sample, **kwargs):
        """
        Run simulation from VerifAI/Scenic sample

        This is the main interface method called by VerifAI's falsifier.

        Args:
            sample: Dictionary with sampled parameters from Scenic/VerifAI
                   Expected keys: lateral_offset, lateral_velocity,
                                 heading_error, heading_rate
            **kwargs: Additional arguments (unused)

        Returns:
            Tuple (result_dict, error_value):
                - result_dict: Full simulation results
                - error_value: Scalar error metric (1.0 if violation, 0.0 if safe)
        """
        if self.K is None:
            return {'error': 'Controller not initialized'}, 1.0

        # Extract initial state from sample
        x0 = np.array([
            sample.get('lateral_offset', 0.0),
            sample.get('lateral_velocity', 0.0),
            sample.get('heading_error', 0.0),
            sample.get('heading_rate', 0.0)
        ])

        # Run simulation
        result = self.simulator.simulate(x0, self.K, t_max=10.0, dt=0.01)

        # Compute error metric for VerifAI
        # error = 1.0 means counterexample found (violation)
        # error = 0.0 means specification satisfied (safe)
        error_value = 1.0 if result['violation'] else 0.0

        # Add error to result dictionary
        result['error'] = error_value

        return result, error_value

    def get_controller_info(self):
        """
        Get current controller information

        Returns:
            dict: Controller parameters and properties
        """
        if self.K is None:
            return {'status': 'not_initialized'}

        A_cl = self.A - self.B @ self.K
        eigenvalues = np.linalg.eigvals(A_cl)

        return {
            'status': 'initialized',
            'K': self.K.tolist(),
            'Q': self.Q.tolist() if self.Q is not None else None,
            'R': self.R.tolist() if self.R is not None else None,
            'eigenvalues': eigenvalues.tolist(),
            'stable': bool(np.all(np.real(eigenvalues) < 0))
        }


# Alternative: Simple function-based interface (if not using VerifAI Server class)
def create_simulation_function(simulator, A, B):
    """
    Create a simulation function for use with VerifAI without Server class

    Args:
        simulator: LaneKeepingSimulator instance
        A, B: System matrices

    Returns:
        Function that takes (sample, Q, R) and returns (result, error)
    """
    def simulation_function(sample, Q, R):
        """
        Simulation function for VerifAI

        Args:
            sample: Dict with initial condition parameters
            Q, R: Controller parameters

        Returns:
            Tuple (result, error_value)
        """
        # Compute LQR gain
        try:
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P

            # Check stability
            A_cl = A - B @ K
            if not np.all(np.real(np.linalg.eigvals(A_cl)) < 0):
                return {'error': 'unstable'}, 1.0

        except:
            return {'error': 'computation_failed'}, 1.0

        # Extract initial state
        x0 = np.array([
            sample.get('lateral_offset', 0.0),
            sample.get('lateral_velocity', 0.0),
            sample.get('heading_error', 0.0),
            sample.get('heading_rate', 0.0)
        ])

        # Simulate
        result = simulator.simulate(x0, K, t_max=10.0)
        error_value = 1.0 if result['violation'] else 0.0

        return result, error_value

    return simulation_function


if __name__ == '__main__':
    # Test the server
    print("Testing LaneKeepingServer...")

    if not VERIFAI_AVAILABLE:
        print("VerifAI not installed - Server class available but untested")
    else:
        print(" VerifAI available - Server class ready")

    # Simple test
    from simulator import LaneKeepingSimulator

    v = 15.0
    b = 0.5
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, v, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B = np.array([[0], [0], [0], [b]])

    simulator = LaneKeepingSimulator(A, B, lane_width=3.5)
    server = LaneKeepingServer(simulator, A, B)

    # Set controller
    Q = np.diag([100, 10, 50, 5])
    R = np.array([[1.0]])

    success = server.set_controller_params(Q, R)
    print(f"Controller initialization: {' Success' if success else ' Failed'}")

    # Test simulation
    sample = {
        'lateral_offset': 0.3,
        'lateral_velocity': 0.05,
        'heading_error': 0.1,
        'heading_rate': 0.02
    }

    result, error = server.simulate(sample)
    print(f"Test simulation: error = {error:.1f} ({'violation' if error > 0.5 else 'safe'})")

    print("\n Server class test complete")
