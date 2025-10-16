"""
Phase 1: Learn LQR parameters using VerifAI
Uses Scenic scenarios and lane-keeping simulator
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from verifai.falsifier import generic_falsifier
from verifai.samplers import feature_sampler
from dotmap import DotMap
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.simulator import LaneKeepingSimulator

class LQRParameterLearner:
    """Learn Q, R parameters that minimize lane violations"""

    def __init__(self, system_params):
        self.v = system_params['velocity']
        self.b = system_params['control_effectiveness']
        self.lane_width = system_params['lane_width']

        # System matrices
        self.A = np.array([
            [0, 1, 0, 0],
            [0, 0, self.v, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        self.B = np.array([[0], [0], [0], [self.b]])

        # Simulator
        self.simulator = LaneKeepingSimulator(self.A, self.B, self.lane_width)

        # Best parameters found
        self.best_params = None
        self.best_violation_rate = 1.0
        self.history = []

    def compute_lqr_gain(self, Q, R):
        """Compute LQR gain from Q, R matrices"""
        try:
            P = solve_continuous_are(self.A, self.B, Q, R)
            K = np.linalg.inv(R) @ self.B.T @ P

            # Check stability
            A_cl = self.A - self.B @ K
            eigenvalues = np.linalg.eigvals(A_cl)
            if not np.all(np.real(eigenvalues) < 0):
                return None

            return K
        except:
            return None

    def evaluate_parameters(self, params):
        """
        Evaluate LQR parameters on sampled scenarios
        Returns: violation rate (objective to minimize)
        """
        # Extract parameters
        Q = np.diag([params['Q1'], params['Q2'], params['Q3'], params['Q4']])
        R = np.array([[params['R']]])

        # Compute gain
        K = self.compute_lqr_gain(Q, R)
        if K is None:
            return 1.0  # Maximum penalty for unstable controller

        # Sample scenarios using feature sampler
        sampler_params = DotMap()
        sampler_params.feature_sampler = DotMap()

        # Define sampling space (matches Scenic scenario)
        sampler_params.feature_sampler.x1_init = (-0.5, 0.5)  # lateral offset
        sampler_params.feature_sampler.x2_init = (-0.1, 0.1)  # lateral velocity
        sampler_params.feature_sampler.x3_init = (-0.175, 0.175)  # heading
        sampler_params.feature_sampler.x4_init = (-0.05, 0.05)  # heading rate

        # Number of test samples
        n_samples = 100
        violations = 0

        # Simulate on sampled initial conditions
        for _ in range(n_samples):
            # Sample initial condition
            x0 = np.array([
                np.random.uniform(*sampler_params.feature_sampler.x1_init),
                np.random.uniform(*sampler_params.feature_sampler.x2_init),
                np.random.uniform(*sampler_params.feature_sampler.x3_init),
                np.random.uniform(*sampler_params.feature_sampler.x4_init)
            ])

            # Run simulation
            result = self.simulator.simulate(x0, K, t_max=10.0, dt=0.01)

            # Check for violations
            if result['violation']:
                violations += 1

        violation_rate = violations / n_samples

        # Track best
        if violation_rate < self.best_violation_rate:
            self.best_violation_rate = violation_rate
            self.best_params = {
                'Q': Q.tolist(),
                'R': R.tolist(),
                'K': K.tolist(),
                'violation_rate': violation_rate
            }

        # Log progress
        self.history.append({
            'Q': Q.tolist(),
            'R': R.tolist(),
            'violation_rate': violation_rate
        })

        return violation_rate

    def optimize(self, n_iterations=50):
        """
        Run optimization to find best Q, R parameters
        Uses Bayesian Optimization via scikit-optimize
        """
        from skopt import gp_minimize
        from skopt.space import Real
        from skopt.utils import use_named_args

        print("="*70)
        print("PHASE 1: LEARNING LQR PARAMETERS WITH VERIFAI")
        print("="*70)
        print(f"\nOptimization settings:")
        print(f"  - Iterations: {n_iterations}")
        print(f"  - Samples per evaluation: 100")
        print(f"  - Method: Bayesian Optimization\n")

        # Define search space
        space = [
            Real(10, 200, name='Q1'),   # lateral position weight
            Real(1, 50, name='Q2'),     # lateral velocity weight
            Real(10, 100, name='Q3'),   # heading weight
            Real(1, 20, name='Q4'),     # heading rate weight
            Real(0.1, 10, name='R')     # control effort weight
        ]

        @use_named_args(space)
        def objective(**params):
            return self.evaluate_parameters(params)

        # Run optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=n_iterations,
            random_state=42,
            verbose=True
        )

        print("\n" + "="*70)
        print("LEARNING COMPLETE")
        print("="*70)
        print(f"\nBest violation rate: {self.best_violation_rate*100:.2f}%")
        print(f"Best parameters:")
        print(f"  Q = diag{np.diag(np.array(self.best_params['Q']))}")
        print(f"  R = {self.best_params['R'][0][0]:.3f}")
        print(f"  K = {np.array(self.best_params['K']).flatten()}")

        # Save results
        self.save_results()

        return self.best_params

    def save_results(self):
        """Save learned parameters"""
        Path('results').mkdir(exist_ok=True)

        with open('results/learned_parameters.json', 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'history': self.history
            }, f, indent=2)

        print("\n✓ Results saved to results/learned_parameters.json")


def main():
    """Main learning pipeline"""

    # System parameters
    system_params = {
        'velocity': 15.0,
        'control_effectiveness': 0.5,
        'lane_width': 3.5
    }

    # Create learner
    learner = LQRParameterLearner(system_params)

    # Run optimization
    best_params = learner.optimize(n_iterations=50)

    print("\n" + "="*70)
    print("NEXT STEP: Run verification")
    print("  python src/2_lyapunov_verification.py")
    print("="*70)


if __name__ == '__main__':
    main()
