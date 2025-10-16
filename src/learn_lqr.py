# learn_lqr_parameters.py

import numpy as np
from verifai.samplers import ScenicSampler
from verifai.falsifier import generic_falsifier
from dotmap import DotMap
import json

class LQRLearner:
    def __init__(self, A, B, initial_Q, initial_R):
        self.A = A
        self.B = B
        self.best_Q = initial_Q
        self.best_R = initial_R
        self.best_violation_rate = 1.0
        self.history = []

    def objective(self, params):
        """
        Objective function: violation rate given Q, R parameters
        """
        # Extract parameters
        Q = np.diag([params['Q11'], params['Q22'],
                     params['Q33'], params['Q44']])
        R = np.array([[params['R']]])

        # Compute LQR gain
        try:
            P = solve_continuous_are(self.A, self.B, Q, R)
            K = np.linalg.inv(R) @ self.B.T @ P
        except np.linalg.LinAlgError:
            return {'violation_rate': 1.0, 'valid': False}

        # Check stability
        A_cl = self.A - self.B @ K
        eigenvalues = np.linalg.eigvals(A_cl)
        if not np.all(np.real(eigenvalues) < 0):
            return {'violation_rate': 1.0, 'valid': False}

        # Run simulations
        violations = 0
        n_samples = 100

        scenic_sampler = ScenicSampler.fromScenario(
            'lane_keeping_scenario.scenic',
            maxIterations=n_samples
        )

        for i in range(n_samples):
            sample = scenic_sampler.nextSample()
            result = self.simulate(K, sample)
            if result['violation']:
                violations += 1

        violation_rate = violations / n_samples

        # Track best
        if violation_rate < self.best_violation_rate:
            self.best_Q = Q
            self.best_R = R
            self.best_violation_rate = violation_rate

        self.history.append({
            'Q': Q.tolist(),
            'R': R.tolist(),
            'violation_rate': violation_rate
        })

        return {'violation_rate': violation_rate, 'valid': True}

    def simulate(self, K, scenario_params):
        """
        Simulate single trajectory and check for violations
        """
        from scipy.integrate import odeint

        def dynamics(x, t):
            u = -K @ x
            return (self.A @ x + self.B @ u).flatten()

        # Initial condition from scenario
        x0 = np.array([
            scenario_params['ego_init_offset'],
            0.0,  # assume zero initial lateral velocity
            scenario_params['ego_init_heading'],
            0.0   # assume zero initial heading rate
        ])

        # Simulate
        t_span = np.linspace(0, 10, 1000)
        trajectory = odeint(dynamics, x0, t_span)

        # Check violations
        lane_width = 3.5
        violation = np.any(np.abs(trajectory[:, 0]) > lane_width/2)

        return {'violation': violation, 'trajectory': trajectory}

    def optimize(self, n_iterations=100):
        """
        Run optimization loop
        """
        from scipy.optimize import differential_evolution

        print("Starting LQR parameter optimization...")

        bounds = [
            (10, 200),   # Q11
            (1, 20),     # Q22
            (10, 100),   # Q33
            (1, 10),     # Q44
            (0.1, 10)    # R
        ]

        def objective_wrapper(x):
            params = {
                'Q11': x[0], 'Q22': x[1],
                'Q33': x[2], 'Q44': x[3],
                'R': x[4]
            }
            result = self.objective(params)
            return result['violation_rate']

        result = differential_evolution(
            objective_wrapper,
            bounds,
            maxiter=n_iterations,
            popsize=15,
            disp=True
        )

        print(f"\nOptimization complete!")
        print(f"Best violation rate: {self.best_violation_rate*100:.2f}%")
        print(f"Best Q: diag({np.diag(self.best_Q)})")
        print(f"Best R: {self.best_R[0,0]:.3f}")

        # Save results
        self.save_results()

        return self.best_Q, self.best_R

    def save_results(self):
        """
        Save learned parameters and history
        """
        results = {
            'best_Q': self.best_Q.tolist(),
            'best_R': self.best_R.tolist(),
            'best_violation_rate': self.best_violation_rate,
            'history': self.history
        }

        with open('results/learned_parameters.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("\n✓ Results saved to results/learned_parameters.json")

# Main execution
if __name__ == '__main__':
    # System definition
    v = 15.0
    b = 0.5
    A = np.array([[0, 1, 0, 0],
                  [0, 0, v, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])
    B = np.array([[0], [0], [0], [b]])

    # Initial guess
    Q_init = np.diag([50, 5, 25, 2])
    R_init = np.array([[1.0]])

    # Learn
    learner = LQRLearner(A, B, Q_init, R_init)
    Q_learned, R_learned = learner.optimize(n_iterations=50)
