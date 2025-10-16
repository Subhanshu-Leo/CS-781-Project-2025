"""
Phase 3: Statistical Analysis with Confidence Intervals
Run Monte Carlo simulation on unverified region
"""

import numpy as np
from scipy import stats
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.simulator import LaneKeepingSimulator

class StatisticalAnalyzer:
    """Statistical analysis for unverified regions"""

    def __init__(self, system_params, learned_params):
        self.v = system_params['velocity']
        self.b = system_params['control_effectiveness']
        self.lane_width = system_params['lane_width']

        # System
        self.A = np.array([
            [0, 1, 0, 0],
            [0, 0, self.v, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        self.B = np.array([[0], [0], [0], [self.b]])

        # Controller
        self.K = np.array(learned_params['K'])

        # Simulator
        self.simulator = LaneKeepingSimulator(self.A, self.B, self.lane_width)

    def monte_carlo_analysis(self, initial_bounds, n_samples=10000,
                             verified_region=None):
        """
        Monte Carlo simulation with confidence intervals

        Args:
            initial_bounds: Dict with x1, x2, x3, x4 ranges
            n_samples: Number of Monte Carlo samples
            verified_region: If provided, only sample outside this (P, c_max)

        Returns:
            dict with violation rate and confidence intervals
        """
        print("="*70)
        print("PHASE 3: STATISTICAL ANALYSIS")
        print("="*70)
        print(f"\nMonte Carlo settings:")
        print(f"  - Samples: {n_samples}")
        print(f"  - Confidence level: 95%\n")

        violations = 0
        samples_analyzed = 0
        violation_times = []
        max_laterals = []

        # If verified region provided, filter samples
        if verified_region:
            P, c_max = verified_region
            print(f"Focusing on UNVERIFIED region (V(x) > {c_max:.2f})")

        np.random.seed(42)  # Reproducibility

        for i in range(n_samples):
            # Sample initial condition
            x0 = np.array([
                np.random.uniform(*initial_bounds['x1']),
                np.random.uniform(*initial_bounds['x2']),
                np.random.uniform(*initial_bounds['x3']),
                np.random.uniform(*initial_bounds['x4'])
            ])

            # Check if in verified region
            if verified_region:
                P, c_max = verified_region
                V = x0.T @ P @ x0
                if V <= c_max:
                    continue  # Skip verified samples

            samples_analyzed += 1

            # Simulate
            result = self.simulator.simulate(x0, self.K, t_max=10.0)

            if result['violation']:
                violations += 1
                violation_times.append(result['violation_time'])

            max_laterals.append(result['max_lateral'])

            # Progress
            if (i+1) % 1000 == 0:
                print(f"  Processed {i+1}/{n_samples} samples...")

        if samples_analyzed == 0:
            print("\n✓ All samples in verified region!")
            return None

        # Compute statistics
        violation_rate = violations / samples_analyzed

        # Wilson score confidence interval (better for proportions)
        z = stats.norm.ppf(0.975)  # 95% CI
        n = samples_analyzed
        p_hat = violation_rate

        denominator = 1 + z**2/n
        center = (p_hat + z**2/(2*n)) / denominator
        margin = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denominator

        ci_lower = max(0, center - margin)
        ci_upper = min(1, center + margin)

        print("\n" + "="*70)
        print("STATISTICAL RESULTS")
        print("="*70)
        print(f"\nSamples analyzed: {samples_analyzed}")
        print(f"Violations found: {violations}")
        print(f"\nViolation Rate: {violation_rate*100:.3f}%")
        print(f"95% Confidence Interval: [{ci_lower*100:.3f}%, {ci_upper*100:.3f}%]")

        if violations > 0:
            print(f"\nViolation Statistics:")
            print(f"  Mean time to violation: {np.mean(violation_times):.2f} s")
            print(f"  Earliest violation: {np.min(violation_times):.2f} s")

        # Save results
        results = {
            'samples_analyzed': samples_analyzed,
            'violations': violations,
            'violation_rate': violation_rate,
            'confidence_interval_95': {
                'lower': ci_lower,
                'upper': ci_upper
            },
            'method': 'Wilson Score Interval',
            'statistics': {
                'mean_max_lateral': float(np.mean(max_laterals)),
                'std_max_lateral': float(np.std(max_laterals)),
                'max_lateral_observed': float(np.max(max_laterals))
            }
        }

        if violations > 0:
            results['violation_statistics'] = {
                'mean_time': float(np.mean(violation_times)),
                'std_time': float(np.std(violation_times)),
                'min_time': float(np.min(violation_times)),
                'max_time': float(np.max(violation_times))
            }

        # Save
        with open('results/statistical_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("\n✓ Results saved to results/statistical_analysis.json")

        # Visualization
        self.plot_statistics(max_laterals, violation_rate)

        return results

    def plot_statistics(self, max_laterals, violation_rate):
        """Plot statistical analysis results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of maximum lateral deviations
        ax1.hist(max_laterals, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(x=self.lane_width/2, color='red', linestyle='--',
                   linewidth=2, label='Lane Boundary')
        ax1.set_xlabel('Maximum Lateral Deviation (m)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Maximum Lateral Deviations', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Violation rate visualization
        categories = ['Safe', 'Violations']
        values = [(1-violation_rate)*100, violation_rate*100]
        colors = ['green', 'red']
        ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_title(f'Safety Analysis\n({violation_rate*100:.2f}% violation rate)',
                     fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('results/statistical_analysis.png', dpi=300)
        print("✓ Visualization saved to results/statistical_analysis.png")
        plt.close()


def main():
    """Run statistical analysis"""

    # Load learned parameters
    with open('results/learned_parameters.json', 'r') as f:
        data = json.load(f)
        learned_params = data['best_params']

    # Load verification results (if exists)
    verified_region = None
    try:
        with open('results/verification_certificate.json', 'r') as f:
            cert = json.load(f)
            if cert['verification_result']['status'] == 'PARTIAL':
                # Need to load P matrix and c_max
                # This requires running verification first
                print("Partial verification detected - analyzing unverified region")
    except FileNotFoundError:
        print("No verification results found - analyzing full input space")

    # System parameters
    system_params = {
        'velocity': 15.0,
        'control_effectiveness': 0.5,
        'lane_width': 3.5
    }

    # Initial condition bounds
    initial_bounds = {
        'x1': (-0.5, 0.5),
        'x2': (-0.1, 0.1),
        'x3': (-0.175, 0.175),
        'x4': (-0.05, 0.05)
    }

    # Run analysis
    analyzer = StatisticalAnalyzer(system_params, learned_params)
    results = analyzer.monte_carlo_analysis(
        initial_bounds,
        n_samples=10000,
        verified_region=verified_region
    )


if __name__ == '__main__':
    main()
