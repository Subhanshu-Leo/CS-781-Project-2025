"""
Baseline Comparison: Compare learned controller with hand-tuned baseline
Shows improvement achieved through VerifAI learning
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.simulator import LaneKeepingSimulator
from src.utils import load_learned_parameters


def evaluate_controller(Q, R, simulator, n_samples=1000):
    """
    Evaluate a controller on random initial conditions

    Args:
        Q: State cost matrix
        R: Control cost matrix
        simulator: LaneKeepingSimulator instance
        n_samples: Number of test samples

    Returns:
        dict with violation_rate, max_lateral, mean_lateral
    """
    from scipy.linalg import solve_continuous_are

    # Compute LQR gain
    A = simulator.A
    B = simulator.B

    try:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P

        # Check stability
        A_cl = A - B @ K
        if not np.all(np.real(np.linalg.eigvals(A_cl)) < 0):
            return {
                'violation_rate': 1.0,
                'max_lateral': float('inf'),
                'mean_lateral': float('inf'),
                'stable': False
            }
    except:
        return {
            'violation_rate': 1.0,
            'max_lateral': float('inf'),
            'mean_lateral': float('inf'),
            'stable': False
        }

    # Sample initial conditions
    np.random.seed(42)
    violations = 0
    max_laterals = []

    for _ in range(n_samples):
        x0 = np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.175, 0.175),
            np.random.uniform(-0.05, 0.05)
        ])

        result = simulator.simulate(x0, K, t_max=10.0, dt=0.01)

        if result['violation']:
            violations += 1

        max_laterals.append(result['max_lateral'])

    return {
        'violation_rate': violations / n_samples,
        'max_lateral': np.max(max_laterals),
        'mean_lateral': np.mean(max_laterals),
        'std_lateral': np.std(max_laterals),
        'stable': True,
        'K': K.tolist()
    }


def compare_with_baselines():
    """
    Compare learned controller with multiple baseline controllers
    """
    print("\n" + "="*70)
    print("BASELINE COMPARISON ANALYSIS")
    print("="*70)

    # Setup system
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

    # Load learned parameters
    try:
        Q_learned, R_learned, K_learned, learned_violation_rate = load_learned_parameters()
        print("\nLoaded learned controller from Phase 1")
    except FileNotFoundError:
        print("\n Error: Learned parameters not found")
        print("Please run Phase 1 first: python src/1_learn_parameters.py")
        return

    # Define baseline controllers
    baselines = {
        'Baseline 1 (Conservative)': {
            'Q': np.diag([50, 5, 25, 2]),
            'R': np.array([[1.0]]),
            'description': 'Conservative hand-tuned from literature'
        },
        'Baseline 2 (Aggressive)': {
            'Q': np.diag([150, 15, 75, 10]),
            'R': np.array([[0.5]]),
            'description': 'Aggressive control with high state penalties'
        },
        'Baseline 3 (Balanced)': {
            'Q': np.diag([100, 10, 50, 5]),
            'R': np.array([[1.0]]),
            'description': 'Balanced baseline (typical default)'
        },
        'Baseline 4 (Minimal Control)': {
            'Q': np.diag([30, 3, 15, 1]),
            'R': np.array([[2.0]]),
            'description': 'High control penalty (energy efficient)'
        }
    }

    # Evaluate all controllers
    print("\n" + "-"*70)
    print("Evaluating controllers on 1000 random scenarios...")
    print("-"*70)

    results = {}

    # Evaluate learned controller
    print("\nLearned Controller (VerifAI Optimized):")
    learned_result = evaluate_controller(Q_learned, R_learned, simulator, n_samples=1000)
    results['Learned (VerifAI)'] = learned_result
    print(f"  Violation rate: {learned_result['violation_rate']*100:.2f}%")
    print(f"  Max lateral deviation: {learned_result['max_lateral']:.4f} m")
    print(f"  Mean lateral deviation: {learned_result['mean_lateral']:.4f} m")

    # Evaluate baselines
    for name, config in baselines.items():
        print(f"\n{name}:")
        print(f"  {config['description']}")
        result = evaluate_controller(config['Q'], config['R'], simulator, n_samples=1000)
        results[name] = result

        if result['stable']:
            print(f"  Violation rate: {result['violation_rate']*100:.2f}%")
            print(f"  Max lateral deviation: {result['max_lateral']:.4f} m")
            print(f"  Mean lateral deviation: {result['mean_lateral']:.4f} m")
        else:
            print(f"   UNSTABLE CONTROLLER")

    # Compute improvements
    print("\n" + "="*70)
    print("IMPROVEMENT ANALYSIS")
    print("="*70)

    for name, result in results.items():
        if name == 'Learned (VerifAI)' or not result['stable']:
            continue

        improvement = (result['violation_rate'] - learned_result['violation_rate']) / result['violation_rate'] * 100

        print(f"\nLearned vs {name}:")
        print(f"  Violation rate improvement: {improvement:.1f}%")
        print(f"  Baseline: {result['violation_rate']*100:.2f}% violations")
        print(f"  Learned:  {learned_result['violation_rate']*100:.2f}% violations")

    # Find best baseline
    stable_baselines = {k: v for k, v in results.items()
                       if k != 'Learned (VerifAI)' and v['stable']}

    if stable_baselines:
        best_baseline_name = min(stable_baselines, key=lambda k: stable_baselines[k]['violation_rate'])
        best_baseline = stable_baselines[best_baseline_name]

        print("\n" + "="*70)
        print(f"BEST BASELINE: {best_baseline_name}")
        print(f"  Violation rate: {best_baseline['violation_rate']*100:.2f}%")
        print(f"\nLEARNED CONTROLLER:")
        print(f"  Violation rate: {learned_result['violation_rate']*100:.2f}%")

        if learned_result['violation_rate'] < best_baseline['violation_rate']:
            improvement = (best_baseline['violation_rate'] - learned_result['violation_rate']) / best_baseline['violation_rate'] * 100
            print(f"\n✓LEARNED CONTROLLER IS BETTER")
            print(f"  Improvement: {improvement:.1f}%")
        else:
            print(f"\n⚠ Best baseline performs better")

    # Visualization
    generate_comparison_plots(results)

    # Save results
    Path('results').mkdir(exist_ok=True)

    # Prepare for JSON serialization
    results_serializable = {}
    for name, result in results.items():
        results_serializable[name] = {
            'violation_rate': float(result['violation_rate']),
            'max_lateral': float(result['max_lateral']),
            'mean_lateral': float(result['mean_lateral']),
            'stable': bool(result['stable'])
        }

    with open('results/baseline_comparison.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print("\nResults saved to results/baseline_comparison.json")
    print("Plots saved to results/baseline_comparison.png")


def generate_comparison_plots(results):
    """Generate comparison visualizations"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Filter stable controllers
    names = []
    violation_rates = []
    max_laterals = []
    colors = []

    for name, result in results.items():
        if result['stable']:
            names.append(name.replace(' (VerifAI)', '\n(VerifAI)').replace('Baseline ', 'B'))
            violation_rates.append(result['violation_rate'] * 100)
            max_laterals.append(result['max_lateral'])

            # Color learned controller differently
            if 'Learned' in name or 'VerifAI' in name:
                colors.append('green')
            else:
                colors.append('steelblue')

    # Plot 1: Violation rates
    bars1 = ax1.bar(range(len(names)), violation_rates, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Controller', fontsize=12)
    ax1.set_ylabel('Violation Rate (%)', fontsize=12)
    ax1.set_title('Violation Rate Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Annotate bars
    for i, (bar, rate) in enumerate(zip(bars1, violation_rates)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

    # Plot 2: Max lateral deviation
    bars2 = ax2.bar(range(len(names)), max_laterals, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=1.75, color='red', linestyle='--', linewidth=2, label='Lane Boundary')
    ax2.set_xlabel('Controller', fontsize=12)
    ax2.set_ylabel('Max Lateral Deviation (m)', fontsize=12)
    ax2.set_title('Maximum Lateral Deviation', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    # Annotate bars
    for i, (bar, lateral) in enumerate(zip(bars2, max_laterals)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{lateral:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('results/baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    compare_with_baselines()
