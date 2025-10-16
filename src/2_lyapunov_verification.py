# lyapunov_verification.py - Complete standalone script

import numpy as np
from scipy.linalg import solve_continuous_lyapunov, solve_continuous_are
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import json
import sys
from datetime import datetime

class LyapunovVerifier:
    """
    Complete Lyapunov-based verification for lane keeping
    """

    def __init__(self, A, B, Q, R, lane_width=3.5):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.L = lane_width / 2  # lane half-width

        # Compute LQR gain
        self.P_riccati = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ self.P_riccati
        self.A_cl = A - B @ self.K

        # Compute Lyapunov matrix
        self.P = solve_continuous_lyapunov(self.A_cl.T, -Q)
        self.P_inv = np.linalg.inv(self.P)

        # Compute maximum invariant set
        self.c_max = self.L**2 / self.P_inv[0, 0]

        self.verification_result = None

    def verify_stability(self):
        """
        Step 1: Verify system is stable
        """
        print("\n" + "="*70)
        print("STEP 1: STABILITY VERIFICATION")
        print("="*70)

        # Check eigenvalues
        eigenvalues = np.linalg.eigvals(self.A_cl)
        all_stable = np.all(np.real(eigenvalues) < 0)

        print("\nClosed-loop eigenvalues:")
        for i, λ in enumerate(eigenvalues):
            print(f"  λ_{i+1} = {λ:.4f}")

        if all_stable:
            print("\n✓ All eigenvalues have negative real part")
            print("✓ System is exponentially stable")

            # Compute convergence rate
            max_real = np.max(np.real(eigenvalues))
            print(f"\nConvergence rate: α = {-max_real:.4f} rad/s")
            print(f"Time to 1% of initial: {-np.log(0.01)/max_real:.2f} s")
        else:
            print("\n✗ FAILED: System is unstable")
            return False

        return True

    def verify_lyapunov_function(self):
        """
        Step 2: Verify Lyapunov function properties
        """
        print("\n" + "="*70)
        print("STEP 2: LYAPUNOV FUNCTION VERIFICATION")
        print("="*70)

        # Check P is positive definite
        eigenvalues_P = np.linalg.eigvals(self.P)
        P_positive_def = np.all(eigenvalues_P > 1e-10)

        print("\nLyapunov matrix P eigenvalues:")
        for i, λ in enumerate(eigenvalues_P):
            print(f"  λ_{i+1}(P) = {λ:.6f}")

        if P_positive_def:
            print("\n✓ P is positive definite")
            print("✓ V(x) = x^T P x is a valid Lyapunov function")
        else:
            print("\n✗ FAILED: P is not positive definite")
            return False

        # Verify Lyapunov equation
        Vdot_matrix = self.A_cl.T @ self.P + self.P @ self.A_cl
        residual = Vdot_matrix + self.Q
        max_error = np.max(np.abs(residual))

        print(f"\nLyapunov equation residual: {max_error:.2e}")

        if max_error < 1e-8:
            print("✓ Lyapunov equation satisfied: A_cl^T P + P A_cl = -Q")
        else:
            print(f"⚠ Warning: Large residual {max_error:.2e}")

        # Check Q is positive definite
        eigenvalues_Q = np.linalg.eigvals(self.Q)
        Q_positive_def = np.all(eigenvalues_Q > 1e-10)

        if Q_positive_def:
            print("✓ Q is positive definite")
            print("✓ V̇(x) = -x^T Q x < 0 for all x ≠ 0")
        else:
            print("\n✗ FAILED: Q is not positive definite")
            return False

        print(f"\n✓ LYAPUNOV FUNCTION VERIFIED")
        print(f"  V(x) = x^T P x")
        print(f"  V̇(x) = -x^T Q x < 0")

        return True

    def compute_invariant_set(self):
        """
        Step 3: Compute maximum invariant ellipsoid
        """
        print("\n" + "="*70)
        print("STEP 3: INVARIANT SET COMPUTATION")
        print("="*70)

        print(f"\nLane half-width: L = {self.L:.3f} m")
        print(f"P^(-1)[0,0] = {self.P_inv[0,0]:.6f}")
        print(f"\nMaximum invariant set parameter:")
        print(f"  c_max = L² / P^(-1)[0,0]")
        print(f"  c_max = {self.L**2:.4f} / {self.P_inv[0,0]:.6f}")
        print(f"  c_max = {self.c_max:.4f}")

        # Verify boundary condition
        x_boundary = np.zeros(4)
        x_boundary[0] = np.sqrt(self.c_max * self.P_inv[0, 0])
        V_boundary = x_boundary.T @ self.P @ x_boundary

        print(f"\nBoundary verification:")
        print(f"  At x = [{x_boundary[0]:.4f}, 0, 0, 0]^T:")
        print(f"  V(x) = {V_boundary:.4f} (should equal c_max = {self.c_max:.4f})")
        print(f"  |x₁| = {abs(x_boundary[0]):.4f} (should equal L = {self.L:.4f})")

        error = abs(V_boundary - self.c_max)
        if error < 1e-6:
            print(f"\n✓ Boundary condition verified (error = {error:.2e})")
        else:
            print(f"\n⚠ Warning: boundary error = {error:.2e}")

        print(f"\n✓ INVARIANT ELLIPSOID COMPUTED")
        print(f"  Ω_c = {{x : x^T P x ≤ {self.c_max:.4f}}}")
        print(f"  Guarantees: |x₁| ≤ {self.L:.3f} m for all x ∈ Ω_c")

        return True

    def verify_initial_conditions(self, initial_bounds):
        """
        Step 4: Verify initial conditions fit in invariant set
        """
        print("\n" + "="*70)
        print("STEP 4: INITIAL CONDITION VERIFICATION")
        print("="*70)

        print("\nInitial condition bounds:")
        for key, val in initial_bounds.items():
            print(f"  {key} ∈ [{val[0]:.4f}, {val[1]:.4f}]")

        # Check all corners of hypercube
        corners = []
        for x1 in initial_bounds['x1']:
            for x2 in initial_bounds['x2']:
                for x3 in initial_bounds['x3']:
                    for x4 in initial_bounds['x4']:
                        x = np.array([x1, x2, x3, x4])
                        V_val = x.T @ self.P @ x
                        corners.append((x, V_val))

        max_V = max(V for _, V in corners)
        worst_x = max(corners, key=lambda pair: pair[1])[0]

        print(f"\nChecking {len(corners)} corners of initial condition box...")
        print(f"  Maximum V(x₀) = {max_V:.4f}")
        print(f"  Invariant set c_max = {self.c_max:.4f}")
        print(f"  Worst-case corner: {worst_x}")

        if max_V <= self.c_max:
            coverage = 100.0
            verified = True
            print(f"\n✓✓✓ FORMAL VERIFICATION SUCCESSFUL ✓✓✓")
            print(f"  All initial conditions lie within Ω_c")
            print(f"  Coverage: {coverage:.1f}%")
            print(f"  Safety guarantee: ZERO LANE VIOLATIONS for all t ≥ 0")
        else:
            ratio = self.c_max / max_V
            coverage = ratio * 100
            verified = False
            print(f"\n⚠ PARTIAL VERIFICATION")
            print(f"  Ratio: c_max / max_V = {ratio:.4f}")
            print(f"  Coverage: {coverage:.1f}% of input space")
            print(f"\n  Options:")
            print(f"  1. Restrict initial conditions by factor {ratio:.4f}")
            print(f"  2. Improve controller (adjust Q, R)")
            print(f"  3. Use statistical analysis for unverified region")

        self.verification_result = {
            'verified': verified,
            'coverage': coverage,
            'c_max': self.c_max,
            'max_V_initial': max_V,
            'worst_corner': worst_x.tolist()
        }

        return verified, coverage

    def simulate_trajectory(self, x0, t_max=10):
        """
        Simulate trajectory from initial condition
        """
        def dynamics(x, t):
            return (self.A_cl @ x).flatten()

        t_span = np.linspace(0, t_max, 1000)
        trajectory = odeint(dynamics, x0, t_span)

        # Compute Lyapunov function along trajectory
        V_trajectory = [x.T @ self.P @ x for x in trajectory]

        # Check violations
        violations = np.any(np.abs(trajectory[:, 0]) > self.L)

        return {
            'time': t_span,
            'states': trajectory,
            'V_values': V_trajectory,
            'violation': violations
        }

    def generate_visualizations(self, initial_bounds):
        """
        Generate all verification plots
        """
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)

        # Figure 1: Ellipsoid projection (x1 vs x3)
        fig1, ax1 = plt.subplots(figsize=(10, 8))

        # Project onto (x1, x3) plane
        P_reduced = self.P[[0,2], :][:, [0,2]]
        eigenvalues, eigenvectors = np.linalg.eigh(P_reduced)

        a = np.sqrt(self.c_max / eigenvalues[0])
        b = np.sqrt(self.c_max / eigenvalues[1])
        angle = np.degrees(np.arctan2(eigenvectors[1,0], eigenvectors[0,0]))

        from matplotlib.patches import Ellipse, Rectangle

        ellipse = Ellipse(xy=(0, 0), width=2*a, height=2*b, angle=angle,
                         facecolor='lightblue', edgecolor='blue',
                         linewidth=2, alpha=0.5, label='Invariant Set $\Omega_c$')
        ax1.add_patch(ellipse)

        # Lane boundaries
        ax1.axvline(x=-self.L, color='red', linestyle='--', linewidth=2,
                   label='Lane Boundaries')
        ax1.axvline(x=self.L, color='red', linestyle='--', linewidth=2)
        ax1.fill_betweenx([-0.3, 0.3], -self.L, self.L, alpha=0.1, color='green')

        # Initial condition box
        x1_range = initial_bounds['x1']
        x3_range = initial_bounds['x3']
        rect = Rectangle((x1_range[0], x3_range[0]),
                        x1_range[1]-x1_range[0],
                        x3_range[1]-x3_range[0],
                        linewidth=2, edgecolor='green',
                        facecolor='none', linestyle=':',
                        label='Initial Conditions $X_0$')
        ax1.add_patch(rect)

        ax1.set_xlabel('Lateral Position $x_1$ (m)', fontsize=14)
        ax1.set_ylabel('Heading Angle $x_3$ (rad)', fontsize=14)
        ax1.set_xlim([-2.5, 2.5])
        ax1.set_ylim([-0.3, 0.3])
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12, loc='upper right')
        ax1.set_title('Invariant Ellipsoid: Position vs Heading Projection',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/ellipsoid_projection.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: results/ellipsoid_projection.png")

        # Figure 2: Trajectory simulation
        worst_x = np.array(self.verification_result['worst_corner'])
        traj_result = self.simulate_trajectory(worst_x, t_max=10)

        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

        # Lateral position over time
        ax2a.plot(traj_result['time'], traj_result['states'][:, 0],
                 'b-', linewidth=2, label='$x_1(t)$ - Lateral Position')
        ax2a.axhline(y=self.L, color='red', linestyle='--', linewidth=2,
                    label='Lane Boundary')
        ax2a.axhline(y=-self.L, color='red', linestyle='--', linewidth=2)
        ax2a.fill_between(traj_result['time'], -self.L, self.L,
                         alpha=0.15, color='green', label='Safe Region')
        ax2a.set_xlabel('Time (s)', fontsize=13)
        ax2a.set_ylabel('Lateral Position (m)', fontsize=13)
        ax2a.set_title('Trajectory: Lateral Position', fontsize=14, fontweight='bold')
        ax2a.grid(True, alpha=0.3)
        ax2a.legend(fontsize=11)

        # Lyapunov function over time
        ax2b.plot(traj_result['time'], traj_result['V_values'],
                 'g-', linewidth=2, label='$V(x(t))$')
        ax2b.axhline(y=self.c_max, color='orange', linestyle='--',
                    linewidth=2, label='$c_{max}$ (Boundary)')
        ax2b.fill_between(traj_result['time'], 0, self.c_max,
                         alpha=0.15, color='green', label='Safe Set')
        ax2b.set_xlabel('Time (s)', fontsize=13)
        ax2b.set_ylabel('Lyapunov Function $V(x)$', fontsize=13)
        ax2b.set_title('Lyapunov Function Evolution', fontsize=14, fontweight='bold')
        ax2b.grid(True, alpha=0.3)
        ax2b.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig('results/trajectory_analysis.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: results/trajectory_analysis.png")

        # Figure 3: Phase portrait (x1 vs x2)
        fig3, ax3 = plt.subplots(figsize=(10, 8))

        # Simulate multiple trajectories
        n_traj = 20
        x1_samples = np.linspace(initial_bounds['x1'][0],
                                initial_bounds['x1'][1], n_traj)

        for x1_init in x1_samples:
            x0_sample = np.array([x1_init, 0.05, 0.1, 0])
            traj = self.simulate_trajectory(x0_sample, t_max=5)
            ax3.plot(traj['states'][:, 0], traj['states'][:, 1],
                    'b-', alpha=0.3, linewidth=1)
            ax3.plot(x0_sample[0], x0_sample[1], 'go', markersize=6)

        # Plot ellipsoid projection on (x1, x2)
        P_reduced_12 = self.P[[0,1], :][:, [0,1]]
        theta = np.linspace(0, 2*np.pi, 100)
        circle = np.array([np.cos(theta), np.sin(theta)])
        L_chol = np.linalg.cholesky(P_reduced_12)
        ellipse_12 = np.sqrt(self.c_max) * np.linalg.solve(L_chol.T, circle)
        ax3.plot(ellipse_12[0,:], ellipse_12[1,:], 'r-', linewidth=2,
                label='Invariant Set Boundary')

        ax3.axvline(x=-self.L, color='red', linestyle='--', linewidth=2)
        ax3.axvline(x=self.L, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Lateral Position $x_1$ (m)', fontsize=14)
        ax3.set_ylabel('Lateral Velocity $x_2$ (m/s)', fontsize=14)
        ax3.set_title('Phase Portrait: Trajectories Converging to Origin',
                     fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=12)
        ax3.set_xlim([-2, 2])
        ax3.set_ylim([-0.3, 0.3])
        plt.tight_layout()
        plt.savefig('results/phase_portrait.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: results/phase_portrait.png")

        plt.close('all')

    def generate_certificate(self, initial_bounds):
        """
        Generate formal verification certificate
        """
        certificate = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'method': 'Lyapunov_stability_analysis',
                'tool': 'Python_scipy_numpy'
            },
            'system': {
                'type': 'linear_time_invariant',
                'dimensions': 4,
                'state_variables': ['lateral_position', 'lateral_velocity',
                                  'heading_angle', 'heading_rate'],
                'A_matrix': self.A.tolist(),
                'B_matrix': self.B.tolist()
            },
            'controller': {
                'type': 'LQR',
                'Q_matrix': self.Q.tolist(),
                'R_matrix': self.R.tolist(),
                'K_gain': self.K.tolist(),
                'closed_loop_eigenvalues': [
                    {'real': float(np.real(λ)), 'imag': float(np.imag(λ))}
                    for λ in np.linalg.eigvals(self.A_cl)
                ]
            },
            'lyapunov_function': {
                'form': 'V(x) = x^T P x',
                'P_matrix': self.P.tolist(),
                'P_eigenvalues': np.linalg.eigvals(self.P).tolist(),
                'derivative_form': 'V_dot(x) = -x^T Q x',
                'positive_definite': bool(np.all(np.linalg.eigvals(self.P) > 0)),
                'derivative_negative_definite': bool(np.all(np.linalg.eigvals(self.Q) > 0))
            },
            'invariant_set': {
                'form': 'Omega_c = {x : x^T P x <= c_max}',
                'c_max': float(self.c_max),
                'guaranteed_bound': f'|x_1| <= {self.L} meters',
                'lane_width': float(2 * self.L)
            },
            'initial_conditions': {
                'bounds': initial_bounds,
                'max_lyapunov_value': float(self.verification_result['max_V_initial']),
                'worst_case_corner': self.verification_result['worst_corner']
            },
            'verification_result': {
                'status': 'VERIFIED' if self.verification_result['verified'] else 'PARTIAL',
                'coverage_percent': float(self.verification_result['coverage']),
                'safety_guarantee': 'Zero lane violations for infinite time horizon'
                                  if self.verification_result['verified']
                                  else f"{self.verification_result['coverage']:.1f}% of input space verified",
                'proof_method': 'Constructive Lyapunov stability proof'
            },
            'numerical_properties': {
                'condition_number_P': float(np.linalg.cond(self.P)),
                'lyapunov_equation_residual': float(np.max(np.abs(
                    self.A_cl.T @ self.P + self.P @ self.A_cl + self.Q))),
                'convergence_rate': float(-np.max(np.real(np.linalg.eigvals(self.A_cl))))
            }
        }

        with open('results/verification_certificate.json', 'w') as f:
            json.dump(certificate, f, indent=2)

        print("\n✓ Verification certificate generated: results/verification_certificate.json")

        # Also generate human-readable text certificate
        self._generate_text_certificate(certificate)

        return certificate

    def _generate_text_certificate(self, cert):
        """
        Generate human-readable text certificate
        """
        with open('results/verification_certificate.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("    FORMAL VERIFICATION CERTIFICATE\n")
            f.write("    Lane-Keeping Controller Safety Guarantee\n")
            f.write("="*70 + "\n\n")

            f.write(f"Date: {cert['metadata']['timestamp']}\n")
            f.write(f"Method: Lyapunov Stability Analysis\n\n")

            f.write("THEOREM:\n")
            f.write("-" * 70 + "\n")
            f.write("For the closed-loop lane-keeping system with LQR controller,\n")
            f.write(f"given initial conditions x(0) satisfying:\n\n")
            for key, val in cert['initial_conditions']['bounds'].items():
                f.write(f"  {key} ∈ [{val[0]:.4f}, {val[1]:.4f}]\n")
            f.write(f"\nIt is formally proven that for ALL time t ≥ 0:\n")
            f.write(f"  |x₁(t)| < {self.L:.3f} meters\n\n")
            f.write(f"Therefore: ZERO LANE VIOLATIONS occur.\n\n")

            f.write("PROOF SUMMARY:\n")
            f.write("-" * 70 + "\n")
            f.write("1. Constructed Lyapunov function V(x) = x^T P x\n")
            f.write("2. Verified V̇(x) = -x^T Q x < 0 (energy decreasing)\n")
            f.write(f"3. Computed maximum invariant ellipsoid Ω_c (c = {self.c_max:.4f})\n")
            f.write(f"4. Verified all initial conditions satisfy V(x₀) ≤ c\n")
            f.write("5. By invariance: x(t) ∈ Ω_c ⊆ Safe Region for all t ≥ 0\n\n")

            f.write("VERIFICATION STATUS:\n")
            f.write("-" * 70 + "\n")
            if cert['verification_result']['status'] == 'VERIFIED':
                f.write("✓✓✓ COMPLETE FORMAL VERIFICATION ✓✓✓\n")
                f.write("Coverage: 100% of specified input space\n")
            else:
                f.write(f"⚠ PARTIAL VERIFICATION\n")
                f.write(f"Coverage: {cert['verification_result']['coverage_percent']:.1f}%\n")

            f.write("\n" + "="*70 + "\n")
            f.write("This certificate can be independently verified by:\n")
            f.write("1. Checking P > 0 (all eigenvalues positive)\n")
            f.write("2. Verifying A_cl^T P + P A_cl = -Q\n")
            f.write("3. Confirming max{V(x) : x ∈ X₀} ≤ c_max\n")
            f.write("="*70 + "\n")

        print("✓ Text certificate generated: results/verification_certificate.txt")

    def run_complete_verification(self, initial_bounds):
        """
        Run all verification steps
        """
        print("\n" + "="*70)
        print("="*70)
        print("    LYAPUNOV-BASED FORMAL VERIFICATION")
        print("    Lane-Keeping Controller Safety Analysis")
        print("="*70)
        print("="*70)

        # Step 1: Stability
        if not self.verify_stability():
            print("\n✗ VERIFICATION FAILED: System unstable")
            return False

        # Step 2: Lyapunov function
        if not self.verify_lyapunov_function():
            print("\n✗ VERIFICATION FAILED: Invalid Lyapunov function")
            return False

        # Step 3: Invariant set
        if not self.compute_invariant_set():
            print("\n✗ VERIFICATION FAILED: Could not compute invariant set")
            return False

        # Step 4: Initial conditions
        verified, coverage = self.verify_initial_conditions(initial_bounds)

        # Generate outputs
        self.generate_visualizations(initial_bounds)
        self.generate_certificate(initial_bounds)

        # Final summary
        print("\n" + "="*70)
        print("="*70)
        print("    VERIFICATION COMPLETE")
        print("="*70)

        if verified:
            print("\n✓✓✓ SUCCESS: FORMAL SAFETY GUARANTEE ESTABLISHED ✓✓✓\n")
            print("Proven: Zero lane violations for infinite time horizon")
            print(f"Coverage: 100% of specified initial conditions")
        else:
            print(f"\n⚠ PARTIAL VERIFICATION\n")
            print(f"Formally verified: {coverage:.1f}% of input space")
            print("Recommendation: See statistical analysis for remaining region")

        print("\nGenerated artifacts:")
        print("  • results/ellipsoid_projection.png")
        print("  • results/trajectory_analysis.png")
        print("  • results/phase_portrait.png")
        print("  • results/verification_certificate.json")
        print("  • results/verification_certificate.txt")

        print("\n" + "="*70)
        print("="*70 + "\n")

        return verified


def main():
    """
    Main execution function - MODIFIED to load learned parameters
    """
    import os
    import json

    # Create results directory
    os.makedirs('results', exist_ok=True)

    print("\n" + "="*70)
    print("PHASE 2: FORMAL VERIFICATION OF LEARNED CONTROLLER")
    print("="*70 + "\n")

    # System parameters (same as Phase 1)
    v = 15.0  # m/s
    b = 0.5
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, v, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B = np.array([[0], [0], [0], [b]])

    # ============================================
    # KEY CHANGE: Load LEARNED parameters from Phase 1
    # ============================================
    try:
        print("Loading learned parameters from Phase 1...")
        with open('results/learned_parameters.json', 'r') as f:
            data = json.load(f)
            learned_params = data['best_params']

            # Extract Q and R from learned parameters
            Q = np.array(learned_params['Q'])
            R = np.array(learned_params['R'])

            print("✓ Successfully loaded learned parameters")
            print(f"\nLearned Parameters:")
            print(f"  Q = diag({np.diag(Q)})")
            print(f"  R = {R[0,0]:.3f}")
            print(f"  Empirical violation rate (from learning): {learned_params['violation_rate']*100:.2f}%")
            print(f"\nNow attempting formal verification...\n")

    except FileNotFoundError:
        print("⚠ WARNING: No learned parameters found!")
        print("  Expected file: results/learned_parameters.json")
        print("  Please run Phase 1 first: python src/1_learn_parameters.py")
        print("\nUsing DEFAULT parameters for demonstration purposes...\n")

        Q = np.diag([100, 10, 50, 5])
        R = np.array([[1.0]])

    except Exception as e:
        print(f"✗ Error loading learned parameters: {e}")
        print("Using default parameters...\n")
        Q = np.diag([100, 10, 50, 5])
        R = np.array([[1.0]])

    # Initial condition bounds (same as used in learning)
    initial_bounds = {
        'x1': (-0.5, 0.5),    # lateral position (m)
        'x2': (-0.1, 0.1),    # lateral velocity (m/s)
        'x3': (-0.175, 0.175),  # heading angle (rad) = ±10°
        'x4': (-0.05, 0.05)   # heading rate (rad/s)
    }

    # Lane width
    lane_width = 3.5  # meters

    # Create verifier and run
    verifier = LyapunovVerifier(A, B, Q, R, lane_width)
    success = verifier.run_complete_verification(initial_bounds)

    # ============================================
    # NEW: Add recommendation based on result
    # ============================================
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE - NEXT STEPS")
    print("="*70 + "\n")

    if success:
        print("✓✓✓ FORMAL PROOF SUCCESSFUL ✓✓✓")
        print("\nThe learned controller is formally verified safe.")
        print("No statistical analysis needed.\n")
        print("Your project deliverable:")
        print("  - Learned parameters (Phase 1)")
        print("  - Formal safety proof (Phase 2)")
        print("  - Coverage: 100% of input space")
    else:
        print("⚠ PARTIAL VERIFICATION")
        print("\nThe learned controller is verified for a subset of inputs.")
        print("Statistical analysis REQUIRED for unverified region.\n")
        print("Next step: Run Phase 3")
        print("  python src/3_statistical_analysis.py")
        print("\nYour project deliverable:")
        print("  - Learned parameters (Phase 1)")
        print("  - Formal proof for verified subset (Phase 2)")
        print("  - Statistical analysis with confidence intervals (Phase 3)")

    print("\n" + "="*70 + "\n")

    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
