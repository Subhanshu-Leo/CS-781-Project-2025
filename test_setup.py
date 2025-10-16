"""Quick test to verify setup is working"""

import numpy as np
from scipy.linalg import solve_continuous_lyapunov
import matplotlib.pyplot as plt

def test_numpy():
    """Test NumPy"""
    A = np.array([[1, 2], [3, 4]])
    print(f"✓ NumPy working: Created {A.shape} array")
    return True

def test_scipy():
    """Test SciPy"""
    A = np.array([[-1, 1], [0, -2]])
    Q = np.eye(2)
    P = solve_continuous_lyapunov(A.T, -Q)
    print(f"✓ SciPy working: Solved Lyapunov equation")
    return True

def test_matplotlib():
    """Test Matplotlib"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1, 2], [0, 1, 4], 'b-')
    ax.set_title('Test Plot')
    plt.savefig('test_plot.png', dpi=100)
    plt.close()
    print(f"✓ Matplotlib working: Created test_plot.png")
    return True

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Testing VS Code Setup for Verification Project")
    print("="*50 + "\n")

    try:
        test_numpy()
        test_scipy()
        test_matplotlib()
        print("\n✓✓✓ All tests passed! Setup is complete ✓✓✓\n")
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
