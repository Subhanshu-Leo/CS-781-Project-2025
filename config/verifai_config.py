"""
VerifAI configuration for parameter learning
"""

from dotmap import DotMap

# Sampler configuration
def get_sampler_config():
    """Get feature sampler configuration"""
    sampler_params = DotMap()

    # Initial condition ranges (same as used in simulator)
    sampler_params.x1_init = (-0.5, 0.5)      # lateral position (m)
    sampler_params.x2_init = (-0.1, 0.1)      # lateral velocity (m/s)
    sampler_params.x3_init = (-0.175, 0.175)  # heading angle (rad)
    sampler_params.x4_init = (-0.05, 0.05)    # heading rate (rad/s)

    return sampler_params

# Optimization configuration
def get_optimizer_config():
    """Get optimizer configuration"""
    optimizer_params = DotMap()

    optimizer_params.method = 'bayesian'
    optimizer_params.n_iterations = 50
    optimizer_params.n_samples_per_iteration = 100
    optimizer_params.random_state = 42

    return optimizer_params

# Parameter search space
def get_parameter_space():
    """Get LQR parameter search space"""
    return {
        'Q1': (10, 200),    # lateral position weight
        'Q2': (1, 50),      # lateral velocity weight
        'Q3': (10, 100),    # heading weight
        'Q4': (1, 20),      # heading rate weight
        'R': (0.1, 10)      # control effort weight
    }

# System parameters
def get_system_params():
    """Get system parameters"""
    return {
        'velocity': 15.0,
        'control_effectiveness': 0.5,
        'lane_width': 3.5
    }
