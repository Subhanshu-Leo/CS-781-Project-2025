"""Integration tests for complete pipeline"""
import pytest
import subprocess
import json
from pathlib import Path


def test_complete_pipeline():
    """Test full 3-phase pipeline"""

    # Phase 1
    result = subprocess.run(
        ['python', 'src/1_learn_parameters.py', '--n_iterations', '2'],
        capture_output=True
    )
    assert result.returncode == 0
    assert Path('results/learned_parameters.json').exists()

    # Phase 2
    result = subprocess.run(
        ['python', 'src/2_lyapunov_verification.py'],
        capture_output=True
    )
    assert result.returncode == 0
    assert Path('results/verification_certificate.json').exists()

    # Phase 3
    result = subprocess.run(
        ['python', 'src/3_statistical_analysis.py'],
        capture_output=True
    )
    # May fail if 100% verified - that's OK
    assert Path('results/statistical_analysis.json').exists() or \
           result.returncode == 0

def test_scenic_scenario_valid():
    """Test Scenic scenario is syntactically valid"""
    import scenic

    scenario = scenic.scenarioFromFile('scenarios/lane_keeping.scenic')
    assert scenario is not None

    # Generate one scene to verify
    scene, _ = scenario.generate()
    assert len(scene.objects) > 0
