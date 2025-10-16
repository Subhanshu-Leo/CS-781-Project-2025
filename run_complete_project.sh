#!/bin/bash
# Complete project pipeline

echo "========================================================================"
echo "LANE KEEPING VERIFICATION - COMPLETE PIPELINE"
echo "========================================================================"
echo ""

# Activate environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Phase 1: Learn Parameters
echo "PHASE 1: Learning LQR Parameters with VerifAI"
echo "----------------------------------------------------------------"
python src/1_learn_parameters.py
if [ $? -ne 0 ]; then
    echo "Error in Phase 1"
    exit 1
fi
echo ""

# Phase 2: Formal Verification
echo "PHASE 2: Lyapunov-Based Formal Verification"
echo "----------------------------------------------------------------"
python src/2_lyapunov_verification.py
if [ $? -ne 0 ]; then
    echo "Error in Phase 2"
    exit 1
fi
echo ""

# Phase 3: Statistical Analysis (if needed)
echo "PHASE 3: Statistical Analysis with Confidence Intervals"
echo "----------------------------------------------------------------"
python src/3_statistical_analysis.py
if [ $? -ne 0 ]; then
    echo "Warning: Phase 3 had issues (may be OK if 100% verified)"
fi
echo ""

# Display results
echo "========================================================================"
echo "PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved in results/ directory:"
ls -lh results/
echo ""
echo "Open visualizations:"
echo "  - results/ellipsoid_projection.png"
echo "  - results/trajectory_analysis.png"
echo "  - results/phase_portrait.png"
echo "  - results/statistical_analysis.png"
echo ""
echo "Certificates:"
echo "  - results/verification_certificate.txt"
echo "  - results/learned_parameters.json"
echo "  - results/statistical_analysis.json"
echo ""
