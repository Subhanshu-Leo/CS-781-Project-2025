#!/bin/bash
# Complete demonstration for course evaluation

echo "======================================================"
echo "CS781 Project Demo: LQR Lane-Keeping Verification"
echo "Student: [Your Name]"
echo "======================================================"

# 1. Show Scenic scenario
echo -e "\n[1/6] Scenic Scenario Definition"
cat scenarios/lane_keeping.scenic | head -20
read -p "Press Enter to continue..."

# 2. Run learning with VerifAI
echo -e "\n[2/6] Phase 1: Learning with VerifAI (5 iterations demo)"
python src/1_learn_parameters.py --n_iterations 5 --demo

# 3. Show learned parameters
echo -e "\n[3/6] Learned Parameters"
cat results/learned_parameters.json | jq '.best_params'
read -p "Press Enter to continue..."

# 4. Run formal verification
echo -e "\n[4/6] Phase 2: Formal Verification"
python src/2_lyapunov_verification.py

# 5. Show certificate
echo -e "\n[5/6] Verification Certificate"
cat results/verification_certificate.txt | head -40
read -p "Press Enter to continue..."

# 6. Statistical analysis
echo -e "\n[6/6] Phase 3: Statistical Analysis"
python src/3_statistical_analysis.py

echo -e "\n======================================================"
echo "Demo Complete! Check results/ for all outputs."
echo "======================================================"
