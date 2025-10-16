# Lane Keeping Controller - Formal Verification Project

Formal verification of LQR lane-keeping controller using:
1. **VerifAI** for parameter learning
2. **Lyapunov stability theory** for formal proof
3. **Statistical analysis** with confidence intervals (if needed)

## Project Structure
```
lane_keeping_verification/
├── src/
│   ├── 1_learn_parameters.py       # Phase 1: VerifAI learning
│   ├── 2_lyapunov_verification.py  # Phase 2: Formal verification
│   ├── 3_statistical_analysis.py   # Phase 3: Statistical analysis
│   └── simulator.py                # Lane-keeping simulator
├── results/                        # Auto-generated outputs
├── tests/                          # Unit tests
└── config/                         # Configuration files
```

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Your Setup
```bash
python test_setup.py
# Should output: "✓✓✓ All tests passed!"
```

### 3. Run Complete Pipeline
```bash
# Option A: Run all phases automatically
chmod +x run_complete_project.sh
./run_complete_project.sh

# Option B: Run phases individually
python src/1_learn_parameters.py     # Learning (20-30 min)
python src/2_lyapunov_verification.py  # Verification (< 1 min)
python src/3_statistical_analysis.py   # Statistics (5-10 min)
```

### 4. View Results
```bash
ls results/
# - learned_parameters.json
# - verification_certificate.txt
# - ellipsoid_projection.png
# - trajectory_analysis.png
# - statistical_analysis.json
```

## Three-Phase Pipeline

### Phase 1: Parameter Learning
- Uses VerifAI with Bayesian Optimization
- Minimizes empirical lane violation rate
- Samples 100 scenarios per evaluation
- Outputs: `results/learned_parameters.json`

### Phase 2: Formal Verification
- Lyapunov-based stability proof
- Computes maximum safe invariant set
- Attempts to verify 100% of input space
- Outputs: `results/verification_certificate.txt`

### Phase 3: Statistical Analysis (if needed)
- Monte Carlo with 10,000 samples
- Wilson score 95% confidence intervals
- Only runs on unverified region
- Outputs: `results/statistical_analysis.json`

## Project Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- VerifAI, Scenic (for learning phase)
- Ubuntu 20.04+ recommended

## Development

### Run Tests
```bash
pytest tests/ -v
```

### Format Code
```bash
black src/ tests/
```

### VS Code
Open project in VS Code for:
- Debugging configurations
- Task automation
- IntelliSense

## Project Deliverables

1. **Learned controller parameters** (Phase 1)
2. **Formal safety proof** OR **partial verification** (Phase 2)
3. **Statistical confidence intervals** (Phase 3, if needed)

## References

- VerifAI: https://github.com/BerkeleyLearnVerify/VerifAI
- Scenic: https://github.com/BerkeleyLearnVerify/Scenic
- Project documentation: `docs/`

## Author

[Your Name]

## License

[Your License]
