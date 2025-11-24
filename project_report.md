# Lane-Keeping Controller: Learning and Formal Verification

## Abstract
This project uses the VerifAI toolkit to learn optimal LQR controller
parameters for autonomous vehicle lane-keeping, followed by formal
verification using Lyapunov stability theory.

## 1. Introduction
### 1.1 Problem Statement
In this project, we will use the VerifAI toolkit to "learn" the parameters of a linear quadratic regulator (LQR) controller that minimizes the count of lane violations of an ego car. Subsequently, you are required to attempt a proof that the learnt controller results in no lane violations.
The Scenic probablistic-program based sampler allows you to specify statistical variations of higher-level semantic features, based on which test samples are created. The simulator follows the control rules of the parameterized LQR controller and determines the trajectory of the ego car with the sampled values of the semantic features.
You are required to learn key parameters of the controller using this setup, and then try to attempt a proof (any technique is fine) that these values of parameters lead to zero violations of lane crossings. If your "learned" LQR controller does not yield a proof of correctness, you must provide a statistical estimate (along with a confidence interval) of what percentage of inputs within the allowed variations will lead to lane violations.
Please see here for extensive documentation on VerifAI, and also worked out examples, explained in tutorial style. This project will be building on the case study titled "Lane keeping with inbuilt simulator".

### 1.2 Approach Overview
- Phase 1: Parameter learning with VerifAI
- Phase 2: Formal verification via Lyapunov analysis
- Phase 3: Statistical confidence intervals (if needed)

## 2. Background
### 2.1 VerifAI Toolkit
### 2.2 Scenic Probabilistic Programming
### 2.3 LQR Control Theory
### 2.4 Lyapunov Stability

## 3. Methodology
### 3.1 Scenic Scenario Design
[Scenic code and explanation]

### 3.2 VerifAI-Based Learning
[Explain optimization setup]

### 3.3 Formal Verification Procedure
[Detail Lyapunov analysis]

## 4. Implementation
### 4.1 System Architecture
### 4.2 Key Components
### 4.3 Integration with VerifAI Case Study

## 5. Results
### 5.1 Learned Parameters
Q = diag([...])
R = ...
Empirical violation rate: X%

### 5.2 Verification Results
Coverage: X%
Status: COMPLETE/PARTIAL

### 5.3 Statistical Analysis
Confidence interval: [X%, Y%]

## 6. Discussion
### 6.1 Comparison with Baseline
### 6.2 Limitations
### 6.3 Future Work

## 7. Conclusion

## References
[Include VerifAI, Scenic papers]
