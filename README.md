# Bachelor's Thesis – Symbolic Regression and Physics-Informed Modeling

This repository contains the source code for my Bachelor's thesis, conducted by Thomas Lutz. 
The thesis explores and benchmarks modern Symbolic Regression (SR) methods in combination
with physics-based knowledge integration, such as Physics-Informed Neural Networks (PINNs), 
Reinforcement Learning (RL), and Bayesian optimization techniques.

---

## 📁 Project Structure

```
Bachelor_Arbeit_Codes Kopie/
├── PySR_Benchmark/
│   ├── 1.PySR_Test_Functions.py
│   ├── 2.PySr_Rod_PINN.py
│   └── 3.PySR_Beam_PINN.py
│
├── SR+GB_Benchmark/
│   └── GP+DGL+Boundary+Rod.py
│
├── Bingo GPSR Bayesia/
│   └── bingo.py
│
├── SR+RL/
│   ├── 1.Test_Functions/
│   ├── 2.Sr+RL_Rod_PINN/
│   ├── 3.Sr+RL_Beam_PINN/
│   └── SR+RL_Rod_No_PINN/
│
├── Neu_Non_DIM/
    ├── Rod.py
    └── Beam.py
```





---

## 🧪 Methods Overview

- **PySR (Symbolic Regression)**  
  Genetic programming for interpretable mathematical expression discovery.

- **Bingo**  
  A symbolic regression framework with Bayesian optimization components.

- **Symbolic Regression + Reinforcement Learning**  
  Hybrid SR models that incorporate RL strategies for model generation.

- **Physics-Informed Neural Networks (PINNs)**  
  Neural networks that solve differential equations by embedding physical laws.

---

## ⚙️ Requirements

You can install all required packages using `pip`. A `requirements.txt` file is recommended (but not included yet). Example:

```bash
pip install pysr torch numpy matplotlib 


---

## 🧪 Methods Overview

- **PySR (Symbolic Regression)**  
  Genetic programming for interpretable mathematical expression discovery.

- **Bingo**  
  A symbolic regression framework with Bayesian optimization components.

- **Symbolic Regression + Reinforcement Learning**  
  Hybrid SR models that incorporate RL strategies for model generation.

- **Physics-Informed Neural Networks (PINNs)**  
  Neural networks that solve differential equations by embedding physical laws.

---

🎯 Thesis Goal
The main objective of this thesis is to:
Benchmark symbolic regression tools
Compare performance and interpretability
Integrate physical constraints via PINNs
Evaluate robustness, generalizability, and computation time


📄 License
This code was developed as part of a Bachelor's thesis at [Insert your university name here]. It is provided for research and educational purposes only.
© 2025 Thomas Lutz. All rights reserved.
