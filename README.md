### SE-PINN

This is the official repository for _Solving the Schrödinger Equation via Physics-Informed Machine Learning_.

Here a physics-informed neural network is developed and evaluated for solving the Schrödinger equation of quantum mechanics. The model is constrained to predict quantum-mechanical states that respect the physical properties of symmetry, normality, and orthogonality as well as information from the Schrödinger equation itself via a custom loss function and a custom architectural layer in PyTorch. In addition, the model learns not via supervised learning but via reinforcement learning (RL) from feedback from the Schrödinger equation.

This research was in collaboration with Alexander Ahrens and under the supervision of Prof. Ipek Oguz (https://engineering.vanderbilt.edu/bio/ipek-oguz) at Vanderbilt University.

---

### Demonstration

__Figure 1__ and __Figure 2__ are animations of the ground state (left) and the energy of the ground state (right) that are predicted by the model as it trains. The physical system is the quantum harmonic oscillator, which is used to model diatomic molecules such as molecules of hydrogen halides, nitrogen, and oxygen. The enforcement of exact symmetry on the prediction of the ground state via a particular architectural layer -- a "hub layer" -- improves the convergence of the model to the correct energy.

| **Figure 1**: No Enforcement of Exact Symmetry via Architecture |
| --- |
| ![Animation of PINN](assets/no_enforcement_of_symmetry.gif) |

| **Figure 2**: Enforcement of Exact Symmetry via Architecture |
| --- |
| ![Animation of PINN](assets/enforcement_of_symmetry.gif) |

---

### Usage

Installation on operating system:

```
pip install sepinn
```

Usage in Python:

```
from sepinn.wrappedpinn import WrappedPINN

model = WrappedPINN()
```

---

### Documentation

__Google Colab__ (Interactive)

https://colab.research.google.com/github/Tiger-Du/SE-PINN/blob/main/docs/quantum_harmonic_oscillator.ipynb

__nbviewer__ (Non-interactive)

https://nbviewer.org/github/Tiger-Du/SE-PINN/blob/main/docs/quantum_harmonic_oscillator.ipynb

---

### Citation

```
@techreport{DuAhrensOguz2023,
  author={Du, Tiger and Ahrens, Alexander and Oguz, Ipek},
  institution={Vanderbilt University},
  title={Solving the Schrodinger Equation via Physics-Informed Machine Learning},
  year={2023}
}
```

---

### License

This repository is licensed under the GPL-3.0 license.
