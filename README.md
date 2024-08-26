### SE-PINN

This is the official repository for _Solving the Schrödinger Equation via Physics-Informed Machine Learning_.

Here a physics-informed neural network is developed in PyTorch for solving the Schrödinger equation of quantum mechanics. The model is constrained to predict quantum-mechanical states that respect the mathematical-physical properties of __symmetry__, __normality__, and __orthogonality__ — all via __(1) a custom loss function__ and __(2) a custom architectural layer__. In addition, the model learns not through supervised learning but through __reinforcement learning (RL)__ via feedback from the Schrödinger equation itself.

This research was in collaboration with Alexander Ahrens and under the supervision of Prof. Ipek Oguz (https://engineering.vanderbilt.edu/bio/ipek-oguz) at Vanderbilt University.

---

### Demonstration

__Figure 1__ and __Figure 2__ are animations of the ground state (_left_) and the energy of the ground state (_right_) that are predicted by the model as it trains. The physical system of interest is the __quantum harmonic oscillator__, which is used to model diatomic molecules such as diatomic nitrogen, diatomic oxygen, and the hydrogen halides.

The enforcement of exact symmetry on the prediction of the ground state via a special architectural layer of the model — a __"hub layer"__ — improves its convergence to the correct energy, as visualized in __Figure 2__.

| **Figure 1**: Without Architectural Enforcement of Exact Symmetry |
| --- |
| ![Animation of PINN](assets/no_enforcement_of_symmetry.gif) |

| **Figure 2**: With Architectural Enforcement of Exact Symmetry |
| --- |
| ![Animation of PINN](assets/enforcement_of_symmetry.gif) |

---

### Usage

__Installation from PyPI via pip__:

```
pip install sepinn
```

__Usage in Python__:

```
from sepinn.wrappedpinn import WrappedPINN

model = WrappedPINN(...)

model.train(...)
```

---

### Documentation

A Jupyter notebook is included as an example in `docs` and is also available through Google Colab and nbviewer.

__Google Colab__ (Interactive)

https://colab.research.google.com/github/Tiger-Du/SE-PINN/blob/main/docs/quantum_harmonic_oscillator.ipynb

__nbviewer__ (Non-interactive)

https://nbviewer.org/github/Tiger-Du/SE-PINN/blob/main/docs/quantum_harmonic_oscillator.ipynb

---

### Citation

SE-PINN is citable via the BibTeX entry below.

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

SE-PINN is available under the GPL-3.0 license.
