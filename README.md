### SE-PINN

This is the official GitHub repository for _Solving the Schrödinger Equation via Physics-Informed Machine Learning_.

A PyTorch model was developed and evaluated to solve the Schrodinger equation of quantum mechanics. The predictions of the model are constrained to respect the physical properties of symmetry, normality, and orthogonality as well as information from the Schrodinger equation itself.

This research was in collaboration with Alexander Ahrens and under the supervision of Prof. Ipek Oguz (https://engineering.vanderbilt.edu/bio/ipek-oguz) at Vanderbilt University.

---

### Demonstration

__Figure 1__ and __Figure 2__ are animations of the energy eigenvector and the energy eigenvalue that are predicted by the model as it trains. The enforcement of exact symmetry of the predicted energy eigenvector via an architectural layer of the model improves its convergence to the correct energy eigenvalue.

| **Figure 1**: No Enforcement of Exact Symmetry via Architecture |
| --- |
| ![Animation of PINN](assets/animation%20(no%20symmetry).gif) |

| **Figure 2**: Enforcement of Exact Symmetry via Architecture |
| --- |
| ![Animation of PINN](assets/animation.gif) |

---

### Usage

__Google Colab__ (Interactive): https://colab.research.google.com/github/Tiger-Du/SE-PINN/blob/main/SE_PINN.ipynb

__nbviewer__ (Non-interactive): https://nbviewer.org/github/Tiger-Du/SE-PINN/blob/main/SE_PINN.ipynb

__GitHub__ (Non-interactive): https://github.com/Tiger-Du/SE-PINN/blob/main/SE_PINN.ipynb

---

### License

This repository is under the GPL-3.0 license.
