### SE-PINN

This is the official GitHub repository for _Solving the Schrödinger Equation via Physics-Informed Machine Learning_.

In this research, a PyTorch model was developed and evaluated to solve the Schrodinger equation of quantum mechanics. The predictions of the model are constrained to respect the physical properties of symmetry, normality, and orthogonality as well as information from the Schrodinger equation itself via a custom loss function and a custom architectural layer in PyTorch.

This research was in collaboration with Alexander Ahrens and under the supervision of Prof. Ipek Oguz (https://engineering.vanderbilt.edu/bio/ipek-oguz) at Vanderbilt University.

---

### Demonstration

__Figure 1__ and __Figure 2__ are animations of the energy eigenvector and the energy eigenvalue that are predicted by the model as it trains. The enforcement of exact symmetry of the predicted energy eigenvector via an architectural layer, a "hub layer," of the model improves its convergence to the correct energy eigenvalue.

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

### Citation

```
@misc{
  title={Solving the Schrodinger Equation via Physics-Informed Machine Learning},
  author={Tiger Du and Alexander Ahrens and Ipek Oguz},
  year={2023}
}
```

---

### License

This repository is under the GPL-3.0 license.
