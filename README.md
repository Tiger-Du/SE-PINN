### SE-PINN

[Webpage](https://tiger-du.github.io/sepinn.html) | [GitHub](https://github.com/Tiger-Du/SE-PINN) | [PyPI](https://pypi.org/project/sepinn/)

SE-PINN is a physics-informed neural network in PyTorch that solves the Schrödinger equation of quantum mechanics.

The model is constrained to predict quantum-mechanical states that respect the mathematical-physical properties of __symmetry__, __normality__, and __orthogonality__ — all via __(1) a custom loss function__ and __(2) a custom neural-network layer__. In addition, the model learns not through supervised learning but through __reinforcement learning (RL)__ via feedback from the Schrödinger equation itself.

SE-PINN was developed at Vanderbilt University in collaboration with Alexander Ahrens and under the supervision of Prof. Ipek Oguz (https://engineering.vanderbilt.edu/bio/?pid=ipek-oguz).

The design of SE-PINN is based on https://arxiv.org/abs/2203.00451 and https://arxiv.org/abs/1904.08991.

In addition, SE-PINN has the following features:
- L-BFGS optimization
- A class that supports deterministic training, checkpointing of training, and visualization
- `pip install sepinn` — A Python package on PyPI with CI/CD
- Documentation as a [webpage](https://tiger-du.github.io/sepinn.html) and a [Google Colab notebook](https://colab.research.google.com/github/Tiger-Du/SE-PINN/blob/main/docs/quantum_harmonic_oscillator.ipynb
)

---

### Demonstration

__Figure 1__ and __Figure 2__ are visualizations of the ground state (_left_) and the energy of the ground state (_right_) that are predicted by the model as it trains. The physical system of interest is the __quantum harmonic oscillator__, which is used to model diatomic molecules such as diatomic nitrogen, diatomic oxygen, and the hydrogen halides.

The enforcement of symmetry on the prediction of the ground state via a special architectural layer of the model — a __"hub layer"__ — improves its convergence to the correct energy, as visualized in __Figure 2__.

| **Figure 1**: SE-PINN without Enforcement of Symmetry |
| --- |
| <img src=https://raw.githubusercontent.com/Tiger-Du/SE-PINN/main/assets/no_enforcement_of_symmetry.gif> |

| **Figure 2**: SE-PINN with Enforcement of Symmetry |
| --- |
| <img src=https://raw.githubusercontent.com/Tiger-Du/SE-PINN/main/assets/enforcement_of_symmetry.gif> |

---

### Usage

__1. Install from PyPI__

```
pip install sepinn
```

__2. Import into Python__

```python
from sepinn.wrappedpinn import WrappedPINN

model = WrappedPINN(...)

model.train(...)
```

---

### Documentation

A Jupyter notebook is available for reference in the `docs` folder as well as through Google Colab and nbviewer.

__Google Colab__ (_Interactive_):

https://colab.research.google.com/github/Tiger-Du/SE-PINN/blob/main/docs/quantum_harmonic_oscillator.ipynb

__nbviewer__ (_Non-interactive_):

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
