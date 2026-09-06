<div align="center">

# Matrix Decomposition Implemenations

**A hands-on marimo built, math-first implementations of Matrix Decomposition Functions,**  
**find the notebooks; hosting on molab & hf-spaces.**

<br/>

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_TAVLehyiE58b5RDzjxFxSW/app)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/PragyanTiwari/Gram-Schmidt-Orthonormal-Basis)
[![Python](https://img.shields.io/badge/Python-3.11%2B-4F46E5?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0%2B-7C3AED?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-1E293B?style=flat&logo=apache&logoColor=white)](https://opensource.org/licenses/Apache-2.0)

<img src=".assets/01.gif" alt="Matrix Decompositions Demo" width="620"/>

</div>

## Table of Contents

  - [Overview](#overview)
  - [Marimo Apps](#marimo-apps)
  - [Quickstart](#quickstart)
  - [Implementation Notes](#implementation-notes)
  - [Contributing](#contributing)
  - [Resources \& Acknowledgements](#resources--acknowledgements)


## Overview

A curated set of [marimo](https://marimo.io) notebooks based on **Matrix Decomposition** functions, written in Python, each pairing a mathematical derivation with annotated Python including an interactive visualization, inside a single reactive environment.

The series is a progressive build, starting from orthogonalization fundamentals and working toward full matrix factorizations and applications:

`Gram-Schmidt` → `QR` → `LU` → `Householder` → `SVD` → `PCA`

> These functions reduce computationally expensive operations i.e. inversion, least squares, eigensolving, into sequences of simpler, numerically stable factors.

>> Applications such as **noise reduction, signal processing, image compression** and more will be covered as the series progresses.

## Marimo Apps

| Notebook | Open in molab | Open in HF Spaces |
|---|:---:|:---:|
| **Gram-Schmidt Orthogonalization** | [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_TAVLehyiE58b5RDzjxFxSW/app) | [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/PragyanTiwari/Gram-Schmidt-Orthonormal-Basis) |
| **QR Decomposition** |  [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_UqB7KaRLi2dar9bLeThKJd) | 🔜 |
| **Householder Reflection & Bidiagonalization** | 🔜 | 🔜 |

## Quickstart

Requires Python `>= 3.12` and [`uv`](https://docs.astral.sh/uv/).

**1. Clone and install dependencies**

```bash
git clone https://github.com/prgyn8/Matrix-Decomposition-Implementations.git
uv sync
```

**2. Run a marimo app**, (eg. gram-schmidt process)

```bash
uvx marimo run apps/gs_process.py       # you can find the available notebooks in the apps directory.
```

**3. Optionally, run a notebook in sandbox environment**

```bash
# Run the app
uvx marimo run --sandbox apps/gs_process.py

# Or open for editing
uvx marimo edit --sandbox apps/gs_process.py
```

---

## Implementation Notes

<details>

<summary><strong>Gram-Schmidt Orthogonalization</strong></summary>

<br/>

```python
## snippet from the notebook : https://molab.marimo.io/notebooks/nb_TAVLehyiE58b5RDzjxFxSW
def gram_schmidt(X:np.ndarray)->np.ndarray:

    """
    original -> orthogonal -> orthonormal
    args:
        A set of linearly independent vectors stored in columns in the array X.
    returns:
        Returns matrix Q of the shape of X, having orthonormal vectors for the given vectors.
    """
    Q = np.copy(X).astype("float64")
    n_vecs = Q.shape[1]

    # defining a function to compute the L2-norm
    length = lambda x: np.linalg.norm(x)

    # iteration with each vector in the matrix X
    for nth_vec in range(n_vecs):

        # iteratively removing each preceding projection from nth vector
        for k_proj in range(nth_vec):

            # the dot product would be the scaler coefficient 
            scaler = Q[:,nth_vec] @ Q[:,k_proj]
            projection = scaler * Q[:,k_proj]
            Q[:,nth_vec] -= projection                 # removing the Kth projection

        norm = length(Q[:,nth_vec])

        # handling the case if the loop encounters linearly dependent vectors. 
        # Since, they come already under the span of vector space, hence their value will be 0.
        if np.isclose(norm,0, rtol=1e-15, atol=1e-14, equal_nan=False):
            Q[:,nth_vec] = 0
        else:
            # making orthogonal vectors -> orthonormal
            Q[:,nth_vec] = Q[:,nth_vec] / norm

    return Q
```

```python
# Verification: Q.T @ Q ≈ I
A = np.array([[1, 0, 0], [2, 0, 3], [4, 5, 6]]).T
assert np.allclose(gram_schmidt(A).T @ gram_schmidt(A), np.eye(3))  # ✓
```

> 💬 Questions on implementation or numerical stability? Start a thread in [Discussions](https://github.com/prgyn8/Matrix-Decomposition-Implementations/discussions).

</details>

---

## Contributing

Contributions are welcome, whether it's a bug report, a new decomposition technique, or a clearer explanation of the math.

1. **Fork** the repository
2. **Sync** dependencies: `uv sync`
3. **Create a branch** for your changes
4. **Open a Pull Request** — maintainers will review it

For questions, suggestions, or discussion of the mathematics:

- 💬 [Discussion Board](https://github.com/prgyn8/Matrix-Decomposition-Implementations/discussions)
- 🐛 [Open an Issue](https://github.com/prgyn8/Matrix-Decomposition-Implementations/issues)

---

## Resources & Acknowledgements

- [**Wikipedia** — Gram-Schmidt Process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process) — foundational definitions and mathematical references
- [**DataCamp** — Orthogonal Matrices](https://www.datacamp.com/tutorial/orthogonal-matrix) — accessible article on orthogonality
- [**MIT OpenCourseWare** — Lecture 17](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-17-orthogonal-matrices-and-gram-schmidt/) — in-depth treatment by *Prof. Gilbert Strang*
- [**Steve Brunton**](https://www.youtube.com/@Eigensteve) — original spark for this project; exceptional intuition on engineering applications of linear algebra
- [**Graphical Linear Algebra**](https://graphicallinearalgebra.net/2017/08/09/orthogonality-and-projections/) — visual treatment of orthogonality and projections

---

<div align="center">

<br/>

[⭐ Star this repo](https://github.com/prgyn8/Matrix-Decomposition-Implementations/stargazers) &nbsp;·&nbsp;
[💬 Join the Discussion](https://github.com/prgyn8/Matrix-Decomposition-Implementations/discussions) &nbsp;·&nbsp;
[Author →](https://github.com/prgyn8)

</div>
