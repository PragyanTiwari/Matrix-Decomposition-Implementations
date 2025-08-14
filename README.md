# Matrix Decompositions Implementation for SVD & PCA

A comprehensive implementation of matrix decomposition algorithms with a focus on Singular Value Decomposition (SVD) and Principal Component Analysis (PCA). This repository provides educational implementations of fundamental linear algebra techniques used in machine learning and data analysis.

## 📚 Overview

This project implements various matrix decomposition methods from scratch, providing detailed mathematical explanations and practical demonstrations. The implementations are designed for educational purposes and include:

- **Gram-Schmidt Orthogonalization & QR Decomposition**
- **Householder Reflection Methods**
- **SVD Demonstrations**

## 🚀 Features

### Gram-Schmidt QR Decomposition (`Gram_Schmidt_QR_Decomposition.py`)

An interactive Marimo notebook that implements the Gram-Schmidt orthogonalization process and QR decomposition:

- **Orthogonalization Process**: Converts linearly independent vectors into orthonormal basis
- **QR Decomposition**: Decomposes matrix A into orthonormal matrix Q and upper triangular matrix R
- **Mathematical Explanations**: Detailed mathematical formulations and geometric interpretations
- **Interactive Visualizations**: Step-by-step process demonstration

#### Key Functions:
- `gs_Orthogonalization(X)`: Performs Gram-Schmidt orthogonalization
- `gs_QR_Decomposition(X)`: Complete QR decomposition using Gram-Schmidt
- `is_Orthonormal(Q)`: Validates orthonormality of matrix Q

#### Mathematical Foundation:
For a set of linearly independent vectors {v₁, v₂, ..., vₘ}, the process constructs orthonormal basis {w₁, w₂, ..., wₘ} where:

```
A = QR
```

Where:
- Q ∈ ℝⁿˣᵏ contains orthonormal columns
- R ∈ ℝᵏˣᵏ is upper triangular matrix
- Q^T Q = I (orthonormality condition)

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.7+
- NumPy
- Marimo (for interactive notebooks)

### Setup
```bash
# Clone the repository
git clone https://github.com/PragyanTiwari/Matrix-Decompositions-Implementation-for-SVD-PCA.git
cd Matrix-Decompositions-Implementation-for-SVD-PCA

# Install dependencies
pip install numpy marimo

# Run the Gram-Schmidt notebook
marimo run Gram_Schmidt_QR_Decomposition.py
```

### Quick Example
```python
import numpy as np
from Gram_Schmidt_QR_Decomposition import gs_QR_Decomposition

# Define a matrix with linearly independent columns
A = np.array([[1, 0, 0], 
              [2, 0, 3], 
              [4, 5, 6]]).T

# Perform QR decomposition
Q, R = gs_QR_Decomposition(A)

# Verify decomposition: A ≈ Q @ R
print("Original matrix A:")
print(A)
print("\nOrthonormal matrix Q:")
print(Q)
print("\nUpper triangular matrix R:")
print(R)
print(f"\nVerification (A ≈ Q@R): {np.allclose(A, Q @ R)}")
```

## 📖 Mathematical Background

### Gram-Schmidt Process
The Gram-Schmidt process transforms a set of linearly independent vectors into an orthonormal set by:

1. **Orthogonalization**: Remove projections of each vector onto previous ones
2. **Normalization**: Scale to unit length

Mathematical formulation:
```
u₁ = v₁
w₁ = u₁/‖u₁‖

For i = 2, 3, ..., m:
uᵢ = vᵢ - Σⱼ₌₁ⁱ⁻¹ proj_wⱼ(vᵢ)
wᵢ = uᵢ/‖uᵢ‖
```

### QR Decomposition
Every matrix A with linearly independent columns can be factored as A = QR, where Q has orthonormal columns and R is upper triangular.

## 📁 File Structure

```
Matrix-Decompositions-Implementation-for-SVD-PCA/
├── Gram_Schmidt_QR_Decomposition.py    # Interactive Gram-Schmidt & QR implementation
├── Houholder-reflection.py             # Householder reflection methods
├── svd_demo.py                         # SVD demonstration
└── README.md                           # This file
```

## 🎯 Educational Goals

This repository aims to provide:
- **Intuitive Understanding**: Clear explanations of mathematical concepts
- **Implementation Details**: From-scratch implementations without relying on built-in functions
- **Geometric Interpretation**: Visual understanding of orthogonality and matrix transformations
- **Practical Applications**: Real-world relevance in machine learning and data analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the implementations or add new decomposition methods.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 References

This project is undertaken through many resources, the topmost resources I learnt from,

- [Wikipedia](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process) – for foundational definitions and mathematical references. 
- [DataCamp](https://www.datacamp.com/tutorial/orthogonal-matrix) – for providing informational article upon Orthogonality.
- [MIT OpenCourseWare](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-17-orthogonal-matrices-and-gram-schmidt/) – for refurbishing the in-depth knowledge of Gram-Schmidt Process, taught by *Prof. Gilbert Strang*.
- [Steve Burton (*Amazing Guy*)](https://www.google.com/search?q=steve+brunton&sca_esv=55a910f019e63594&rlz=1C1GCEA_enIN1112IN1112&sxsrf=AE3TifMoAjuMLl0MOCAV5lyl_Ga8KboiEg%3A1755118367776&ei=H_ucaP-UL_Of4-EPrsmB8QY&ved=0ahUKEwi_oOa21YiPAxXzzzgGHa5kIG4Q4dUDCBA&uact=5&oq=steve+brunton&gs_lp=Egxnd3Mtd2l6LXNlcnAiDXN0ZXZlIGJydW50b24yBBAjGCcyCxAuGIAEGJECGIoFMgsQABiABBiRAhiKBTIKEAAYgAQYQxiKBTIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABEiZC1CRBljLCHABeACQAQCYAaoBoAGvAqoBAzAuMrgBA8gBAPgBAZgCA6ACwgLCAggQABiwAxjvBcICCxAAGIAEGLADGKIEwgIKEC4YgAQYQxiKBZgDAIgGAZAGBZIHAzEuMqAHuROyBwMwLjK4B7sCwgcDMi0zyAcP&sclient=gws-wiz-serp)  – for sparking the interest, this is from where I started this project. *He has a great interest in Physics Implementation of every engineering field.*

---

*Built with ❤️ for educational purposes in linear algebra and matrix decompositions*
