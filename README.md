#### **A comprehensive implementation of core **Matrix Decomposition techniques** in Linear Algebra, providing an efficient computation approach, helps in uncovering hidden relationships in data, and lets you power many scalable applications in various fields of science and engineering.**

______________________________________________________________________

To gain a deeper understanding of how Orthogonalization & Matrices Decomposition works in real-life applications, & how they save bunch of time through an approach of vectorization, you'll find such techniques used in;

- **📡 Signal Processing**
- **🤖 Control Systems and Robotics**
- **🖼️ Image Processing**
- **➗ Solving Linear Systems i.e. *AX = B***

With certain mathematical intuitions (*having visual introspections*),this project simplifies most of the abstract concepts and becomes easier to grasp and connect with practical applications.

![a snippet of notebook](.assets/matrix_snippet.gif)

> You'll yet to see more implementations—such as **Householder Reflection**, **Bidiagonalization**, **LU Decomposition**, on this repo, and others—*these will be added soon*.

## What's Inside

The **Gram-Schmidt Orthogonalization** is one of the fundamental process in Linear Algebra to achieve *Orthonormal Vectors* for a given vector space. The Orthonormal Basis are produced by iteratively removing vector projections — also known as the *Vector Projection Elimination method*.

**Terms like Orthogonality, QR Decomposition are being discussed in the — [🗨️Discussion section](https://github.com/PragyanTiwari/Matrix-Decompositions-Implementation-for-SVD-PCA/discussions).**

Here's a snippet;

```bash
def gs_Orthogonalization(X:np.ndarray)->np.ndarray:

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

To edit the notebook in a sandbox environment, run this;

```bash
uvx marimo edit --sandbox notebooks\Gram_Schmidt_QR_Decomposition.py
```

## 🧪 Testing

The updates made on this project, can be tested for deployment, (and for personal experimentation) by the following;

- Fork the repository.

- Run uv sync to install dependencies (*uv lockfile will help*)

```bash
uv sync
```

- To test the export process, we'll run `.github/scripts/build.py` from the root directory through a symlink.

```bash
uv run build.py
```

This will export all notebooks in a folder called `_site/` in the root directory

## 🌱 Contribution Guide

- If you find a bug or have a feature request, please open an [Issue](https://github.com/PragyanTiwari/Matrix-Decompositions-Implementation-for-SVD-PCA/issues).

- PR will be reviewed by the maintainers.

- Questions & Suggestions can be queried on the [Discussion section](https://github.com/PragyanTiwari/Matrix-Decompositions-Implementation-for-SVD-PCA/discussions).

______________________________________________________________________
