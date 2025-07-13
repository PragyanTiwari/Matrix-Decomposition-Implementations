import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    return mo, np


@app.cell
def _(np):
    X = np.random.rand(5,3)
    X
    return (X,)


@app.cell
def _(X, np):
    U, S, V = np.linalg.svd(X, full_matrices=True)
    U
    return


@app.cell
def _():
    from sklearn.decomposition import PCA
    # auto is default
    pca = PCA(svd_solver="full")
    pca = PCA(svd_solver="randomized")
    pca = PCA(svd_solver="covariance_eigh")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Singular Value Decomposition (SVD) Overview""")
    return


@app.cell
def _(np):
    # original matrix
    A = np.array([[2,1],[1,2]])

    # transformation 1
    x1 = np.array([0,1])

    # transformation 2
    x2 = np.array([1,0])
    return A, x1


@app.cell
def _(A):
    A
    return


@app.cell
def _(A, x1):
    x1 @ A
    return


@app.cell
def _(mo):
    mo.md("""## Orthogonal Matrix""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    From a geometric perspective, an orthogonal matrix represents a linear transformation that preserves the length of vectors and the angles between them. Such transformations include rotations, reflections, or combinations of these operations in n-dimensional space. **This means that when we multiply a vector by an orthogonal matrix, only its orientation changes — not its magnitude.**

    Properties of Orthogonal Matrix (U):

    - U^T = U^-1
    -
    - U.U^T = U^T.U = Identity Matrix
    -
    """
    )
    return


@app.cell
def _(np):
    # identity matrix
    num = 100

    a = np.eye(3,dtype=int)
    b = np.eye(3,k=1)
    return a, b


@app.cell
def _(a, b, np):
    np.dot(a,b)
    return


@app.cell
def _(a, b, np):
    np.matmul(a,b)
    return


@app.cell
def _(a, b):
    a @ b
    return


@app.cell
def _(np):
    def is_orthogonal(Q:np.ndarray)->bool:
        # is shape equal??
        if Q.shape[0] != Q.shape[1]:
            return False

        # classical implementation
        Q_T = np.transpose(Q)
        I = np.eye(N=Q.shape[0],dtype=Q.dtype)
        return np.allclose(Q @ Q_T, I)
    return (is_orthogonal,)


@app.cell(hide_code=True)
def _(is_orthogonal, np):
    theta = np.pi/3  # 60 degrees
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]])


    print("Rotation matrix:")
    print(rotation_matrix)
    print("Is orthogonal:", is_orthogonal(rotation_matrix))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
