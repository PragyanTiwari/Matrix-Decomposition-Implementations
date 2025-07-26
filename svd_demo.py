import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium", css_file="custom.css")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Hare Krishna 🦚""").center()
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Singular Value Decomposition (SVD) Overview""")
    return


@app.cell
def _(np):
    # original matrix
    ZZ = np.array([[2,1],[1,2]])

    # transformation 1
    x1 = np.array([0,1])

    # transformation 2
    x2 = np.array([1,0])
    return ZZ, x1


@app.cell
def _(ZZ):
    ZZ
    return


@app.cell
def _(ZZ, x1):
    x1 @ ZZ
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
def _(np):
    # let Z matrix be the following
    Z = np.array([[1,2,4],[0,0,5],[0,3,6]])

    # column-vectors of mat Z
    a = Z[:,0]
    b = Z[:,1]
    c = Z[:,2]

    Z
    return a, b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### **Orthonormal Vectors** (unit vector having orthogonality)""")
    return


@app.cell
def _(np):
    unit_vector = lambda x: x / np.linalg.norm(x,ord=2)
    return (unit_vector,)


@app.cell
def _(a, unit_vector):
    q1 = a
    unit_vector(q1)
    return (q1,)


@app.cell
def _(b, np, q1, unit_vector):
    q2 = b - ((np.dot(b,q1))*q1)
    unit_vector(q2)
    return


@app.cell
def _(b, np, q1):
    np.dot(b,q1)
    return


@app.cell
def _(b, q1):
    b - ((b @ q1) * q1)
    return


@app.cell
def _(np):
    def gram_schmidt(Z:np.ndarray):

        # copy of the original
        new_matrix = np.copy(Z).astype("float64")
        length = lambda x: np.linalg.norm(x)

        for _ in range(Z.shape[1]):

            for p in range(_):
                # removing individual projections iteratively
                projection = (new_matrix[:,p] @ new_matrix[:,_]) * new_matrix[:,p]
                new_matrix[:,_] -= projection

            # to deal with linear dependence
            if np.isclose(length(new_matrix[:, _]), 0, rtol=1e-15, atol=1e-14, equal_nan=False):
                new_matrix[:,_] = np.zeros(new_matrix.shape[0])
            else:    
                # making orthogonal vectors -> orthonormal
                new_matrix[:,_] = new_matrix[:,_] / length(new_matrix[:,_])

        return new_matrix

    return (gram_schmidt,)


@app.cell
def _(np):
    P = np.array([[1, 1, 0], [-1, 2, 1], [0, 1, 1]]).T
    P
    return


@app.cell
def _(np):
    A = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [2, 1, 1, 2], [1, 0, 0, 1]]).T
    return (A,)


@app.cell
def _(A, gram_schmidt):
    ortho = gram_schmidt(A)
    ortho
    return (ortho,)


@app.cell
def _(is_orthogonal, ortho):
    is_orthogonal(ortho)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
