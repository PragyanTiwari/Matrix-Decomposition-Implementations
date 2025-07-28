import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Gram-Schmidt Orthogonalization
    ---
    """).center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### **_Talking about Orthogonality_**

    ##### From a geometric perspective, an orthogonal matrix represents a linear transformation that preserves the length of vectors and the angles between them. Such transformations include rotations, reflections, or combinations of these operations in n-dimensional space. **This means that when we multiply a vector by an orthogonal matrix, only its orientation changes — not its magnitude.**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    statement = mo.md("""
    *here is the call! and it simply means,*

    ##### **Perpendicular Vectors == Orthogonal Vectors**, 
    where, the dot product of any two vectors in vector space is *0*.
    """)

    mo.accordion({"Don't know what orthogonality is ???":statement})
    return


@app.cell
def _(np):
    # vector space having linear independent vectors

    # A = np.array([[1,0,0],[2,0,3],[4,5,6]]).T
    # A = np.array([[1, 1, 0], [-1, 2, 1], [0, 1, 1]]).T
    A = np.array([[1, 1, 0],
                  [1, 0, 1],
                  [1, 1, 1]], dtype=float).T
    return (A,)


@app.cell
def _(A, np):
    QT,RT = np.linalg.qr(A)
    RT
    return


@app.cell
def _(A, np):
        # the gram-schmidt process iteration

    Q = np.copy(A).astype("float64")
    R = np.zeros(A.shape).astype("float64")
    length = lambda x: np.linalg.norm(x)


    for col_index in range(A.shape[1]):

        for proj_index in range(col_index): 

            scaler = Q[:,col_index] @ Q[:,proj_index]
            projection = scaler * Q[:,proj_index]
            Q[:,col_index] -= projection
            R[proj_index,col_index] = scaler

        norm = length(Q[:,col_index])

        if np.isclose(norm,0, rtol=1e-15, atol=1e-14, equal_nan=False):
            Q[:,col_index] = 0
        else:
            Q[:,col_index] = Q[:,col_index] / norm
            R[col_index,col_index] = norm



    return Q, R


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The matrix Q:""")
    return


@app.cell
def _(Q):
    Q
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The matrix R:""")
    return


@app.cell
def _(R):
    R
    return


@app.cell
def _(A, Q, R, np):
    np.allclose(A,Q @ R)
    return


@app.cell
def _(Q, np):
    np.allclose(Q.T @ Q, np.eye(3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(A):
    A[:,0] @ A[:,0].T
    return


@app.cell
def _(np):
    np.eye(3)
    return


if __name__ == "__main__":
    app.run()
