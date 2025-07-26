import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Talking about Orthogonality

    #### From a geometric perspective, an orthogonal matrix represents a linear transformation that preserves the length of vectors and the angles between them. Such transformations include rotations, reflections, or combinations of these operations in n-dimensional space. **This means that when we multiply a vector by an orthogonal matrix, only its orientation changes — not its magnitude.**
    """
    )
    return


@app.cell
def _(np):
    # vector space having linear independent vectors

    # A = np.array([[1,0,0],[2,0,3],[4,5,6]]).T
    # A
    A = np.array([[1, 1, 0],
                  [1, 0, 1],
                  [1, 1, 1]], dtype=float).T
    return (A,)


@app.cell
def _(A, np):
    # the gram-schmidt process iteration

    Q = np.copy(A).astype("float64")
    R = np.zeros(A.shape).astype("float64")
    length = lambda x: np.linalg.norm(x)


    for col_index in range(A.shape[1]):

        for proj_index in range(col_index): 

            scaler = Q[:,col_index] @ A[:,proj_index]
            projection = scaler * Q[:,proj_index]
            Q[:,col_index] -= projection
            R[proj_index,col_index] = scaler

        norm = length(Q[:,col_index])

        if np.isclose(norm,0, rtol=1e-15, atol=1e-14, equal_nan=False):
            Q[:,col_index] = np.zeros(A.shape[1])
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
def _(A, Q, R):
    A == Q @ R
    return


@app.cell
def _(Q):
    Q.T @ Q
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""---""")
    return


@app.cell
def _(A, np):
    np.zeros(A.shape)
    return


if __name__ == "__main__":
    app.run()
