import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
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


@app.cell
def _(mo):
    statement = mo.md("""
    *here is the call!*

    ##### **Perpendicular Vectors == Orthogonal Vectors**, 
    such that, the dot product of any two vectors in vector space is *0*.
    """)

    mo.accordion({"Don't know what orthogonality is ???":statement})
    return


app._unparsable_cell(
    r"""
    # imaging having a vector space A
    import matplotlib.pyplot as plt

    def draw_vector(v, origin=[0, 0], **kwargs):
        plt.quiver(*origin, *v, angles='xy', scale_units='xy', scale=1, **kwargs)

    # Step 1: Original Vectors
    a = np.array([2, 2])
    b = np.array([2, 0])

    # Step 2: Gram-Schmidt
    u1 = a
    e1 = u1 / np.linalg.norm(u1)

    proj = np.dot(b, e1) * e1
    u2 = b - proj
    e2 = u2 / np.linalg.norm(u2)

    # Step 3: Plot
    plt.figure(figsize=(6, 6))
    draw_vector(a, color='blue', label='a')
    draw_vector(b, color='red', label='b')
    draw_vector(proj, color='gray' label='proj_b_on_a')
    draw_vector(u2, color='orange', label='orthogonal b\'')
    draw_vector(e1, color='green', label='e1 (unit)')
    draw_vector(e2, color='purple', label='e2 (unit)')

    # Plot settings
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title(\"Gram-Schmidt Orthogonalization\")
    plt.show()

    """,
    name="_"
)


@app.cell(hide_code=True)
def _(np, plt):

    import seaborn as sns

    def gram_schmidt(V):
        QT = []
        for v in V:
            for q in QT:
                v -= np.dot(v, q) * q
            QT.append(v / np.linalg.norm(v))
        return np.array(QT)

    # 5D vectors
    V = np.random.randn(5, 5)
    QTT = gram_schmidt(V)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(V, ax=axs[0], cmap='coolwarm', annot=True)
    axs[0].set_title("Original Vectors (V)")

    sns.heatmap(QTT, ax=axs[1], cmap='coolwarm', annot=True)
    axs[1].set_title("Orthogonalized Vectors (Q)")
    plt.tight_layout()
    plt.show()

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


@app.cell
def _(Q, mo, np):
    are_these_similar = lambda x,y: np.allclose(x,y)
    are_these_similar(Q,Q.T)
    mo.show_code()
    return


if __name__ == "__main__":
    app.run()
