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
def _(mo):
    mo.md(
        r"""
    ##### For a vector space having basis \( \{ \vec{v}_1, \ldots, \vec{v}_m \} \) of a subspace \( S \subset \mathbb{R}^n \), the **Gram–Schmidt** process constructs an _**orthonormal basis**_ \( \{ \vec{w}_1, \vec{w}_2, \ldots, \vec{w}_m \} \), such that:

    \[
    \operatorname{gram\_schmidt} \left( \left\{ \vec{v}_1, \vec{v}_2, \ldots, \vec{v}_m \right\} \right)
    \longrightarrow \left\{ \vec{w}_1, \vec{w}_2, \ldots, \vec{w}_m \right\}
    \]

    ##### where each \( \vec{w}_i \) is orthonormal, and constructed via the following steps:

    Set:

    \[
    \vec{u}_1 = \vec{v}_1, \quad \vec{w}_1 = \frac{\vec{u}_1}{\|\vec{u}_1\|}
    \]

    For each \( i = 2, 3, \ldots, m \), compute:

    \[
    \vec{u}_i = \vec{v}_i - \sum_{j=1}^{i-1} \operatorname{proj}_{\vec{w}_j}(\vec{v}_i)
    = \vec{v}_i - \sum_{j=1}^{i-1} \left( \frac{\vec{w}_j^\top \vec{v}_i}{\vec{w}_j^\top \vec{w}_j} \right) \vec{w}_j
    \]

    in other words,

    \[
    \begin{aligned}
    \vec{u}_1 &= \vec{v}_1, &
    \vec{w}_1 &= \frac{\vec{u}_1}{\|\vec{u}_1\|}, \\[8pt]
    \vec{u}_2 &= \vec{v}_2 - \operatorname{proj}_{\vec{u}_1}(\vec{v}_2)
              = \vec{v}_2 - \frac{\vec{u}_1^{\top}\vec{v}_2}{\vec{u}_1^{\top}\vec{u}_1} \vec{u}_1, &
    \vec{w}_2 &= \frac{\vec{u}_2}{\|\vec{u}_2\|}, \\[8pt]
    \vec{u}_3 &= \vec{v}_3 - \operatorname{proj}_{\vec{u}_1}(\vec{v}_3) - \operatorname{proj}_{\vec{u}_2}(\vec{v}_3), &
    \vec{w}_3 &= \frac{\vec{u}_3}{\|\vec{u}_3\|}, \\[6pt]
    &\;\vdots & &\;\vdots \\
    \vec{u}_k &= \vec{v}_k - \sum_{j=1}^{k-1} \operatorname{proj}_{\vec{u}_j}(\vec{v}_k), &
    \vec{w}_k &= \frac{\vec{u}_k}{\|\vec{u}_k\|}.
    \end{aligned}
    \]

    Then normalize:

    \[
    \vec{w}_i = \frac{\vec{u}_i}{\|\vec{u}_i\|}
    \]

    ##### These vectors \( \{ \vec{w}_1, \ldots, \vec{w}_m \} \) satisfy the orthonormality condition:

    \[
    \vec{w}_i^\top.\vec{w}_j =
    \begin{cases}
    1 & \text{if } i = j, \\
    0 & \text{if } i \neq j
    \end{cases}
    \]

    ##### and such orthonormal vectors can be assembled into the columns which build an **Orthonormal Matrix \( Q \in \mathbb{R}^{n \times m} \),** such that:

    \[
    Q^T. Q = I
    \]

    ##### In practical numerical implementations (due to rounding errors), we often get:

    \[
    Q^T. Q \approx I
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    side_note_for_norm = mo.md(r"""
    💡 **For you info..** 

    In the Gram–Schmidt process, the norm \( \| \cdot \| \) used is the **Euclidean norm** (also known as the **\(\ell^2 \)** norm).

    \[
    \| \vec{v} \| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \left( \sum_{i=1}^n v_i^2 \right)^{1/2}
    \]


    This measures the **Euclidean distance** of a vector \( \vec{v} \in \mathbb{R}^n \) from the origin.
    """)
    mo.callout(side_note_for_norm,kind="info")
    return


@app.cell
def _(np):
    # vector space having linear independent vectors

    A = np.array([[1,0,0],[2,0,3],[4,5,6]]).T
    # A = np.array([[1, 1, 0], [-1, 2, 1], [0, 1, 1]]).T
    # A = np.array([[1, 1, 0],
    #               [1, 0, 1],
    #               [1, 1, 1]], dtype=float).T
    A
    return (A,)


@app.cell
def _(A, np):
    QT,RT = np.linalg.qr(A)
    RT
    return


@app.cell
def _(A, mo, np):
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


    mo.show_code()
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
