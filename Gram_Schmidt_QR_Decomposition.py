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


    ##### In _**Gram-Schmidt Orthogonalization**_, we take a set of linearly independent vectors and systematically construct an orthonormal basis from them. The process removes projections of each vector onto the ones before it, ensuring orthogonality, and then normalizes the result to unit length.
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


@app.cell(hide_code=True)
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

    In the Gram–Schmidt process, the norm \( \| \cdot \| \) used here is the **Euclidean norm** (also known as the **\(\ell^2 \)** norm).

    \[
    \| \vec{v} \| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \left( \sum_{i=1}^n v_i^2 \right)^{1/2}
    \]


    Measuring the **Euclidean distance** of a vector \( \vec{v} \in \mathbb{R}^n \) from the origin.
    """)
    mo.callout(side_note_for_norm,kind="info")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## **Here's the Scratch Implementation ✍️**
    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""defining a vector space A,""")
    return


@app.cell
def _(mo, np):
    # a vector space A having independent linearity

    A = np.array([[1,0,0], [2,0,3], [4,5,6]]).T
    mo.show_code(print(A))
    return (A,)


@app.cell
def _(A):
    print(A)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""###### `gs_Orthogonalization` func. which uses the Gram-Schmidt Process,""")
    return


@app.cell(hide_code=True)
def _(mo, np):
    # defining the gram-schmidt process

    def gs_Orthogonalization(X:np.ndarray)->np.ndarray:

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

    mo.show_code()
    return (gs_Orthogonalization,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ###### Now, we'll define a func. `is_orthonormal` to check the orthonormality of a Matrix satisfying the following fundamental step,
    \[
    Q^T. Q = I
    \]
    """
    )
    return


@app.cell
def _(A, gs_Orthogonalization, mo, np):

    def is_Orthonormal(Q: np.ndarray)->bool:
        """
        Checks if the columns of Q are orthonormal.
        For Q with shape (m, n), this checks if Q.T @ Q == I_n
        """
        Q_TQ = Q.T @ Q
        I = np.eye(Q.shape[1], dtype=Q.dtype)
        return np.allclose(Q_TQ, I)

    Q_A = gs_Orthogonalization(A)

    is_Orthonormal(Q_A)

    mo.show_code()

    return Q_A, is_Orthonormal


@app.cell(hide_code=True)
def _(Q_A, is_Orthonormal, mo):
    mo.plain_text(is_Orthonormal(Q_A))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##### **Therefore,**""")
    return


@app.cell
def _(A, Q, mo):

    v1_stack = mo.vstack([
        mo.md("#### **Original Vectors**"),
        mo.plain_text(A)
    ], align="center")

    v2_stack = mo.vstack([
        mo.md("#### **Orthonormal Vectors**"),
        mo.plain_text(Q)
    ],align="center")

    mo.hstack([v1_stack,mo.md("### turns into ➡️"), v2_stack],
             align="center",gap=0)
    return


@app.cell
def _(mo):
    mo.callout(mo.md("##### 🎯 Here, we got the success, `Q_A` storing the orthonormal vectors satisfies the eq."),kind="success").style({
        "font-size": "0.85rem",   # smaller text
        "padding": "0.2rem",      # less inner spacing
        "margin": "0.2rem 0",     # less space around
        "border-radius": "100px"    # optional: slightly tighter corners
    }).center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(""" """)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ###### **Now, let's have a look at QR Decompostion using Gram-Schmidt** 
    ## **QR Decomposition via the Gram–Schmidt Process 📐**
    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We compute the **QR decomposition** of a matrix \( A \in \mathbb{R}^{n \times k} \) as:

    \[
    A = QR
    \]

    - \( Q \in \mathbb{R}^{n \times k} \) contains **orthonormal columns** derived from \( A \),
    - \( R \in \mathbb{R}^{k \times k} \) is an **upper triangular matrix** that stores:

      - The **projection coefficients** used to subtract previous directions (above the diagonal), and  
      - The **norms** used to normalize each orthogonalized vector (on the diagonal).

    Each column of \( A \) is processed by removing its projections onto all previously computed orthonormal vectors and then normalized to form the columns of \( Q \). These coefficients naturally fill the entries of \( R \), making the decomposition exact:

    \[
    A = QR
    \]

    We compute the **QR decomposition** of a matrix \( A \in \mathbb{R}^{n \times k} \) as:

    \[
    A = QR
    \]

    Here:
    - \( Q \) contains **orthonormal columns** \( \vec{w}_1, \ldots, \vec{w}_k \),
    - \( R \) is an upper triangular matrix that:
      - Stores the **projection coefficients** used to subtract previously computed directions (above the diagonal),
      - And holds the **norms** used to normalize (on the diagonal).

    So the full decomposition is:

    \[
    A = 
    \begin{bmatrix}
    | & | &        & | \\
    \vec{v}_1 & \vec{v}_2 & \cdots & \vec{v}_k \\
    | & | &        & |
    \end{bmatrix}
    =
    \begin{bmatrix}
    | & | &        & | \\
    \vec{w}_1 & \vec{w}_2 & \cdots & \vec{w}_k \\
    | & | &        & |
    \end{bmatrix}
    \begin{bmatrix}
    r_{11} & r_{12} & \cdots & r_{1k} \\
    0 & r_{22} & \cdots & r_{2k} \\
    \vdots & \ddots & \ddots & \vdots \\
    0 & \cdots & 0 & r_{kk}
    \end{bmatrix}
    \]
    """
    )
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


@app.cell
def _():
    # def gs_orthogonalization()
    return


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
