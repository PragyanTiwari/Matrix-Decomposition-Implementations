import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    return mo, np, plt


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
    <br>
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
    mo.md(r"""### Mathematical Overview""")
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
def _(A, Q_A, mo):

    v1_stack = mo.vstack([
        mo.md("#### **Original Vectors**"),
        mo.plain_text(A)
    ], align="center")

    v2_stack = mo.vstack([
        mo.md("#### **Orthonormal Vectors**"),
        mo.plain_text(Q_A)
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
def _(A, Q_A, np, plt):
    # comparison plot

    # Standard basis vectors
    basis = np.eye(3)

    # Apply transformations
    _transformed_A = A @ basis
    _transformed_Q = Q_A @ basis

    # Create figure with adjusted layout
    fig2 = plt.figure(figsize=(14, 5))
    fig2.suptitle('Matrix Transformation (A v/s Q)', y=1.05, fontsize=14)

    # Plot for Original Matrix A
    _ax1 = fig2.add_subplot(121, projection='3d')
    _ax1.set_title("Original Matrix Transformation (A)", fontsize=12, pad=12)
    _ax1.set_xlim([0, 10])
    _ax1.set_ylim([-10, 0])
    _ax1.set_zlim([0, 10])
    arrows_A = _ax1.quiver(*np.zeros((3, 3)), *_transformed_A, 
                        color=['r', 'g', 'b'], 
                        arrow_length_ratio=0.12,
                        linewidth=2.5,
                        label=['A·i (1st column)', 'A·j (2nd column)', 'A·k (3rd column)'])
    _ax1.legend(handles=[
        plt.Line2D([0], [0], color='r', lw=2, label='A·i (1st col)'),
        plt.Line2D([0], [0], color='g', lw=2, label='A·j (2nd col)'), 
        plt.Line2D([0], [0], color='b', lw=2, label='A·k (3rd col)')
    ], loc='upper left', fontsize=9)
    _ax1.set_box_aspect([1,1,1])
    _ax1.grid(True, alpha=0.3)
    _ax1.set_xlabel('X', fontsize=9)
    _ax1.set_ylabel('Y', fontsize=9)
    _ax1.set_zlabel('Z', fontsize=9)

    # Plot for Orthogonal Matrix Q
    _ax2 = fig2.add_subplot(122, projection='3d')
    _ax2.set_title("Orthogonal Component (Q)", fontsize=12, pad=12)
    _ax2.set_xlim([0, 1.5])
    _ax2.set_ylim([-1.5, 0])
    _ax2.set_zlim([0, -1.5])
    arrows_Q = _ax2.quiver(*np.zeros((3, 3)), *_transformed_Q,
                        color=['r', 'g', 'b'], 
                        arrow_length_ratio=0.12,
                        linewidth=2.5,
                        label=['Q·i', 'Q·j', 'Q·k'])
    _ax2.legend(handles=[
        plt.Line2D([0], [0], color='r', lw=2, label='Q·i (1st col)'),
        plt.Line2D([0], [0], color='g', lw=2, label='Q·j (2nd col)'), 
        plt.Line2D([0], [0], color='b', lw=2, label='Q·k (3rd col)')
    ], loc='upper left', fontsize=9)
    _ax2.set_box_aspect([1,1,1])
    _ax2.grid(True, alpha=0.3)
    _ax2.set_xlabel('X', fontsize=9)
    _ax2.set_ylabel('Y', fontsize=9)
    _ax2.set_zlabel('Z', fontsize=9)

    plt.tight_layout()
    fig2
    return


@app.cell(hide_code=True)
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
    ##### The matrix \( A \in \mathbb{R}^{n \times k} \) can be decomposed and be represented in other form, i.e.:

    \[
    A = QR
    \]

    _where,_

    - ##### \( Q \in \mathbb{R}^{n \times k} \) contains **orthonormal columns** derived from \( A \),
    - ##### \( R \in \mathbb{R}^{k \times k} \) is an **upper triangular matrix** that stores:

        1. The **projection coefficients** used to subtract previous directions (above the diagonal), and
        2. The **norms** used to normalize each orthogonalized vector (on the diagonal).

    Each vector of \( A \) is processed by removing its projections onto all previously computed orthonormal vectors and then normalized to form the columns of \( Q \). These coefficients naturally fill the entries of \( R \), making it an upper triangular matrix.

    ##### **So the full decomposition is:**

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
def _(mo, np):

    def gs_QR_Decomposition(X:np.ndarray):
        """
        An updated version of the above one, performing QR Decomposition using the Gram-Schmidt orthogonalization process
        Args:
            A set of linearly independent vectors stored in columns in the array X.
        Returns:
            Q: matrix carrying orthonormal vectors
            R: matrix having projection coefficients of orthonormal vectors
        """
        Q = np.copy(X).astype("float64")
        R = np.zeros(X.shape).astype("float64")
        n_vecs = X.shape[1]
        length = lambda x: np.linalg.norm(x)

        for nth_vec in range(n_vecs):

            for k_proj in range(nth_vec):

                # the dot product would be the scaler coefficient 
                scaler = Q[:,nth_vec] @ Q[:,k_proj]
                projection = scaler * Q[:,k_proj]

                Q[:,nth_vec] -= projection                 # removing the Kth projection
                R[k_proj,nth_vec] = scaler                 # putting the scaler coeff. in R

            norm = length(Q[:,nth_vec])

            # handling the case if the loop encounters linearly dependent vectors. 
            # Since, they come already under the span of vector space, hence their value will be 0.
            if np.isclose(norm,0, rtol=1e-15, atol=1e-14, equal_nan=False):
                Q[:,nth_vec] = 0
            else:
                # making orthogonal vectors -> orthonormal
                Q[:,nth_vec] = Q[:,nth_vec] / norm
                # the norm will be the scaler coeff of the first projection, (can be proved through system equations)
                R[nth_vec,nth_vec] = norm

        return (Q,R)

    mo.show_code()
    return (gs_QR_Decomposition,)


@app.cell
def _(A, gs_QR_Decomposition, mo):
    QA, RA = gs_QR_Decomposition(A)
    mo.show_code()
    return QA, RA


@app.cell
def _(A, QA, RA, mo):
    _v1_stack = mo.vstack([
        mo.md("#### **Original Vectors (A)**"),
        mo.plain_text(A)
    ], align="center")

    _v2_stack = mo.vstack([
        mo.md("#### **Q**"),
        mo.plain_text(QA)
    ],align="center")

    _v3_stack = mo.vstack([
        mo.md("#### **R**"),
        mo.plain_text(RA)
    ],align="center")


    stack = mo.hstack([_v1_stack,mo.md("### QR Decomposition ➡️").center(), _v2_stack, _v3_stack],
             align="center",gap=0, widths=[0.3,0.5,0.20,0.30])

    mo.show_code(stack,position="above")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    <br>
    Since, the necessary matrices are produced. Let's check whether their dot product i.e. `QA @ RA` found similar to **matrix A**.
    """
    )
    return


@app.cell
def _(A, QA, RA, mo, np):
    true_similarity = np.allclose(A, QA @ RA)
    mo.show_code(true_similarity,position="above")
    return


@app.cell
def _(A, QA, RA, np, plt):
    # orientation plot

    phi = np.linspace(0, np.pi, 80)
    theta = np.linspace(0, 2*np.pi, 80)
    x = np.outer(np.sin(phi), np.cos(theta))
    y = np.outer(np.sin(phi), np.sin(theta))
    z = np.outer(np.cos(phi), np.ones_like(theta))

    sphere_points = np.vstack((x.flatten(), y.flatten(), z.flatten()))

    # Apply transformations

    transformed_A = A @ sphere_points
    transformed_Q = QA @ sphere_points
    transformed_R = RA @ sphere_points

    # Plot
    fig = plt.figure(figsize=(10, 3.5))  # Smaller plots

    # A: Full Transformation
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(
        transformed_A[0].reshape(x.shape),
        transformed_A[1].reshape(y.shape),
        transformed_A[2].reshape(z.shape),
        color='red', alpha=0.6
    )
    ax1.set_title("A: Original",fontsize=10)
    ax1.set_box_aspect([1,1,1])

    # Q: Rotation Only
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(
        transformed_Q[0].reshape(x.shape),
        transformed_Q[1].reshape(y.shape),
        transformed_Q[2].reshape(z.shape),
        color='blue', alpha=0.6
    )
    ax2.set_title("Q: Rotation Only",fontsize=10)
    ax2.set_box_aspect([1,1,1])

    # R: Stretch and Skew
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(
        transformed_R[0].reshape(x.shape),
        transformed_R[1].reshape(y.shape),
        transformed_R[2].reshape(z.shape),
        color='green', alpha=0.6
    )
    ax3.set_title("R: Stretch/Skew",fontsize=10)
    ax3.set_box_aspect([1,1,1])
    # mo.mpl.interactive(fig)
    return (fig,)


@app.cell(hide_code=True)
def _(fig, mo):
    orientation_md = mo.md(
        r"""
    <br>
    ### **Here's the breakdown (_The final Orientation_)**

    ##### The original `matrix (A)` gets transformed into decomposed matrices i.e. `Q` & `R`. The orientation of originality changes such that it preserves some of the properties. Here's the detailed explanation...
    """
    )

    plot = mo.mpl.interactive(fig)

    sidenote = mo.md(r"""**Note:** _The scale is relative here to the transformation (not absolute), but the equation is consistent._""")

    mo.vstack([orientation_md,plot,sidenote],gap=0.5)
    return


@app.cell
def _(mo):
    first = mo.md("###### This description highlights both the geometric intuition and mathematical rigor behind your visualization. Would you like to emphasize any specific aspect (e.g., applications to least-squares problems)?")
    second = first
    third = first

    mo.hstack([first,second,third],gap=3)
    return


if __name__ == "__main__":
    app.run()
