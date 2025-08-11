import marimo

__generated_with = "0.14.16"
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
    # **Orthonormal Basis Constructions with Gram-Schmidt Algorithm**
    ---
    """).center()
    return


@app.cell(hide_code=True)
def _():
    # styling dicts for markdown

    style_dict = {
        "color": "#2d3436",
        "font-family": "Roboto",
        "font-size": "1.05rem",
        "line-height": "1.6",
        "letter-spacing": "0.5px",
        "padding": "12px 18px",
        "border-radius": "8px"
    }

    style_dict_2 = {
        "background-color": "#f9f9f9",
        "padding": "12px",
        "border-radius": "8px",
        "line-height": "1.6"
    }
    return style_dict, style_dict_2


@app.cell(hide_code=True)
def _(mo, style_dict):
    mo.md(
        r"""
    #### **Orthonormal basis are the cornerstone of Linear Algebra — a set of vectors that are not only mutually perpendicular (orthogonal) but also of unit length (normalized). This unique combination makes them exceptionally powerful in simplifying complex problems. In Machine Learning, orthonormal bases serve as the backbone for techniques like Singular Value Decomposition (*SVD*), Principal Component Analysis (*PCA*), and various feature engineering methods**
    """
    ).style(style_dict)
    return


@app.cell(hide_code=True)
def _(mo, style_dict_2):
    mo.md("""
    **In Gram-Schmidt Orthogonalization, to produce Orthonormal VECTORS,  We simply,**

    1. take a set of linearly independent vectors (*stored in a matrix*). Think of it like having mix fruits both apples & bananas 🍎🍌.
    2. We then find and cut down their projection on each other — separating apples from bananas, so nothing overlaps.
    3. and, normalizing and arranging them so that they become Orthogonal — now each fruit gets its own clean basket, ***representing its own unique dimension***.
    """).style(style_dict_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### **In Technical Terms, we understand that,**

    ##### **An orthogonal matrix represents a linear transformation preserving both vector lengths and angles. It could be a rotation, a reflection, or a combination in _n_-dimensional space. The key insight is that multiplying a vector by an orthogonal matrix changes _where_ it points, but not _how long_ it is.**


    **Through this notebook**, you'll build the understanding of the mathematical Intuition along with its scratch implementation in python. Also check out, how this orthogonalization process plays a key role in QR Decomposition, and understand how a matrix’s orientation changes through a transformation.
    """
    ).style({"color": "blue"})
    return


@app.cell
def _(mo):
    # side quest - 1

    statement = mo.md("""
    #### **Still not getting what Orthogonality is???**

    *here is the call! and it simply means,*

    #### **Perpendicular Vectors == Orthogonal Vectors**, 
    where, the dot product of any two vectors in vector space is *0*.

    """).style({'color':'purple'})

    mo.accordion({"side quest 🏴‍☠️":statement}).right()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ## **Getting its Mathematical Overview**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, style_dict_2):
    mo.md(
        r"""
    #### For a vector space having basis \( \{ \vec{v}_1, \ldots, \vec{v}_m \} \) of a subspace \( S \subset \mathbb{R}^n \), the **Gram–Schmidt** process constructs an _**orthonormal basis**_ \( \{ \vec{w}_1, \vec{w}_2, \ldots, \vec{w}_m \} \), such that:

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
    ).style(style_dict_2)
    return


@app.cell(hide_code=True)
def _(mo):
    side_note_for_norm = mo.md(r"""
    ##### **An info. to be pointed out...** 

    In the Gram–Schmidt process, the norm \( \| \cdot \| \) used here is the **Euclidean norm** (also known as the **\(\ell^2 \)** norm).

    \[
    \| \vec{v} \| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \left( \sum_{i=1}^n v_i^2 \right)^{1/2}
    \]


    Measuring the **Euclidean distance** of a vector \( \vec{v} \in \mathbb{R}^n \) from the origin.
    """)
    mo.callout(side_note_for_norm,kind="neutral")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Here's the Scratch Implementation**
    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**first, defining a vector space, calling it A.**""")
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
    mo.md(r"""###### **Now, let's define func. `gs_Orthogonalization` which will utilize the Gram-Schmidt Process,**""")
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
def _(mo, style_dict_2):
    mo.md(
        r"""
    ###### **Now, we'll define a func. `is_orthonormal` to check the orthonormality of a Matrix satisfying the following fundamental step,**
    \[
    Q^T. Q = I
    \]
    """
    ).style(style_dict_2)
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


    # calling the function
    Q_A = gs_Orthogonalization(A)

    # checking the condition
    is_Orthonormal(Q_A)

    mo.show_code()

    return Q_A, is_Orthonormal


@app.cell
def _(Q_A, is_Orthonormal, mo):
    mo.plain(is_Orthonormal(Q_A)).child
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##### **The above justifies the orthogonality achieved by the matrix `Q_A`. You can find the changes we've made till now below,**""")
    return


@app.cell
def _(A, Q_A, mo):
    matrices = {"Original Vectors":[mo.md(to_latex(A)), mo.md("## hmm...").left()],
                "Orthonormal Vectors":[mo.md(to_latex(Q_A.astype("int64"))), mo.md("## Perfect.").left()]}

    radio = mo.ui.radio(options=matrices,
                value="Original Vectors",
                label="#### **select the matrix**")
    return (radio,)


@app.cell
def _(mo, radio, style_dict):
    mo.hstack([radio.center(), radio.value[0].center(), radio.value[1].left()],
              widths=[1,2,1],
              align="center").style(style_dict)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(""" """)
    return


@app.cell
def _(A, Q_A, mo, np, plt, style_dict_2):
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

    mo.accordion({"want a BETTER intuition of this, click here...":fig2}).style(style_dict_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <br>
    ## **QR Decomposition via the Gram–Schmidt Process**
    ---
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### **From a Broader Perspective,**

    ##### **The Gram–Schmidt process doesn’t just give us the orthonormal basis, it naturally leads to the bigger picture,**

    ##### **_QR Decomposition_, a proficient way to represent `matrix A` in the form of Orthogonality & Upper-Triangularity...**

    ##### **This powerful decomposition technique is computationaly practical, helping us solve linear system & least squares problems, and many ML algorithms...**

    **You can learn more about QR Decomposition [here](https://en.wikipedia.org/wiki/QR_decomposition#:~:text=In%20linear%20algebra%2C%20a%20QR,is%20the%20basis%20for%20a).**

    <br>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, style_dict_2):
    mo.md(
        r"""
    #### **A simple understanding of its working is written here,**
    ---

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
    ).style(style_dict_2)
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


@app.function
def to_latex(A):
    """
    rendering the matrix into LaTEX code.
    """
    rows = [" & ".join(map(str, row)) for row in A]
    mat = r"\begin{bmatrix}" + r" \\".join(rows) + r"\end{bmatrix}"
    return r"\[" + mat + r"\]"


@app.cell
def _(A, QA, RA, mo, style_dict):
    _v1_stack = mo.vstack([
        mo.md("#### **Original Vectors (A)**"),
        mo.md(to_latex(A))
    ], align="center")

    _v2_stack = mo.vstack([
        mo.md("#### **Orthonormal (Q)**"),
        mo.md(to_latex(QA.astype("int64")))
    ],align="center")

    _v3_stack = mo.vstack([
        mo.md("#### **Upper Triangular (R)**"),
        mo.md(to_latex(RA.astype("int64")))
    ],align="center")


    stack = mo.hstack([_v1_stack,mo.md("## **QR Decomposition** ➡️").center(), _v2_stack, _v3_stack],
             align="center",gap=0, widths=[0.3,0.5,0.20,0.30]).style(style_dict)

    stack
    return


@app.cell
def _(mo, style_dict):
    mo.md(
        r"""
    <br>
    **Since, the necessary matrices are produced. Let's check whether their dot product i.e. `QA @ RA` found similar to matrix A.**
    """
    ).style(style_dict)
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
    fig = plt.figure(figsize=(9.5, 5))  # Smaller plots
    fig.suptitle("Orientation Figures of the Transformations")
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


    return (fig,)


@app.cell
def _(fig, mo):
    # description
    orientation_md = mo.md(
        r"""
    <br>
    ### **Here's the breakdown (_The final Orientation_)**

    ##### The original `matrix (A)` gets transformed into decomposed matrices i.e. `Q` & `R`. The orientation of originality changes such that it preserves some of the properties. Here's the detailed explanation...
    """
    )

    # interactive plot
    plot = mo.mpl.interactive(fig).callout()


    # notice
    sidenote = mo.md(
        r"""**Note:** The scale is relative here to the transformation (_not absolute_), but the equation is consistent."""
    ).style({"color": "red"})


    # creating bullet points for interpretation
    _first = mo.md("""
    ### **The Original 🔴**

    The ellipsoid shape here illustrating the orientation of **matrix A**, looks stretched and carrying the real vector values.
    """).center()

    _second = mo.md("""
    ### **The pure rotation 🔵**

    The pure non-linear transformation done by Gram-Schmidt Orthogonalization, taking the orientation of **unit sphere**, containing Orthonormal Vectors, preserving the magnitude ratio and structure of the original matrix.
    """).center()

    _third = mo.md("""
    ### **The Upper Triangular 🟢**

    The stretched/skewed ellipsoid containing the necessary coefficients for the Matrix Q. _It is skewed in general, since the lower triangular values are 0._
    """).center()

    bullet_pts = mo.hstack([_first,_second,_third],gap=2)

    # stacking
    mo.vstack([orientation_md, sidenote, plot,bullet_pts]).center()
    return


if __name__ == "__main__":
    app.run()
