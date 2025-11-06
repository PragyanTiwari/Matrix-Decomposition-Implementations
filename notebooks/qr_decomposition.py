import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium", css_file="")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

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

    style_dict_3 = {
        "border": "2px solid black",
        "padding": "8px",
        "border-radius": "4px",
        "display": "inline-block"
    }

    # utilities

    def to_latex(A):
        """
        rendering the matrix into LaTEX code.
        """
        rows = [" & ".join(map(str, row)) for row in A]
        mat = r"\begin{bmatrix}" + r" \\".join(rows) + r"\end{bmatrix}"
        return r"\[" + mat + r"\]"

    return mo, np, plt, style_dict, style_dict_2, to_latex


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # **QR Decomposition of Matrix with Gram-Schmidt Process**
    ---
    """).center()
    return


@app.cell
def _(mo, style_dict):
    mo.md(
        r"""

    ### **From a Broader Perspective,**

    #### **The Gram–Schmidt process doesn’t just give us the orthonormal basis, it naturally leads to the bigger picture,**
    #### **QR Decomposition, a proficient way to represent `matrix A` in the form of Orthogonality & Upper-Triangularity...**
    #### **This powerful decomposition technique is computationaly practical, helping us solve linear system & least squares problems, and many ML algorithms...**
    """
    ).style(style_dict)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
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
def _(np):
    A = np.array([[1,0,0], [2,0,3], [4,5,6]]).T
    return (A,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ```python {.marimo}
    import numpy as np
    ```

    ```python {.marimo}
    # a vector space A
    A = np.array([[1,0,0], [2,0,3], [4,5,6]]).T
    ```
    """
    )
    return


@app.cell
def _(A, np):
    def gs_QR_Decomposition(X:np.ndarray):
        """
        An updated function of the gs_Orthogonalization, adding the functionality of QR Decomposition.
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

    QA, RA = gs_QR_Decomposition(A)
    return QA, RA


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ```python {.marimo}
    def gs_QR_Decomposition(X:np.ndarray):
        '''
        An updated version of the above one, performing QR Decomposition using the Gram-Schmidt orthogonalization process
        Args:
            A set of linearly independent vectors stored in columns in the array X.
        Returns:
            Q: matrix carrying orthonormal vectors
            R: matrix having projection coefficients of orthonormal vectors
        '''
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
    ```

    ```python
    QA, RA = gs_QR_Decomposition(A)
    ```
    """
    )
    return


@app.cell
def _(A, QA, RA, mo, style_dict, to_latex):
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
def _(mo):
    mo.md(
        r"""
    ```python {.marimo}
    true_similarity = np.allclose(A, QA @ RA)
    ```

    ```python {.marimo}
    print(true_similarity)
    ```
    """
    )
    return


@app.cell
def _(A, QA, RA, np):
    np.allclose(A, QA @ RA)
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
    fig = plt.figure(figsize=(13, 5))  # Smaller plots
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
def _(fig, mo, style_dict, style_dict_2):
    # description
    orientation_md = mo.md(
        r"""
    #### **Orientation Figures from QR Decomposition**
    ---

    """
    ).style(style_dict)

    desc_md = mo.md("""
    ##### **The original `matrix (A)` gets transformed into decomposed matrices i.e. `Q` & `R`. The orientation of originality changes such that it preserves some of the properties. Here's the detailed explanation...**
    """).style(style_dict)

    # interactive plot
    plot = mo.mpl.interactive(fig)


    # notice
    sidenote = mo.md(
        r"""**NOTE:** The scale is relative here to the transformation (_not absolute_), but the equation is consistent."""
    ).style({"color": "blue"})


    # creating bullet points for interpretation
    first_ = mo.md("""
    ### **The Original 🔴**

    ##### **The red ellipsoid shape here illustrates the orientation of `matrix A`, looking stretched and reflecting how vectors are distributed in space.**
    """).style(style_dict_2).center()

    second_ = mo.md("""
    ### **The Pure Rotation 🔵**

    ##### **After extracting the orthogonal component Q, the transformation becomes a pure rotation. This preserves lengths and angles, so the shape turns into a perfect unit sphere — showing that the vectors are now absolutely orthogonal without any stretching in any direction.**
    """).style(style_dict_2).center()

    third_ = mo.md("""
    ### **The Upper Triangular 🟢**

    ##### **Even visually, matrix R being filled with values only in upper triangular proportion, the orientation will be skewed/stretched to a certain axis, containing all those vector coefficients.**
    """).style(style_dict_2).center()

    bullet_pts = mo.hstack([first_,second_,third_], align="stretch").center()

    # stacking
    mo.vstack([orientation_md, desc_md, sidenote, plot,bullet_pts]).center()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
