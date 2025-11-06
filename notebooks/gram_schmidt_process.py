import marimo

__generated_with = "0.14.16"
app = marimo.App(css_file="")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt
    from wigglystuff import Matrix
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


    # additional utility functions
    def to_latex(A):
        """
        rendering the matrix into LaTEX code.
        """
        rows = [" & ".join(map(str, row)) for row in A]
        mat = r"\begin{bmatrix}" + r" \\".join(rows) + r"\end{bmatrix}"
        return r"\[" + mat + r"\]"
    return Matrix, mo, np, pd, plt, px, style_dict, style_dict_2, to_latex


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # **Orthonormal Basis Constructions with Gram-Schmidt**
    ---
    """).center()
    return


@app.cell
def _(mo, style_dict):
    mo.md(
        r"""
    #### **Orthonormal basis are the cornerstone of Linear Algebra — a set of vectors which are not only mutually perpendicular (orthogonal) but also of unit length (normalized).** 

    #### **This unique combination makes them exceptionally powerful in simplifying complex problems while dealing with large equations.**

    #### **In the context of Machine Learning, orthonormal bases serve as the backbone for techniques like Singular Value Decomposition (*SVD*), Principal Component Analysis (*PCA*), and various feature engineering methods, which are yet to be shown.**
    """
    ).style(style_dict)
    return


@app.cell
def _(mo):
    mo.md(r"""<br>""")
    return


@app.cell
def _(mo, style_dict_2):
    mo.md(
        r"""
    **In Gram-Schmidt Orthogonalization, to produce Orthonormal Vectors,**  
    **We follow this simple approach,**

    1. **take a set of [linearly independent vectors](https://www.wikiwand.com/en/articles/Linear_independence) (*stored in a matrix*)** — think of it like having mix fruits both apples & bananas 🍎🍌.
    2. **We then find and cut down their projection on each other** — separating apples from bananas, so nothing overlaps.
    3. **and, normalizing and arranging them so that they become Orthogonal** — now each fruit gets its own clean basket, *representing its own unique dimension*
    """
    ).style(style_dict_2)
    return


@app.cell
def _(mo):
    # side quest - 1

    statement = mo.md("""
    #### **Still not getting what Orthogonality is???**

    *So, here is the call!*

    *it simply means,*

    #### **Perpendicular Vectors are Orthogonal Vectors**, 
    where, the dot product of any two vectors in vector space is *0*.

    """).style({'color':'purple'})

    mo.accordion({"side quest 🏴‍☠️":statement}).right()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---

    ## **A mathematical Intuition,**
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


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    ## **Python Implemenation Of Gram-Schmidt**

    looking at the python implementation of gram-schimdt process step wise, and defining a function for it, which then, we'll use it later too.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**1. firstly, defining a vector space, calling it A.**""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ```python {.marimo}
    import numpy as np
    ```

    ```python {.marimo}
    # a vector space A having independent linearity

    A = np.array([[1,0,0], [2,0,3], [4,5,6]]).T
    ```

    ```python {.marimo}
    print(A)
    ```
    """
    )
    return


@app.cell
def _(np):
    # a vector space A having independent linearity

    A = np.array([[1,0,0], [2,0,3], [4,5,6]]).T
    return (A,)


@app.cell(hide_code=True)
def _(A, mo, to_latex):
    mo.md(to_latex(A)).left()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**2. Now, let's define a func. `gram_schmidt` utilizing the Gram-Schmidt Process,**""")
    return


@app.cell
def _(np):
    # defining the gram-schmidt process

    def gram_schmidt(X:np.ndarray)->np.ndarray:

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

    return (gram_schmidt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ```python
    def gram_schmidt(X:np.ndarray)->np.ndarray:
        '''
        original -> orthogonal -> orthonormal
        args:
            A set of linearly independent vectors stored in columns in the array X.
        returns:
            Returns matrix Q of the shape of X, having orthonormal vectors for the given vectors.
        '''
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
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, style_dict_2):
    mo.md(
        r"""
    **3. To check whether our function is producing Orthonormal Vectors,**

    **Defining `is_Orthonormal`to check the ortho-normality of Matrix Q,**

    \[
    Q^T. Q = I
    \]
    """
    ).style(style_dict_2)
    return


@app.cell
def _(A, gram_schmidt, np):
    def is_Orthonormal(Q: np.ndarray)->bool:
        """
        Checks if the columns of Q are orthonormal.
        For Q with shape (m, n), this checks if Q.T @ Q == I_n
        """
        Q_TQ = Q.T @ Q
        I = np.eye(Q.shape[1], dtype=Q.dtype)
        return np.allclose(Q_TQ, I)


    # calling the function
    Q_A = gram_schmidt(A)

    return Q_A, is_Orthonormal


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ```python {.marimo}
    def is_Orthonormal(Q: np.ndarray)->bool:
        '''
        Checks if the columns of Q are orthonormal.
        For Q with shape (m, n), this checks if Q.T @ Q == I_n
        '''
        Q_TQ = Q.T @ Q
        I = np.eye(Q.shape[1], dtype=Q.dtype)
        return np.allclose(Q_TQ, I)


    # calling the function
    Q_A = gram_schmidt(A)

    # checking the condition
    is_Orthonormal(Q_A)
    ```
    """
    )
    return


@app.cell
def _(Q_A, is_Orthonormal):
    # checking the condition
    is_Orthonormal(Q_A)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""> The Orthonormal condition satisifies and hence results in TRUE. So, the above justifying the orthogonality of the matrix `Q_A`.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <wbr>
    **4. You can find the transformation we've made so far in the matrix below,**
    <wbr>
    """
    )
    return


@app.cell
def _(A, Q_A, mo, to_latex):
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


@app.cell
def _(A, Q_A, mo, np, plt):
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


    mo.md(
        f"""
    /// details | **Want a better intuition of this through a plot, click here.**
        type: info
    A quiver plot to understand how original vectors differ from Orthonormal Vectors.

    {mo.as_html(fig2)}
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ---
    ## **Try it on your own,**

    Slide the values of Matrix A, experiment with different values and check out their Orthonormal Vectors respectively.

    The radar plot showing the orientations of **Original Matrix** `(A)` and **Orthonormal Matrix** `(Q)`. The radar plot will form triangle for Q for every linear independent vectors in `A`, otherwise, the shape will be distorted.
    """
    )
    return


@app.cell
def _(A, Matrix):
    # defining the matrix from wigglystuff widget
    wiggly_matrix = Matrix(matrix=A, step=0.1, flexible_cols=True)
    return (wiggly_matrix,)


@app.cell
def _(mo, wiggly_matrix):
    # making widget accessible to marimo
    w_mat = mo.ui.anywidget(wiggly_matrix)
    return (w_mat,)


@app.cell
def _(np, w_mat):
    # retrieving the matrix from the widget
    mat = np.array(w_mat.value['matrix'])
    return (mat,)


@app.cell
def _(gram_schmidt, mat):
    # getting the orthonormal matrix using gram-schmidt for mat
    Q_mat = gram_schmidt(mat)
    return (Q_mat,)


@app.cell
def _(Q_mat, mat, mo, np, pd, px):
    # radar plot
    data = {
        'n_vec' : [f"v{i+1}" for i in range(mat.T.shape[1])],
        'A' : [np.linalg.norm(i) for i in mat.T],
        'Q' : [np.linalg.norm(i) for i in Q_mat.T]
    }

    v_df = pd.melt(pd.DataFrame(data),
            id_vars=['n_vec'],
            value_vars=['A','Q'])

    _rd_fig = px.line_polar(
        v_df,
        r='value',
        theta='n_vec',
        color='variable',
        line_close=True,
        start_angle=90,
        markers=True,
        color_discrete_sequence=["#E74C3C", "#2980B9"],
    )

    _rd_fig.update_traces(
        fill="toself",
        opacity=0.7,
        line=dict(width=4),  
        marker=dict(size=8, symbol="circle") 
    )

    _rd_fig.update_layout(title='Original vs Orthonormal', 
                          template='simple_white', 
                          title_x=0.5, title_y=0.85,
                          polar_radialaxis_linecolor='black',
                          legend=dict(
                              yanchor="bottom", y=0.0, 
                              xanchor="right", 
                              x=1.0, orientation="v"))

    rd_fig = mo.ui.plotly(_rd_fig)
    return (rd_fig,)


@app.cell
def _(np):
    # function for checking linear dependence

    def check_linear_independence(X:np.ndarray):
        """
        checks linear independence of given matrix
        """
        rank = np.linalg.matrix_rank(X)
        n_cols = X.shape[1]
        if rank == n_cols:
            return True
        else:
            return False
    return (check_linear_independence,)


@app.cell
def _(Matrix, Q_mat, mo):
    wiggly_Q = mo.ui.anywidget(Matrix(matrix=Q_mat, static=True))
    return (wiggly_Q,)


@app.cell
def _(check_linear_independence, mat, mo, w_mat, wiggly_Q):
    rd_stack = mo.vstack([
        mo.md("#### A").center(),
        w_mat.center(),
        mo.md("<wbr>"),
        mo.md(f"**Linear Independence: {check_linear_independence(mat)}**").center(),
        mo.md("<wbr>"),
        mo.md("#### Q").center(),
        wiggly_Q.center()
    ])
    return (rd_stack,)


@app.cell
def _(mo, rd_fig, rd_stack):
    playground = mo.hstack([rd_stack, rd_fig], widths=[1,1.5],
                          align='center', justify='center')

    additional_info = mo.md(r"""The Q matrix **(denoted with blue in radar plot)** will remain fixed **(having unit length)** in radar plot, for all matrix A having linear independent vectors.""").style({'color':'blue', 'text-align':'center', 'font-size':'2rem', 'font-weight':'500'})
    return additional_info, playground


@app.cell
def _(additional_info, mo, playground):
    mo.vstack([playground, additional_info], gap=0.005)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    ## **Moving to the next Notebook**

    This notebook covered up the basics of gram-schimdt process and how orthonormal vectors are produced through it. 

    One of its basic application is in **QR Decomposition**, which we'll be exploring in the next notebook, and seeing how the matrix A will be decomposed at fixed two matrices (One will be Orthonormal & other to be Upper-Triangular).
    """
    )
    return


@app.cell
def _(mo):
    bt = mo.ui.button("Go to QR Decomposition Notebook").style({"background-color": "#0984e3",
                        "color": "white",
                        "padding": "10px 20px",
                        "border": "none",
                        "border-radius": "5px",
                        "font-size": "1rem",
                        "cursor": "pointer"}).center()


    bt
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## **Acknowledgements (Resources I learnt from...)**

    This project is undertaken through many resources, the topmost resources I learnt from,

    - [Wikipedia](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process) – for providing foundational definitions and mathematical references. 
    - [DataCamp](https://www.datacamp.com/tutorial/orthogonal-matrix) – for providing informational article upon Orthogonality.
    - [MIT OpenCourseWare](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-17-orthogonal-matrices-and-gram-schmidt/) – for refurbishing the in-depth knowledge of Gram-Schmidt Process, taught by *Prof. Gilbert Strang*.
    - [Steve Brunton (*Amazing Guy*)](https://www.google.com/search?q=steve+brunton&sca_esv=55a910f019e63594&rlz=1C1GCEA_enIN1112IN1112&sxsrf=AE3TifMoAjuMLl0MOCAV5lyl_Ga8KboiEg%3A1755118367776&ei=H_ucaP-UL_Of4-EPrsmB8QY&ved=0ahUKEwi_oOa21YiPAxXzzzgGHa5kIG4Q4dUDCBA&uact=5&oq=steve+brunton&gs_lp=Egxnd3Mtd2l6LXNlcnAiDXN0ZXZlIGJydW50b24yBBAjGCcyCxAuGIAEGJECGIoFMgsQABiABBiRAhiKBTIKEAAYgAQYQxiKBTIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABEiZC1CRBljLCHABeACQAQCYAaoBoAGvAqoBAzAuMrgBA8gBAPgBAZgCA6ACwgLCAggQABiwAxjvBcICCxAAGIAEGLADGKIEwgIKEC4YgAQYQxiKBZgDAIgGAZAGBZIHAzEuMqAHuROyBwMwLjK4B7sCwgcDMi0zyAcP&sclient=gws-wiz-serp)  – for sparking the interest, this is from where I started this project. *He has a great interest in Physics Implementation of every engineering field.*
    """
    )
    return


if __name__ == "__main__":
    app.run()
