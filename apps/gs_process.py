# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.18.4",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
#     "pandas==3.0.1",
#     "plotly==6.6.0",
#     "pyzmq>=27.1.0",
#     "wigglystuff>=0.2.5",
# ]
# [tool.marimo.display]
# custom_css = ["public/custom.css"]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Google+Sans+Flex:wght@420;430;500;600&family=Google+Sans+Code:wght@400;500&display=swap');

    :root {
        --marimo-text-font: 'Google Sans Flex', sans-serif !important;
        --marimo-heading-font: 'Geist', sans-serif !important;
        --marimo-monospace-font: 'Google Sans Code', monospace !important;
    }

    .prose {
        font-family: var(--marimo-text-font) !important;
        font-size: 17px !important;
        line-height: 1.5 !important;
    }

    .prose p,
    .prose ul,
    .prose ol,
    .prose li,
    .prose table {
        font-family: var(--marimo-text-font) !important;
        font-size: 16.5px !important;
        line-height: 1.5 !important;
        color: #1f2937;
    }

    .prose h1,
    .prose h2,
    .prose h3 {
        font-family: var(--marimo-heading-font) !important;
        font-weight: 500;
        letter-spacing: -0.02em;
    }

    .prose code {
        font-family: var(--marimo-monospace-font) !important;
        color: #1d4ed8 !important;
        background-color: #f3f4f6 !important;
        padding: 2px 6px !important;
        border-radius: 6px !important;
        border: 1px solid #e5e7eb !important;
        font-size: 15px !important;
        line-height: 1.5 !important;
        font-weight: 400 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        scroll-margin-top: 85px !important;
            }
    </style>
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Orthonormal Basis with Gram-Schmidt
    """).center()
    return


@app.cell
def _(mo):
    progress_state, update_state = mo.state((0, "loading the notebook..."))
    return progress_state, update_state


@app.cell(hide_code=True)
def _(mo, progress_state):
    val, current_status = progress_state._value

    # 2. Dynamically set the colors based on the value
    if val < 100:
        bg_color = "#f8fafc"
        bar_color = "#4338ca"
    else:
        bg_color = "#ecfdf5"
        bar_color = "#059669"

    # 3. Calculate the bar visuals
    bar_length = 40
    filled_blocks = int((val / 100) * bar_length)
    bar_visual = ("█" * filled_blocks) + (" " * (bar_length - filled_blocks))

    # 4. Format the final string
    tqdm_str = f"{val:3}%|{bar_visual}| {val}/100 [{current_status}]"

    # 5. Render the HTML
    mo.Html(f"""
    <div style="background-color: {bg_color}; padding: 14px 20px; border-radius: 8px; border: 1px solid #e2e8f0; width: fit-content; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <span style="color: {bar_color}; font-family: var(--marimo-monospace-font), monospace; font-size: 14.5px; font-weight: 500; white-space: pre;">{tqdm_str}</span>
    </div>
    """).center()
    return


@app.cell
def _(mo):
    # sidebar

    _heading_links = {
        "Orthonormal Basis": "#deriving-orthonormal-basis-using-gram-schmidt-process",
        "Gram-Schmidt Workflow": "#a-simple-workflow-diagram-of-gram-schmidt-process",
        "Mathematical Intuition": "#a-mathematical-intuition-of-gram-schmidt",
        "Python Implementation": "#implementing-in-python",
        "Playground": "#playground-try-on-your-own",
        "Acknowledgements": "#acknowledgements-resources-i-learnt-from",
    }

    _github_repo = "https://github.com/"
    _about_the_author = "https://github.com/prgyn8"

    _structure = {
        _heading_links.get(
            "Orthonormal Basis"
        ): f"{mo.icon('lucide:chart-no-axes-gantt')} Orthonormal Basis",
        _heading_links.get(
            "Gram-Schmidt Workflow"
        ): f"{mo.icon('lucide:workflow')} Gram-Schmidt Workflow",
        _heading_links.get(
            "Mathematical Intuition"
        ): f"{mo.icon('lucide:sigma')} Mathematical Intuition",
        _heading_links.get(
            "Python Implementation"
        ): f"{mo.icon('lucide:code-2')} Python Implementation",
        _heading_links.get(
            "Playground"
        ): f"{mo.icon('lucide:flask-conical')} Playground",
        _heading_links.get(
            "Acknowledgements"
        ): f"{mo.icon('lucide:heart-handshake')} Acknowledgements",
        _github_repo: f"{mo.icon('lucide:github')} Github",
        _about_the_author: f"{mo.icon('lucide:user-round')} Author",
    }

    _orientation = "vertical"

    _title = mo.md("# Contents \n ---").left()

    _width = "320px"

    mo.sidebar([_title, mo.nav_menu(_structure, orientation="vertical")], width=_width)
    return


@app.cell
def _(update_state):
    update_state((10, "loading markdowns..."))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **In the context of Machine Learning or every aspect of data engineering techniques, Orthonormal Basis remained a cornerstone of Linear Algebra,**

    It served as a backbone of pre-processing techniques such as Singular Value Decomposition (SVD), PCA, Image Processing etc.

    Traditional methods of predictions like _Oridinary Least Squares_, models for _Noise Reduction_ & _Signal Processing_, Orthogonality has brought the modernized ways of implementing these models.

    **Then, What are indeed Orthonormal Basis (or, Orthogonality) ?**

    An Orthonormal Basis, is a set of vectors that satisfies two strict conditions:

    - `Orthogonal:` They are mutually perpendicular (90° to each other). 📐
    - `Normalized:` They each have a length of exactly 1. 📏

    **...And, Why does this matter?**

    This unique combination simplifies complex math immensely. When vectors are orthonormal, difficult matrix inversions often turn into simple transposes.

    > This notebook talks about this orthogonality, more precisely the orthogonal vectors which we'll produce for a simple matrix A using a fundamental method called Gram-Schmidt Process.

    Here I provide both the approaches of understanding, i.e. Theory (in the form of Mathematical Intuition) & Practicality (the code implementation in python...)

    You can try the Playground at last to experiment by yourself with different matrices.

    The complete code source is available here : [github-repo](https:/github.com)

    <br>

    ### A simple workflow diagram of Gram-Schmidt Process;
    """)
    return


@app.cell(hide_code=True)
def _():
    # importing library
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt
    from wigglystuff import Matrix

    return Matrix, np, pd, plt, px


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
        "border-radius": "8px",
    }
    style_dict_2 = {
        "background-color": "#f9f9f9",
        "padding": "12px",
        "border-radius": "8px",
        "line-height": "1.6",
    }
    return style_dict, style_dict_2


@app.function(hide_code=True)
# additional utility functions

def to_latex(A):
    """
    rendering the matrix into LaTEX code.
    """
    rows = [" & ".join(map(str, row)) for row in A]
    mat = r"\begin{bmatrix}" + r" \\".join(rows) + r"\end{bmatrix}"
    return r"\[" + mat + r"\]"


@app.cell
def _(update_state):
    update_state((42, "rendering image..."))
    return


@app.cell
def _(mo):
    mo.image(
        src="public/images/gs-01.png",
        width=900,
        height=400,
        caption="take the example of fruits 🍎 🍌",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # side quest - 1

    statement = mo.md("""
    Here are some good resources to understand Orthogonality & Projections, that you can look up to;

    🌟[Graphical Linear Algebra Article](https://graphicallinearalgebra.net/2017/08/09/orthogonality-and-projections/)

    🌟[Orthogonal Vector Calculator](https://onlinemschool.com/math/assistance/vector/orthogonality/)

    """).style({"color": "purple"})

    mo.accordion({"side quest 🏴‍☠️": statement}).right()
    return


@app.cell
def _(update_state):
    update_state((57, "crunching the math..."))
    return


@app.cell(hide_code=True)
def _(mo):
    _sub_title = "Mathematical Intuition of Gram-Schmidt"
    _desc = """

    Before we write the code, let's look at the math driving it. The Gram-Schmidt process transforms standard vectors into an orthonormal basis through two repeating steps:

    1. **Orthogonalization**: Subtracting the projections of a vector onto the previously processed vectors (making them perpendicular).
    2. **Normalization**: Dividing the resulting vector by its magnitude (making it a unit vector).

    >> 💡 **Tip:** The equations below represent the exact sequential logic we will use in our Python loop. For the best understanding, try reading this math block hand-in-hand with the code implementation!
    """

    mo.vstack([mo.md(f"## {_sub_title}\n---"), mo.md(f"{_desc}")])
    return


@app.cell(hide_code=True)
def _(mo, style_dict_2):
    mo.md(
        r"""

    For a vector space having basis \( \{ \vec{v}_1, \ldots, \vec{v}_m \} \) of a subspace \( S \subset \mathbb{R}^n \), the **Gram–Schmidt** process constructs an _**orthonormal basis**_ \( \{ \vec{w}_1, \vec{w}_2, \ldots, \vec{w}_m \} \), such that:

    \[
    \operatorname{gram\_schmidt} \left( \left\{ \vec{v}_1, \vec{v}_2, \ldots, \vec{v}_m \right\} \right)
    \longrightarrow \left\{ \vec{w}_1, \vec{w}_2, \ldots, \vec{w}_m \right\}
    \]

    where each \( \vec{w}_i \) is orthonormal, and constructed via the following steps:

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

    These vectors \( \{ \vec{w}_1, \ldots, \vec{w}_m \} \) satisfy the orthonormality condition:

    \[
    \vec{w}_i^\top.\vec{w}_j =
    \begin{cases}
    1 & \text{if } i = j, \\
    0 & \text{if } i \neq j
    \end{cases}
    \]

    and such orthonormal vectors can be assembled into the columns which build an **Orthonormal Matrix \( Q \in \mathbb{R}^{n \times m} \),** such that:

    \[
    Q^T. Q = I
    \]

    In practical numerical implementations (due to rounding errors), we often get:

    \[
    Q^T. Q \approx I
    \]
    """
    ).style(style_dict_2)
    return


@app.cell(hide_code=True)
def _(mo):
    side_note_for_norm = mo.md(r"""
    ##### **Simple note to be pointed out...** 

    In the Gram–Schmidt process, the norm \( \| \cdot \| \) used here is the **Euclidean norm** (also known as the **\(\ell^2 \)** norm).

    \[
    \| \vec{v} \| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \left( \sum_{i=1}^n v_i^2 \right)^{1/2}
    \]


    Measuring the **Euclidean distance** of a vector \( \vec{v} \in \mathbb{R}^n \) from the origin.
    """)
    mo.callout(side_note_for_norm, kind="neutral")
    return


@app.cell
def _(update_state):
    update_state((75, "compiling python implementation..."))
    return


@app.cell(hide_code=True)
def _(mo):
    _sub_title = "Implementing in Python 🐍"
    _desc = """
    Now, let's bring the math to life! 🚀 We are going to translate the exact formulas from the Mathematical Intuition section into Python, right here: 
    """
    _following_steps = "Follow the below steps ⬇️ :"

    mo.vstack(
        [
            mo.md(f"## {_sub_title} \n ---"),
            mo.md(f"{_desc}"),
            mo.md(f"**{_following_steps}**"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. firstly, defining a vector space, calling it A.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
    return


@app.cell
def _(np):
    # a vector space A having independent linearity

    A = np.array([[1, 0, 0], [2, 0, 3], [4, 5, 6]]).T
    return (A,)


@app.cell
def _(A, mo):
    mo.md(to_latex(A)).left()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2. Now, let's define a func. `gram_schmidt` utilizing the Gram-Schmidt Process,
    """)
    return


@app.cell(hide_code=True)
def _(np):
    # defining the gram-schmidt process

    def gram_schmidt(X: np.ndarray) -> np.ndarray:
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
        def length(x):
            return np.linalg.norm(x)

        # iteration with each vector in the matrix X
        for nth_vec in range(n_vecs):
            # iteratively removing each preceding projection from nth vector
            for k_proj in range(nth_vec):
                # the dot product would be the scaler coefficient
                scaler = Q[:, nth_vec] @ Q[:, k_proj]
                projection = scaler * Q[:, k_proj]
                Q[:, nth_vec] -= projection  # removing the Kth projection

            norm = length(Q[:, nth_vec])

            # handling the case if the loop encounters linearly dependent vectors.
            # Since, they come already under the span of vector space, hence their value will be 0.
            if np.isclose(norm, 0, rtol=1e-15, atol=1e-14, equal_nan=False):
                Q[:, nth_vec] = 0
            else:
                # making orthogonal vectors -> orthonormal
                Q[:, nth_vec] = Q[:, nth_vec] / norm

        return Q

    return (gram_schmidt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3. To check whether our function is producing Orthonormal Vectors, we're defining `is_Orthonormal`to check the ortho-normality of the `Matrix Q`,

    \[
    Q^T. Q = I
    \]
    """)
    return


@app.cell
def _(A, gram_schmidt, np):
    def is_Orthonormal(Q: np.ndarray) -> bool:
        """
        Checks if the columns of Q are orthonormal.
        For Q with shape (m, n), this checks if Q.T @ Q == I_n
        """
        Q_TQ = Q.T @ Q
        Identity = np.eye(Q.shape[1], dtype=Q.dtype)
        return np.allclose(Q_TQ, Identity)

    # calling the function
    Q_A = gram_schmidt(A)
    return (Q_A,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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

    # >> True
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > The Orthonormal condition satisifies and hence results in TRUE. So, the above justifies we've produced the orthogonality for the matrix `Q_A`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4. **You can interact with the toggle below to visually compare the Original Vectors against our newly minted Orthonormal Vectors:**
    """)
    return


@app.cell
def _(update_state):
    update_state((78, "executing matrices..."))
    return


@app.cell
def _(A, Q_A, mo):
    matrices = {
        "Original Vectors": [mo.md(to_latex(A)), mo.md("## hmm...").left()],
        "Orthonormal Vectors": [
            mo.md(to_latex(Q_A.astype("int64"))),
            mo.md("## Perfect.").left(),
        ],
    }

    radio = mo.ui.radio(
        options=matrices,
        value="Original Vectors",
        label="#### **select the matrix 🔽**",
    )
    return (radio,)


@app.cell
def _(mo, radio, style_dict):
    mo.hstack(
        [radio.center(), radio.value[0].center(), radio.value[1].left()],
        widths=[1, 2, 1],
        align="center",
    ).style(style_dict)
    return


@app.cell(hide_code=True)
def _(A, Q_A, mo, np, plt, update_state):
    ## comparison plot (needs better desc. here too...)
    ## ---------------------------------------------------------

    update_state((84, "generating the hidden plot..."))

    # Standard basis vectors
    basis = np.eye(3)

    # Apply transformations
    _transformed_A = A @ basis
    _transformed_Q = Q_A @ basis

    # Create figure with adjusted layout
    fig2 = plt.figure(figsize=(14, 5))
    fig2.suptitle("Matrix Transformation (A v/s Q)", y=1.05, fontsize=14)

    # Plot for Original Matrix A
    _ax1 = fig2.add_subplot(121, projection="3d")
    _ax1.set_title("Original Matrix Transformation (A)", fontsize=12, pad=12)
    _ax1.set_xlim([0, 10])
    _ax1.set_ylim([-10, 0])
    _ax1.set_zlim([0, 10])
    _ax1.quiver(
        *np.zeros((3, 3)),
        *_transformed_A,
        color=["r", "g", "b"],
        arrow_length_ratio=0.12,
        linewidth=2.5,
        label=["A·i (1st column)", "A·j (2nd column)", "A·k (3rd column)"],
    )
    _ax1.legend(
        handles=[
            plt.Line2D([0], [0], color="r", lw=2, label="A·i (1st col)"),
            plt.Line2D([0], [0], color="g", lw=2, label="A·j (2nd col)"),
            plt.Line2D([0], [0], color="b", lw=2, label="A·k (3rd col)"),
        ],
        loc="upper left",
        fontsize=9,
    )
    _ax1.set_box_aspect([1, 1, 1])
    _ax1.grid(True, alpha=0.3)
    _ax1.set_xlabel("X", fontsize=9)
    _ax1.set_ylabel("Y", fontsize=9)
    _ax1.set_zlabel("Z", fontsize=9)

    # Plot for Orthogonal Matrix Q
    _ax2 = fig2.add_subplot(122, projection="3d")
    _ax2.set_title("Orthogonal Component (Q)", fontsize=12, pad=12)
    _ax2.set_xlim([0, 1.5])
    _ax2.set_ylim([-1.5, 0])
    _ax2.set_zlim([0, -1.5])
    _ax2.quiver(
        *np.zeros((3, 3)),
        *_transformed_Q,
        color=["r", "g", "b"],
        arrow_length_ratio=0.12,
        linewidth=2.5,
        label=["Q·i", "Q·j", "Q·k"],
    )
    _ax2.legend(
        handles=[
            plt.Line2D([0], [0], color="r", lw=2, label="Q·i (1st col)"),
            plt.Line2D([0], [0], color="g", lw=2, label="Q·j (2nd col)"),
            plt.Line2D([0], [0], color="b", lw=2, label="Q·k (3rd col)"),
        ],
        loc="upper left",
        fontsize=9,
    )
    _ax2.set_box_aspect([1, 1, 1])
    _ax2.grid(True, alpha=0.3)
    _ax2.set_xlabel("X", fontsize=9)
    _ax2.set_ylabel("Y", fontsize=9)
    _ax2.set_zlabel("Z", fontsize=9)

    plt.tight_layout()

    mo.md(
        f"""
    /// details | Want a better intuition of this through a plot, **click here**.
        type: info
    A quiver plot to understand how original vectors differ from Orthonormal Vectors.

    {mo.as_html(fig2)}
    ///
    """
    )
    return


@app.cell
def _(update_state):
    update_state((84, "creating playgound, standby..."))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <br>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _sub_title = "Playground (Try on your own)"

    mo.vstack([mo.md(f"## {_sub_title} \n ---")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How to use this interactive:**

    Use the [wigglystuff matrix](https://github.com/koaning/wigglystuff) to adjust `Matrix A` and observe how the radar plot reflects the scaling and rotation of your input vectors.As you experiment, watch how the Orthonormal vectors in Matrix Q react in real-time!

    Keep an eye on the **Linear Independence** status, whenever, the vectors are independent, the Q matrix will form a perfect unit triangle❕
    """)
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
    mat = np.array(w_mat.value["matrix"])
    return (mat,)


@app.cell
def _(gram_schmidt, mat):
    # getting the orthonormal matrix using gram-schmidt for mat
    Q_mat = gram_schmidt(mat)
    return (Q_mat,)


@app.cell
def _(Q_mat, mat, mo, np, pd, px):
    ## radar plot
    ## --------------------------------------------------------
    data = {
        "n_vec": [f"v{i + 1}" for i in range(mat.T.shape[1])],
        "A": [np.linalg.norm(i) for i in mat.T],
        "Q": [np.linalg.norm(i) for i in Q_mat.T],
    }

    v_df = pd.melt(pd.DataFrame(data), id_vars=["n_vec"], value_vars=["A", "Q"])

    _rd_fig = px.line_polar(
        v_df,
        r="value",
        theta="n_vec",
        color="variable",
        line_close=True,
        start_angle=90,
        markers=True,
        color_discrete_sequence=["#E74C3C", "#2980B9"],
    )

    _rd_fig.update_traces(
        fill="toself",
        opacity=0.7,
        line=dict(width=4),
        marker=dict(size=8, symbol="circle"),
    )

    _rd_fig.update_layout(
        title="Original vs Orthonormal",
        template="simple_white",
        title_x=0.5,
        title_y=0.95,
        width=500,
        height=500,
        polar_radialaxis_linecolor="black",
        legend=dict(yanchor="bottom", y=0.0, xanchor="right", x=1.2, orientation="v"),
    )

    rd_fig = mo.ui.plotly(_rd_fig)
    return (rd_fig,)


@app.cell
def _(np):
    # function for checking linear dependence

    def check_linear_independence(X: np.ndarray):
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
    rd_stack = mo.vstack(
        [
            mo.md("#### A").center(),
            w_mat.center(),
            mo.md("<wbr>"),
            mo.md(
                f"**Linear Independence: {check_linear_independence(mat)}**"
            ).center(),
            mo.md("<wbr>"),
            mo.md("#### Q").center(),
            wiggly_Q.center(),
        ]
    )
    return (rd_stack,)


@app.cell
def _(update_state):
    update_state((94, "getting closer..."))
    return


@app.cell
def _(mo, rd_fig, rd_stack):
    playground = mo.hstack(
        [rd_stack, rd_fig], widths=[1, 1.5], align="center", justify="center"
    )

    additional_info = mo.md(
        r"""The Q matrix **(denoted with blue in radar plot)** will remain fixed **(having unit length)** in radar plot, for all matrix A having linear independent vectors."""
    ).style({"color": "blue", "text-align": "center"})
    return additional_info, playground


@app.cell
def _(additional_info, mo, playground):
    mo.vstack([playground, additional_info], gap=0.005)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Acknowledgements (Resources I learnt from...)

    This project is undertaken through many resources, the topmost resources I learnt from,

    - [Wikipedia](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process) – for providing foundational definitions and mathematical references.
    - [DataCamp](https://www.datacamp.com/tutorial/orthogonal-matrix) – for providing informational article upon Orthogonality.
    - [MIT OpenCourseWare](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-17-orthogonal-matrices-and-gram-schmidt/) – for refurbishing the in-depth knowledge of Gram-Schmidt Process, taught by *Prof. Gilbert Strang*.
    - [Steve Brunton (*Amazing Guy*)](https://www.google.com/search?q=steve+brunton&sca_esv=55a910f019e63594&rlz=1C1GCEA_enIN1112IN1112&sxsrf=AE3TifMoAjuMLl0MOCAV5lyl_Ga8KboiEg%3A1755118367776&ei=H_ucaP-UL_Of4-EPrsmB8QY&ved=0ahUKEwi_oOa21YiPAxXzzzgGHa5kIG4Q4dUDCBA&uact=5&oq=steve+brunton&gs_lp=Egxnd3Mtd2l6LXNlcnAiDXN0ZXZlIGJydW50b24yBBAjGCcyCxAuGIAEGJECGIoFMgsQABiABBiRAhiKBTIKEAAYgAQYQxiKBTIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABEiZC1CRBljLCHABeACQAQCYAaoBoAGvAqoBAzAuMrgBA8gBAPgBAZgCA6ACwgLCAggQABiwAxjvBcICCxAAGIAEGLADGKIEwgIKEC4YgAQYQxiKBZgDAIgGAZAGBZIHAzEuMqAHuROyBwMwLjK4B7sCwgcDMi0zyAcP&sclient=gws-wiz-serp)  – for sparking the interest, this is from where I started this project. *He has a great interest in Physics Implementation of every engineering field.*
    """)
    return


@app.cell
def _(update_state):
    update_state((100, "Done 🙌"))
    return


if __name__ == "__main__":
    app.run()
