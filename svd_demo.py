import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium", css_file="custom.css")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
def _():
    style_dict = {
        "color": "#1e3d59",
        "font-family": 'Lora',
        "font-size": "1.1rem",
        "line-height": "1.7",
        "background": "linear-gradient(135deg, #f0f8ff, #e6f7ff)",
        "padding": "12px 18px",
        "border-radius": "12px"
    }

    style_dict_2 = {
        "color": "#2d3436",
        "font-family": "Roboto",
        "font-size": "1.05rem",
        "line-height": "1.6",
        "letter-spacing": "0.5px",
        "background-color": "#f8f9fa",
        "padding": "12px 18px",
        "border-radius": "8px"
    }

    return (style_dict_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # **Orthonormal Basis Constructions with Gram-Schmidt Algorithm**
    ---
    """).center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **_Talking about Orthogonality..._**

    #### **Orthonormal basis are the cornerstone of Linear Algebra — a set of vectors that are not only mutually perpendicular (orthogonal) but also of unit length (normalized). This unique combination makes them exceptionally powerful in simplifying complex problems. In Machine Learning, orthonormal bases serve as the backbone for techniques like Singular Value Decomposition (*SVD*), Principal Component Analysis (*PCA*), and various feature engineering methods**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, style_dict_2):
    mo.md("""
    **In Gram-Schmidt Orthogonalization,  We simply,**

    1. take a set of linearly independent vectors (*stored in a matrix*). Think of it like having mix fruits both apples & bananas 🍎🍌.
    2. We then find and cut down their projection on each other — separating apples from bananas, so nothing overlaps.
    3. and, normalizing and arranging them so that they become Orthogonal — now each fruit gets its own clean basket, ***representing its own unique dimension***.
    """).style(style_dict_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### **In Technical Terms,**

    **_An orthogonal matrix_ represents a linear transformation that preserves both vector lengths and angles. It could be a rotation, a reflection, or a combination in _n_-dimensional space. The key insight is that multiplying a vector by an orthogonal matrix changes _where_ it points, but not _how long_ it is.**


    **_Through this notebook_, it'll help you build understanding of the mathematical Intuition along with its scratch implementation in python. Also check out, how this orthogonalization process plays a key role in QR Decomposition, and understand how a matrix’s orientation changes through a transformation.**
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""This notebook is the first draft of the project of  Matrix Decomposition Pipeline & its real-life Applications (you'll see in further update in the [repo](https://github.com/PragyanTiwari/Matrix-Decompositions-Implementation-for-SVD-PCA) .)""")
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

    mo.accordion({"side quest 🏴‍☠️":statement}).center()
    return


@app.cell
def _(np):
    A = np.array([[1,0,0], [2,0,3], [4,5,6]]).T
    Q,_ = np.linalg.qr(A)

    def to_latex(A):
        """
        rendering the matrix into LaTEX code.
        """
        rows = [" & ".join(map(str, row)) for row in A]
        mat = r"\begin{bmatrix}" + r" \\".join(rows) + r"\end{bmatrix}"
        return r"\[" + mat + r"\]"
    return A, Q, to_latex


@app.cell
def _(A, Q, mo, to_latex):
    matrices = {"Original Vectors":mo.md(to_latex(A)),
                "Orthogonal Vectors":mo.md(to_latex(Q))}

    radio = mo.ui.radio(options=matrices,
                value="Original Vectors",
                label="#### select the matrix")
    return (radio,)


@app.cell
def _(mo, radio):
    mo.hstack([radio.center(),radio.value.center()])
    return


if __name__ == "__main__":
    app.run()
