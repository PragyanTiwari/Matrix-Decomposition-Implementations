import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium", css_file="custom.css")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # **Orthonormal Basis ConH
    """).center()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## _Talking about Orthogonality..._

    #### Orthonormal basis are cornerstone of Linear Algebra, especially in Machine Learning techniques such as **SVD**, **PCA**, and **Feature Engineering**.

    #### In **Gram-Schmidt Orthogonalization**,  

    We simply, 

    1. take a set of linearly independent vectors (*stored in a matrix*). Think of it like having mix fruits both apples & bananas 🍎🍌.
    2. We then find and cut down their projection on each other --- separating apples from bananas, so nothing overlaps.
    3. and, normalizing and arranging them so that they become Orthogonal --- now each fruit gets its own clean basket, ***representing its own unique dimension***.

    In Technical Terms,

    #### **_An orthogonal matrix_ represents a linear transformation that preserves both vector lengths and angles. It could be a rotation, a reflection, or a combination in _n_-dimensional space. The key insight is that multiplying a vector by an orthogonal matrix changes _where_ it points, but not _how long_ it is.**


    #### **_Through this notebook_, it'll help you build understanding of the mathematical Intuition along with its scratch implementation in python. Also check out, how this orthogonalization process plays a key role in QR Decomposition, and understand how a matrix’s orientation changes through a transformation.**


    This notebook is the first draft of the project of  Matrix Decomposition Pipeline & its real-life Applications (you'll see further in the updates...).
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
