import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return (np,)


@app.cell
def _(np):
    A = np.array([[1,0,0], [2,0,3], [4,5,6]]).T
    A
    return (A,)


@app.cell
def _(np):
    # QR Decomposition with Gram-Schmidt

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
    return


@app.cell
def _(A, np):
    np.inner(A,A)
    return


@app.cell
def _(A):
    A @ A
    return


@app.cell
def _(np):
    import plotly.graph_objects as go

    # Original vector x
    x = np.array([3, 2])

    # Unit vector u (must be unit length)
    u = np.array([1, 1])
    u = u / np.linalg.norm(u)

    # Compute inner product ⟨x, u⟩
    dot_product = np.dot(x, u)

    # Compute Householder reflection: x - 2⟨x, u⟩u
    reflected = x - 2 * dot_product * u

    # Create figure
    fig = go.Figure()

    # Plot original vector x
    fig.add_trace(go.Scatter(
        x=[0, x[0]],
        y=[0, x[1]],
        mode='lines+markers+text',
        name='x',
        line=dict(color='blue', width=3),
        text=['', 'x'],
        textposition='top right'
    ))

    # Plot unit vector u
    fig.add_trace(go.Scatter(
        x=[0, u[0]],
        y=[0, u[1]],
        mode='lines+markers+text',
        name='u (mirror normal)',
        line=dict(color='green', dash='dot'),
        text=['', 'u'],
        textposition='top right'
    ))

    # Plot reflected vector
    fig.add_trace(go.Scatter(
        x=[0, reflected[0]],
        y=[0, reflected[1]],
        mode='lines+markers+text',
        name="Householder reflection H(x)",
        line=dict(color='red', width=3, dash='dash'),
        text=['', 'H(x)'],
        textposition='bottom left'
    ))

    # Mirror line (orthogonal to u)
    slope = -u[0]/u[1]  # orthogonal slope
    x_line = np.linspace(-4, 4, 100)
    y_line = slope * x_line
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name='Mirror Line',
        line=dict(color='gray', dash='dot')
    ))

    # Layout settings
    fig.update_layout(
        title='Householder Reflection of Vector x',
        xaxis=dict(range=[-4, 4], zeroline=True),
        yaxis=dict(range=[-4, 4], zeroline=True),
        width=700,
        height=700,
        showlegend=True,
        plot_bgcolor='white'
    )

    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    fig.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
