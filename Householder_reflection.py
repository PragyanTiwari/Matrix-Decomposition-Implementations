import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d, Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import FancyArrowPatch
    return np, plt


@app.cell
def _(np, plt):



    # Define the matrix (transposed to match your input)
    A = np.array([[1,0,0], [2,0,3], [4,5,6]]).T

    # QR decomposition
    Q, _ = np.linalg.qr(A)

    # Standard basis vectors
    basis = np.eye(3)

    # Apply transformations
    transformed_A = A @ basis
    transformed_Q = Q @ basis

    # Create figure with adjusted layout
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle('Matrix Transformation (A v/s Q)', y=1.05, fontsize=14)

    # Plot for Original Matrix A
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Original Matrix Transformation (A)", fontsize=12, pad=12)
    ax1.set_xlim([0, 10])
    ax1.set_ylim([-10, 0])
    ax1.set_zlim([0, 10])
    arrows_A = ax1.quiver(*np.zeros((3, 3)), *transformed_A, 
                        color=['r', 'g', 'b'], 
                        arrow_length_ratio=0.12,
                        linewidth=2.5,
                        label=['A·i (1st column)', 'A·j (2nd column)', 'A·k (3rd column)'])
    ax1.legend(handles=[
        plt.Line2D([0], [0], color='r', lw=2, label='A·i (1st col)'),
        plt.Line2D([0], [0], color='g', lw=2, label='A·j (2nd col)'), 
        plt.Line2D([0], [0], color='b', lw=2, label='A·k (3rd col)')
    ], loc='upper left', fontsize=9)
    ax1.set_box_aspect([1,1,1])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X', fontsize=9)
    ax1.set_ylabel('Y', fontsize=9)
    ax1.set_zlabel('Z', fontsize=9)

    # Plot for Orthogonal Matrix Q
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Orthogonal Component (Q)", fontsize=12, pad=12)
    ax2.set_xlim([0, 1.5])
    ax2.set_ylim([-1.5, 0])
    ax2.set_zlim([0, -1.5])
    arrows_Q = ax2.quiver(*np.zeros((3, 3)), *transformed_Q,
                        color=['r', 'g', 'b'], 
                        arrow_length_ratio=0.12,
                        linewidth=2.5,
                        label=['Q·i', 'Q·j', 'Q·k'])
    ax2.legend(handles=[
        plt.Line2D([0], [0], color='r', lw=2, label='Q·i (1st col)'),
        plt.Line2D([0], [0], color='g', lw=2, label='Q·j (2nd col)'), 
        plt.Line2D([0], [0], color='b', lw=2, label='Q·k (3rd col)')
    ], loc='upper left', fontsize=9)
    ax2.set_box_aspect([1,1,1])
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X', fontsize=9)
    ax2.set_ylabel('Y', fontsize=9)
    ax2.set_zlabel('Z', fontsize=9)

    plt.tight_layout()
    fig
    return A, transformed_A


@app.cell
def _(transformed_A):
    transformed_A
    return


@app.cell
def _():
    # # Example QR decomposition

    # A = np.array([[1, 1, 0],
    #               [0, 1, 1],
    #               [1, 0, 1]])
    # Q, R = np.linalg.qr(A)

    # # Create a sphere
    # phi = np.linspace(0, np.pi, 50)
    # theta = np.linspace(0, 2*np.pi, 50)
    # x = np.outer(np.sin(phi), np.cos(theta))
    # y = np.outer(np.sin(phi), np.sin(theta))
    # z = np.outer(np.cos(phi), np.ones_like(theta))

    # sphere_points = np.vstack((x.flatten(), y.flatten(), z.flatten()))

    # # Transformations
    # transformed_A = A @ sphere_points
    # transformed_Q = Q @ sphere_points
    # transformed_R = R @ sphere_points

    # # Plot
    # fig = plt.figure(figsize=(20, 6))

    # # A: Full Transformation
    # ax1 = fig.add_subplot(131, projection='3d')
    # ax1.plot_surface(
    #     transformed_A[0].reshape(x.shape),
    #     transformed_A[1].reshape(y.shape),
    #     transformed_A[2].reshape(z.shape),
    #     color='orange', alpha=0.6
    # )
    # ax1.set_title("A: Full Transformation (Q*R)")
    # ax1.set_box_aspect([1,1,1])

    # # Q: Rotation Only
    # ax2 = fig.add_subplot(132, projection='3d')
    # ax2.plot_surface(
    #     transformed_Q[0].reshape(x.shape),
    #     transformed_Q[1].reshape(y.shape),
    #     transformed_Q[2].reshape(z.shape),
    #     color='blue', alpha=0.6
    # )
    # ax2.set_title("Q: Rotation Only")
    # ax2.set_box_aspect([1,1,1])

    # # R: Stretch and Skew
    # ax3 = fig.add_subplot(133, projection='3d')
    # ax3.plot_surface(
    #     transformed_R[0].reshape(x.shape),
    #     transformed_R[1].reshape(y.shape),
    #     transformed_R[2].reshape(z.shape),
    #     color='green', alpha=0.6
    # )
    # ax3.set_title("R: Stretch/Skew")
    # ax3.set_box_aspect([1,1,1])

    # plt.tight_layout()

    # mo.mpl.interactive(fig)
    # plt.show()
    return


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
def _():
    # import plotly.graph_objects as go

    # # Original vector x
    # x = np.array([3, 2])

    # # Unit vector u (must be unit length)
    # u = np.array([1, 1])
    # u = u / np.linalg.norm(u)

    # # Compute inner product ⟨x, u⟩
    # dot_product = np.dot(x, u)

    # # Compute Householder reflection: x - 2⟨x, u⟩u
    # reflected = x - 2 * dot_product * u

    # # Create figure
    # fig = go.Figure()

    # # Plot original vector x
    # fig.add_trace(go.Scatter(
    #     x=[0, x[0]],
    #     y=[0, x[1]],
    #     mode='lines+markers+text',
    #     name='x',
    #     line=dict(color='blue', width=3),
    #     text=['', 'x'],
    #     textposition='top right'
    # ))

    # # Plot unit vector u
    # fig.add_trace(go.Scatter(
    #     x=[0, u[0]],
    #     y=[0, u[1]],
    #     mode='lines+markers+text',
    #     name='u (mirror normal)',
    #     line=dict(color='green', dash='dot'),
    #     text=['', 'u'],
    #     textposition='top right'
    # ))

    # # Plot reflected vector
    # fig.add_trace(go.Scatter(
    #     x=[0, reflected[0]],
    #     y=[0, reflected[1]],
    #     mode='lines+markers+text',
    #     name="Householder reflection H(x)",
    #     line=dict(color='red', width=3, dash='dash'),
    #     text=['', 'H(x)'],
    #     textposition='bottom left'
    # ))

    # # Mirror line (orthogonal to u)
    # slope = -u[0]/u[1]  # orthogonal slope
    # x_line = np.linspace(-4, 4, 100)
    # y_line = slope * x_line
    # fig.add_trace(go.Scatter(
    #     x=x_line,
    #     y=y_line,
    #     mode='lines',
    #     name='Mirror Line',
    #     line=dict(color='gray', dash='dot')
    # ))

    # # Layout settings
    # fig.update_layout(
    #     title='Householder Reflection of Vector x',
    #     xaxis=dict(range=[-4, 4], zeroline=True),
    #     yaxis=dict(range=[-4, 4], zeroline=True),
    #     width=700,
    #     height=700,
    #     showlegend=True,
    #     plot_bgcolor='white'
    # )

    # fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    # fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    # fig.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
