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
def _():
    return


if __name__ == "__main__":
    app.run()
