import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    return mo, pd


@app.cell
def _(pd):
    df = pd.read_csv("parkinsons_updrs.csv")
    df.head()
    return (df,)


@app.cell
def _(df, mo):
    table = mo.ui.table(df.corr().round(decimals=2), show_column_summaries=False)
    table
    return


@app.cell
def _(df, pd):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    x_df = df.drop(["subject#","motor_UPDRS","total_UPDRS"],axis=1)
    X = add_constant(x_df)

    vif_data = {
        "features":X.columns.tolist(),
        "vif score":[variance_inflation_factor(X.values,i) for i in range(len(X.columns.tolist()))]
    }

    vif_df = pd.DataFrame(vif_data)
    return X, vif_df, x_df


@app.cell
def _(vif_df):
    vif_df.sort_values(by=["vif score"], ascending=False)
    return


@app.cell
def _(x_df):
    features_lst = x_df.columns.tolist()

    shimmer_features = [f for f in features_lst if f.startswith("Shimmer")]
    jitter_features = [j for j in features_lst if j.startswith("Jitter")]
    return (shimmer_features,)


@app.cell
def _(X, shimmer_features):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize the data
        ('pca', PCA(n_components=2))               # Apply PCA
    ])

    pca_shimmer = pipeline.fit_transform(X[shimmer_features].to_numpy())

    # The pipeline can now be used like any other scikit-learn estimator
    # For example, to fit the pipeline to your data:
    # pipeline.fit(X_train)

    # And to transform your data:
    # X_transformed = pipeline.transform(X_train)

    # Or to fit and transform in one step:
    # X_transformed = pipeline.fit_transform(X_train)
    return (pipeline,)


@app.cell
def _(pipeline):
    pipeline.pca
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
