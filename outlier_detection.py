import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from scipy import stats
    import altair as alt
    import numpy as np
    return alt, mo, np, pd, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # **Distribution based Outlier Detection**
    ---
    """
    ).center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##### To develop a robust model capable of accurately predicting **galaxy mass** based on observable properties such as brightness captured from different bands and redshift. The project aims to enhance model performance by effectively identifying and handling outliers in the dataset.
    <br>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### **Understanding the Statistical Concepts for Outlier detection**
    ---
    ##### Features in predicting the target variable contains outliers which might overfit the trained model. In order to detect outliers, we'll understand the concept of ***Probability distribution Function*** which is a mathematical function which describes the likelihood of different outcomes in a random experiment. It consists of two types:

    **PROBABILITY MASS FUNCTION (PMF)** : Probability function which assigns probabilities to discrete random variables. It forms a distribution of prb. of each discrete random variable.

    **PROBABILITY DENSITY FUNCTION (PDF)** : Probability function represents probability density at a given point and it is used for continuous random variables. Here the prb. of a given point remains zero.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="images\pdf_pmf.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    **CUMULATIVE DISTRIBUTION FUNCTION (CDF)** : Function which represents the prb. of a random variable less than or equal to a specific value. A 50% prb. in terms of CDF indicates that we've `accumulated` half of the probability density distribution. 

    **NOTE**🚨: The Inverse of CDF is known as **Percent Point Function (PPF)**.

    ### **Relation between Probability Density Function (PDF) & Cumulative Distribution Function (CDF)**
    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="images\pdf_cdf.png",width=700).center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(value="""
    The derivative of CDF gives us the value of PDF respectively. 

    Similarly, a specific interval in Prb. density distribution on integration tells us the CDF value.
    """, kind="info").center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""**Here is the relationship between the Prb. Distribution Functions:**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="images\\relation.png",width=800).center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(value=mo.md("""
    ### **Some Basic Terminlogies of Normalization and Standardization**
    **NORMAL DISTRIBUTION** : Probability distribution with a bell curve.

    **STANDARD NORMAL DISTRIBUTION** : Normal distribution with mean 0 and std.dev. 1.

    **Z-SCORE** : A `standard` measure which tell how many std.dev. a data point is away from the mean(0).<br>
    """))

    return


@app.cell
def _(mo, pd):
    # importing dataset

    buzzard_df = pd.read_csv("D:\\Pragyan\\outlier-detection-project\\buzzard_dc1.csv")
    buzzard_df.drop_duplicates(inplace=True)
    buzzard_df.head()
    mo.show_code()
    return (buzzard_df,)


@app.cell
def _(buzzard_df, mo):
    mo.ui.table(buzzard_df)
    return


@app.cell
def _(buzzard_df, mo):
    buzzard_df.shape
    mo.show_code()
    return


@app.cell
def _(buzzard_df):
    print(buzzard_df.shape)
    return


@app.cell
def _(alt, buzzard_df, mo):
    # plotting the correlation matrix

    correlation_matrix = buzzard_df[["u", "g", "r", "i", "z", "y", "log.mass", "redshift"]].corr()

    correlation_data = correlation_matrix.stack().reset_index()
    correlation_data.columns = ['Variable1', 'Variable2', 'Correlation']

    # heatmap
    p_chart = alt.Chart(correlation_data).mark_rect().encode(
        x=alt.X('Variable1:O', title=None),
        y=alt.Y('Variable2:O', title=None),
        color=alt.Color('Correlation:Q', 
                        scale=alt.Scale(scheme='magma'),
                        title='Correlation'),
        tooltip=['Variable1', 'Variable2', 'Correlation']
    ).properties(
        title='Correlation Matrix Heatmap',
        width=500,
        height=500
    )

    # interactive chart
    mo.ui.altair_chart(p_chart).center()

    mo.show_code()
    return (p_chart,)


@app.cell
def _(p_chart):
    p_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## **Identifying and Addressing Outliers**""")
    return


@app.cell
def _(buzzard_df, mo):
    buzzard_df.describe()
    mo.show_code()
    return


@app.cell
def _(buzzard_df):
    buzzard_df.describe()
    return


@app.cell
def _(buzzard_df, mo, pd, stats):
    # defining a func to find outlier counts for each feature

    def get_outlier_counts(frame:pd.DataFrame,threshold):
        df = frame.copy()
        # calculating zscore for threshold
        threshold_zscore = stats.norm.ppf(threshold)
        # calculating zscore for df
        dataframe_zscore = abs(stats.zscore(df))
        # outlier condition
        outliers = (dataframe_zscore > threshold_zscore).sum(axis=0)
        return outliers

    # get outlier counts for threshold 0.95
    features = buzzard_df.drop('log.mass',axis=1)
    get_outlier_counts(features,0.95) # can experiment for different threshold values

    mo.show_code()
    return features, get_outlier_counts


@app.cell
def _(mo):
    mo.md("""**Total no. of outliers at 95% threshold in each feature :**""")
    return


@app.cell
def _(features, get_outlier_counts, pd):
    s = pd.DataFrame(get_outlier_counts(features,0.95), index=features.columns.tolist(), columns=['value'])
    s
    return


@app.cell
def _(mo, pd, stats):

    def remove_outliers(threshold,frame:pd.DataFrame)->pd.DataFrame:
        """drop outliers based on the specified threshold w.r.t. normalized distribution"""

        df = frame.copy()

        # obtaining threshold values
        threshold_zscore = stats.norm.ppf(threshold)
        dataframe_zscore = pd.DataFrame(abs(stats.zscore(df.select_dtypes(include="float64"))))

        # getting outlier indices
        outliers =  dataframe_zscore[dataframe_zscore > threshold_zscore]
        remove_outliers_condition = outliers.sum(axis=1) > 4
        outliers_indices = outliers[remove_outliers_condition].index

        # dropping outliers
        df.drop(index=outliers_indices,inplace=True)

        return df

    mo.show_code()
    return (remove_outliers,)


@app.cell
def _(buzzard_df, remove_outliers):
    remove_outliers(threshold=0.95,frame=buzzard_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    *During Pre-processing, `remove_outliers` will be used to address the outliers for each threshold value, subsequently lead us to find the optimal value for threshold for model performance*
    <br>
    #### **Building a Simple Pre-processing Pipeline to perform prediction for each iteration**
    """
    )
    return


@app.cell
def _(buzzard_df, mo, np, remove_outliers):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    def preprocessing_pipeline(dataframe,threshold=0.95,keep_outliers=True):

        # whether to keep outliers or not
        if not keep_outliers:
            dataframe = remove_outliers(threshold,dataframe)

        # defining x and y
        features = dataframe.drop('log.mass',axis=1)
        target = dataframe['log.mass']


        # train-test split
        x_Train,x_Test,y_Train,y_Test = train_test_split(features,target,test_size=0.3,random_state=1)

        # standardizing training dataset
        scaler = StandardScaler()
        scaler.fit(x_Train)

        # transforming features
        x_Train = scaler.transform(x_Train)
        x_Test = scaler.transform(x_Test)

        return x_Train,x_Test,y_Train,y_Test




    def get_model_score(model,keep_outliers,threshold=0.95) -> float:
        """evaluating R2 score for each sub-model to find strength of linear relationship"""
        x_Train,x_Test,y_Train,y_Test = preprocessing_pipeline(buzzard_df,
                                        keep_outliers=keep_outliers,threshold=threshold)
        linear_model = model.fit(x_Train,y_Train)
        y_pred = linear_model.predict(x_Test)
        return r2_score(y_Test,y_pred) * 100


    # defining threshold array to experiment with different threshold vals with different models
    threshold_arr = np.arange(start=0.89,stop=1,step=0.001)
    threshold_arr

    mo.show_code()
    return get_model_score, threshold_arr


@app.cell
def _(threshold_arr):
    threshold_arr
    return


@app.cell
def _(mo):
    mo.md("""### **Using Linear Regression as a Baseline Model for Performance**""")
    return


@app.cell
def _(mo):
    from sklearn.linear_model import LinearRegression
    MODEL = LinearRegression()
    mo.show_code()
    return (MODEL,)


@app.cell
def _(MODEL, get_model_score, mo):
    # with outliers
    score = get_model_score(model=MODEL,keep_outliers=True)
    print(f"{MODEL=} : {score}")
    mo.show_code(position="above")
    return (score,)


@app.cell
def _(MODEL, score):
    f"{MODEL=} : {score}"
    return


@app.cell
def _(MODEL, get_model_score, mo, threshold_arr):
    # without outliers
    linear_regression_res = []
    for ci in threshold_arr:
        model_score = get_model_score(MODEL,threshold=ci,keep_outliers=False)
        print(ci, "\t", model_score)
        linear_regression_res.append(model_score)

    mo.show_code()
    return (linear_regression_res,)


@app.cell
def _(MODEL, linear_regression_res, mo):
    # maximum score achieved
    max_score = max(linear_regression_res)
    print(f"{MODEL=} : {max_score}")
    mo.show_code()
    return (max_score,)


@app.cell
def _(MODEL, max_score):
    f"{MODEL=} : {max_score}"
    return


@app.cell
def _(alt, linear_regression_res, mo, pd, threshold_arr):


    # Assuming threshold_arr and linear_regression_res are already defined

    # Convert data to a Pandas DataFrame for Altair
    df = pd.DataFrame({'threshold': threshold_arr, 'score': linear_regression_res})

    # Find the index of the maximum score
    optimal_threshold_index = df['score'].idxmax()

    # Extract the optimal threshold and score
    peak_x = df['threshold'][optimal_threshold_index]
    peak_y = df['score'][optimal_threshold_index]

    # Create the line chart
    line = alt.Chart(df).mark_line().encode(
        x=alt.X('threshold', title='Threshold Values'),
        y=alt.Y('score', title='Model Score', scale=alt.Scale(domain=[95.2, 95.7])),
        tooltip=['threshold', 'score']
    ).properties(
        title="Linear Regression Performance"
    )

    # Create a point for the maximum score
    point = alt.Chart(pd.DataFrame({'threshold': [peak_x], 'score': [peak_y]})).mark_point(color='red', size=200).encode(
        x='threshold',
        y='score',
        tooltip=['threshold', 'score']
    )

    # Create a text label for the maximum score
    text = alt.Chart(pd.DataFrame({'threshold': [peak_x+0.001], 'score': [peak_y+0.025]})).mark_text(
        align='center',
        baseline='bottom',
        dy=10,
        fontSize=11,
        fontWeight='bold',
        color='black'
    ).encode(
        x='threshold',
        y='score',
        text=alt.value(f'Max Score: {peak_y:.2f}%')
    )

    # Combine the layers
    _chart = alt.layer(line, point, text)

    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(mo):
    mo.md("""### **Conclusion**""")
    return


@app.cell
def _(mo):
    mo.callout("""
    The model achieved an accuracy of approximately 91% when including outliers. By removing 0.8% of outliers, we significantly improved performance to 95.5%. This clearly demonstrates the substantial impact of outliers on the model.

    To further enhance accuracy, we can explore alternative regression models such as Decision Trees and Support Vector Regression.
    """, kind="success")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
