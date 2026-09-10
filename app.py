import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


st.set_page_config(
    page_title="Insurance linear regression analysis",
    layout="wide"
)

st.write("Linear Regression Report")



st.header("Data review and preprocessing")
df = pd.read_csv("insurance.csv")
df["gender"]= df["sex"].map({"male":0, "female":1})
df["is_smoker"]= df["smoker"].map({"yes":1, "no":0})
st.header("1. Dataset Overview")

st.write(
    "This dataset contains information about age, BMI, "
    "number of children, gender, smoking status, and insurance charges."
    "And there is no null or missing values in the dataset."
)

st.dataframe(df.head())




st.header("Linear regression model")

X = df[
    ["age", "bmi", "children", "gender", "is_smoker"]
]

y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=123
)

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)



st.header("Model Coefficients")

st.write(f"Intercept: {lr.intercept_:.2f}")

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_
})

st.dataframe(coef_df)




st.header("Model performance")

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

col1, col2, col3 = st.columns(3)

col1.metric(
    "R² Score",
    f"{r2:.3f}"
)

col2.metric(
    "MSE",
    f"{mse:,.2f}"
)

col3.metric(
    "MAE",
    f"{mae:,.2f}"
)


st.header("Residual")

residuals = y_test - y_pred

fig, ax = plt.subplots()

ax.scatter(
    y_pred,
    residuals
)

ax.axhline(
    y=0,
    linestyle="--"
)

ax.set_xlabel("Predicted Charges")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Predicted Values")

st.pyplot(fig)



fig, ax = plt.subplots()

pd.Series(
    residuals
).plot(
    kind="kde",
    ax=ax
)

ax.axvline(
    x=0,
    linestyle="--"
)

ax.set_xlabel("Residual")
ax.set_title("KDE Plot of Residuals")

st.pyplot(fig)

