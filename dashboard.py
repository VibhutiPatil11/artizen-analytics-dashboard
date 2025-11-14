import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.title("📊 Artizen Analytics Dashboard")

df = pd.read_csv("data/sales.csv")
st.write("### Raw Sales Data")
st.dataframe(df)

# 1. Regional Demand Chart
fig = px.bar(df, x="region", y="sales", color="product")
st.plotly_chart(fig)

# 2. Price vs Sales Scatter
fig2 = px.scatter(df, x="price", y="sales", color="product")
st.plotly_chart(fig2)

# 3. ML Prediction (Random Forest)
st.subheader("🔮 Predict Demand Based on Price")

model = RandomForestRegressor()
model.fit(df[["price"]], df["sales"])

input_price = st.slider("Select a price to predict demand", 100, 2000)

prediction = model.predict([[input_price]])[0]

st.success(f"Expected Sales at ₹{input_price}: {int(prediction)} units")
