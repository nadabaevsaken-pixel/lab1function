import streamlit as st
import json
import pandas as pd

# Деректерді оқу
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

st.title("🛒 Simple Shop Statistics")

# Барлық тауарларды шығару
st.subheader("All Products")
st.dataframe(df)

# Категория бойынша санау
st.subheader("📊 Products by Category")
st.bar_chart(df["category"].value_counts())

# Орташа баға
st.subheader("💰 Average Price by Category")
avg_price = df.groupby("category")["price"].mean()
st.bar_chart(avg_price)

# Орташа рейтинг
st.subheader("⭐ Average Rating")
avg_rating = df["rating"].mean()
st.success(f"Average rating across all products: {avg_rating:.2f}")

# Баға мен рейтинг scatter plot
st.subheader("📈 Price vs Rating")
st.scatter_chart(df[["price", "rating"]])
